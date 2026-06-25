# Grounded Multi-Task VQA SFT Loss Improvement Spec (RL excluded) — v3

Target: Qwen3-VL 8B Instruct, LoRA SFT, curriculum pipeline (OBS -> ... -> CHR).
Goal: train **mixed** multi-task QA where some samples carry `grounding` boxes and some do
not (`grounding: []`), so the model is accurate on (a) the final `answer`, (b) localization
(IoU >= 0.80) of **referenced** objects, and (c) **which camera view** each box belongs to
(`image_idx` in 1..6). RL is out of scope; everything below is SFT-time (teacher-forced) only.

### Scope note — referring-grounding only (no detection)
The task is **referring grounding inside multi-task VQA**: the model answers a question and
grounds the objects its reasoning *references* (e.g., a truck + barriers it used to decide
"adjacent lane"). It does **not** need exhaustive object detection. Consequences:
- No detection metrics (mAP, scene-level recall/precision/F1) and no set/Hungarian training
  loss. Those were removed from the earlier draft.
- Multi-box note: a single sample may still carry several referenced boxes (the example has
  4). When that happens, eval needs a **lightweight matching** purely to compute per-box IoU
  correctly (predicted box count/order can differ from GT). This matching is keyed on
  `(image_idx, object label)` — far simpler than detection matching — and is about *correct
  measurement*, not recall optimization (Section 7).

---

## 0. Premise — current implementation (from analysis md, Part 2)

- `WSDTrainer` (`train_nuscenes_qwen3vl.py:481`) subclasses HF `Trainer` and overrides **only**
  `create_scheduler`. It does **not** override `compute_loss`. The loss is exactly what
  `Qwen3VLForConditionalGeneration.forward(..., labels=labels)` returns: standard causal-LM
  cross-entropy, shift-by-one, `ignore_index=-100`, mean reduction over non-ignored positions,
  with `num_items_in_batch` forwarded so gradient accumulation equals a larger batch.
- Label masking in `preprocess_with_system_prompt` (`:133-209`): per-turn ChatML tokenization;
  `system`/`human` -> all `IGNORE_INDEX`; `gpt` -> tokens copied except the first 3
  (`<|im_start|>`, `assistant`, `\n`). Loss target = the whole assistant JSON string
  `{"reasoning": ..., "grounding": [...], "answer": ...}` plus `<|im_end|>\n`.
- `NuScenesDataCollator.__call__` (`:345`) pads `labels` with `IGNORE_INDEX`.
- No token/field weighting, no auxiliary losses, no per-category balancing, no special handling
  for empty grounding. Optimizer `adamw_torch`, `weight_decay=0.01` (`qwenvl/train/trainer.py:316`),
  `max_grad_norm=1.0`, `bf16`, LoRA dropout 0.05, r=64, alpha=128, peak LR 2e-4,
  WSD ratios [0.10, 0.70, 0.20], 1 epoch/stage, per_device=1 x grad_accum=4.
- Stage-end eval (`sft_model_tester.run_per_category_eval`) greedily generates, extracts the
  `answer`, **string-matches** GT. bbox quality and `image_idx` are not measured today.

Ref (base objective): standard VLM SFT next-token-prediction — LLaVA, *Visual Instruction
Tuning* (Liu et al., NeurIPS 2023); Qwen2.5-VL / Qwen3-VL technical reports. [standard]

### Data format (authoritative — match exactly)
Grounding entries use Qwen3-VL native special tokens (present in 8B vocab:
`<|object_ref_start|>`=151646, `<|object_ref_end|>`=151647, `<|box_start|>`=151648,
`<|box_end|>`=151649):

```json
"grounding": [
  { "image_idx": 1, "camera": "Front-left",
    "ref": "<|object_ref_start|>truck<|object_ref_end|><|box_start|>(488,484),(744,668)<|box_end|>" }
]
```

- Coordinates `(x1,y1),(x2,y2)` in **0-1000 normalized** space inside `<|box_start|>..<|box_end|>`.
- `image_idx` in {1: Front-left, 2: Front, 3: Front-right, 4: Back-left, 5: Back, 6: Back-right};
  rear views (4,5,6) horizontally flipped before normalization. IoU/eval must use the same
  flipped convention as training.
- `grounding` may be `[]`; a sample may contain multiple entries across different `image_idx`.

---

## 1. Target design — composite, per-sample-gated loss

Principle: **base CE on all samples; grounding-specific terms gated per sample by a
`has_grounding` mask and per token by role masks.** Non-grounding samples receive only base CE
+ answer weighting + presence-gate; they are excluded from view/IoU/KLAL terms via mask
multiplication (no Python branching on data — Section 8).
Ref (per-sample task gating in multi-task instruction tuning): LISA SEG-token gating
(Lai et al., CVPR 2024). [known practice]

```
L = L_ce         (all assistant tokens; unchanged semantics)
  + w_ans   * L_ans   (answer-field tokens; ALL samples)
  + w_gate  * L_gate  (presence-gate token: "[]" vs "[{"; ALL samples)
  + lam_view * L_view (6-way classification on image_idx token; grounded samples)
  + lam_iou  * L_iou  (IoU-aware CE on coord-token spans; grounded; per-image_idx)
  + lam_klal * L_klal (KL attention loss to GT attention map; grounded; via image_idx)
```

Default weights (starting point): `w_ans=2.0`, `w_gate=1.5`, `lam_view=0.5`, `lam_iou=0.5`,
`lam_klal=0.1`. All configurable (Section 10).

---

## 2. Loss components (with references)

### 2.1 Base CE (`L_ce`) — unchanged objective, preserved grad-accum
Compute manually (do not pass `labels` to the model) to preserve `num_items_in_batch`:

```python
def _base_ce(logits, labels, num_items_in_batch):
    sl = logits[:, :-1, :].contiguous()            # predicts labels[:, 1:]
    st = labels[:, 1:].contiguous()
    loss = F.cross_entropy(sl.view(-1, sl.size(-1)).float(), st.view(-1),
                           ignore_index=-100, reduction="sum")
    denom = num_items_in_batch if num_items_in_batch is not None \
            else (st != -100).sum().clamp_min(1)
    return loss / denom
```
Ref: HF `ForCausalLMLoss`; LLaVA SFT (Liu et al., NeurIPS 2023). [standard]

### 2.2 Answer-field weighting (`L_ans`) — ALL samples
Extra CE on `"answer"` tokens to tighten loss <-> answer-accuracy correspondence; applies to
grounded and non-grounded samples (answer is the primary objective regardless of grounding).
Masked CE via `answer_token_mask`, normalized by answer-token count.
Ref: project-internal analysis (md Section 2.9 item 1); standard token-reweighting idea —
**no single canonical paper**. [project heuristic]

### 2.3 Presence gate (`L_gate`) — ALL samples
Extra CE on the decision token after `[`: target `]` (empty) vs `{` (box). Symmetric
box-presence signal; suppresses hallucinated boxes on empty samples, encourages commitment on
grounded ones. Token boundary is tokenizer-dependent, so the mask is built at preprocessing
time (Section 3).
Ref: design heuristic; object-hallucination-reduction motivation cf. Ferret (You et al.,
ICLR 2024). [heuristic + related motivation]

### 2.4 View classification (`L_view`) — grounded samples, per box
6-way classification on the `image_idx` value token. Restrict logits to the 6 single-token
digit ids "1".."6"; cross-entropy vs class `image_idx - 1`. Sharper than full-vocab CE;
supports class weights for view imbalance and direct view-accuracy monitoring.

```python
self.view_token_ids = torch.tensor([tok("1"), tok("2"), tok("3"),
                                    tok("4"), tok("5"), tok("6")])
self.view_class_weight = None   # optional (6,) inverse-frequency weights

def view_cls_loss(self, logits, image_idx_mask, image_idx_target):
    sl_mask = image_idx_mask[:, 1:]                 # shift to match logits[:, :-1]
    sl_tgt  = image_idx_target[:, 1:]
    sel = logits[:, :-1, :][sl_mask]                # (N_boxes, V)
    if sel.numel() == 0:
        return logits.new_zeros(())
    sel6 = sel[:, self.view_token_ids].float()      # (N_boxes, 6), fp32
    tgt  = sl_tgt[sl_mask]                           # (N_boxes,) in 0..5
    return F.cross_entropy(sel6, tgt, weight=self.view_class_weight)
```
Cost: gather + tiny CE; no extra forward/attention. Rank early in rollout.
Ref: **project-specific** to this 6-view egocentric setup — no direct source paper. Closest
context for multi-view / image-indexed grounding in driving VQA: DriveLM (Sima et al.,
ECCV 2024), OmniDrive (Wang et al., CVPR 2025). [project-specific + domain context]

### 2.5 IoU-aware CE (`L_iou`) — grounded coord spans, per-image_idx
Inject IoU-space smoothness that plain digit-token CE lacks (token CE is not metric-aware:
small coordinate-value errors can flip multiple digit tokens, large ones may not).
- At collation, for each GT box generate `M` pseudo boxes within a GIoU band, tokenize each in
  native `<|box_start|>(x1,y1),(x2,y2)<|box_end|>` form, precompute each pseudo box's GIoU
  weight vs its GT box (no model prediction needed for the weight).
- Coordinate-digit token CE is weighted by the box's GIoU, aligning the gradient with IoU.
- IoU/GIoU computed **per image_idx** only (cross-view matching undefined).

Tiers (document which is active):
- **Tier 1 (faithful, preferred):** concatenate `M` pseudo boxes with the GT label in a single
  forward; mask attention so pseudo boxes do not attend to each other; share the GT box's
  positional embedding (single prediction at inference). Needs attention-mask + `position_ids`
  surgery.
- **Tier 2 (fallback):** scalar GIoU-based weight on the existing GT coord-token CE.

`coord_token_mask` (Section 3) marks coordinate digits between `<|box_start|>`/`<|box_end|>`.
Ref: **R-VLM**, IoU-aware weighted cross-entropy + pseudo-box single-pass (ACL 2025 Findings).
Coordinate-as-text limitation context: Pix2Seq (Chen et al., ICLR 2022); VLM-FO1 (arXiv 2025).
[paper-backed: R-VLM verified]

### 2.6 KLAL — KL Attention Loss (`L_klal`) — grounded samples
Align attention from answer/grounding tokens to visual tokens against a GT attention map
derived from the bbox, via KL divergence, combined with NTP.
- GT attention map: project each GT box onto the ViT patch grid of its `image_idx` view (use
  `box_image_idx` to select the correct camera's visual-token sub-block in the concatenated
  6-view sequence), with a small smoothing kernel.
- Average KL over a **subset** of LLM layers (`klal_layers`); compute in fp32.
- Cost: attentions force eager attention (FlashAttention disabled) — heavy at 6 views; restrict
  to a few layers and rank last. Decide `output_attentions` from `lam_klal > 0` **once**
  (Section 8), never data-dependently.
Ref: **KLAL**, Esmaeilkhani & Latecki, *Direct Visual Grounding by Directing Attention of
Visual Tokens* (WACV 2026). Practical knobs (layer averaging, small smoothing kernel, lambda
not too large or fluency degrades) per the same paper. [paper-backed: verified]

---

## 3. Data-prep & collator contract

Build token-role masks where `labels` are constructed (`preprocess_with_system_prompt`), all
aligned to the **label** sequence (same length/padding as `labels`).

Per-sample fields the collator (`NuScenesDataCollator.__call__`) must emit:
- `labels` (B,T) long — unchanged.
- `answer_token_mask`, `gate_token_mask`, `coord_token_mask`, `image_idx_mask` (B,T) bool.
- `image_idx_target` (B,T) long — class 0..5 at `image_idx` positions, `-100` elsewhere.
- `has_grounding` (B,) bool.
- `gt_boxes` (ragged, per box `[x1,y1,x2,y2]` 0-1000, list of tensors), `box_image_idx`
  (ragged, 1..6 per box), `box_label` (ragged, object label string per box; used by eval
  matching in Section 7).
- `pseudo_boxes` (ragged: pseudo coord spans + precomputed GIoU weights; grounded only).

Optional stabilization (multi-box samples): emit GT boxes in a **deterministic order** (e.g.,
`image_idx`, then descending area, then `x1`) so autoregressive teacher forcing does not waste
capacity on an arbitrary permutation. This is a training-stability nicety for referring with
several boxes, not a detection requirement.

Notes:
- Emit ragged fields as **lists of tensors** (so `_prepare_inputs` moves them to device);
  pure-Python lists stay on CPU and must be moved in `compute_loss`.
- Assert at construction: `coord_token_mask.any(dim=1) == has_grounding` and
  `image_idx_mask.any(dim=1) == has_grounding`.

---

## 4. `compute_loss` integration

```python
class WSDTrainer(Trainer):
    def __init__(self, *args, w_ans=2.0, w_gate=1.5,
                 lam_view=0.5, lam_iou=0.5, lam_klal=0.1, klal_layers=(-1,), **kw):
        super().__init__(*args, **kw)
        self.w_ans, self.w_gate = w_ans, w_gate
        self.lam_view, self.lam_iou, self.lam_klal = lam_view, lam_iou, lam_klal
        self.klal_layers = klal_layers
        self.view_token_ids = ...        # precompute (Section 2.4)
        self.view_class_weight = None

    # create_scheduler: keep existing WSD override unchanged.

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        answer_mask   = inputs.pop("answer_token_mask")
        gate_mask     = inputs.pop("gate_token_mask")
        coord_mask    = inputs.pop("coord_token_mask")
        image_idx_m   = inputs.pop("image_idx_mask")
        image_idx_t   = inputs.pop("image_idx_target")
        has_grounding = inputs.pop("has_grounding")
        gt_boxes      = inputs.pop("gt_boxes")
        box_image_idx = inputs.pop("box_image_idx")
        pseudo        = inputs.pop("pseudo_boxes")
        labels        = inputs["labels"]

        need_attn = self.lam_klal > 0                       # decided ONCE (DDP-safe)
        outputs = model(**inputs, output_attentions=need_attn)
        logits  = outputs.logits

        loss = self._base_ce(logits, labels, num_items_in_batch)
        loss = loss + self.w_ans  * self._masked_token_ce(logits, labels, answer_mask)
        loss = loss + self.w_gate * self._masked_token_ce(logits, labels, gate_mask)
        if self.lam_view > 0:
            loss = loss + self.lam_view * self.view_cls_loss(logits, image_idx_m, image_idx_t)
        if self.lam_iou > 0:
            loss = loss + self.lam_iou * self.iou_aware_ce(logits, pseudo, coord_mask,
                                                           box_image_idx, has_grounding)
        if need_attn:
            loss = loss + self.lam_klal * self.klal_loss(outputs.attentions, self.klal_layers,
                                                         gt_boxes, box_image_idx, has_grounding)
        return (loss, outputs) if return_outputs else loss

    @staticmethod
    def _masked_token_ce(logits, labels, mask):
        sl, st, m = logits[:, :-1, :], labels[:, 1:], mask[:, 1:]
        if m.sum() == 0:
            return logits.new_zeros(())
        return F.cross_entropy(sl[m].float(), st[m], reduction="mean")
```
Every aux helper returns 0 on an empty mask (no NaN).

---

## 5. Curriculum & data mixture (non-RL)
- **Grounding warmup stage** before OBS (localization + view-selection prior first). Gate the
  transition on the IoU metric (Section 7), not CE.
- WSD needs total step count for decay placement; for the warmup stage use **cosine/constant**
  (clean early stop) OR a fixed budget with checkpoint selection at the IoU-threshold step.
- **Keep grounding aux ON across all 10 stages** (small `lam_*`) + a **grounding floor** per
  stage (minimum fraction of grounding-bearing samples) — first-line forgetting defense, as
  there is no replay.
- Optional: adaptive task weighting instead of fixed `lam_*` (Section 6).
Ref: curriculum learning (Bengio et al., ICML 2009); grounding cold-start before reasoning cf.
GETok (arXiv 2025); mixing grounding + non-grounding to preserve general ability — TWIST & SCOUT
(Bhowmik et al., ICCV 2025), Data Mixing Laws (Ye et al., ICLR 2025); LoRA forgetting-avoidance
cf. GeoChat (Kuckreja et al., CVPR 2024). [known/verified]

---

## 6. Forgetting safeguards (fixed r=64, sequential, no replay)
- Cheapest: grounding floor (Section 5).
- Bigger change (later): modular LoRA experts (grounding vs understanding) to avoid the two
  skills overwriting the same low-rank subspace.
- Optional adaptive task weighting to auto-balance grounding vs answer/reasoning.
- Keep watching the aggregator forgetting column (`aggregate_curriculum_reports.py`).
Ref: modular LoRA — MixLoRA (Shen et al., arXiv 2024, venue unconfirmed), D-MoLE (arXiv 2025,
venue unconfirmed); adaptive task weighting — uncertainty weighting (Kendall, Gal & Cipolla,
CVPR 2018), Adaptive Task Balancing for Visual Instruction Tuning (arXiv 2024, venue
unconfirmed); emergent grounding alternative — DiffLMM (ICCV 2025). [mixed confidence]

---

## 7. Evaluation changes (prerequisite — do first; referring-scoped)
Extend `sft_model_tester.run_per_category_eval` to report, as **separate columns**:
- `answer_acc` (existing string-match).
- `view_acc` — fraction of predicted boxes with correct `image_idx`.
- `grounding_acc@0.8` — referring-style accuracy: a referenced object is correct if its matched
  predicted box has IoU >= 0.8 with the GT box **in the same `image_idx`**. For **single-box**
  samples this is a direct IoU. For **multi-box** samples, match predicted -> GT boxes by
  `(image_idx, object label)` first, then IoU within matched pairs (greedy on IoU when a label
  repeats within a view). Matching exists only to compute IoU correctly when count/order differ
  from GT — it is not a detection metric.
- `grounding_format_valid` — fraction of samples whose box tokens parse.
- (optional) `referring_completeness` — fraction of referenced GT objects that received a
  matched box >= 0.8 (plus a count of spurious boxes). Lightweight referring-set check, not mAP.

These decompose answer vs view vs localization vs format so each loss term's effect is isolated,
and supply the warmup transition gate (Section 5).
Ref: referring expression comprehension accuracy@IoU tradition — RefCOCO (Yu et al., ECCV 2016;
Kazemzadeh et al., EMNLP 2014). [known]

---

## 8. Numerical / distributed correctness requirements
- **Aux normalization by the relevant count**, never full batch: view by box count, IoU by
  grounded-coord count, KLAL by grounded-sample count; `clamp_min(1)` so empty batches give 0
  not NaN. Otherwise effective `lam_*` silently scales with batch composition.
- **Grad accumulation:** `num_items_in_batch` corrects only base CE; aux terms are
  per-microbatch means (mild approximation across the window) — document it.
- **Shift alignment:** all token-position losses use `logits[:, :-1]` predicting `labels[:, 1:]`;
  masks taken in the shifted frame (`mask[:, 1:]`).
- **DDP/ZeRO:** set `output_attentions` from `lam_klal > 0` once; never toggle on data-dependent
  conditions (diverging forward graphs across ranks can deadlock).
- **dtype:** all aux terms in fp32 (IoU areas, KL stability).
- **KLAL cost:** restrict to `klal_layers`; consider recomputing only the
  (text-query x visual-key) attention slice.
- **Outlier-long samples (md 2.9 item 3):** at `model_max_length=16384` plus vision tokens, the
  longest OBS sample (~14877 chars) can be tail-truncated, silently dropping loss targets — add
  a pre-filter/assertion.

---

## 9. Phased rollout (each step measurable in isolation via Section 7)
1. **Evaluation metrics (Section 7)** — eval-only, zero training risk; unblocks the rest.
2. **Answer + presence-gate weighting** — cheapest loss change; helps all samples; suppresses FP boxes.
3. **View classification (6-way)** — cheap (no extra forward/attention); fixes view-selection.
4. **IoU-aware CE (per-image_idx)** — medium cost; within-view localization.
5. **KLAL** — most expensive; layer-subset; last / optional.
6. **Warmup stage + grounding floor** — curriculum/mixture change.
7. **Modular LoRA** — later; structural forgetting fix.

---

## 10. Config additions (`curriculum_v1.yaml`)
```yaml
loss:
  w_ans: 2.0
  w_gate: 1.5
  lam_view: 0.5
  lam_iou: 0.5
  lam_klal: 0.1
  klal_layers: [-1]
  iou_pseudo_M: 4
  iou_giou_band: [0.5, 0.95]
  view_class_weight: auto       # inverse-frequency from train image_idx histogram

curriculum:
  warmup_grounding_stage: true
  warmup_schedule: cosine
  warmup_gate_metric: grounding_acc@0.8   # referring-style (Section 7)
  warmup_gate_threshold: 0.6
  grounding_floor: 0.15
```
Wire into `qwenvl/curriculum/config.py` and per-stage env emission so each stage can override
`lam_*` and `grounding_floor`.

---

## References (consolidated, with confidence)

Paper-backed (verified in this work):
- **R-VLM** — IoU-aware weighted cross-entropy for VLM grounding. ACL 2025 Findings. (2.5)
- **KLAL / Direct Visual Grounding by Directing Attention of Visual Tokens** — Esmaeilkhani &
  Latecki. WACV 2026. (2.6)
- **TWIST & SCOUT** — balancing grounding vs understanding / forget-free tuning. Bhowmik et al.,
  ICCV 2025. (5, 6)
- **DiffLMM** — emergent grounding without grounding supervision. ICCV 2025. (6)
- **Data Mixing Laws** — Ye et al. ICLR 2025. (5)

Paper-backed (well-established):
- **LLaVA / Visual Instruction Tuning** — Liu et al. NeurIPS 2023. (0, 2.1)
- **Ferret** — You et al. ICLR 2024. (2.3 motivation)
- **LISA** — Lai et al. CVPR 2024. (1 gating)
- **Pix2Seq** — Chen et al. ICLR 2022. (2.5 context)
- **RefCOCO** — Yu et al. ECCV 2016; Kazemzadeh et al. EMNLP 2014. (7 referring eval)
- **Curriculum Learning** — Bengio et al. ICML 2009. (5)
- **Uncertainty weighting** — Kendall, Gal & Cipolla. CVPR 2018. (6)
- **DriveLM** — Sima et al. ECCV 2024; **OmniDrive** — Wang et al. CVPR 2025;
  **GeoChat** — Kuckreja et al. CVPR 2024. (2.4, 5 domain context)

Referenced but venue unconfirmed (arXiv; verify before citing):
- **VLM-FO1** (arXiv 2025) — region-token paradigm. (2.5 context)
- **GETok** (arXiv 2025) — grounding cold-start + GRPO (RL part out of scope). (5)
- **MixLoRA** (Shen et al., arXiv 2024), **D-MoLE** (arXiv 2025) — modular LoRA. (6)
- **Adaptive Task Balancing for Visual Instruction Tuning** (arXiv 2024). (6)

Project-internal / no single canonical source (honest marking):
- **Answer-field weighting** (2.2) — md Section 2.9 item 1; generic token reweighting.
- **Presence gate** (2.3) — design heuristic (hallucination-suppression motivation only).
- **View classification on image_idx** (2.4) — project-specific to the 6-view setup; no direct
  source paper (domain context: DriveLM, OmniDrive).
