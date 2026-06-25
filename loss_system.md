# Loss System Report — Curriculum-v2 Composite SFT Loss

Authoritative reference for the loss currently computed by
`CompositeWSDTrainer.compute_loss` in `train_nuscenes_qwen3vl_v2.py`,
backed by helper functions in `qwen-vl-finetune/qwenvl/train/composite_loss.py`.

This is the **frozen invariant** of every ablation experiment (B0, F1, F2,
F3, L1, L2, S7, C01–C06, C07, C08, C12). It is never varied across runs;
only data scheduling (GF, ER) and LR scheduling (per_stage_wsd, global_wsd,
per_stage_wsd_relaxed) change.

---

## 1. Top-level form

For each microbatch (per_device_train_batch_size = 1, gradient_accumulation_steps = 4
⇒ 4 microbatches per optimizer step), the trainer computes a single scalar loss:

```
L_total  =  L_base_ce
          + w_ans   · L_ans
          + w_gate  · L_gate
          + λ_view  · L_view
          + λ_iou   · L_iou
          + λ_klal  · L_klal
```

This is the scalar passed to `backward()`. By construction, when every
per-component breakdown is logged it satisfies:

```
loss == loss_base_ce + loss_answer_w + loss_gate_w + loss_view_w + loss_iou_w + loss_klal_w
```

(within float precision). If the breakdown ever diverges from `loss`, the
bug is in `compute_loss` itself, not in any config.

---

## 2. Active coefficients

From `qwen-vl-finetune/configs/curriculum_v2.yaml:loss`:

| Symbol | YAML key | Value | What the term measures |
|---|---|---|---|
| `w_ans`    | `loss.w_ans`    | **2.0** | Extra weight on the answer-field tokens |
| `w_gate`   | `loss.w_gate`   | **1.5** | Extra weight on the presence-gate token |
| `λ_view`   | `loss.lam_view` | **0.5** | 6-way restricted CE on the `image_idx` digit |
| `λ_iou`    | `loss.lam_iou`  | **0.5** | IoU-aware re-weighting of coord-token CE (Tier 2) |
| `λ_klal`   | `loss.lam_klal` | **0.0** | KL attention loss (scaffolded, disabled) |
| `iou_alpha` | `loss.iou_alpha` | **1.0** | Scaling inside the IoU weight formula `w_b = 1 + α·max(0, 1−GIoU_b)` |
| `klal_layers` | `loss.klal_layers` | `"-1"` | Layer subset for KLAL when enabled |
| `view_class_weight` | `loss.view_class_weight` | `auto` | Inverse-frequency class weights from the train `image_idx` histogram |

These are loaded via `CompositeLossConfig` (`composite_loss.py:38`) and
held constant; no stage in `curriculum_v2.yaml`'s `stages:` block
overrides them.

---

## 3. Per-term specification

### 3.1 `L_base_ce` — base causal-LM cross-entropy

Standard next-token CE on the assistant body. Computed manually rather
than via `model(labels=…)` so the divisor is `num_items_in_batch`
(matching HF transformers ≥4.46's `ForCausalLMLoss`), which makes
gradient accumulation mathematically exact instead of an
average-of-averages approximation.

**Implementation** (`composite_loss.py:base_ce`):

```python
def base_ce(logits, labels, num_items_in_batch=None):
    sl = logits[:, :-1, :].contiguous()           # shift-by-one
    st = labels[:, 1:].contiguous()
    loss = F.cross_entropy(
        sl.view(-1, sl.size(-1)).float(),         # fp32
        st.view(-1),
        ignore_index=IGNORE_INDEX,                # -100
        reduction="sum",
    )
    denom = (num_items_in_batch
             if num_items_in_batch and num_items_in_batch > 0
             else (st != IGNORE_INDEX).sum().clamp_min(1))
    return loss / denom
```

**What's masked** (set to `IGNORE_INDEX = -100`, contributing nothing):

- All `system`-turn tokens
- All `human`-turn tokens (including the expanded `<\|vision_pad\|>` blocks)
- The first 3 tokens of the `gpt` turn (`<\|im_start\|>`, `assistant`, `\n`)
- All collator padding

**What's targeted**: the assistant content tokens (the entire JSON envelope
— reasoning + grounding + answer) plus the closing `<\|im_end\|>\n`.

`L_base_ce` is the **mean NLL per non-ignored target token** across the
whole grad-accum window.

### 3.2 `L_ans` — answer-field weighting (all samples)

Extra CE applied only to tokens inside the JSON `"answer"` field's value.
Tightens the loss ↔ answer-accuracy correspondence — answers are ~1–2%
of assistant tokens but determine stage-end accuracy.

**Mask construction** (`token_role_masks.py:build_assistant_masks`): scans
the assistant turn for the 3-gram landmark `(9217, 788, 330)` = (`answer`,
`":`, ` "`), then walks forward marking tokens until one decodes with a
`"` character (the closing quote). Robust to merged-quote tokens like
`",\n` or `"\n}`.

**Loss helper** (`composite_loss.py:masked_token_ce`):

```python
def masked_token_ce(logits, labels, mask):
    sl = logits[:, :-1, :]; st = labels[:, 1:]
    m = mask[:, 1:]                                  # shift mask
    if m.sum() == 0:
        return logits.new_zeros((), dtype=torch.float32)
    return F.cross_entropy(sl[m].float(), st[m], reduction="mean")
```

**Normalization**: mean over the marked positions in this microbatch — NOT
by `num_items_in_batch`. Magnitude is per-token NLL of the answer body.

**Why `w_ans = 2.0`**: doubles the per-token weight of the answer field
relative to `L_base_ce`. Effective contribution per microbatch ≈
`2.0 × mean_NLL_over_answer_tokens`.

### 3.3 `L_gate` — presence-gate single-token CE (all samples)

Single-token CE at the position immediately after `"grounding":`. The
dataset is built with `json.dumps(..., indent=2)`, so this token is one
of exactly two ids:

| Token id | Decoded | Meaning |
|---:|---|---|
| 2278  | `' [\n'`   | grounded (non-empty list) |
| 10239 | `' [],\n'` | empty grounding |

The mask is built by finding the 3-gram landmark `(1951, 287, 788)` =
(`ground`, `ing`, `":`) and marking the next position if it's one of
those two gate ids.

Same `masked_token_ce` machinery as `L_ans`. **Normalization**: mean over
the single gate position per sample. Effective contribution per sample ≈
`1.5 × CE` on that one token.

**Why this term**: suppresses two failure modes simultaneously:

- Hallucinated boxes on empty-GT samples (model emits `' [\n'`, then
  invents entries)
- Refused grounding when GT has it (model emits `' [],\n'`, skipping the
  boxes)

### 3.4 `L_view` — 6-way view classification on `image_idx`

Restricted softmax CE over the 6 single-token view digit IDs `'1'..'6'`
(IDs `16..21`). Fires only for **grounded samples** (mask is empty
otherwise). Each grounding entry's `image_idx` value position contributes
one CE term.

**Implementation** (`composite_loss.py:view_classification_loss`):

```python
def view_classification_loss(logits, image_idx_mask, image_idx_target,
                             view_token_ids, class_weight=None):
    m = image_idx_mask[:, 1:]
    if m.sum() == 0:
        return logits.new_zeros((), dtype=torch.float32)
    sl = logits[:, :-1, :]; tgt = image_idx_target[:, 1:]
    sel = sl[m]                                     # (N_boxes, V)
    sel6 = sel[:, view_token_ids].float()           # (N_boxes, 6) fp32
    tgt_cls = tgt[m]                                # 0..5
    if class_weight is not None:
        class_weight = class_weight.to(dtype=sel6.dtype)
    return F.cross_entropy(sel6, tgt_cls,
                           weight=class_weight, reduction="mean")
```

**Mask landmark**: 5-gram `(330, 1805, 7258, 788, 220)` = (` "`, `image`,
`_idx`, `":`, ` `) → the following token is the view digit; class label =
`digit_id − 16` ∈ {0..5}.

**Why a restricted 6-way softmax** (instead of full-vocab CE): sharper
gradient signal, decouples view prediction from the rest of the
vocabulary, and supports `class_weight` for view imbalance.

**`view_class_weight: auto`**: at startup `_compute_view_inverse_frequency`
walks the training JSON and produces smoothed inverse-frequency weights
normalized to mean 1. nuScenes is heavily front-camera-biased
(`image_idx=2`), so this up-weights the rare rear views. The buffer is
built in `torch.float32` regardless of model dtype to satisfy
`F.cross_entropy`'s requirement that `weight.dtype == input.dtype`.

**Normalization**: mean over N_boxes in the microbatch. A sample with 3
boxes contributes 3 terms. Returns 0 when no grounded sample is in the
microbatch (TSS/RML scenario).

### 3.5 `L_iou` — IoU-aware coord-token CE (Tier 2)

R-VLM-style IoU-weighting on the predicted box coordinate tokens. Greedy-
decodes the predicted box at each (sample, box) group's coord-token
positions, parses it, computes GIoU vs the GT box (same `image_idx`),
detaches GIoU, and uses it as a per-box scalar weight on that box's
coord-token CE.

**Implementation** (`composite_loss.py:iou_aware_ce_tier2`):

```python
def iou_aware_ce_tier2(logits, labels, coord_token_mask, coord_box_id,
                       gt_boxes, box_image_idx, tokenizer, iou_alpha=1.0):
    ce_per_pos = F.cross_entropy(sel_logits.float(), sel_targets,
                                 reduction="none")        # (N_coord,)
    # Group coord positions by (sample, box_id)
    groups = defaultdict(list)
    for i, (s_i, b_i) in enumerate(zip(pos_sample, pos_box)):
        if b_i >= 0:
            groups[(s_i, b_i)].append(i)
    weighted_sum = torch.zeros((), dtype=torch.float32)
    for (s_i, b_i), positions in groups.items():
        gt_box = tuple(gt_boxes[s_i][b_i])
        # Greedy-decode predicted box at THIS box's coord positions
        pred_ids_b = [pred_ids[p] for p in positions]
        pred_box = parse_box_token_ids(tokenizer, pred_ids_b)
        giou = giou_xyxy(pred_box, gt_box) if pred_box else -1.0
        w_b = 1.0 + iou_alpha * max(0.0, 1.0 - giou)     # scalar in [1, 3]
        weighted_sum += ce_per_pos[idx_tensor].sum() * float(w_b)
    return weighted_sum / total_coord_tokens.clamp_min(1.0)
```

**Mask construction**:

- `coord_token_mask`: marks tokens **between** `<\|box_start\|>` and
  `<\|box_end\|>` in any grounding entry (the coordinate digits, commas,
  parentheses).
- `coord_box_id`: 0-indexed box id at each coord position, computed by
  counting `<\|box_start\|>` occurrences in token order. Used to group
  positions by their owning box.

**Per-box weight**: `w_b = 1 + α · max(0, 1 − GIoU_b)`, clamped to
`[1, 1 + 2α]`. With `α = 1.0`:

- Perfect prediction (GIoU = 1.0): `w_b = 1`
- Half-overlapping prediction (GIoU = 0.0): `w_b = 2`
- Disjoint or parse-failure (GIoU = −1.0): `w_b = 3`

The GIoU is **detached** (it's a Python float), so no gradient flows
through the IoU computation — only through the per-position CE values.
Each box's per-position CE is summed and scaled by `w_b`; the final loss
is `Σ w_b · CE_b / total_coord_tokens`.

**Normalization**: total grounded-coord token count across the
microbatch. Returns 0 when no grounded sample is in the microbatch.

**Why "Tier 2"**: the spec lists two implementation tiers
(`loss_improvement_design_v3.md:2.5`). **Tier 1** (R-VLM paper) augments
the forward pass with M pseudo-boxes + attention-mask + positional-
embedding surgery. **Tier 2** (here) skips augmentation and applies a
scalar GIoU-derived weight on the existing coord-token CE — simpler;
preserves IoU smoothness without touching the forward graph.

### 3.6 `L_klal` — KL Attention Loss (scaffolded, currently disabled)

Currently `λ_klal = 0.0`. The trainer's `__init__` reads this once and
sets `self._needs_output_attentions = (λ_klal > 0)`. Since it's 0, the
forward pass runs with `output_attentions=False`, keeping FlashAttention
active.

The function `klal_loss_stub` returns a zero tensor. To enable:

1. Replace the stub with the real KL-vs-GT-attention computation per
   Esmaeilkhani & Latecki (WACV 2026): project each GT box onto the ViT
   patch grid of its `image_idx` view, build a smoothed target attention
   map, take KL vs the model's attention weights at answer/grounding
   token positions (averaged over `klal_layers`).
2. Set `lam_klal > 0` in the YAML.

Step 2 alone triggers `output_attentions=True` which forces eager
attention (FlashAttention disabled), with a significant speed cost — so
only flip it on AFTER step 1 is implemented.

---

## 4. Token-role masks — how the loss knows where to apply each term

Built once per sample at preprocess time in
`train_nuscenes_qwen3vl_v2.py:preprocess_with_system_prompt_v2`, which
calls `qwenvl/train/token_role_masks.py:build_assistant_masks`. Each is a
boolean tensor (or long for targets) the same length as `labels`:

| Field | Shape | Purpose | Drives |
|---|---|---|---|
| `answer_token_mask` | `(B, T)` bool | True at every answer-field token | `L_ans` |
| `gate_token_mask`   | `(B, T)` bool | True at the single token after `"grounding":` | `L_gate` |
| `coord_token_mask`  | `(B, T)` bool | True at every coord-digit token inside `<\|box_start\|>...<\|box_end\|>` | `L_iou` |
| `image_idx_mask`    | `(B, T)` bool | True at every view-digit token (the `image_idx` value) | `L_view` |
| `image_idx_target`  | `(B, T)` long | Class 0..5 at `image_idx_mask` positions; `IGNORE_INDEX` elsewhere | `L_view` target |
| `coord_box_id`      | `(B, T)` long | 0-indexed box id at each coord position; `IGNORE_INDEX` elsewhere | `L_iou` per-box grouping |

Built by scanning the tokenized assistant turn for hard-coded landmark
sequences and the special box-delimiter token IDs `151648` / `151649`.
A `verify_landmarks()` startup check confirms the tokenizer vocab still
matches; on drift it fails fast with a clear error.

**Runtime assertion** (defensive): in
`NuScenesVQADatasetV2._get_item`, if `has_grounding == True` but
`coord_token_mask.any()` or `image_idx_mask.any()` is False, the loader
raises immediately with the sample's image paths and first GT box dumped
— so a future tokenizer drift cannot silently zero out `L_view` / `L_iou`.

---

## 5. Aggregation flow inside `compute_loss`

```python
# train_nuscenes_qwen3vl_v2.py:CompositeWSDTrainer.compute_loss

# 1) Forward (output_attentions decided once at trainer init from lam_klal>0)
outputs = model(**model_inputs, output_attentions=self._needs_output_attentions)
logits = outputs.logits

# 2) Base CE — exact gradient-accumulation semantics
loss = base_ce(logits, labels, num_items_in_batch=num_items_in_batch)
breakdown = {"loss_base_ce": float(loss.detach())}

# 3) Diagnostic — fraction of microbatch samples with any grounded coord token
breakdown["grounded_frac"] = float(coord_mask.any(dim=1).float().mean().detach())

# 4) Each aux term: lambda * L_aux added to total. Both raw and weighted
#    contributions logged so loss == base_ce + Σ loss_*_w by construction.
if self.loss_config.w_ans > 0:
    l_ans   = masked_token_ce(logits, labels, answer_mask)
    contrib = self.loss_config.w_ans * l_ans
    loss    = loss + contrib
    breakdown["loss_answer"]   = float(l_ans.detach())   # raw L_ans
    breakdown["loss_answer_w"] = float(contrib.detach()) # w_ans · L_ans

# (same pattern for w_gate, lam_view, lam_iou, lam_klal)

self._last_loss_breakdown = breakdown
return (loss, outputs) if return_outputs else loss
```

The per-component values land in the trainer's log dict at every
`logging_steps` interval; eval-step logs are excluded (the `log()`
override skips when any log key starts with `eval_`).

---

## 6. Per-component logging — what appears in TensorBoard / console

| Logged key | Meaning |
|---|---|
| `loss` | Total scalar passed to `backward()` (= base + Σ weighted contribs) |
| `loss_base_ce` | Base causal-LM CE term |
| `loss_answer`   | **Raw** `L_ans` (per-token mean over answer tokens) |
| `loss_answer_w` | **Contribution to total** = `w_ans · L_ans` = `2.0 · L_ans` |
| `loss_gate`     | Raw `L_gate` |
| `loss_gate_w`   | Contribution = `w_gate · L_gate` = `1.5 · L_gate` |
| `loss_view`     | Raw `L_view` (per-box mean view-CE; 0 when no grounded sample) |
| `loss_view_w`   | Contribution = `λ_view · L_view` = `0.5 · L_view` |
| `loss_iou`      | Raw `L_iou` |
| `loss_iou_w`    | Contribution = `λ_iou · L_iou` = `0.5 · L_iou` |
| `loss_klal`     | Raw `L_klal` (always 0 with `λ_klal = 0`) |
| `loss_klal_w`   | Contribution (always 0) |
| `grounded_frac` | Fraction of microbatch samples with any grounded coord token |
| `learning_rate` | WSD-scheduled LR for this step |
| `grad_norm`, `epoch`, … | Standard HF Trainer fields |

`grounded_frac` is the diagnostic that distinguishes the **two zero
modes** of the grounding-bearing aux terms (`L_view`, `L_iou`):

- `grounded_frac > 0` and `loss_view = 0` → **real bug** (the runtime
  assertion should fire first; investigate if it doesn't)
- `grounded_frac = 0` and `loss_view = 0` → **expected** (no grounded
  sample in this microbatch — common during sequential training of
  TSS/RML at ~4% natural grounded rate, less common with `--grounding_floor`
  active)

The `loss_*_w` vs `loss_*` distinction matters: `loss_*` shows the **raw
term magnitude** (useful for tracking trend within that subtask);
`loss_*_w` shows the **actual contribution** to the total loss (useful
for verifying the invariant `loss = loss_base_ce + Σ loss_*_w`).

---

## 7. Numerical / distributed correctness

| Requirement | Where handled |
|---|---|
| **fp32 for all aux losses** | Every aux helper does `.float()` on its slice before CE; `view_class_weight` built as `torch.float32` regardless of model dtype |
| **`num_items_in_batch` only on base CE** | `base_ce` uses it; aux terms use their per-mask denominator. Aux-term grad-accum-window approximation is documented (md §8) |
| **Shift alignment** | Every term uses `logits[:, :-1]` predicting `labels[:, 1:]`; masks taken in the shifted frame |
| **DDP-safe `output_attentions`** | Decided ONCE at trainer init from `lam_klal > 0`. Never toggled data-dependently → identical forward graph across ranks → no NCCL deadlock |
| **Zero-not-NaN on empty masks** | Every aux helper has an early-return `return logits.new_zeros((), dtype=torch.float32)` |
| **`clamp_min(1)` denominators** | `base_ce` and `iou_aware_ce_tier2` clamp denominators to ≥ 1 |
| **Outlier-long sample truncation** | `NuScenesVQADatasetV2` pre-filters samples whose assistant tokens exceed `max_assistant_tokens = 6000`, surfaced in the startup log |

---

## 8. What's NOT in the loss path (explicit scope)

- **No KL-distillation loss** — base model is frozen, only LoRA adapters
  train, no teacher signal.
- **No replay buffer or EWC-style regulariser** built into the loss.
  Forgetting is fought via DATA scheduling (`--replay`, `--grounding_floor`)
  + the per-stage WSD schedule, never via the loss itself.
- **No auxiliary head losses on bbox coords** — boxes are predicted purely
  as next-token over digit/comma tokens. No separate IoU-regression head.
- **No special handling for empty grounding** beyond the presence-gate
  term (`L_gate`).
- **No label smoothing** — `label_smoothing_factor = 0` throughout.
- **No KLAL execution** — wired but disabled via `λ_klal = 0` so
  FlashAttention stays on.

---

## 9. LR ↔ loss interaction

The composite loss interacts with the LR scheduler at the parameter-update
step, not in the forward pass:

```
Δθ = −LR_t · ∇θ L_total
```

For all three LR variants:

| Schedule | Behavior |
|---|---|
| **`per_stage_wsd`** (B0, F1, F2, S7) | Independent trapezoid per stage. Ratios `[0.10, 0.70, 0.20]`. LR returns to 0 at every stage boundary. |
| **`global_wsd`** (L1, C01–C03, F3, C07, C08, C12) | One continuous WSD across the sum of per-stage step budgets. Spec-pinned `warmup_frac = 0.03`, `decay_frac = 0.10`. No per-stage reset. |
| **`per_stage_wsd_relaxed`** (L2, C04–C06) | Per-stage WSD with relaxed `[0.0, 0.80, 0.20]` ratios (stage 0 keeps 0.10 warmup). Plus a sparse-stage peak factor of `0.3` for stages with grounded fraction < 0.15 (TSS, RML). |

The `loss_*_w` numbers in the logs are **pre-LR-scaling**. Step 0 of any
stage has LR = 0 (warmup start) — `loss` is computed, gradients are
computed, but the parameter update is multiplied by 0. This is why
`loss` can be nontrivial at step 0 while parameters change minimally.

---

## 10. One-paragraph plain-English summary

The model is trained with a six-term cross-entropy loss: standard
next-token CE over the entire assistant JSON envelope, plus **2.0×**
extra CE on the `answer` field's value tokens, plus **1.5×** extra CE on
the single token that decides empty-vs-non-empty `grounding` (a
hallucination/refusal gate), plus **0.5×** a sharper 6-way CE restricted
to the 6 view-digit token IDs `'1'..'6'` (with inverse-frequency class
weighting to compensate for the front-camera skew), plus **0.5×** an
IoU-aware re-weighting of the coordinate-token CE inside each box
(GIoU(pred, GT) determines the per-box weight, capped at `1 + 2·α`).
KLAL is wired but disabled (`λ_klal = 0`), so FlashAttention stays on
and there's no extra compute cost from attention-output materialization.
All aux terms compute in fp32 (cast at slicing), gate cleanly on their
λ coefficient (setting any to 0 makes that term a no-op zero tensor),
normalize by their relevant counts (per-token, per-box, or per-coord —
never full batch), and short-circuit to zero on empty masks (no NaN).
Per-component raw and weighted values are logged at every training step
so the invariant `loss = loss_base_ce + Σ loss_*_w` is mechanically
checkable. The same composite loss runs identically on training and
evaluation forward passes — HF Trainer's `prediction_step` routes
through the same `compute_loss` override.

---

## 11. Source locations

| Component | File | Key symbol |
|---|---|---|
| Coefficient defaults | `qwen-vl-finetune/configs/curriculum_v2.yaml` | `loss:` block |
| Loss config dataclass | `qwen-vl-finetune/qwenvl/train/composite_loss.py` | `CompositeLossConfig` |
| Per-term implementations | `qwen-vl-finetune/qwenvl/train/composite_loss.py` | `base_ce`, `masked_token_ce`, `view_classification_loss`, `iou_aware_ce_tier2`, `klal_loss_stub` |
| Mask construction | `qwen-vl-finetune/qwenvl/train/token_role_masks.py` | `build_assistant_masks`, `verify_landmarks` |
| Trainer integration | `train_nuscenes_qwen3vl_v2.py` | `CompositeWSDTrainer.compute_loss`, `.log` |
| Runtime mask assertion | `train_nuscenes_qwen3vl_v2.py` | `NuScenesVQADatasetV2._get_item` |

---

## 12. Frozen-invariant guarantee

This loss surface is held constant across all 16 canonical ablation
experiments. Any change here invalidates the comparison across runs —
including, critically, B0's bit-for-bit reproducibility of the v2
baseline. The ablation framework only varies:

- **data scheduling**: `--grounding_floor`, `--replay`
- **LR scheduling**: `--lr_schedule`
- **training topology**: `--mode {sequential, mixed}`

These three knobs are orthogonal to everything documented above.
