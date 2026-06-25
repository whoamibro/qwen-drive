# Curriculum-v2 Loss — Complete Specification

This is the entire loss surface as it currently runs in
`train_nuscenes_qwen3vl_v2.py:CompositeWSDTrainer.compute_loss`, with the
active config from `curriculum_v2.yaml`.

---

## 1. Top-level formula

For each microbatch (per_device_train_batch_size=1, grad_accum=4 ⇒ 4
microbatches per optimizer step), the trainer computes:

```
L_total = L_base_ce
        + w_ans   * L_ans
        + w_gate  * L_gate
        + λ_view  * L_view
        + λ_iou   * L_iou
        + λ_klal  * L_klal
```

**Active config values** (from `curriculum_v2.yaml:loss`):

| Coefficient         | Value     | What it does |
|---------------------|-----------|--------------|
| `w_ans`             | **2.0**   | Extra weight on the answer-field tokens |
| `w_gate`            | **1.5**   | Extra weight on the presence-gate token |
| `lam_view`          | **0.5**   | 6-way classification on the `image_idx` digit token |
| `lam_iou`           | **0.5**   | IoU-aware re-weighting of coord-token CE (Tier 2) |
| `lam_klal`          | **0.0**   | KLAL disabled — keeps FlashAttention on |
| `iou_alpha`         | **1.0**   | Scaling inside the IoU weight formula |
| `klal_layers`       | `"-1"`    | Layer subset (unused while lam_klal=0) |
| `view_class_weight` | `auto`    | Inverse-frequency class weights from train histogram |

These are the **stage 0 defaults**; every stage inherits them unless
overridden by `stages[i].loss`. Currently no stage overrides them, so every
stage uses the same lambda mix.

---

## 2. Per-component details

### 2.1 `L_base_ce` — base causal-LM cross-entropy

Standard next-token CE on the assistant body, **not** computed inside the
model's forward. Implementation in `qwenvl/train/composite_loss.py:base_ce`:

```python
def base_ce(logits, labels, num_items_in_batch=None):
    sl = logits[:, :-1, :].contiguous()      # predict labels[:, 1:]
    st = labels[:, 1:].contiguous()
    loss = F.cross_entropy(
        sl.view(-1, sl.size(-1)).float(),    # cast to fp32
        st.view(-1),
        ignore_index=IGNORE_INDEX,           # -100
        reduction="sum",
    )
    denom = num_items_in_batch if num_items_in_batch > 0 \
            else (st != -100).sum().clamp_min(1)
    return loss / denom
```

Critical detail: the divisor is `num_items_in_batch` (the total number of
non-ignored target tokens across the **whole grad-accum window**), forwarded
by HF Trainer. This makes gradient accumulation mathematically equivalent to
a larger batch — matches HF's `ForCausalLMLoss` semantics.

**What's masked** (set to `IGNORE_INDEX=-100`):

- All `system`-turn tokens
- All `human`-turn tokens (including `<image>` placeholders → expanded
  vision tokens)
- The first 3 tokens of the `gpt` turn (`<|im_start|>`, `assistant`, `\n`)
- Padding (set by the collator)

**What's targeted**: the assistant content tokens plus the closing
`<|im_end|>\n`. So `L_base_ce` is averaged NLL over the whole assistant JSON
envelope — reasoning + grounding + answer.

### 2.2 `L_ans` — answer-field weighting (ALL samples)

Extra mean CE on the JSON `answer` field's value tokens. From
`composite_loss.py:masked_token_ce`:

```python
def masked_token_ce(logits, labels, mask):
    sl = logits[:, :-1, :]
    st = labels[:, 1:]
    m  = mask[:, 1:]                         # shift mask into label frame
    if m.sum() == 0:
        return logits.new_zeros((), dtype=torch.float32)
    return F.cross_entropy(sl[m].float(), st[m], reduction="mean")
```

**Mask construction** (`qwenvl/train/token_role_masks.py:build_assistant_masks`):
scans the tokenized assistant turn for the landmark 3-gram `(9217, 788, 330)`
= (`answer`, `":`, ` "`); then walks forward marking tokens until it hits
one whose decoded form contains `"` (the closing quote). Robust to merged
closing-quote tokens like `'",\n'` or `'"\n}'`.

**Normalization**: mean over the marked positions in this microbatch — **not**
by `num_items_in_batch`. So the magnitude is roughly per-token NLL of the
answer body. Typical token count: 1–8 depending on category (single MCQ
letter for OBS/IDN, 2–4 letters for AAS, multi-word for open-ended).

**Why w_ans=2.0**: answers carry the highest informational density in the
loss but only ~1–2% of assistant tokens. Doubling them at the loss level
tightens the loss ↔ answer-accuracy correspondence without affecting
`num_items_in_batch` (which is base CE's denominator).

### 2.3 `L_gate` — presence-gate (ALL samples)

Single-token CE at the token immediately following `"grounding":`. The
dataset is built with `json.dumps(..., indent=2)`, so this token is one of
exactly two: **`' [\n'` (id 2278)** for grounded samples, or
**`' [],\n'` (id 10239)** for empty-grounding samples.

Same `masked_token_ce` machinery; mask is built by finding the landmark
3-gram `(1951, 287, 788)` = (`ground`, `ing`, `":`) and marking the next
position if it's one of the two gate-target tokens.

**Why a binary gate signal**: suppresses two failure modes simultaneously:

- Hallucinated boxes on empty-GT samples (model emits `' [\n'` then makes
  up entries)
- Refused grounding when there should be some (model emits `' [],\n'`
  skipping the boxes)

**Normalization**: also mean over the gate position(s). One position per
sample → mean over 1 token per sample → effectively `1.5 × CE` per sample
on that single position.

### 2.4 `L_view` — 6-way view classification on `image_idx`

This term is **grounded-samples only** (mask is empty otherwise). For each
box in each grounded sample, restricts the softmax to the 6 single-digit
view-id tokens `'1'..'6'` (token IDs 16–21) and computes cross-entropy
against the GT class.

From `composite_loss.py:view_classification_loss`:

```python
def view_classification_loss(logits, image_idx_mask, image_idx_target,
                             view_token_ids, class_weight=None):
    m = image_idx_mask[:, 1:]
    if m.sum() == 0:
        return logits.new_zeros((), dtype=torch.float32)
    sl = logits[:, :-1, :]
    tgt = image_idx_target[:, 1:]

    sel = sl[m]                                      # (N_boxes, V)
    sel6 = sel[:, view_token_ids].float()            # (N_boxes, 6) fp32
    tgt_cls = tgt[m]                                 # class 0..5
    if class_weight is not None:
        class_weight = class_weight.to(dtype=sel6.dtype)
    return F.cross_entropy(sel6, tgt_cls,
                           weight=class_weight, reduction="mean")
```

**Mask landmark**: 5-gram `(330, 1805, 7258, 788, 220)` = (` "`, `image`,
`_idx`, `":`, ` `) → the next token is a view digit (16–21) which becomes
a class label `digit_id - 16` ∈ {0..5}.

**Why a restricted 6-way softmax instead of full-vocab CE**: sharper
gradient, decouples view-prediction from the rest of the vocabulary, and
supports per-class weighting for view imbalance.

**`view_class_weight: auto`**: at startup `_compute_view_inverse_frequency`
walks the training JSON, counts the histogram over `image_idx ∈ {1..6}`,
and produces smoothed inverse-frequency weights normalized to mean 1. Cars
heavily skew toward CAM_FRONT (view 2), so this weights the rarer views
(rear cameras, view 4/5/6) up. The buffer is built as `torch.float32`
regardless of model dtype to match `F.cross_entropy`'s requirement
(the bf16 → fp32 fix from a previous session).

**Normalization**: mean over `N_boxes` in the batch. For a sample with 3
boxes (3 image_idx positions), contributes 3 terms.

### 2.5 `L_iou` — IoU-aware coord-token CE (Tier 2)

This is the most involved term. From `composite_loss.py:iou_aware_ce_tier2`:

```python
def iou_aware_ce_tier2(logits, labels, coord_token_mask, coord_box_id,
                      gt_boxes, box_image_idx, tokenizer, iou_alpha=1.0):
    # Per-position CE at every coord-digit position.
    ce_per_pos = F.cross_entropy(sel_logits.float(), sel_targets,
                                 reduction="none")        # (N_coord,)

    # Group positions by (sample, box_id) using coord_box_id.
    groups = defaultdict(list)
    for i, (s_i, b_i) in enumerate(zip(pos_sample, pos_box)):
        if b_i >= 0:
            groups[(s_i, b_i)].append(i)

    weighted_sum = torch.zeros((), dtype=torch.float32)
    for (s_i, b_i), positions in groups.items():
        gt_box = tuple(gt_boxes[s_i][b_i])
        # Greedy-decode the predicted box at THIS box's coord positions.
        pred_ids_b = [pred_ids[p] for p in positions]
        pred_box = parse_box_token_ids(tokenizer, pred_ids_b)
        giou = giou_xyxy(pred_box, gt_box) if pred_box else -1.0
        w_b = 1.0 + iou_alpha * max(0.0, 1.0 - giou)      # scalar in [1, 3]

        weighted_sum += ce_per_pos[idx_tensor].sum() * float(w_b)

    return weighted_sum / total_coord_tokens.clamp_min(1.0)
```

**What's happening**:

1. **`coord_token_mask`** marks the coordinate-digit tokens *between*
   `<|box_start|>` and `<|box_end|>` for every box in the GT JSON.
2. **`coord_box_id`** says which box (0-indexed within the sample) each
   coord token belongs to — built at preprocess time by counting
   `<|box_start|>` occurrences. Lets us group coord-token positions by
   their owning box.
3. For each (sample, box) group, we **argmax-decode** the predicted token
   IDs at those positions (teacher-forced — input is the GT prefix; logits
   are the model's prediction for the next token), parse them back to
   `(x1, y1, x2, y2)` via `parse_box_token_ids`, and compute GIoU vs the
   GT box.
4. The per-box scalar weight is `w_b = 1 + α · max(0, 1 - GIoU_b)`, clamped
   non-negative. With `α=1` and identical boxes (GIoU=1), `w_b=1` (no
   extra weight). Bad predictions (GIoU=0 or negative) get `w_b` up to 3
   (when GIoU=-1, dropped to `w_b=1+α·2 = 3`). Parse failure ⇒ treated as
   GIoU=-1.
5. The GIoU is **detached** (it's `float(w_b)`, not a tensor) so no
   gradient flows through the IoU computation itself — only through the
   per-position CE values.
6. Each box's per-position CE is summed and scaled by `w_b`. The final
   loss is `sum / total_coord_tokens`.

**Why "Tier 2"**: the design spec (`loss_improvement_design_v3.md:2.5`)
lists two implementation tiers. **Tier 1** (R-VLM paper) appends `M`
pseudo-boxes to a single forward with attention-mask surgery and shared
positional embeddings. **Tier 2** (what we have) skips the pseudo-box
augmentation and just uses GIoU-derived per-box weights on the existing
coord-token CE. Simpler; preserves IoU smoothness without modifying the
forward graph.

**Normalization**: total grounded-coord token count (sum of all positions
in `coord_token_mask` across the batch). So when no sample in the batch is
grounded → returns 0, no NaN.

### 2.6 `L_klal` — KL attention loss (DISABLED)

Currently `lam_klal=0.0`. The trainer's `__init__` reads this once and sets
`self._needs_output_attentions = (lam_klal > 0)`. Since it's 0, the forward
pass runs with `output_attentions=False`, keeping FlashAttention active.

The stub `klal_loss_stub` returns a zero tensor. To enable KLAL you'd need
to (a) replace the stub with a real KL-vs-GT-attention computation, and
(b) set `lam_klal > 0` in the YAML — that triggers `output_attentions=True`,
which forces eager-attention fallback (FA off) and slows the forward pass.

---

## 3. Token-role masks — how the loss knows where to apply each term

Built once per sample at preprocess time in
`train_nuscenes_qwen3vl_v2.py:preprocess_with_system_prompt_v2` calling
`qwenvl/train/token_role_masks.py:build_assistant_masks`. Each is a boolean
tensor (or long for the targets) the same length as `labels`:

| Field               | Shape          | Purpose |
|---------------------|----------------|---------|
| `answer_token_mask` | `(B, T)` bool  | True at every answer-field token |
| `gate_token_mask`   | `(B, T)` bool  | True at the single token after `"grounding":` |
| `coord_token_mask`  | `(B, T)` bool  | True at every coord-digit token inside `<\|box_start\|>...<\|box_end\|>` |
| `image_idx_mask`    | `(B, T)` bool  | True at every view-digit token (the value of `image_idx`) |
| `image_idx_target`  | `(B, T)` long  | Class 0..5 at `image_idx_mask` positions; `IGNORE_INDEX` elsewhere |
| `coord_box_id`      | `(B, T)` long  | 0-indexed box id at each coord position; `IGNORE_INDEX` elsewhere |

These are built by scanning the tokenized assistant turn for hard-coded
landmark sequences (`token_role_masks.py:GROUNDING_LANDMARK`,
`ANSWER_LANDMARK`, `IMAGE_IDX_LANDMARK`) and the special box-delimiter
token IDs `151648`/`151649`. A `verify_landmarks()` startup check confirms
the tokenizer vocab still matches; on drift it fails fast with a clear
error.

**Runtime assertion** (added during the bug-fix pass): in
`NuScenesVQADatasetV2._get_item`, if `has_grounding=True` but
`coord_token_mask.any()` or `image_idx_mask.any()` is False, the loader
raises immediately with the sample's image paths and first GT box dumped —
so a future tokenizer drift can't silently zero out `L_view` / `L_iou`.

---

## 4. Aggregating the terms

The flow in `CompositeWSDTrainer.compute_loss`:

```python
# 1) Forward
outputs = model(**model_inputs, output_attentions=self._needs_output_attentions)
logits = outputs.logits
device = logits.device

# 2) Base CE — exact gradient-accumulation semantics via num_items_in_batch
loss = base_ce(logits, labels, num_items_in_batch=num_items_in_batch)
breakdown = {"loss_base_ce": float(loss.detach())}

# 3) grounded_frac monitoring — fraction of microbatch samples with any
#    grounded coord token; useful for spotting "loss_view=0 because no
#    grounding" vs "loss_view=0 because of a bug".
grounded_in_batch = float(coord_mask.any(dim=1).float().mean().detach())
breakdown["grounded_frac"] = grounded_in_batch

# 4) Each aux term: lambda * L_aux added to total, both raw and weighted
#    contributions logged so they SUM to total.
if self.loss_config.w_ans > 0:
    l_ans = masked_token_ce(logits, labels, answer_mask)
    contrib = self.loss_config.w_ans * l_ans
    loss = loss + contrib
    breakdown["loss_answer"]   = float(l_ans.detach())     # raw L_ans
    breakdown["loss_answer_w"] = float(contrib.detach())   # w_ans * L_ans

# (same pattern for w_gate, lam_view, lam_iou, lam_klal)
```

**Invariant by construction**:

```
loss == loss_base_ce + loss_answer_w + loss_gate_w + loss_view_w + loss_iou_w + loss_klal_w
```

(within float precision). If you ever see this break in a log, the bug is
in `compute_loss`, not in your config.

---

## 5. Per-component logging — how to read it

The `_last_loss_breakdown` dict is merged into HF Trainer's `log()` calls,
but **only on training-step logs**. Eval-step logs (anything with a key
starting `eval_`) are skipped to avoid the cross-category bleed-through bug
from a previous session.

For each training step at `logging_steps=1`, TensorBoard/console will show:

| Logged key              | What it is |
|-------------------------|------------|
| `loss`                  | Total scalar passed to `backward()` (= sum of weighted contributions) |
| `loss_base_ce`          | Base causal-LM CE term |
| `loss_answer`           | Raw `L_ans` (per-token mean over answer tokens) |
| `loss_answer_w`         | Contribution to total = `w_ans × L_ans` |
| `loss_gate`             | Raw `L_gate` |
| `loss_gate_w`           | Contribution = `w_gate × L_gate` |
| `loss_view`             | Raw `L_view` (per-box mean view-CE; 0 when no grounded samples in batch) |
| `loss_view_w`           | Contribution = `lam_view × L_view` |
| `loss_iou`              | Raw `L_iou` |
| `loss_iou_w`            | Contribution = `lam_iou × L_iou` |
| `grounded_frac`         | Fraction of microbatch samples with any grounded coord token |
| `learning_rate`         | WSD-scheduled LR for this step |
| `epoch`, `grad_norm`, … | Standard HF Trainer fields |

`grounded_frac` is the diagnostic that distinguishes legitimate-zero from
broken-zero on the aux terms:

- `grounded_frac > 0` and `loss_view = 0` ⇒ real bug (the runtime assertion
  should fire first)
- `grounded_frac = 0` and `loss_view = 0` ⇒ expected (no grounded sample in
  this microbatch — common for TSS/RML at 4% grounding rate)

---

## 6. Numerical correctness — what's already handled

All from the spec's §8 ("Numerical / distributed correctness requirements"):

| Requirement                          | Where handled |
|--------------------------------------|---------------|
| **fp32 for all aux losses**          | Every aux helper does `.float()` on its slice before CE; `view_class_weight` buffer built as `torch.float32` regardless of model dtype |
| **`num_items_in_batch` only on base CE** | Only `base_ce` uses it; aux terms use their per-mask denominator (md §8 documents this as a mild approximation across grad-accum window) |
| **Shift alignment**                  | Every term uses `logits[:, :-1]` predicting `labels[:, 1:]` with masks taken in the shifted frame |
| **DDP-safe `output_attentions`**     | Decided ONCE at trainer init from `lam_klal > 0`. Never toggled data-dependently. All ranks build identical forward graphs. |
| **0-not-NaN on empty masks**         | Every aux helper has an early-return `return logits.new_zeros((), dtype=torch.float32)` |
| **`clamp_min(1)` denominators**      | `base_ce` and `iou_aware_ce_tier2` clamp their denominators to ≥1 |
| **Outlier-long sample truncation**   | `NuScenesVQADatasetV2` pre-filters samples whose assistant tokens exceed `max_assistant_tokens=6000` (md §8 item 6); also visible in the startup log |

---

## 7. What `compute_loss` does NOT include

Explicitly out of scope, documented:

- **No KL-distillation loss** — base model is frozen, only LoRA adapters
  train, no teacher signal.
- **No replay buffer / EWC against forgetting**. Each stage only sees its
  own category's slice; forgetting is measured (by the aggregator's
  `forgetting` column) but not prevented.
- **No auxiliary head losses on bbox coords** — boxes are predicted purely
  as next-token over digit/comma tokens. No IoU-regression head.
- **No special handling for empty grounding** beyond the presence-gate
  term.
- **No label smoothing** — `label_smoothing_factor=0` everywhere.
- **No KLAL** — wired but execution disabled.
- **No grounding-floor / data-mixture rebalancing** — deferred follow-up
  (md §9 step 6 / P2-5).

---

## 8. How LR shapes the actual update

The composite loss interacts with the WSD scheduler in a non-obvious way.
For curriculum-v2 each stage runs its own LR cycle:

```
peak_lr = 2e-4
wsd = [0.10, 0.70, 0.20]    # warmup 10% → stable 70% → decay 20%
```

So at step 0 of every stage, LR=0 ⇒ first step's `loss` value moves
nothing. Through warmup the per-component contributions are scaled by an LR
factor sweeping 0→1. Decay symmetrically sweeps 1→0. The `loss_*_w` numbers
you see in the log are the **pre-scaling** contributions; the actual
parameter update is `LR_t × ∇loss`.

That's why you might see `loss` numbers look identical at step 0 and step 1
(warmup) but parameters change minimally — the LR multiplier is the
bottleneck early.

---

## 9. Practical summary in one paragraph

You're training Qwen3-VL 8B's LoRA adapters with a six-term cross-entropy
loss: standard next-token CE over the whole assistant JSON, plus **2.0×**
extra CE on the answer-field tokens, plus **1.5×** extra CE on the single
token that decides empty-vs-non-empty grounding, plus **0.5×** a sharper
6-way view-classification CE restricted to the 6 view-digit token IDs (with
inverse-frequency class weighting to compensate for the front-camera skew),
plus **0.5×** an IoU-aware re-weighting of the coordinate-token CE inside
each box (predicted vs GT GIoU determines the per-box weight, capped at +2).
KLAL is wired but turned off so FlashAttention stays on. All aux terms are
computed in fp32, gated by their lambdas (set 0 to disable cleanly),
normalized by their relevant counts (per-token, per-box, or per-coord —
never full batch), and protected against NaN on empty masks. Per-component
values plus their weighted contributions are logged at every training step
so you can verify `loss = loss_base_ce + sum(loss_*_w)` holds and watch
`grounded_frac` to distinguish "no grounding in this microbatch" from
"broken masks". All the same logic is identical at train and eval time
(HF Trainer routes through `compute_loss` in both).
