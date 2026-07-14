# Claude Code Task — Two Loss-Design Gaps: Grounding Completeness & View-Selection

## Context & framing (read first)

Two gaps were identified by inspecting eval sanity dumps on **multi-box, multi-view**
samples. Neither is a numerical bug — the existing composite loss is computed correctly.
Both are **missing supervision axes**: the loss optimizes the wrong thing relative to the
observed failures.

Concrete failing case (OBS val_idx=1):
```
GT boxes:   img4·car, img5·car, img2·trailer     (3 boxes across 3 views)
PRED boxes: img2·trailer (IoU 1.000 OK), + img2·truck, img2·car  (2 spurious)
Result:     img4 car MISSED, img5 car MISSED, answer flipped No→Yes
```
The model nails the front-view box but **omits both rear-view boxes** and **hallucinates
two front-view boxes**, then gets the answer wrong because it never "saw" the rear cars.

**Root cause — why the current loss does not catch this:**
- `L_view` and `L_iou` are **teacher-forced**: they only act on token positions where a GT
  box *exists* in the label sequence. A box the model fails to emit has no coord tokens and
  no image_idx token, so these terms apply **zero penalty** to omissions.
- `L_gate` (presence gate) only decides `[]` vs `[{` — i.e. "any box vs none." This sample
  emitted a box, so the gate passes. It is blind to "2 of 3 boxes missing."
- `base CE` is the *only* term that touches the missing boxes (it teacher-forces the full GT
  sequence), but that signal is diluted across reasoning/answer tokens.

**Scope guardrail (do not cross):** the spec is deliberately *referring-grounding only — no
detection, no mAP, no set-level recall optimization, no Hungarian/set loss* (spec §scope
note). The tasks below add **lightweight per-view presence signal + measurement +
data-sampling**, NOT a detection head or set-matching training loss. If an implementation
starts needing Hungarian matching or a detection loss, STOP — that is out of scope; raise it
instead.

**Hard invariants (unchanged):** do not alter base CE semantics, the existing
`w_ans/w_gate/lam_view/lam_iou` term math, grad-accum (`num_items_in_batch`), shift
alignment, DDP `output_attentions` single-decision, or fp32 aux. All new behavior is
**config-gated, default-off**, so an unchanged config reproduces current training
bit-for-bit.

---

## Task ordering (do in this order — stabilize, then measure, then optimize)

```
T0  Deterministic    →  T1  Measurement   →  T2  Per-view presence  →  T3  View-stratified
    box ordering         (eval-only)           loss                     sampling
(data preproc,           (no training)         (smallest loss change)   (data, no loss change)
 no loss, no eval)
        └─────────────── re-evaluate after EACH, in isolation ───────────────┘
```

---

## T0 — Deterministic GT box ordering (data preprocessing) [P0, do first]

**Why.** Grounding boxes are semantically **permutation-invariant** — `img4·car, img5·car,
img2·trailer` means the same thing in any order. Eval already respects this (matching is keyed
on `(image_idx, label)`, order-independent). But **teacher-forced SFT is order-dependent**: the
model generates boxes as an autoregressive token sequence and is trained to reproduce the GT's
*specific* order via next-token prediction. If the GT order is arbitrary (e.g. whatever order
the Qwen3-VL-235B teacher emitted during auto-labeling), the model is forced to memorize a
meaningless permutation and wastes capacity on it.

**Fix.** Sort the GT grounding boxes into a **deterministic canonical order** at collation /
data-prep time, before tokenization:

```
sort key:  (image_idx asc, area desc, x1 asc)
```
so the ordering becomes a *predictable function* the model can learn, instead of noise. This is
exactly the "optional stabilization" already noted in loss spec §3 — promote it to a real step.

**Constraints:**
- **Data preprocessing only** — does NOT touch the loss math and does NOT change eval. The
  `(image_idx, label)` eval matching is order-independent, so metrics are unaffected by design;
  this only changes the *target token order* the model is trained to produce.
- Apply the SAME sort consistently to: the serialized `grounding` JSON in the label sequence
  AND every per-box ragged field (`gt_boxes`, `box_image_idx`, `box_label`, `pseudo_boxes`) so
  all per-box masks/targets stay aligned to the reordered sequence.
- Config-gated `data.deterministic_box_order: true` (default **false** = current behavior), so
  an off config reproduces current training bit-for-bit. Recommend turning it **on** as the new
  default once verified.
- This is **not** a permutation-invariant set loss. The point is to make order learnable by
  *fixing* it, which is far cheaper than Hungarian/set matching and keeps the autoregressive
  token paradigm intact (see scope guardrail).

**First check before implementing:** inspect a few multi-box GT samples and determine whether
boxes are *already* emitted in a stable order or in the teacher's arbitrary order. If already
stable, T0 is a no-op (record that and skip). If arbitrary, T0 is a near-free stability win.

**Acceptance:** with the flag on, multi-box GT label sequences are sorted by the canonical key;
all per-box ragged fields remain index-aligned to the reordered boxes (add an assertion);
eval metrics are unchanged vs flag-off on a fixed adapter (proves eval order-independence);
training loss curves are equal or smoother (no capacity wasted on permutation).

---

## T1 — Measurement first (eval-only, zero training risk) [P0]

The completeness failure is currently invisible — it is blended into `grounding_acc@0.8`.
Turn on and surface the per-component breakdown so the gap is quantified before any loss
change. Extend `sft_model_tester.run_per_category_eval`.

Add these **per-category** columns (some already specced in §7 as "optional" — make them
first-class):

| Column | Definition |
|---|---|
| `referring_completeness` | mean over samples of (# GT referenced objects that got a matched pred box with IoU≥0.8 in the same image_idx) / (# GT referenced objects). Missing boxes drive this down. |
| `n_missing_boxes` | mean per-sample count of GT boxes with no matched pred box. |
| `n_spurious_boxes` | mean per-sample count of pred boxes not matched to any GT box. |
| `per_view_recall` | 6-length vector: for each image_idx 1..6, fraction of GT boxes in that view that got a matched pred box (IoU≥0.8, same view). **This isolates the rear-view-omission pattern.** |
| `view_confusion[6×6]` | counts of (GT image_idx → PRED image_idx) over matched-by-label boxes, so front-bias / adjacent-view confusion is visible. |

Matching stays exactly as in §7: by `(image_idx, label)`, greedy on IoU when a label repeats
— **for measurement only**, not a detection metric. Do not add mAP.

**Acceptance:** running eval on the existing F3/baseline adapters prints these columns;
`per_view_recall` shows the rear views (4,5,6) lower than front (2), and `view_confusion`
shows mass on the image_idx=2 column. This confirms the diagnosis quantitatively before any
loss change.

---

## T2 — Per-view presence loss `L_vpresence` (lightweight completeness signal) [P1]

A **6-way per-view presence** term that penalizes omitting a box in a view that should have
one (and emitting one in a view that should not) — WITHOUT set matching or detection.

**Definition (per sample):**
- GT supplies a 6-dim binary target `view_has_box[1..6]` = does this sample have ≥1 GT box in
  view v.
- The model must expose a comparable 6-dim presence prediction. Implement as the **cheapest
  faithful proxy**: from the teacher-forced grounding region, derive per-view presence from
  the `image_idx` tokens the model commits to, OR add 6 lightweight presence read-outs at the
  `grounding` field start. **Prefer the token-derived proxy** to avoid architecture changes;
  if that is not cleanly separable, raise it before adding heads.
- Loss = mean binary cross-entropy over the 6 views: `BCE(pred_view_presence, view_has_box)`.
  This is symmetric → penalizes **both** missing a needed view (the rear-view omission) and
  adding an un-needed view (front-view spurious).

**Critical constraints:**
- This is a **per-view binary** signal (6 values), NOT a per-box or set-level signal. It does
  not match individual boxes. It says only "this view should/shouldn't contain grounding."
  That keeps it inside the no-detection scope while directly attacking the omission failure.
- Config-gated: `loss.lam_vpresence` (default **0.0** = off). Add to the composite as
  `+ lam_vpresence * L_vpresence`. Suggested first non-zero value: **0.5**.
- Normalize by number of samples (clamp_min 1); fp32; no data-dependent branching (mask
  multiplication only) — same DDP rules as other aux terms.
- It applies to **all samples** (empty-grounding samples have `view_has_box = all-zeros`, so
  the term also discourages hallucinated boxes there — complements `L_gate`).
- Build the `view_has_box` target in the collator alongside the existing masks; assert
  `view_has_box.any(dim=1) == has_grounding`.

**Acceptance:** with `lam_vpresence=0.5`, `referring_completeness` and `per_view_recall[4,5,6]`
(from T1) rise vs the `lam_vpresence=0` run, while `answer_acc` does not regress. Log
`loss_vpresence` (raw) and `loss_vpresence_w` (=lam·L) like the other components.

---

## T3 — View-stratified grounding sampling (data, no loss change) [P1]

The omissions concentrate in **rear views (4,5,6)** — a data-exposure imbalance, not just a
loss problem. Extend the existing grounding-floor / sampling machinery so rear-view grounding
is not starved.

- Add `data.grounding_floor.view_stratified: bool` (default **false**). When true, the
  grounding-floor top-up pool is drawn **per-view-uniform**: instead of sampling grounding
  -bearing samples uniformly, stratify by the GT box's `image_idx` so each of the 6 views gets
  a fair share of the floor budget. Reuse the per-category-uniform two-stage sampler pattern
  (the same fix applied in the sampler issue report) but keyed on `image_idx`.
- This is purely a **sampling** change; it does not touch the loss. Budget invariant: same
  total optimizer steps; it re-weights *which* grounding samples appear, not how many.
- Verify with an offline exposure check (extend `verify_sampler`): realized per-view share of
  grounding-bearing draws is ~uniform across the 6 views within tolerance.

**Acceptance:** `verify_sampler` shows per-view grounding exposure ~1/6 each (within
tolerance); after training, `per_view_recall` for views 4/5/6 improves relative to T2-only.

---

## What is explicitly NOT in scope (reaffirm)

- No detection head, no coordinate regression head.
- No Hungarian / set-level matching **in the training loss** (matching stays eval-only, for
  IoU measurement).
- No mAP / scene-level recall-precision **optimization** (measurement columns in T1 are fine;
  optimizing them directly is not).
- No change to `L_iou`'s within-view math. (Note for the record: `L_iou` optimizes within-view
  coordinate accuracy, which is already strong — trailer IoU 1.000 in the failing case. The
  binding constraints are **view selection** and **completeness**, which T1–T3 target; `L_iou`
  cannot reach either, so do not increase `lam_iou` as a fix for this.)

---

## Relationship to existing view work

T2/T3 complement the earlier view-selection plan (class-balanced `L_view`, view-aware
encoding). Ordering across the whole effort:
1. `L_view` class-balanced + view confusion matrix (already specced).
2. **T1 measurement** (this doc) — quantify completeness + per-view recall.
3. **T2 per-view presence** — penalize omission/hallucination per view.
4. **T3 view-stratified sampling** — fix rear-view data starvation.
5. (later) view-aware positional encoding — deeper view-perception fix.

`L_view` fixes *which view a emitted box claims*; `L_vpresence` (T2) fixes *whether a box is
emitted per view at all*. They are complementary, not redundant.

---

## Config additions (all default-off)

```yaml
data:
  deterministic_box_order: false   # T0: sort GT boxes (image_idx, area desc, x1); recommend true
loss:
  lam_vpresence: 0.0          # T2: per-view (6-way) box-presence BCE; try 0.5
data:
  grounding_floor:
    view_stratified: false    # T3: stratify floor top-up per image_idx (1..6)
eval:
  report_completeness: true   # T1: emit referring_completeness, n_missing/spurious,
                              #     per_view_recall, view_confusion
```

## Acceptance summary
- T0: GT boxes sorted by canonical key with on flag; per-box ragged fields stay index-aligned;
  eval metrics unchanged vs flag-off (proves eval order-independence); loss curves equal/smoother.
- T1: completeness/per-view-recall/confusion columns appear; rear-view recall measurably
  lower than front — diagnosis quantified.
- T2: `lam_vpresence=0.5` raises completeness + rear recall, no answer regression; components
  logged raw and weighted.
- T3: per-view grounding exposure ~uniform in `verify_sampler`; rear recall improves further.
- Global: all knobs default-off reproduce current training bit-for-bit; no Hungarian/detection
  loss introduced.
```
