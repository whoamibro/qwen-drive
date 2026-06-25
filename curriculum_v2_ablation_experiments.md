# Curriculum-v2 — Forgetting & LR Ablation Experiments

Implementation spec for training-technique experiments on top of the
**curriculum-v2 composite loss**, targeting two observed problems:

1. **Catastrophic forgetting of the grounding skill** in grounding-sparse
   stages. Macro `format_valid` collapses to ≈0.03 and macro
   `grounding_acc@0.8` to ≈0.005 right after the two grounding-sparse stages
   (TSS, 7 grounded / RML, 10 grounded), then recovers at the next
   grounding-rich stage (DRA, 124 grounded). `answer_acc` does **not** collapse
   — the failure is grounding-specific.
2. **Per-stage WSD LR schedule** (10 independent warmup→stable→decay cycles,
   each re-warming to peak 2e-4) amplifies (1): a full high-LR cycle on a
   ~3.5%-grounded stage overwrites the shared LoRA weights.

> **Hard invariant — loss is frozen for every run.** All experiments use the
> curriculum-v2 `Curri_Loss Setup`
> (`L_base_ce + w_ans·L_ans + w_gate·L_gate + λ_view·L_view + λ_iou·L_iou +
> λ_klal·L_klal`) with active config
> `w_ans=2.0, w_gate=1.5, λ_view=0.5, λ_iou=0.5, λ_klal=0.0,
> view_class_weight=auto`. We vary **only** data scheduling and LR scheduling.
> Do not modify `composite_loss.py` semantics or these coefficients.

---

## 0. Code touch-points

| Concern | File / symbol |
|---|---|
| Stage orchestration | `qwen-vl-finetune/scripts/run_curriculum.sh` |
| Config schema + loader | `qwen-vl-finetune/configs/curriculum_v2.yaml`, `qwenvl/curriculum/config.py` |
| Trainer | `train_nuscenes_qwen3vl_v2.py::CompositeWSDTrainer` |
| LR scheduler | `qwenvl/train/wsd_scheduler.py` + `CompositeWSDTrainer.create_scheduler` |
| Dataset / sampler | `NuScenesVQADatasetV2` (+ collator) |
| Splits / eval subsets | `nuscenes_pipeline/postprocessing/split_sft_by_category.py`, `build_eval_subset.py` |
| Stage-end eval | `nuscenes_pipeline/modules/sft_model_tester.py::run_per_category_eval` |
| Cross-run aggregator | `hf_dataset_train/aggregate_curriculum_reports.py` |

All new behavior is **config-gated and default-off** — an empty/old config
reproduces `B0` exactly.

---

## 1. Baseline (`B0`)

- **Training mode:** `sequential` — 10 stages
  `OBS→IDN→AAS→SRO→TSS→RML→DRA→RWP→ESC→CHR`, each warm-started from the previous
  adapter, 1 epoch on that category's slice only.
- **Loss:** `Curri_Loss Setup` (frozen).
- **LR schedule:** `per_stage_wsd` — each stage warmup(0.10)→stable(0.70)→
  decay(0.20), peak `2e-4`, LR→0 at every stage boundary.

---

## 2. Building blocks (orthogonal knobs)

An experiment = `training.mode` + `lr.schedule` + `grounding_floor{on,off}` +
`replay{on,off}`.

### 2.1 Training mode — `training.mode`
```yaml
training:
  mode: sequential          # baseline: 10 per-category stages, warm-started
  # mode: mixed             # all categories sampled every step, ONE run
  mixed:
    sampler: uniform        # or: difficulty_weighted (Data Mixing Laws, ICLR 2025)
    mixture_weights: null   # dict {OBS:w,...}; null = uniform. See Appendix A for
                            # the difficulty-weighted initial-weight design.
```

### 2.2 Grounding Floor (GF) — `data.grounding_floor`
```yaml
data:
  grounding_floor:
    enabled: false
    min_fraction: 0.25      # ≥25% of each batch is grounding-bearing
    pool: global            # floor samples drawn from grounding-bearing samples
                            # across ALL 10 categories
    sweep: [0.15, 0.25, 0.35]
```
Weighted sampler up-samples grounding-bearing samples (target has ≥1 valid box;
reuse the `has_grounding` flag the loss masks use) to `min_fraction`. The rest
follows the stage's natural distribution. **Total optimizer steps unchanged**
(resampling, not extra steps).

### 2.3 Experience Replay (ER) — `data.replay`
```yaml
data:
  replay:
    enabled: false
    fraction: 0.10          # 10% of each stage's stream from PRIOR categories
    buffer: prior_stages    # sequential: categories 0..i-1 ; stage 0 = no replay
    policy: uniform
    sweep: [0.05, 0.10, 0.15]
```

### 2.4 LR schedule — `lr.schedule`
```yaml
lr:
  schedule: per_stage_wsd            # baseline
  # schedule: global_wsd
  # schedule: per_stage_wsd_relaxed

  global_wsd:                        # ONE cycle over the whole run
    warmup_frac: 0.03                # warmup once over first 3% of TOTAL steps
    decay_frac:  0.10                # single decay over LAST 10% of TOTAL steps
    peak_lr: 2.0e-4                  # stable in between; scheduler ignores stage boundaries

  per_stage_wsd_relaxed:             # keep per-stage cycles, but relax
    warmup_ratio: 0.0                # warm-started stages need ~no fresh warmup (stage 0 may keep 0.10)
    stable_ratio: 0.80
    decay_ratio:  0.20
    peak_lr: 2.0e-4
    sparse_stage_peak_factor: 0.3    # grounding-sparse stages use peak×0.3
    sparse_threshold: 0.15           # "sparse" = stage grounded-fraction < 0.15 (TSS≈0.035, RML≈0.05)
```
**`global_wsd` caveat:** intermediate (stage-boundary / step-fraction) eval
checkpoints come from the high-LR stable phase and are **not annealed** — their
absolute numbers run noisier/slightly worse than an annealed checkpoint.
Compare *trends* and the *final annealed* checkpoint, not raw intermediates,
when ranking against per-stage-WSD runs.

> **Mixed mode has no per-category stages**, so `per_stage_*` schedules are
> undefined there. **The only well-defined LR for `mixed` is `global_wsd`**,
> which is therefore mixed mode's mandatory/default schedule.

---

## 3. Experiment matrix — factorial grid

Rows = data scheduling, columns = LR schedule. Cells = experiment IDs.
`—` = not applicable. This grid makes coverage (and one gap) visible at a
glance.

### Sequential block (`training.mode = sequential`)

| Data scheduling      | LR: per-stage WSD<br>(**baseline**) | LR: global WSD | LR: per-stage relaxed |
|----------------------|:-----------------------------------:|:--------------:|:---------------------:|
| **none**             | `B0`                                | `L1`           | `L2`                  |
| **+ GF**             | `F1`                                | `C01`          | `C04`                 |
| **+ ER**             | `F2`                                | `C02`          | `C05`                 |
| **+ GF + ER**        | `S7`                                | `C03`          | `C06`                 |

The sequential block is now a **complete 4×3 factorial** (no empty cells).

### Mixed block (`training.mode = mixed`, LR is always global WSD)

| Data scheduling | LR: global WSD (only option) |
|-----------------|:----------------------------:|
| **none**        | `F3` *(= mixed baseline)*     |
| **+ GF**        | `C07`                        |
| **+ ER**        | `C08`                        |
| **+ GF + ER**   | `C12`                        |

### ID → user-list mapping (so nothing is lost)

| Your item | ID | Your item | ID |
|---|---|---|---|
| Forgetting (1) GF | `F1` | Combo (4) base+GF+PSW-R | `C04` |
| Forgetting (2) ER | `F2` | Combo (5) base+ER+PSW-R | `C05` |
| Forgetting (3) Mixed | `F3` | Combo (6) base+GF+ER+PSW-R | `C06` |
| LR (1) global WSD | `L1` | Combo (7) Mixed+GF | `C07` |
| LR (2) per-stage relaxed | `L2` | Combo (8) Mixed+ER | `C08` |
| Combo (1) base+GF+GW | `C01` | Combo (9) Mixed+GW | `C09` **= F3** |
| Combo (2) base+ER+GW | `C02` | Combo (10) Mixed+GF+GW | `C10` **= C07** |
| Combo (3) base+GF+ER+GW | `C03` | Combo (11) Mixed+ER+GW | `C11` **= C08** |
| | | Combo (12) Mixed+GF+ER+GW | `C12` |
| | | **`S7`** (supplement, §4.2) | base+GF+ER+**per-stage WSD** |

---

## 4. Logical-consistency analysis (requested)

### 4.1 Redundancies — three duplicate pairs in the mixed block
Because mixed mode's only LR is `global_wsd` (§2.4):

- **Combo (9) `Mixed + global WSD` = the mixed baseline `F3`.** Same config.
- **Combo (10) `Mixed + GF + global WSD` = Combo (7) `Mixed + GF`** → `C07`.
- **Combo (11) `Mixed + ER + global WSD` = Combo (8) `Mixed + ER`** → `C08`.

**Resolution:** run each once; alias the duplicate IDs (`C09→F3`, `C10→C07`,
`C11→C08`). This removes 3 redundant runs. Nothing is lost — the global-WSD
label is already implied by `mixed`.

### 4.2 Coverage gap — FILLED by `S7`
The sequential block was missing one factorial cell: `GF+ER` on the **baseline
per-stage WSD** LR (`GF+ER` otherwise appears only with global WSD `C03` and
per-stage relaxed `C06`). Without it you could not isolate *"does GF+ER help on
its own, before any LR change?"* from the LR effect.

**`S7` = `baseline + GF + ER + per-stage WSD`** now fills this cell. The
sequential block is a **complete 4×3 factorial** (12 cells), so data-scheduling
effects (`none / GF / ER / GF+ER`) and LR effects (`PSW / GW / PSW-R`) are fully
separable.

### 4.3 Single-knob controls are complete
`B0` (control), `F1`/`F2`/`F3` (each forgetting technique alone), `L1`/`L2`
(each LR change alone) cover every main effect. Good — every combo has its
constituent single-knob runs for attribution.

### 4.4 Optional supplements worth considering (not required)
- **Mixed LR contrast.** Mixed has only one LR here (global WSD). If you want an
  LR contrast *within* mixed, the meaningful comparison is **global WSD vs
  constant-LR (no final decay)** — add only if interested; not in your list.
- **Difficulty-weighted mixed sampler** is listed as an option; treat it as a
  *sweep on top of* the uniform mixed runs (`F3`, `C07`, `C08`, `C12`), not as
  separate baseline runs, to avoid combinatorial blow-up.
- **Multi-seed** on sparse categories (see §6) — strongly recommended for any
  ranking claim, not a new experiment.

### 4.5 No other logical errors found
Budget invariance (§5), GF/ER composition (§5), and eval comparability (§6) are
handled by the rules below. With §4.1 dedup and §4.2 filled by `S7`, the matrix
is internally consistent and the sequential factorial is complete.

---

## 5. Composition rules & invariants

1. **Budget invariance.** Every run uses the **same total optimizer steps** as
   `B0` (= sum over the 10 stages at 1 epoch). GF/ER change batch *composition*,
   not step count. A `mixed` run is set to the same total step count.
2. **GF ∩ ER overlap.** When both on, apply replay selection first, then let GF
   top up grounding-bearing samples; a replayed sample that is itself
   grounding-bearing counts toward both quotas (no double-count, never exceed
   100%). Step count stays fixed.
3. **Mixed redundancy of GF/ER.** In `mixed`, all categories are present every
   step, so grounding starvation is largely gone by construction and replay is
   mostly subsumed by mixing. Implement as requested, but expect GF to act only
   as deliberate grounding *over*-sampling and ER to have a small effect; note
   this when interpreting `C07`/`C08`/`C12`.

---

## 6. Eval protocol & metrics

Produce a **checkpoint × category** matrix, same shape as the current baseline
table.
- **Sequential runs:** evaluate after each of the 10 stages (10 checkpoints).
- **Mixed runs:** evaluate at **10 evenly-spaced step checkpoints** (10%…100%
  of total steps) so the progression is comparable to the 10 stage-evals.

Per cell: `answer_acc`, `view_acc`, `grounding_acc@0.8`, `format_valid`.

**Headline metrics for ranking runs:**
1. **Final macro** (mean over 10 categories at the last checkpoint) — ↑ better;
   watch `grounding_acc@0.8` and `format_valid`.
2. **Macro forgetting** = mean over categories of
   `(best_earlier_checkpoint_acc − final_acc)` — ↓ better (reuse the
   aggregator's forgetting column; report per metric).
3. **Collapse indicator** = `min over checkpoints of (macro format_valid)` —
   ↑ better. **Baseline ≈ 0.030** (the RML-stage collapse). A good forgetting
   fix keeps this high → the central success signal for this study.
4. **Answer non-regression** = final macro `answer_acc` must not drop vs `B0`.

> **Eval-noise warning:** TSS/RML grounding metrics are computed over only
> 7/10 grounded samples and swing hard. Rank on **macro** metrics with
> **≥2 seeds** for `B0`, `F1`, `F3`, `L1`, and the top combos. Do not over-read
> a single sparse-category cell.

---

## 7. Outputs & naming
```
output/curriculum_v2_exp/<EXP_ID>__seed<S>/
  stage_00_OBS/ … stage_09_CHR/     # sequential
  ckpt_010/ … ckpt_100/             # mixed (step-fraction checkpoints)
  eval_report.json                  # per checkpoint
  run_manifest.json                 # exact knob settings (mode, lr+params, GF, ER, seed, total_steps, frozen loss)
  summary.json                      # §6 headline metrics
output/curriculum_v2_exp/_master_comparison.{csv,md}
```
Every run reproducible from `run_manifest.json` alone.

---

## 8. Recommended run order

1. **`B0`** — re-confirm baseline (box-extractor fix must be in place so
   `n_pred_boxes>0`).
2. **Single knobs:** `F1` (GF), `F2` (ER), `F3` (mixed baseline), `L1` (global
   WSD), `L2` (per-stage relaxed). Isolates each main effect.
3. **Most promising sequential combos:** `C01` (GF+GW), `C04` (GF+PSW-R),
   `C03` (GF+ER+GW), `C06` (GF+ER+PSW-R), `S7` (GF+ER+baseline-LR).
4. **Remaining sequential:** `C02`, `C05`.
5. **Mixed combos:** `C07` (GF), `C08` (ER), `C12` (GF+ER).

**Distinct runs after dedup:** sequential 12 (`B0,L1,L2,F1,F2,C01–C06,S7`)
+ mixed 4 (`F3,C07,C08,C12`) = **16**.

**A-priori expectation:** `F1` (GF) alone should remove the RML-stage collapse
(collapse indicator ↑); `F3` (mixed) is the strongest *structural* fix
(removes stage boundaries entirely); `L1`/`L2` reduce the LR-driven overwrite.

---

## 9. Acceptance check
- Default-off config reproduces `B0` bit-for-bit (same loss curve, same
  per-stage eval) — proves the frozen loss is untouched.
- With `F1` at `min_fraction=0.25`, the logged `grounded_frac` on sparse stages
  is `≳0.25` instead of ~0.
- `_master_comparison` reproduces the baseline table for `B0` and reports the
  collapse indicator (`min macro format_valid`) for every run.

---

## Appendix A — Mixed difficulty-weighted mixture: initial-weight design

Applies only to `training.mode: mixed` with `mixed.sampler: difficulty_weighted`.
The mixture weight `w_c` sets the per-step sampling probability of category `c`.
**This is a category-level lever and is orthogonal to the grounding floor** (GF
controls the grounding-bearing *fraction* across categories; the mixture controls
*which categories* are drawn). Use both together; do not conflate them.

### A.1 Stage the design — do not hand-pick "the" weights
Run in this order; each later stage uses the earlier as its control:

- **v0 — uniform (`mixture_weights: null`, `w_c = 0.10`).** This is the mixed
  baseline (`F3`). **Always run v0 first** — you cannot claim difficulty
  weighting helps without the uniform mixed run to compare against.
- **v1 — size-temperature.** Corrects raw dataset-size imbalance only.
- **v2 — size-temperature × difficulty.** Adds the accuracy-based difficulty
  signal. This is "difficulty-weighted" proper.
- **v3 — fit (Data Mixing Laws).** Replace the hand-set prior with an empirically
  fitted mixture (A.4). v1/v2 are priors/anchors for the proxy runs.

### A.2 Weight formula
```
w_c  ∝  (n_c ^ τ)  ·  (d_c ^ γ)          # then normalize, then clamp, then renormalize
  n_c = train-set size of category c
  d_c = 1 − acc_c                          # "difficulty"; acc_c from the B0 final model
  τ   = size temperature   (v1/v2: 0.5)    # τ=1 → size-proportional, τ=0 → ignore size
  γ   = difficulty exponent(v1: 0; v2: 0.5)# γ=0 → ignore difficulty (pure size-temp)
clamp each normalized w_c to [0.5, 2.0] × (1/N), then renormalize  # no category is starved
```

**Choose `acc_c = answer_acc` (B0 final), NOT grounding_acc**, for the mixture
weight: answer_acc is the stable, category-level "how much does this category
still need training" signal. Grounding scarcity/quality is handled separately by
the GF knob, and grounding_acc on TSS/RML is too noisy (7/10 grounded samples) to
drive a sampling weight. (A blended `acc_c` is offered below only as an option.)

**Clamping is mandatory.** Up-weighting steals samples from other categories under
the fixed step budget (§5.1); the `[0.5, 2.0]×` clamp prevents any category from
being starved — which would re-introduce exactly the forgetting this study fights.

### A.3 Worked numbers from the B0 final model (answer_acc)
Difficulty `d_c = 1 − answer_acc_c`, clamped to `[0.5, 2.0]×` uniform
(uniform = 10.0% / 1.00×). **Size term omitted here** because only OBS (32,551)
and CHR (19,446) sizes are on hand — fill real `n_c` for the size term.

| Category | B0 answer_acc | v2 weight, γ=1.0 | v2 weight, γ=0.5 *(recommended start)* |
|---|---|---|---|
| OBS | 0.745 | 6.5%  (0.65×) | 8.2%  (0.82×) |
| IDN | 0.290 | 18.1% (1.81×) | 13.6% (1.36×) |
| AAS | 0.680 | 8.2%  (0.82×) | 9.1%  (0.91×) |
| SRO | 0.490 | 13.0% (1.30×) | 11.5% (1.15×) |
| TSS | 0.715 | 7.3%  (0.73×) | 8.6%  (0.86×) |
| RML | 0.605 | 10.1% (1.01×) | 10.2% (1.02×) |
| DRA | 0.635 | 9.3%  (0.93×) | 9.8%  (0.98×) |
| RWP | 0.660 | 8.7%  (0.87×) | 9.4%  (0.94×) |
| ESC | 0.715 | 7.3%  (0.73×) | 8.6%  (0.86×) |
| CHR | 0.540 | 11.7% (1.17×) | 11.0% (1.10×) |

`γ=0.5` is the **recommended starting point**: it up-weights the genuinely weak
categories (IDN 1.36×, SRO 1.15×, CHR 1.10×) without aggressively starving the
strong ones. `γ=1.0` is a more aggressive variant for a sweep. To add the size
term, multiply each by `n_c^0.5` and re-normalize+clamp.

*(Optional blended difficulty `acc_c = 0.5·answer + 0.5·grounding`, γ=0.5, flattens
to ≈0.86–1.14× — much milder, because grounding is uniformly low. Only use if you
specifically want grounding-weakness to influence category sampling; otherwise
keep answer-only and let GF handle grounding.)*

### A.4 v3 — fit with Data Mixing Laws (ICLR 2025)
The hand-set weights are a **prior**, not the answer. To actually optimize:
1. Pick a target metric `M` = macro over categories of e.g. `answer_acc` (or a
   blend with `grounding_acc`).
2. Run **3–5 short proxy runs** (reduced steps) at spread-out mixtures: `v0`
   (uniform), `v1` (size-temp), `v2` (γ=1.0), and 1–2 interpolations between them.
3. Fit a simple mixing law `M(w)` (per-category exponential / power form) to the
   proxy `(mixture → M)` points.
4. Predict the `w*` that maximizes `M`, subject to the `[0.5, 2.0]×` clamp.
5. Verify `w*` with one full-length run; compare against `v0` and `v2`.

### A.5 Caveats
- Difficulty up-weighting **helps only if the bottleneck is data quantity.** If a
  category is hard due to intrinsic ambiguity or label noise (suspect IDN, with
  answer_acc 0.29), more sampling amplifies noise rather than helping — verify
  against the v0 uniform run before trusting it.
- **Keep weights static** within a run for v1/v2 (matches Data Mixing Laws).
  Online/dynamic reweighting (update weights from running per-category accuracy)
  is a heavier, separate option — out of scope for this round.
- Mixture weighting is **mixed-mode only**; it has no meaning in the sequential
  runs (those train one category per stage).
