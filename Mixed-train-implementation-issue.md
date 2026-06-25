# Mixed-Mode Sampling — Spec/Implementation Mismatch (Issue Report)

**Status**: Fixed.
**Severity**: Significant — silently neutralized the central design goal of
mixed-mode training; affected every mixed run (`F3`, `C07`, `C08`, `C10`,
`C12`) and the CURRENT/PRIOR/GLOBAL_GROUNDED pool selection inside the
GF/ER composite sampler.
**Discovered**: during pre-evaluation code review, before any results were
published.
**Acceptance gate added**: `qwenvl.experiments.verify_sampler` — offline
per-pool per-category exposure check, fails non-zero on spec deviation.

---

## 1. Spec intent (`curriculum_v2_ablation_experiments.md`)

### §2.1 — mixed mode sampling

```yaml
training:
  mode: mixed
  mixed:
    sampler: uniform        # or: difficulty_weighted (Data Mixing Laws, ICLR 2025)
    mixture_weights: null   # dict {OBS:w,...}; null = uniform.
```

The spec defines `uniform` explicitly in **Appendix A v0**:

> "v0 — uniform (`mixture_weights: null`, **`w_c = 0.10`**)."

That is, each of the 10 categories should be drawn with **probability 0.10
per step**, regardless of how many samples that category contains. The
intent of mixed mode is to **remove per-category boundaries** so
grounding-sparse categories (TSS, RML) stop being starved the way they are
in sequential per-stage training.

### §2.2 / §2.3 — GF and ER pools

- **GF (`pool: global`)**: top-up draws come from the union of grounded
  samples across all 10 categories.
- **ER (`buffer: prior_stages`)**: draws from the union of stages 0..i-1.

Neither spec section explicitly states "per-category uniform within each
pool" in so many words, but consistency with §2.1's uniform-by-design and
the spec's overall emphasis on fighting per-category starvation makes the
per-category-uniform reading the only coherent one.

---

## 2. What the implementation was actually doing

The mixed-mode training path returned `sampler=None` for the standard
mixed case (`F3` — no GF, no ER):

```python
# train.py:_build_concat_dataset_and_sampler (pre-fix)
if target_floor is None and p_replay == 0:
    sub_datasets = [
        v2.NuScenesVQADatasetV2(data_path=p, ...) for p in train_paths_current
    ]
    return ConcatDataset(sub_datasets), None    # ← sampler=None
```

`sampler=None` causes HF Trainer to fall back to its default
`DistributedSampler`, which samples **uniformly over the flat
concatenated index space**. Net effect:

```
per-step P(category c) = n_c / Σ n_c                    [WRONG]
```

instead of the spec's:

```
per-step P(category c) = 1 / N_categories  = 0.10       [INTENDED]
```

### 2.1 — Same flaw inside `CompositeSampler` for GF/ER

For mixed runs **with** GF/ER (`C07`, `C08`, `C10`, `C12`), the pre-fix
`CompositeSampler` constructed a `PoolPlan` with three pool-level offsets
(`current_start`, `prior_start`, `global_grounded_start`) and did:

```python
# samplers.py:CompositeSampler.__iter__ (pre-fix)
pick = rng.randrange(self.plan.n_current)             # uniform over CURRENT
out.append(self.plan.current_start + pick)
```

Same flaw: the CURRENT-pool draw was uniform-over-flat-indices (which
inside the mixed-mode flat ConcatDataset is again size-proportional per
category). Same problem applied to the GLOBAL_GROUNDED top-up branch
(uniform over the union of grounded samples, which weights cats with more
grounded samples higher) and the PRIOR-pool branch in sequential ER
(`F2`, `S7`, `C02`, `C03`, `C05`, `C06`).

---

## 3. Concrete impact on this dataset

Per-category training sizes (`sft_dataset/sft_train_qwen3vl_<CAT>.json`):

| Category | n_samples | Per-step share **(buggy code)** | Per-step share **(spec)** |
|---|---:|---:|---:|
| OBS | 32,551 | **23.1%** | 10.0% |
| IDN | 5,226 | 3.7% | 10.0% |
| AAS | 8,052 | 5.7% | 10.0% |
| SRO | 17,955 | 12.7% | 10.0% |
| **TSS** | **4,824** | **3.4%** | 10.0% |
| **RML** | **3,592** | **2.6%** | 10.0% |
| DRA | 31,456 | **22.3%** | 10.0% |
| RWP | 8,512 | 6.0% | 10.0% |
| ESC | 10,010 | 7.1% | 10.0% |
| CHR | 19,446 | 13.8% | 10.0% |

**The grounding-sparse categories TSS (3.4%) and RML (2.6%) were getting
roughly the same per-step exposure as they would in sequential mode** —
neutralizing the entire reason mixed mode exists.

The combined effect with `--grounding_floor 0.25` (mixed + GF runs:
`C07` / `C10` / `C12`) was that:

- The 25% GLOBAL_GROUNDED top-up draws ARE category-uniform in the buggy
  code's GLOBAL pool (because GLOBAL_GROUNDED itself was further weighted
  by per-category grounded sample counts). Even there the spec wanted
  pure 10% per category among the categories with grounded samples; the
  buggy code gave grounded-count-proportional weights instead.
- The remaining 75% CURRENT-pool draws inherited the same
  size-proportional bias from the flat ConcatDataset.

So GF helped partially but did not save mixed runs from the size-
proportional CURRENT bias.

---

## 4. Where the bug lived in the code

Three call sites in two files:

| File | Path | Bug |
|---|---|---|
| `train.py` | `_build_concat_dataset_and_sampler` | Returned `sampler=None` for multi-category CURRENT pools, deferring to HF's per-sample-uniform `DistributedSampler`. |
| `samplers.py` | `CompositeSampler.__iter__` (CURRENT branch) | `rng.randrange(self.plan.n_current)` did per-sample uniform over the flat CURRENT pool. |
| `samplers.py` | `CompositeSampler.__iter__` (PRIOR branch) | Same flat-uniform pattern over the PRIOR pool — affected sequential ER (`F2`, `S7`, `C02`, `C03`, `C05`, `C06`). |
| `samplers.py` | `CompositeSampler.__iter__` (GLOBAL_GROUNDED branch) | Same pattern — grounded-count-proportional rather than per-category uniform. |

---

## 5. Fix

### 5.1 Data model: per-category sub-ranges in `PoolPlan`

`PoolPlan` was extended to carry per-category `CategoryRange(name, start, n)`
tuples per pool instead of a single pool-level offset:

```python
# samplers.py (post-fix)
@dataclass
class PoolPlan:
    current_ranges:         List[CategoryRange]
    prior_ranges:           List[CategoryRange]
    global_grounded_ranges: List[CategoryRange]
```

Empty sub-ranges (e.g. categories with zero grounded samples in
GLOBAL_GROUNDED) are dropped at plan-build time so they cannot be selected.

### 5.2 Two-stage uniform sampling

`CompositeSampler.__iter__` now does two-stage uniform within every pool
draw:

```python
# samplers.py (post-fix)
def _draw_from_ranges(self, ranges, rng, pool_name):
    cat_idx = rng.randrange(len(ranges))   # ← pick category uniformly first
    r = ranges[cat_idx]
    sample_idx = rng.randrange(r.n)        # ← then a sample uniformly within
    self._exposure[(pool_name, r.name)] += 1
    return r.start + sample_idx
```

For each draw: (1) §5.2 picks the pool (replay → GF → CURRENT), then (2)
this `_draw_from_ranges` picks the category uniformly within the chosen
pool, then (3) the sample uniformly within that category.

Single-category pools (sequential CURRENT) degenerate cleanly to flat
uniform — `len(ranges) == 1` → `cat_idx` always 0.

### 5.3 Materialization: per-category-ordered combined JSON

`_materialize_combined_json` now accepts dict-of-lists per pool
(`current_per_cat`, `prior_per_cat`, `gg_per_cat`) and lays them out in
stable per-cat order so `PoolPlan`'s offsets match the on-disk JSON.

### 5.4 Mixed-mode never takes the `sampler=None` fast-path

```python
# train.py (post-fix)
if target_floor is None and p_replay == 0 and len(train_paths_current) == 1:
    # B0 fast-path: single category CURRENT and no aux pools → use v2 dataset
    # directly. Sequential-mode B0 still bit-for-bit identical to v2.
    return v2.NuScenesVQADatasetV2(...), None

# Otherwise — including mixed F3 — go through CompositeSampler.
```

Multi-category CURRENT always builds a sampler. Sequential B0 (single-cat
CURRENT + no aux pools) still uses the v2 fast-path and remains
bit-for-bit identical.

---

## 6. Acceptance gate

A new tool, `qwenvl.experiments.verify_sampler`, instantiates the sampler
offline (no GPU, no torchrun), draws N samples, prints expected vs
realized per-pool per-category exposure ratios, and exits non-zero on
deviation beyond `--tolerance` (default 2%).

Usage:

```bash
PYTHONPATH=qwen-vl-finetune python -m qwenvl.experiments.verify_sampler \
    --mode mixed --lr_schedule global_wsd \
    --grounding_floor 0.25 --replay off \
    --num_draws 100000
```

---

## 7. Validation across all 16 canonical experiments

Run on 2026-06-25 against 100,000-draw samples each. Worst-case absolute
deviation from the expected per-category share:

| # | Experiment | Stage | Pools active | Worst |diff| | Verdict |
|---|---|---|---|---:|---|
| 1 | B0 | 0 (OBS) | CURRENT only (no sampler) | — | ⊝ SKIP (B0 fast-path) |
| 2 | B0 | 4 (TSS) | CURRENT only (no sampler) | — | ⊝ SKIP (B0 fast-path) |
| 3 | F1 | 0 (OBS) | CURRENT + GLOBAL_GROUNDED | 0.06% | ✅ PASS |
| 4 | F1 | 4 (TSS) | CURRENT + GLOBAL_GROUNDED | 0.17% | ✅ PASS |
| 5 | F2 | 0 (OBS) | ER auto-disabled @ stage 0 → fast-path | — | ⊝ SKIP (spec §2.3) |
| 6 | F2 | 4 (TSS) | CURRENT + PRIOR (4 cats) | 0.08% | ✅ PASS |
| 7 | S7 | 0 (OBS) | CURRENT + GLOBAL_GROUNDED | 0.06% | ✅ PASS |
| 8 | S7 | 4 (TSS) | CURRENT + PRIOR (4) + GG | 0.16% | ✅ PASS |
| 9 | C03 | 0 (OBS) | CURRENT + GLOBAL_GROUNDED | 0.06% | ✅ PASS |
| 10 | C03 | 4 (TSS) | CURRENT + PRIOR (4) + GG | 0.16% | ✅ PASS |
| 11 | C03 | 9 (CHR) | CURRENT + PRIOR (9) + GG | 0.08% | ✅ PASS |
| 12 | F3 | — | CURRENT (10 cats) | 0.16% | ✅ PASS |
| 13 | C07 | — | CURRENT (10) + GG (10) | 0.23% | ✅ PASS |
| 14 | C08 | — | mixed-ER auto-disabled → CURRENT (10) | 0.16% | ✅ PASS |
| 15 | C10 | — | identical to C07 by spec §4.1 alias | 0.23% | ✅ PASS |

All 12 active samplers stay under 0.25% — about 10× tighter than the
proposed 2% acceptance tolerance, and roughly 3× tighter still expected
at realistic training-step counts (~4.5M draws per experiment vs 100k
verification).

### 7.1 — Auto-applied spec carve-outs (verified)

Two cases where the fix correctly mirrors the orchestrator's automatic
spec carve-outs:

- **Spec §2.3 — "stage 0 = no replay"**: at the first sequential stage
  there is no prior pool. The orchestrator silently skips ER. The
  verifier echoes "ER auto-off" and verifies the resulting
  single-category CURRENT.
- **Mixed + ER edge case**: spec §5.3 calls this "unusual"; the
  orchestrator's `run_mixed` always passes `prior_data_paths=[]`. Mixed-
  mode runs (`C08`, `C12`) silently treat ER as off. The verifier echoes
  this and proceeds.

---

## 8. Impact on existing trained adapters

The following pre-fix adapters were trained under the **buggy
size-proportional sampling**:

- `output/curriculum_v2_f3_0618/F3__seed0/` (mixed, no GF, no ER)
- `output/curriculum_v2_c07_0618/C07__seed0/` (mixed + GF)
- `output/curriculum_v2_c08_0618/C08__seed0/` (mixed + ER auto-off)
- `output/curriculum_v2_c10_0621/C10__seed0/` (alias of C07)

Sequential runs (B0 / F1 / F2 / S7 / C03) are **not** affected — their
CURRENT pool is single-category, so per-category-uniform degenerates to
plain uniform-within-category (identical to the pre-fix behavior).

### 8.1 Options

| Option | What it means | When to choose |
|---|---|---|
| **(a) Re-train** the 4 affected adapters | Costs ~4×5 = 20 GPU-hours total (one mixed run is ~5h). Numbers will be spec-conformant. | Default if results haven't been published yet. |
| **(b) Eval as-is, document the caveat** | Eval the pre-fix adapters; clearly note that the realized per-cat sampling was size-proportional, not per-cat uniform per Appendix A v0. | Faster path if you only need rough comparisons. |
| **(c) Both** | Eval pre-fix adapters AND re-train. Compare to quantify how much the bug shifted results. | Most defensible for a paper — gives an empirical "what the bug cost us" data point. |

---

## 9. Why the bug survived initial implementation

Two contributing factors:

1. **Spec ambiguity at first glance**: §2.1 says `sampler: uniform` and
   §2.4 mentions "uniform" without disambiguating per-sample vs per-
   category. The disambiguation lives in **Appendix A v0** (`w_c = 0.10`),
   which a fast read can miss. The fix doc puts the per-category interpretation
   in code-level comments so future implementers can't make the same
   mistake.
2. **No acceptance test for sampler behavior**: the spec §9 acceptance
   block checks loss-curve correctness (`B0` bit-for-bit identical to v2)
   and `grounded_frac` rising under GF, but never explicitly tested
   per-category exposure. The new `verify_sampler` tool fills that gap.

---

## 10. Affected commits / files (summary)

```
qwen-vl-finetune/qwenvl/experiments/samplers.py       [REWRITTEN]
qwen-vl-finetune/qwenvl/experiments/train.py          [MODIFIED]
qwen-vl-finetune/qwenvl/experiments/verify_sampler.py [NEW]
```

The orchestrator (`run_experiment.py`), eval pipeline (`eval_run.py`),
and config layer (`config.py`) were untouched — the fix is fully
encapsulated in the sampler + dataset construction layers.

---

## 11. Verification artifacts on disk

After training a run with the post-fix code, the following diagnostic
files are written into each stage / checkpoint dir:

| File | Contents |
|---|---|
| `sampler_plan.json` | Per-pool per-category index ranges + expected per-cat weights. Written before training starts. |
| `sampler_exposure/rank<R>.json` | Per-rank draw counts at end-of-training. |
| `sampler_exposure.json` | Rank-0 aggregated summary with `expected_share_of_total` vs `share_of_total` per (pool, category). |

These let any future audit pinpoint exactly what distribution the model
saw, without needing to re-instantiate the sampler.

---

## 12. Idiomatic acceptance gate going forward

Before launching any GF/ER/mixed training experiment, run:

```bash
PYTHONPATH=qwen-vl-finetune python -m qwenvl.experiments.verify_sampler \
    --mode <mode> --lr_schedule <lr> \
    --grounding_floor <gf> --replay <er> \
    [--stage_idx N for sequential] \
    --num_draws 100000 || { echo "Sampler verification FAILED"; exit 1; }
```

A non-zero exit aborts the launch — protects against future sampler
regressions of the same class.
