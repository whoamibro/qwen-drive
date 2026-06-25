"""Offline sampler verification — instantiate the sampler for a given
ExpConfig (+ optional stage_idx) and print expected vs realized per-pool
per-category exposure ratios. No GPU, no torchrun, no training.

Use this to (a) catch sampler/spec mismatches before committing GPU time
and (b) generate the §9 acceptance evidence: "realized per-category
exposure within ±tolerance of 1/N".

Usage:
    # Verify F3 (mixed, no GF, no ER) — should show ~10% per category
    python -m qwenvl.experiments.verify_sampler \\
        --mode mixed --lr_schedule global_wsd \\
        --grounding_floor off --replay off \\
        --num_draws 100000

    # Verify F1 at stage TSS (sequential, GF=0.25) — should show CURRENT=TSS,
    # GF top-up with ~10% per cat across 10 grounded sub-pools
    python -m qwenvl.experiments.verify_sampler \\
        --mode sequential --lr_schedule per_stage_wsd \\
        --grounding_floor 0.25 --replay off \\
        --stage_idx 4 --num_draws 100000

    # Verify C03 at stage CHR (sequential, GF+ER, with 9 priors)
    python -m qwenvl.experiments.verify_sampler \\
        --mode sequential --lr_schedule global_wsd \\
        --grounding_floor 0.25 --replay 0.10 \\
        --stage_idx 9 --num_draws 200000

Exits non-zero (and prints which cells violate) if any pool's per-category
realized ratio differs from expected by more than `--tolerance` (default
0.02 = ±2%).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "qwen-vl-finetune"))

from qwenvl.experiments.samplers import (   # noqa: E402
    build_sampler, compute_grounding_flags,
)


CURRICULUM_ORDER = ["OBS", "IDN", "AAS", "SRO", "TSS",
                    "RML", "DRA", "RWP", "ESC", "CHR"]


def _read_grounded_counts(train_dir: str, cat: str) -> Tuple[int, int]:
    """Return (n_total, n_grounded) for a category's training JSON."""
    path = os.path.join(train_dir, f"sft_train_qwen3vl_{cat}.json")
    if not os.path.exists(path):
        return 0, 0
    with open(path) as f:
        data = json.load(f)
    n_g = sum(compute_grounding_flags(data))
    return len(data), n_g


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m qwenvl.experiments.verify_sampler",
        description=__doc__,
    )
    p.add_argument("--mode", required=True, choices=["sequential", "mixed"])
    p.add_argument("--lr_schedule", required=True,
                   choices=["per_stage_wsd", "global_wsd", "per_stage_wsd_relaxed"])
    p.add_argument("--grounding_floor", default="off",
                   help="'off' or float in (0, 1)")
    p.add_argument("--replay", default="off",
                   help="'off' or float in (0, 1)")
    p.add_argument("--stage_idx", type=int, default=0,
                   help="Sequential mode only: which stage (0..9) the sampler "
                        "would be built for. Determines PRIOR pool composition. "
                        "Mixed mode ignores this.")
    p.add_argument("--num_draws", type=int, default=100000,
                   help="Total samples to draw (default 100k for good statistics).")
    p.add_argument("--train_dir", default="sft_dataset")
    p.add_argument("--tolerance", type=float, default=0.02,
                   help="Allowed absolute deviation from expected per-category "
                        "share (default 0.02 = 2%%).")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    # Parse knobs the same way the CLI does.
    gf = None if args.grounding_floor.strip().lower() == "off" else float(args.grounding_floor)
    er = None if args.replay.strip().lower() == "off" else float(args.replay)
    if args.mode == "mixed" and args.lr_schedule != "global_wsd":
        print("ERROR: mode=mixed requires lr_schedule=global_wsd.", file=sys.stderr)
        return 2

    # Build the per-pool per-category size lists.
    if args.mode == "mixed":
        cats_current = CURRICULUM_ORDER[:]
        cats_prior = []          # mixed-mode never has PRIOR — orchestrator's
                                  # run_mixed passes prior_data_paths=[] always.
        if er is not None:
            print("[verify_sampler] note: mode=mixed silently skips ER per the "
                  "orchestrator's behavior (mixed has no per-stage notion of "
                  "'prior categories'). Treating --replay as off for this check.")
            er = None
    else:
        cats_current = [CURRICULUM_ORDER[args.stage_idx]]
        if er is not None and args.stage_idx == 0:
            print("[verify_sampler] note: stage 0 has no priors → orchestrator "
                  "skips ER on stage 0 (spec §2.3). Treating --replay as off.")
            cats_prior = []
            er = None
        else:
            cats_prior = CURRICULUM_ORDER[:args.stage_idx] if er is not None else []

    sizes = {cat: _read_grounded_counts(args.train_dir, cat) for cat in CURRICULUM_ORDER}
    current_per_cat = [(c, sizes[c][0]) for c in cats_current]
    prior_per_cat = [(c, sizes[c][0]) for c in cats_prior]
    global_grounded_per_cat = (
        [(c, sizes[c][1]) for c in CURRICULUM_ORDER]
        if gf is not None else []
    )

    print("=" * 72)
    print(f"Sampler verification — mode={args.mode}, lr_schedule={args.lr_schedule}, "
          f"gf={gf!r}, er={er!r}, stage_idx={args.stage_idx if args.mode == 'sequential' else 'n/a'}")
    print("=" * 72)
    print()
    print(f"  CURRENT pool        ({len(current_per_cat)} cats): "
          f"{[(c, n) for c, n in current_per_cat]}")
    print(f"  PRIOR pool          ({len(prior_per_cat)} cats): "
          f"{[(c, n) for c, n in prior_per_cat]}")
    print(f"  GLOBAL_GROUNDED pool ({len(global_grounded_per_cat)} cats): "
          f"{[(c, n) for c, n in global_grounded_per_cat]}")
    print()

    sampler, plan = build_sampler(
        current_per_cat=current_per_cat,
        prior_per_cat=prior_per_cat,
        global_grounded_per_cat=global_grounded_per_cat,
        target_floor=gf,
        replay_fraction=er,
        num_training_samples=args.num_draws,
        seed=args.seed,
    )

    if sampler is None:
        print("[verify_sampler] B0 fast-path (sampler=None). No per-category "
              "weighting is applied — HF Trainer's DistributedSampler does "
              "uniform-over-flat-indices, which is equivalent to per-sample "
              "uniform within the single CURRENT category. Nothing to verify.")
        return 0

    # Draw the samples (no DDP — runs single-process; sampler's
    # _ddp_rank_world returns (0, 1) when distributed isn't initialized).
    print(f"Drawing {args.num_draws} samples...")
    _ = list(iter(sampler))   # populates _exposure as a side effect
    summary = sampler.exposure_summary()

    print()
    print(f"  total draws: {summary['total_draws']}")
    print()

    failures: List[str] = []
    for pool_name in ("current", "prior", "global_grounded"):
        pool = summary["per_pool"].get(pool_name, {})
        cats = pool.get("categories") or {}
        if not cats:
            continue
        pool_total = pool["pool_total"]
        print(f"  {pool_name.upper()} pool  ({pool_total} draws, "
              f"{pool['share_of_total']*100:.2f}% of total)")
        print(f"    {'cat':>4}  {'count':>8}  {'share_of_pool':>16}  "
              f"{'share_of_total':>16}  {'expected':>10}  {'diff':>8}")
        for cat, info in sorted(cats.items()):
            exp = info["expected_share_of_total"]
            obs = info["share_of_total"]
            diff = obs - exp
            ok = abs(diff) <= args.tolerance
            mark = "✓" if ok else "✗"
            if not ok:
                failures.append(
                    f"{pool_name}.{cat}: expected {exp*100:.2f}%, got "
                    f"{obs*100:.2f}% (diff {diff*100:+.2f}%, tolerance {args.tolerance*100:.1f}%)"
                )
            print(f"    {cat:>4}  {info['count']:>8}  {info['share_of_pool']*100:>15.2f}%  "
                  f"{obs*100:>15.2f}%  {exp*100:>9.2f}%  {diff*100:>+7.2f}% {mark}")
        print()

    if failures:
        print("=" * 72)
        print(f"FAIL — {len(failures)} cell(s) outside ±{args.tolerance*100:.1f}% tolerance:")
        for f in failures:
            print(f"  - {f}")
        print("=" * 72)
        return 1

    print("=" * 72)
    print(f"PASS — all per-category shares within ±{args.tolerance*100:.1f}% of expected.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
