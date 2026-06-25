"""Pre-materialize the GLOBAL_GROUNDED pool used by --grounding_floor.

Per spec §2.2 (`pool: global`), GF top-up draws are sourced from the union of
grounded samples across ALL 10 categories. Without pre-materialization, the
trainer re-reads + re-filters the full per-category train JSONs at every
stage startup (~100 MB of JSON parsing per stage × 10 stages = ~100-300s
overhead per experiment).

This script reads the 10 per-category training files once, filters to
grounded samples, and writes a single consolidated JSON:

    sft_dataset/sft_train_qwen3vl_GLOBAL_GROUNDED.json

The orchestrator (`qwenvl.experiments.run_experiment`) auto-detects this
file and prefers it over the per-category fallback when it exists. If the
file is missing, the orchestrator falls back to the per-category list (the
current default), so this script is OPTIONAL for correctness — just a
performance optimization.

Idempotency:
    A `.built_<n_samples>` marker file is written alongside the pool so a
    re-run with `--force` is required to rebuild after the training data
    changes. Useful for detecting staleness without re-parsing the file.

Usage:
    python -m nuscenes_pipeline.postprocessing.build_global_grounded_pool

    # Rebuild after train data changes:
    python -m nuscenes_pipeline.postprocessing.build_global_grounded_pool --force

    # Custom paths:
    python -m nuscenes_pipeline.postprocessing.build_global_grounded_pool \\
        --train_dir sft_dataset_v9 \\
        --output_path sft_dataset_v9/sft_train_qwen3vl_GLOBAL_GROUNDED.json
"""

import argparse
import glob
import json
import os
import sys
from typing import List


# Canonical curriculum order — matches CURRICULUM_ORDER everywhere else.
CURRICULUM_ORDER: List[str] = [
    "OBS", "IDN", "AAS", "SRO", "TSS",
    "RML", "DRA", "RWP", "ESC", "CHR",
]


def has_grounding(sample) -> bool:
    """True iff `sample`'s GPT response carries a non-empty `grounding` list.

    Matches `qwenvl.experiments.samplers.compute_grounding_flags` so the
    pre-materialized pool is identical to what the trainer would build at
    runtime.
    """
    try:
        obj = json.loads(sample["conversations"][2]["value"])
        return bool(obj.get("grounding"))
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return False


def main():
    p = argparse.ArgumentParser(prog="python -m nuscenes_pipeline.postprocessing.build_global_grounded_pool",
                                description=__doc__)
    p.add_argument("--train_dir", default="sft_dataset",
                   help="Directory containing sft_train_qwen3vl_<CAT>.json files.")
    p.add_argument("--output_path", default=None,
                   help="Output JSON path. Default: <train_dir>/sft_train_qwen3vl_GLOBAL_GROUNDED.json")
    p.add_argument("--force", action="store_true",
                   help="Rebuild even if the output file already exists.")
    args = p.parse_args()

    out_path = args.output_path or os.path.join(
        args.train_dir, "sft_train_qwen3vl_GLOBAL_GROUNDED.json"
    )

    if os.path.exists(out_path) and not args.force:
        print(f"[skip] {out_path} already exists. Pass --force to rebuild.")
        # Show the marker contents for diagnostic clarity.
        markers = glob.glob(out_path + ".built_*")
        if markers:
            print(f"       marker: {os.path.basename(markers[0])}")
        return 0

    if not os.path.isdir(args.train_dir):
        print(f"ERROR: train_dir does not exist: {args.train_dir}", file=sys.stderr)
        return 2

    all_grounded = []
    print(f"Building global grounded pool from {args.train_dir}/")
    print()
    print(f"  {'CAT':>4}   {'total':>7}   {'grounded':>9}   {'rate':>6}")
    print(f"  {'---':>4}   {'-----':>7}   {'--------':>9}   {'----':>6}")

    missing = []
    for cat in CURRICULUM_ORDER:
        path = os.path.join(args.train_dir, f"sft_train_qwen3vl_{cat}.json")
        if not os.path.exists(path):
            print(f"  {cat:>4}   {'MISSING':>7}   {'-':>9}   {'-':>6}    ({path})")
            missing.append(cat)
            continue
        with open(path) as f:
            data = json.load(f)
        grounded = [s for s in data if has_grounding(s)]
        all_grounded.extend(grounded)
        rate = len(grounded) / max(len(data), 1)
        print(f"  {cat:>4}   {len(data):>7}   {len(grounded):>9}   {rate:>5.1%}")

    print(f"  {'---':>4}   {'-----':>7}   {'--------':>9}   {'----':>6}")
    print(f"  {'TOTAL':>4}   {'':>7}   {len(all_grounded):>9}")
    print()

    if not all_grounded:
        print("ERROR: no grounded samples found across any category. Refusing to "
              "write an empty pool.", file=sys.stderr)
        return 3

    # Write atomically: temp file + rename so a concurrent reader can't see a
    # partial file. (The orchestrator's training launches read this path.)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp_path = f"{out_path}.tmp.{os.getpid()}"
    with open(tmp_path, "w") as f:
        json.dump(all_grounded, f)
    os.replace(tmp_path, out_path)

    # Clean up any old marker(s), write a fresh one.
    for old in glob.glob(out_path + ".built_*"):
        os.remove(old)
    marker = f"{out_path}.built_{len(all_grounded)}"
    with open(marker, "w") as f:
        f.write("")

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Wrote {len(all_grounded)} grounded samples ({size_mb:.1f} MB) -> {out_path}")
    print(f"Marker: {os.path.basename(marker)}")
    if missing:
        print(f"\n[warn] {len(missing)} categories missing: {missing}")
        print(f"       Pool covers only {10 - len(missing)} of 10 categories.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
