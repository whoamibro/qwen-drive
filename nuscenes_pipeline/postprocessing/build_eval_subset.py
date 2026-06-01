"""
Build a stratified per-category validation subset for in-training eval and
end-of-stage generation eval during curriculum training.

Reads each `sft_val_qwen3vl_{CAT}.json` in `--input_dir`, samples up to
`--n_per_cat` examples per category with a fixed seed, and writes
`sft_val_qwen3vl_{CAT}_subset.json` files plus a `manifest.json` into
`--output_dir` (default: `<input_dir>/eval_subset_{N}`).

Idempotent: writes a `.subset_built_{N}` marker; re-running with the same N
skips work unless `--force` is passed.

Usage:
    python -m nuscenes_pipeline.postprocessing.build_eval_subset \
        --input_dir sft_dataset \
        --n_per_cat 200
"""

import argparse
import glob
import json
import os
import random
import re
from typing import List


CATEGORY_RE = re.compile(r"sft_val_qwen3vl_([A-Z]+)\.json$")


def discover_categories(input_dir: str) -> List[str]:
    paths = sorted(glob.glob(os.path.join(input_dir, "sft_val_qwen3vl_*.json")))
    cats = []
    for p in paths:
        m = CATEGORY_RE.search(os.path.basename(p))
        if m and "_subset" not in p:
            cats.append(m.group(1))
    return sorted(set(cats))


def build_subset(input_dir: str, output_dir: str, n_per_cat: int, seed: int, force: bool):
    os.makedirs(output_dir, exist_ok=True)

    marker = os.path.join(output_dir, f".subset_built_{n_per_cat}")
    if os.path.exists(marker) and not force:
        print(f"[skip] marker exists: {marker} (use --force to rebuild)")
        return

    cats = discover_categories(input_dir)
    if not cats:
        raise FileNotFoundError(
            f"No sft_val_qwen3vl_*.json files found under {input_dir}"
        )
    print(f"Discovered {len(cats)} categories: {cats}")

    manifest = {"n_per_cat_requested": n_per_cat, "seed": seed, "categories": {}}
    rng = random.Random(seed)

    for cat in cats:
        in_path = os.path.join(input_dir, f"sft_val_qwen3vl_{cat}.json")
        with open(in_path) as f:
            data = json.load(f)

        sample_n = min(n_per_cat, len(data))
        # Reseed per-category so adding/removing a category does not change
        # the selections of other categories.
        cat_rng = random.Random(f"{seed}-{cat}")
        subset = cat_rng.sample(data, sample_n) if sample_n < len(data) else list(data)

        out_path = os.path.join(output_dir, f"sft_val_qwen3vl_{cat}_subset.json")
        with open(out_path, "w") as f:
            json.dump(subset, f, ensure_ascii=False)

        manifest["categories"][cat] = {
            "source": os.path.abspath(in_path),
            "subset_path": os.path.abspath(out_path),
            "n_total": len(data),
            "n_sampled": sample_n,
        }
        print(f"  {cat}: {sample_n}/{len(data)} -> {out_path}")

    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(marker, "w") as f:
        f.write("")

    print(f"\nWrote {len(cats)} category subsets to {output_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", default="sft_dataset")
    p.add_argument("--output_dir", default=None,
                   help="Default: <input_dir>/eval_subset_<N>")
    p.add_argument("--n_per_cat", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    out_dir = args.output_dir or os.path.join(args.input_dir, f"eval_subset_{args.n_per_cat}")
    build_subset(args.input_dir, out_dir, args.n_per_cat, args.seed, args.force)


if __name__ == "__main__":
    main()
