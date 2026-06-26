"""T0 probe — measure whether GT grounding boxes in sft_train_qwen3vl_*.json
are already in canonical order (image_idx asc, area desc, x1 asc), or in the
teacher's arbitrary regex-match order.

If >=90% of multi-box samples are already canonical, T0 is a near-no-op and we
can skip the converter flag. Otherwise, the converter sort is a real win.

Usage:
    python -m nuscenes_pipeline.postprocessing.probe_box_order \
        --data_dir sft_dataset --pattern 'sft_train_qwen3vl_*.json'
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import Counter
from typing import List, Optional, Tuple

BOX_RE = re.compile(r"<\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>")


def parse_box_from_ref(ref: str) -> Optional[Tuple[int, int, int, int]]:
    if not isinstance(ref, str):
        return None
    m = BOX_RE.search(ref)
    if not m:
        return None
    x1, y1, x2, y2 = (int(g) for g in m.groups())
    return x1, y1, x2, y2


def canonical_key(box: Tuple[int, int, int, int], image_idx: int) -> Tuple[int, int, int]:
    """(image_idx asc, -area desc, x1 asc)."""
    x1, y1, x2, y2 = box
    area = max(0, x2 - x1) * max(0, y2 - y1)
    return (image_idx, -area, x1)


def is_canonical(grounding: List[dict]) -> bool:
    keys = []
    for g in grounding:
        box = parse_box_from_ref(g.get("ref"))
        if box is None:
            return False  # treat unparseable as not-canonical, conservative
        keys.append(canonical_key(box, int(g["image_idx"])))
    # already canonical iff sorted non-decreasing
    return all(keys[i] <= keys[i + 1] for i in range(len(keys) - 1))


def scan_file(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    stats = Counter()
    box_count_dist = Counter()
    for sample in data:
        gpt_turn = next((t for t in sample.get("conversations", [])
                         if t.get("from") == "gpt"), None)
        if not gpt_turn:
            continue
        try:
            obj = json.loads(gpt_turn["value"])
        except Exception:
            stats["unparseable_gpt"] += 1
            continue
        grounding = obj.get("grounding") or []
        stats["total"] += 1
        n = len(grounding)
        box_count_dist[n] += 1
        if n == 0:
            stats["empty_grounding"] += 1
        elif n == 1:
            stats["single_box"] += 1
        else:
            stats["multi_box"] += 1
            if is_canonical(grounding):
                stats["multi_box_canonical"] += 1
            else:
                stats["multi_box_arbitrary"] += 1
    return {"stats": dict(stats), "box_count_dist": dict(sorted(box_count_dist.items()))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="sft_dataset")
    p.add_argument("--pattern", default="sft_train_qwen3vl_*.json")
    p.add_argument("--per_file", action="store_true",
                   help="Print per-file breakdown in addition to the aggregate.")
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.join(args.data_dir, args.pattern)))
    # Exclude category-less catch-all files & GLOBAL_GROUNDED (it's a derivative pool).
    paths = [p for p in paths if "GLOBAL_GROUNDED" not in p
             and os.path.basename(p) not in ("sft_train_qwen3vl.json", "sft_val_qwen3vl.json")]
    if not paths:
        print(f"No files matched {args.pattern!r} under {args.data_dir!r}")
        return 1

    agg = Counter()
    agg_box_count: Counter = Counter()
    per_file_rows = []
    for path in paths:
        r = scan_file(path)
        for k, v in r["stats"].items():
            agg[k] += v
        for n, c in r["box_count_dist"].items():
            agg_box_count[n] += c
        cat = os.path.basename(path).replace("sft_train_qwen3vl_", "").replace(".json", "")
        per_file_rows.append((cat, r["stats"]))

    if args.per_file:
        print(f"{'cat':>6}  {'total':>7}  {'empty':>7}  {'1box':>6}  {'multi':>6}  {'canon':>6}  {'%canon':>7}")
        for cat, s in per_file_rows:
            mb = s.get("multi_box", 0)
            ca = s.get("multi_box_canonical", 0)
            pct = (ca / mb * 100.0) if mb else 0.0
            print(f"{cat:>6}  {s.get('total', 0):>7}  {s.get('empty_grounding', 0):>7}  "
                  f"{s.get('single_box', 0):>6}  {mb:>6}  {ca:>6}  {pct:>6.1f}%")
        print()

    total = agg.get("total", 0)
    mb = agg.get("multi_box", 0)
    ca = agg.get("multi_box_canonical", 0)
    ar = agg.get("multi_box_arbitrary", 0)
    print("=" * 64)
    print(f"AGGREGATE  (files: {len(paths)})")
    print("=" * 64)
    print(f"  total samples              : {total:>8,}")
    print(f"  empty grounding            : {agg.get('empty_grounding', 0):>8,}  "
          f"({agg.get('empty_grounding', 0)*100.0/max(total,1):.1f}%)")
    print(f"  single-box samples         : {agg.get('single_box', 0):>8,}  "
          f"({agg.get('single_box', 0)*100.0/max(total,1):.1f}%)")
    print(f"  MULTI-BOX samples (n>=2)   : {mb:>8,}  "
          f"({mb*100.0/max(total,1):.1f}% of total)")
    print(f"    already canonical        : {ca:>8,}  "
          f"({ca*100.0/max(mb,1):.1f}% of multi-box)")
    print(f"    arbitrary order          : {ar:>8,}  "
          f"({ar*100.0/max(mb,1):.1f}% of multi-box)")
    print()
    print("  box-count distribution (multi-box only):")
    for n, c in sorted(agg_box_count.items()):
        if n < 2:
            continue
        print(f"    n={n:<3}: {c:>7,}")
    print()
    pct_canonical = ca * 100.0 / max(mb, 1)
    print("=" * 64)
    if pct_canonical >= 90.0:
        print(f"VERDICT: {pct_canonical:.1f}% multi-box samples already canonical "
              f"(>= 90%). T0 is a near-no-op — skip the converter flag.")
    else:
        print(f"VERDICT: only {pct_canonical:.1f}% multi-box samples already canonical "
              f"(< 90%). T0 sort is a real stability win — implement the converter flag.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
