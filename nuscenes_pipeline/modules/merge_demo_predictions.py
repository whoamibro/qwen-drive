"""
Merge N per-worker `predictions_worker<offset>.json` files (produced by
`demo_tester.py --sample_stride N --sample_offset <offset>`) into a canonical
`predictions.json`.

Sharding contract:
    - Each worker owns frames whose `frame_pos % stride == offset`.
    - Every worker sees the SAME question set in the SAME order (the tester
      reads `demo_questions.json` once per invocation).
    - `frame_pos` values are disjoint across workers → concat + sort by
      `frame_pos` is a lossless merge.

Header fields (scene_token, mode, base_model, lora_path, pkl_path,
resize_factor, max_new_tokens, n_frames, n_questions, demo_questions_path)
are copied from the first worker; the merger sanity-checks that they match
across workers and warns on mismatch.

Usage:
    python -m nuscenes_pipeline.modules.merge_demo_predictions \\
        --worker_dir demo_test_results/<scene>/ \\
        --output demo_test_results/<scene>/predictions.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import List


HEADER_KEYS_TO_CHECK = (
    "scene_token", "mode", "base_model", "lora_path", "pkl_path",
    "resize_factor", "max_new_tokens", "n_frames", "n_questions",
    "demo_questions_path",
    "backend", "api_base", "model_name",
)


def _load_worker_reports(worker_dir: str) -> List[dict]:
    """Load every predictions_worker*.json under worker_dir, sorted by name."""
    paths = sorted(glob.glob(os.path.join(worker_dir, "predictions_worker*.json")))
    if not paths:
        raise FileNotFoundError(
            f"No predictions_worker*.json under {worker_dir!r}. "
            f"Did the worker fan-out run finish?"
        )
    reports = []
    for p in paths:
        with open(p) as f:
            reports.append(json.load(f))
    return reports


def merge_reports(reports: List[dict], strict: bool = False) -> dict:
    """Merge worker reports into a canonical single-report dict."""
    if not reports:
        raise ValueError("merge_reports called with no reports.")

    canonical = {k: reports[0].get(k) for k in HEADER_KEYS_TO_CHECK if k in reports[0]}
    # Sanity: verify all workers agree on header fields.
    mismatches = []
    for i, r in enumerate(reports[1:], start=1):
        for k in HEADER_KEYS_TO_CHECK:
            if k in r and k in canonical and r.get(k) != canonical.get(k):
                mismatches.append((i, k, canonical.get(k), r.get(k)))
    if mismatches:
        msg = "Worker header mismatches:\n" + "\n".join(
            f"  worker[{i}] {k!r}: canonical={c!r}  worker={w!r}"
            for i, k, c, w in mismatches
        )
        if strict:
            raise ValueError(msg)
        print(f"[merge_demo_predictions] WARNING:\n{msg}", file=sys.stderr)

    # Concat + dedup + sort frames.
    seen: dict = {}   # frame_pos -> frame_entry (last-writer wins on dup)
    n_dupes = 0
    for i, r in enumerate(reports):
        for f in r.get("frames", []):
            pos = f.get("frame_pos")
            if pos is None:
                continue
            if pos in seen:
                n_dupes += 1
            seen[pos] = f
    merged_frames = [seen[p] for p in sorted(seen)]
    if n_dupes:
        print(f"[merge_demo_predictions] note: {n_dupes} frame_pos duplicates "
              f"across workers (last-writer wins).", file=sys.stderr)

    canonical["frames"] = merged_frames
    canonical["merged_from_workers"] = len(reports)
    canonical["n_frames_merged"] = len(merged_frames)
    return canonical


def main():
    p = argparse.ArgumentParser(
        prog="python -m nuscenes_pipeline.modules.merge_demo_predictions",
        description="Merge demo_tester worker JSONs into a canonical predictions.json.",
    )
    p.add_argument("--worker_dir", required=True, type=str,
                   help="Directory containing predictions_worker*.json.")
    p.add_argument("--output", type=str, default=None,
                   help="Output path. Default: <worker_dir>/predictions.json.")
    p.add_argument("--strict", action="store_true",
                   help="Error out on header-field mismatch instead of warning.")
    p.add_argument("--keep_workers", action="store_true",
                   help="Do not delete the per-worker JSONs after successful merge.")
    args = p.parse_args()

    reports = _load_worker_reports(args.worker_dir)
    merged = merge_reports(reports, strict=args.strict)

    out_path = args.output or os.path.join(args.worker_dir, "predictions.json")
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)

    n_frames = merged.get("n_frames")
    n_merged = merged["n_frames_merged"]
    print(f"[merge_demo_predictions] merged {len(reports)} workers -> {out_path}")
    print(f"  scene_token: {merged.get('scene_token')}")
    print(f"  frames merged: {n_merged}"
          + (f" (scene has {n_frames})" if n_frames else ""))
    if n_frames and n_merged != n_frames:
        print(f"  WARNING: merged frame count {n_merged} != scene frame count {n_frames}",
              file=sys.stderr)

    if not args.keep_workers:
        for p_ in sorted(glob.glob(os.path.join(args.worker_dir, "predictions_worker*.json"))):
            try:
                os.remove(p_)
            except OSError:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
