"""
Aggregate per-sample prior_disagreements files into a single summary report.

The `answer_generator.py` worker writes a file per sample to
`prior_disagreements/sample_{idx}_disagreements.json` whenever the VLM flags
a disagreement with a Stage 1B (signal) or 1C (sign) ego-applicability claim.

This script scans that directory and produces `_summary.json` in the same
directory, containing aggregate statistics.

Usage:
    # Default: scan prior_disagreements/ and write prior_disagreements/_summary.json
    python -m nuscenes_pipeline.postprocessing.aggregate_disagreements

    # Custom directory
    python -m nuscenes_pipeline.postprocessing.aggregate_disagreements \\
        --disagreement_dir path/to/disagreements
"""

import argparse

# Reuse the aggregation helper from answer_generator to keep logic in one place.
from nuscenes_pipeline.modules.answer_generator import _aggregate_disagreements

import json
import os


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-sample prior_disagreements files into a summary report"
    )
    parser.add_argument(
        "--disagreement_dir", type=str, default="prior_disagreements",
        help="Directory containing sample_{idx}_disagreements.json files (default: prior_disagreements)"
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output summary path (default: <disagreement_dir>/_summary.json)"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.disagreement_dir):
        print(f"ERROR: {args.disagreement_dir} is not a directory.")
        return

    summary = _aggregate_disagreements(args.disagreement_dir)
    if summary is None:
        print(f"No disagreement files found in {args.disagreement_dir}/")
        return

    out_path = args.out or os.path.join(args.disagreement_dir, "_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Pretty-print key metrics to stdout
    print(f"Scanned: {args.disagreement_dir}/")
    print(f"Samples with disagreements: {summary['total_samples_with_disagreements']}")
    print(f"Total disagreements:        {summary['total_disagreements']}")
    print()
    print("By source:")
    for src, n in summary['by_source'].items():
        print(f"  {src:10s}  {n}")
    print()
    print("By category:")
    for cat, n in summary['by_category'].items():
        print(f"  {cat:40s}  {n}")
    print()
    print("By mismatch type:")
    for kind, n in summary['by_mismatch_type'].items():
        print(f"  {kind:60s}  {n}")
    print()
    print(f"Summary written to: {out_path}")


if __name__ == "__main__":
    main()
