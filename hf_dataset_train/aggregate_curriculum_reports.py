"""
Aggregate per-stage eval_report.json files into a single stage x category
accuracy matrix.

Reads <output_root>/stage_*/eval_report.json, writes:
  - <output_root>/curriculum_report.csv   (rows = stages, cols = categories, +macro)
  - <output_root>/curriculum_report.md    (markdown table + per-category
                                            "forgetting" column)

Forgetting column for category C at final stage = max(acc_C across stages 0..N-1)
- acc_C at the final stage. Positive numbers indicate the model lost ground on
that category by the end of the curriculum.

Usage:
    python hf_dataset_train/aggregate_curriculum_reports.py \
        --output_root output/curriculum_v1
"""

import argparse
import csv
import glob
import json
import os
import re
from typing import Dict, List, Tuple


STAGE_DIR_RE = re.compile(r"stage_(\d+)_([A-Za-z]+)$")


def discover_reports(output_root: str) -> List[Tuple[int, str, dict]]:
    """Return [(stage_idx, stage_name, report_dict), ...] sorted by stage idx."""
    out = []
    for sub in sorted(os.listdir(output_root)):
        full = os.path.join(output_root, sub)
        if not os.path.isdir(full):
            continue
        m = STAGE_DIR_RE.match(sub)
        if not m:
            continue
        report_path = os.path.join(full, "eval_report.json")
        if not os.path.exists(report_path):
            print(f"[warn] no eval_report.json in {full}")
            continue
        with open(report_path) as f:
            report = json.load(f)
        out.append((int(m.group(1)), m.group(2), report))
    out.sort(key=lambda x: x[0])
    return out


def collect_categories(reports) -> List[str]:
    cats = set()
    for _, _, r in reports:
        cats.update(r.get("metrics", {}).keys())
    return sorted(cats)


def build_matrix(reports, categories) -> List[Dict]:
    rows = []
    for idx, name, r in reports:
        metrics = r.get("metrics", {})
        row = {"stage_idx": idx, "stage_name": name}
        for cat in categories:
            row[cat] = metrics.get(cat, {}).get("acc")
        # Use stored macro_acc if present, else recompute.
        macro = r.get("macro_acc")
        if macro is None:
            present = [row[c] for c in categories if row[c] is not None]
            macro = sum(present) / len(present) if present else None
        row["macro_acc"] = macro
        rows.append(row)
    return rows


def write_csv(rows, categories, path):
    fieldnames = ["stage_idx", "stage_name"] + categories + ["macro_acc"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k) for k in fieldnames}
            w.writerow(out)


def _fmt(v):
    return "" if v is None else f"{v:.3f}"


def write_markdown(rows, categories, path):
    header = ["stage", "name"] + categories + ["macro"]
    sep = ["---"] * len(header)
    lines = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(sep) + " |")
    for r in rows:
        cells = [str(r["stage_idx"]), r["stage_name"]]
        cells += [_fmt(r[c]) for c in categories]
        cells.append(_fmt(r["macro_acc"]))
        lines.append("| " + " | ".join(cells) + " |")

    # Forgetting column: best-pre-final vs final.
    final = rows[-1] if rows else None
    lines.append("")
    lines.append("## Forgetting (best earlier acc − final-stage acc; higher = more forgetting)")
    lines.append("")
    lines.append("| category | best_earlier | final | forgetting |")
    lines.append("| --- | --- | --- | --- |")
    for cat in categories:
        earlier_vals = [r[cat] for r in rows[:-1] if r[cat] is not None]
        final_val = final[cat] if final else None
        if not earlier_vals or final_val is None:
            lines.append(f"| {cat} | | {_fmt(final_val)} | |")
            continue
        best = max(earlier_vals)
        lines.append(f"| {cat} | {_fmt(best)} | {_fmt(final_val)} | {_fmt(best - final_val)} |")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_root", required=True,
                   help="Curriculum output root (e.g. output/curriculum_v1)")
    args = p.parse_args()

    reports = discover_reports(args.output_root)
    if not reports:
        print(f"No eval_report.json files found under {args.output_root}")
        return

    categories = collect_categories(reports)
    rows = build_matrix(reports, categories)

    csv_path = os.path.join(args.output_root, "curriculum_report.csv")
    md_path = os.path.join(args.output_root, "curriculum_report.md")
    write_csv(rows, categories, csv_path)
    write_markdown(rows, categories, md_path)

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")
    print()
    print(f"Stages: {len(rows)} | Categories: {len(categories)}")
    for r in rows:
        print(f"  stage {r['stage_idx']:02d} ({r['stage_name']:>3s})  macro_acc={_fmt(r['macro_acc'])}")


if __name__ == "__main__":
    main()
