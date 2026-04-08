"""
Count and summarize QA pair statistics from *_qa_results.json files.

Usage:
    python count_qa_stats.py                          # all *_qa_results.json in current dir
    python count_qa_stats.py sample_0_qa_results.json # single file
    python count_qa_stats.py outputs_qa_vllm/         # all *_qa_results.json in a directory
"""

import json
import sys
import glob
import os
from collections import defaultdict, Counter


def detect_version(data):
    """Detect JSON format version: 'v1' has 'qa_pairs', 'v2' has 'qa_results'."""
    if "qa_pairs" in data:
        return "v1"
    elif "qa_results" in data:
        return "v2"
    else:
        raise ValueError(f"Unknown format. Top-level keys: {list(data.keys())}")


def normalize_to_common(data):
    """
    Normalize both v1 and v2 formats into a common structure:
      {
        "sample_idx": int,
        "scene_meta": dict,
        "total_templates_processed": int,
        "total_qa_pairs_generated": int,
        "version": str,
        "qa_pairs": [  # flat list of individual QA pairs
            {
                "template_idx": int,
                "category": str,
                "answer_type": str,
                "positive": dict or None,
                "contrastive": dict or None,
                "contrastive_skip_reason": str or None,  # v2 only
                "pair_id": int or None,  # v2 only
            }, ...
        ]
      }
    """
    version = detect_version(data)

    if version == "v1":
        # v1 is already flat — just pass through with minor normalization
        qa_pairs = []
        for qa in data["qa_pairs"]:
            qa_pairs.append({
                "template_idx": qa["template_idx"],
                "category": qa["category"],
                "answer_type": qa["answer_type"],
                "positive": qa.get("positive"),
                "contrastive": qa.get("contrastive"),
                "contrastive_skip_reason": None,
                "pair_id": None,
            })
        return {
            "sample_idx": data["sample_idx"],
            "scene_meta": data.get("scene_meta", {}),
            "total_templates_processed": data.get("total_templates_processed", 0),
            "total_qa_pairs_generated": data.get("total_qa_pairs_generated", len(qa_pairs)),
            "version": "v1",
            "qa_pairs": qa_pairs,
        }

    else:  # v2
        # v2 groups pairs under qa_results[].pairs[] — flatten them
        qa_pairs = []
        for tmpl in data["qa_results"]:
            for pair in tmpl["pairs"]:
                qa_pairs.append({
                    "template_idx": tmpl["template_idx"],
                    "category": tmpl["category"],
                    "answer_type": tmpl["answer_type"],
                    "positive": pair.get("positive"),
                    "contrastive": pair.get("contrastive"),
                    "contrastive_skip_reason": pair.get("contrastive_skip_reason"),
                    "pair_id": pair.get("pair_id"),
                })
        return {
            "sample_idx": data["sample_idx"],
            "scene_meta": data.get("scene_meta", {}),
            "total_templates_processed": data.get("total_templates_processed", 0),
            "total_qa_pairs_generated": data.get("total_pairs_generated", len(qa_pairs)),
            "version": "v2",
            "qa_pairs": qa_pairs,
        }


def load_qa_files(path=None):
    """Load one or more *_qa_results.json files and return list of normalized dicts."""
    if path is None:
        path = "."

    if os.path.isfile(path):
        files = [path]
    elif os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*_qa_results*.json")))
    else:
        files = sorted(glob.glob(path))

    if not files:
        print(f"No *_qa_results*.json files found at: {path}")
        sys.exit(1)

    data_list = []
    for f in files:
        with open(f) as fh:
            raw = json.load(fh)
        data_list.append(normalize_to_common(raw))
    return data_list, files


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def compute_stats(data_list, file_paths):
    # ── 1. Per-sample summary ─────────────────────────────────
    print_section("Per-Sample Summary")
    print(f"{'File':<45} {'Ver':>4} {'Sample':>6} {'Templates':>10} {'QA Pairs':>9}")
    print("-" * 79)

    total_qa = 0
    total_templates = 0
    for d, fp in zip(data_list, file_paths):
        fname = os.path.basename(fp)
        n_qa = len(d["qa_pairs"])
        n_tmpl = d.get("total_templates_processed", "N/A")
        ver = d.get("version", "?")
        total_qa += n_qa
        if isinstance(n_tmpl, int):
            total_templates += n_tmpl
        print(f"{fname:<45} {ver:>4} {d['sample_idx']:>6} {str(n_tmpl):>10} {n_qa:>9}")

    print("-" * 79)
    print(f"{'TOTAL':<45} {'':>4} {len(data_list):>6} {total_templates:>10} {total_qa:>9}")

    # Flatten all QA pairs
    all_qa = []
    for d in data_list:
        for qa in d["qa_pairs"]:
            qa_entry = {**qa, "sample_idx": d["sample_idx"]}
            all_qa.append(qa_entry)

    # ── 2. QA count per template_idx ──────────────────────────
    print_section("QA Pairs per Template Index")
    template_counter = Counter(qa["template_idx"] for qa in all_qa)
    print(f"{'Template Idx':>12} {'Count':>8} {'Percentage':>12}")
    print("-" * 35)
    for tidx in sorted(template_counter.keys()):
        cnt = template_counter[tidx]
        pct = cnt / len(all_qa) * 100
        print(f"{tidx:>12} {cnt:>8} {pct:>11.1f}%")
    print("-" * 35)
    print(f"{'Total':>12} {len(all_qa):>8} {'100.0%':>12}")

    # ── 3. QA count per category ──────────────────────────────
    print_section("QA Pairs per Category")
    cat_counter = Counter(qa["category"] for qa in all_qa)
    print(f"{'Category':<25} {'Count':>8} {'Percentage':>12}")
    print("-" * 48)
    for cat in sorted(cat_counter.keys()):
        cnt = cat_counter[cat]
        pct = cnt / len(all_qa) * 100
        print(f"{cat:<25} {cnt:>8} {pct:>11.1f}%")
    print("-" * 48)
    print(f"{'Total':<25} {len(all_qa):>8} {'100.0%':>12}")

    # ── 4. QA count per answer_type ───────────────────────────
    print_section("QA Pairs per Answer Type")
    atype_counter = Counter(qa["answer_type"] for qa in all_qa)
    print(f"{'Answer Type':<25} {'Count':>8} {'Percentage':>12}")
    print("-" * 48)
    for atype in sorted(atype_counter.keys()):
        cnt = atype_counter[atype]
        pct = cnt / len(all_qa) * 100
        print(f"{atype:<25} {cnt:>8} {pct:>11.1f}%")
    print("-" * 48)
    print(f"{'Total':<25} {len(all_qa):>8} {'100.0%':>12}")

    # ── 5. Cross-tab: template_idx × answer_type ──────────────
    print_section("Cross-Tab: Template Index × Answer Type")
    cross = defaultdict(Counter)
    for qa in all_qa:
        cross[qa["template_idx"]][qa["answer_type"]] += 1

    all_atypes = sorted(set(qa["answer_type"] for qa in all_qa))
    header = f"{'Template':>10}" + "".join(f"{at:>12}" for at in all_atypes) + f"{'Total':>10}"
    print(header)
    print("-" * len(header))
    for tidx in sorted(cross.keys()):
        row = f"{tidx:>10}"
        row_total = 0
        for at in all_atypes:
            c = cross[tidx][at]
            row_total += c
            row += f"{c:>12}"
        row += f"{row_total:>10}"
        print(row)

    # ── 6. Contrastive pair availability ──────────────────────
    print_section("Contrastive Pair Availability")
    has_contrastive = sum(1 for qa in all_qa if qa.get("contrastive"))
    has_positive = sum(1 for qa in all_qa if qa.get("positive"))
    skipped = sum(1 for qa in all_qa if qa.get("contrastive_skip_reason"))
    print(f"  QA pairs with positive:     {has_positive:>6} / {len(all_qa)}")
    print(f"  QA pairs with contrastive:  {has_contrastive:>6} / {len(all_qa)}")
    print(f"  Contrastive skipped:        {skipped:>6} / {len(all_qa)}")
    print(f"  Total Q-A entries (pos+con): {has_positive + has_contrastive:>5}")

    # ── 7. Confidence distribution ────────────────────────────
    print_section("Confidence Distribution (Positive)")
    conf_counter = Counter()
    for qa in all_qa:
        pos = qa.get("positive", {})
        if pos:
            conf_counter[pos.get("confidence", "N/A")] += 1
    print(f"{'Confidence':<20} {'Count':>8} {'Percentage':>12}")
    print("-" * 43)
    for conf in sorted(conf_counter.keys()):
        cnt = conf_counter[conf]
        pct = cnt / len(all_qa) * 100
        print(f"{conf:<20} {cnt:>8} {pct:>11.1f}%")

    # ── 8. Grand totals ──────────────────────────────────────
    print_section("Grand Totals")
    print(f"  Number of sample files:      {len(data_list)}")
    print(f"  Total templates processed:   {total_templates}")
    print(f"  Total QA pairs generated:    {total_qa}")
    print(f"  Avg QA pairs per sample:     {total_qa / len(data_list):.1f}")
    if total_templates > 0:
        print(f"  QA yield (pairs/templates):  {total_qa / total_templates:.1%}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    data_list, file_paths = load_qa_files(path)
    compute_stats(data_list, file_paths)
