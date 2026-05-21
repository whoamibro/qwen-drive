"""
Split a unified SFT JSON into per-category files for curriculum-learning training.

Each sample in the input is tagged with a `category` field (added by
`prepare_sft_dataset.py`). This script groups samples by category and writes
one file per category using the project's standard 3-letter abbreviations:

    Observation                          -> OBS
    Identification                       -> IDN
    Attributes_and_States                -> AAS
    Spatial_Relationships_and_Occlusion  -> SRO
    Traffic_Signs_and_Signals            -> TSS
    Road_Markings_and_Lane_Configuration -> RML
    Dynamic_Agents_and_Risk_Assessment   -> DRA
    Right_of_Way_and_Planning            -> RWP
    Environmental_and_Sensor_Conditions  -> ESC
    Causal_and_Hypothetical_Reasoning    -> CHR

Output file naming: <input_stem>_<ABBR>.json
e.g.  sft_train_qwen3vl.json  ->  sft_train_qwen3vl_OBS.json, ..., sft_train_qwen3vl_CHR.json

Curriculum-learning usage
-------------------------
The expected training order (easiest perceptual -> hardest reasoning) is:
    OBS -> IDN -> AAS -> SRO -> TSS -> RML -> DRA -> RWP -> ESC -> CHR

Train each stage sequentially, resuming from the previous stage's LoRA checkpoint.

Fallback: matching from qa_results/
-----------------------------------
Older SFT files produced before `prepare_sft_dataset.py` was updated do not
carry the `category` tag. Pass `--qa_results_dir qa_results` to recover the
category for each sample by matching its (image set, TASK question) tuple back
against the original `qa_results/sample_*_qa_results.json` entries.

Usage
-----
    # Standard run (category tag must be present in input samples)
    python -m nuscenes_pipeline.postprocessing.split_sft_by_category \\
        --input sft_dataset/sft_train_qwen3vl.json \\
        --output_dir sft_dataset

    # Both splits in one go
    python -m nuscenes_pipeline.postprocessing.split_sft_by_category \\
        --input sft_dataset/sft_train_qwen3vl.json sft_dataset/sft_val_qwen3vl.json \\
        --output_dir sft_dataset

    # Legacy inputs without category tag — recover via qa_results lookup
    python -m nuscenes_pipeline.postprocessing.split_sft_by_category \\
        --input sft_dataset/sft_train_qwen3vl.json \\
        --output_dir sft_dataset \\
        --qa_results_dir qa_results
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional


CATEGORY_TO_ABBR: Dict[str, str] = {
    "Observation": "OBS",
    "Identification": "IDN",
    "Attributes_and_States": "AAS",
    "Spatial_Relationships_and_Occlusion": "SRO",
    "Traffic_Signs_and_Signals": "TSS",
    "Road_Markings_and_Lane_Configuration": "RML",
    "Dynamic_Agents_and_Risk_Assessment": "DRA",
    "Right_of_Way_and_Planning": "RWP",
    "Environmental_and_Sensor_Conditions": "ESC",
    "Causal_and_Hypothetical_Reasoning": "CHR",
}

# Canonical curriculum order (easy perceptual -> hard reasoning).
CURRICULUM_ORDER: List[str] = [
    "OBS", "IDN", "AAS", "SRO", "TSS",
    "RML", "DRA", "RWP", "ESC", "CHR",
]

# Question text appears after this header in the human turn produced by
# prepare_sft_dataset.build_user_prompt_no_objects.
_TASK_HEADER_RE = re.compile(r"TASK:\s*\n(.*?)(?:\n\nOptions:|$)", re.DOTALL)


def _extract_task_question(human_value: str) -> Optional[str]:
    """Pull the bare question text out of a human turn."""
    m = _TASK_HEADER_RE.search(human_value)
    if not m:
        return None
    return m.group(1).strip()


def _build_question_to_category_lookup(qa_results_dir: str) -> Dict[str, str]:
    """Walk qa_results/ and map every instantiated question text to its
    source category. Multiple QA pairs can share a question, but they all
    come from the same category, so a flat string->category map is safe."""
    lookup: Dict[str, str] = {}
    files = sorted(glob.glob(os.path.join(qa_results_dir, "sample_*_qa_results.json")))
    for fpath in files:
        try:
            with open(fpath) as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        for qa in d.get("qa_results", []):
            category = qa.get("category")
            if not category:
                continue
            for pair in qa.get("pairs", []):
                for slot in ("positive", "contrastive"):
                    blk = pair.get(slot)
                    if blk and blk.get("instantiated_question"):
                        lookup[blk["instantiated_question"].strip()] = category
                for vc in pair.get("vlm_proposed_contrastives", []) or []:
                    if vc and vc.get("instantiated_question"):
                        lookup[vc["instantiated_question"].strip()] = category
    return lookup


def _category_of_sample(sample: Dict, fallback_lookup: Optional[Dict[str, str]]) -> Optional[str]:
    """Resolve a sample's category, preferring the explicit tag and falling
    back to a question-text lookup built from qa_results/."""
    cat = sample.get("category")
    if cat:
        return cat
    if not fallback_lookup:
        return None
    for turn in sample.get("conversations", []):
        if turn.get("from") == "human":
            q = _extract_task_question(turn.get("value", ""))
            if q and q in fallback_lookup:
                return fallback_lookup[q]
            break
    return None


def split_one(input_path: str, output_dir: str,
              fallback_lookup: Optional[Dict[str, str]]) -> Dict[str, int]:
    """Split a single SFT JSON file into per-category files."""
    with open(input_path) as f:
        data = json.load(f)

    bucket: Dict[str, List[Dict]] = defaultdict(list)
    unknown: List[int] = []

    for i, sample in enumerate(data):
        cat = _category_of_sample(sample, fallback_lookup)
        abbr = CATEGORY_TO_ABBR.get(cat) if cat else None
        if abbr is None:
            unknown.append(i)
            continue
        bucket[abbr].append(sample)

    stem = os.path.splitext(os.path.basename(input_path))[0]
    counts: Dict[str, int] = {}
    for abbr in CURRICULUM_ORDER:
        out_name = f"{stem}_{abbr}.json"
        out_path = os.path.join(output_dir, out_name)
        samples = bucket.get(abbr, [])
        with open(out_path, "w") as f:
            json.dump(samples, f, ensure_ascii=False)
        counts[abbr] = len(samples)

    print(f"\n--- {input_path} ({len(data):,} samples) ---")
    for abbr in CURRICULUM_ORDER:
        n = counts[abbr]
        pct = 100.0 * n / len(data) if data else 0.0
        print(f"  {abbr}: {n:>7,d}  ({pct:5.1f}%)")
    if unknown:
        print(f"  UNCATEGORIZED: {len(unknown):,} (not written; first few indices: {unknown[:5]})")

    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Split a unified SFT JSON into per-category files for curriculum learning"
    )
    parser.add_argument(
        "--input", nargs="+", required=True,
        help="One or more SFT JSON files (e.g. sft_dataset/sft_train_qwen3vl.json)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="sft_dataset",
        help="Output directory for the per-category files (default: sft_dataset)"
    )
    parser.add_argument(
        "--qa_results_dir", type=str, default=None,
        help="If the input lacks a 'category' tag, pass the qa_results/ directory "
             "to recover categories by matching question text. Optional."
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    fallback_lookup = None
    if args.qa_results_dir:
        print(f"Building question -> category lookup from {args.qa_results_dir}/ ...")
        fallback_lookup = _build_question_to_category_lookup(args.qa_results_dir)
        print(f"  Loaded {len(fallback_lookup):,} unique questions.")

    for in_path in args.input:
        split_one(in_path, args.output_dir, fallback_lookup)


if __name__ == "__main__":
    main()
