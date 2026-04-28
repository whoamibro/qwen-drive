#!/usr/bin/env python3
"""
Prompt Preview for Answer Generator (Stage 3)

Renders the full Stage 3 system prompt + user prompt for a single template of
a single sample, exactly the way the answer_generator pipeline would. No VLM
inference is performed — the prompt is built and printed only.

Usage:
    # Default: sample 0, first template, answer-first mode
    python show_answer_generator_prompts.py

    # Pick a sample + filter to a specific category / answer_type
    python show_answer_generator_prompts.py --sample_idx 101 --category Observation --answer_type mcq

    # Reasoning-first mode
    python show_answer_generator_prompts.py --sample_idx 101 --answer_mode r_a

    # Save to file instead of stdout
    python show_answer_generator_prompts.py --sample_idx 101 --output prompt.txt
"""

import os
import sys
import json
import argparse

from nuscenes_pipeline.core.nuscenes_data_loader import NuScenesDataLoader
from nuscenes_pipeline.core.qa_utils import SceneAnalyzer, load_question_bank
from nuscenes_pipeline.modules.answer_generator import (
    SYSTEM_PROMPT,
    format_object_positions,
    format_closest_objects,
    enrich_template_with_question_bank,
    pre_instantiate_pairs,
    build_verification_prompt,
)
from nuscenes_pipeline.modules.question_selector import (
    load_risk_assessment_response,
    load_traffic_analysis_response,
    load_traffic_sign_response,
)


def pick_template(sample_block, category=None, answer_type=None):
    """Return the first (cat, qkey, q_data) matching the filters, or None."""
    for cat, qs in sample_block.items():
        if category and cat != category:
            continue
        for qkey, q in qs.items():
            if answer_type and q.get("answer_type") != answer_type:
                continue
            return cat, qkey, q
    return None


def main():
    parser = argparse.ArgumentParser(description="Preview answer-generator (Stage 3) prompts")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Sample index to render (default: 0)")
    parser.add_argument("--category", type=str, default=None,
                        help="Filter to a single category (default: any)")
    parser.add_argument("--answer_type", type=str, default=None,
                        choices=["y_or_n", "mcq", "num_count", "distance", "open_ended"],
                        help="Filter to a single answer_type (default: any)")
    parser.add_argument("--answer_mode", type=str, default="a_r", choices=["a_r", "r_a"],
                        help="a_r=answer-first, r_a=reasoning-first (default: a_r)")
    parser.add_argument("--max_pairs", type=int, default=3,
                        help="Pairs to pre-instantiate (default: 3)")
    parser.add_argument("--filter_distance", type=float, default=50.0,
                        help="Max object distance, m (default: 50)")
    parser.add_argument("--rear_filter", type=float, default=20.0,
                        help="Max rear non-vehicle distance, m (default: 20)")
    parser.add_argument("--pkl_path", type=str,
                        default=os.environ.get("NUSCENES_PKL_PATH",
                                               "./data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl"))
    parser.add_argument("--question_bank", type=str,
                        default=os.environ.get("QUESTION_BANK_PATH",
                                               "./data/question_bank_v4_static.json"))
    parser.add_argument("--stage1_dir", type=str, default="qa_outputs/all",
                        help="Stage-2 (question_selector) output dir (default: qa_outputs/all)")
    parser.add_argument("--risk_results_dir", type=str, default="risk_assessment_results")
    parser.add_argument("--traffic_results_dir", type=str, default="traffic_signal_analysis_results")
    parser.add_argument("--sign_results_dir", type=str, default="traffic_sign_results")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional file path to write the prompt (default: stdout)")
    args = parser.parse_args()

    # Load data sources
    loader = NuScenesDataLoader(args.pkl_path)
    analyzer = SceneAnalyzer(loader)
    question_bank = load_question_bank(args.question_bank)
    scene = analyzer.analyze_sample(
        args.sample_idx,
        max_distance=args.filter_distance,
        rear_filter_distance=args.rear_filter,
    )

    # Stage 2 output for this sample
    stage2_path = os.path.join(args.stage1_dir, f"sample_{args.sample_idx}_applicable_questions.json")
    if not os.path.isfile(stage2_path):
        sys.exit(f"Stage 2 output not found: {stage2_path}")
    stage2 = json.load(open(stage2_path))
    sample_block = stage2.get(str(args.sample_idx))
    if not sample_block:
        sys.exit(f"Sample {args.sample_idx} not present in {stage2_path}")

    chosen = pick_template(sample_block, args.category, args.answer_type)
    if chosen is None:
        sys.exit(f"No template matched filters: category={args.category!r}, answer_type={args.answer_type!r}")
    cat, qkey, q_data = chosen

    # Enrich with question bank metadata
    tmpl = enrich_template_with_question_bank(q_data, cat, question_bank)
    tmpl["q_key"] = qkey

    # Spatial / driving inputs
    driving_command = scene["ego_info"].get("driving_command", "Unknown")
    object_positions_text = format_object_positions(scene)
    closest_objects_text = format_closest_objects(scene)

    # Prior-stage analyses (same loaders the pipeline uses)
    risk = load_risk_assessment_response(args.risk_results_dir, args.sample_idx)
    traffic = load_traffic_analysis_response(args.traffic_results_dir, args.sample_idx)
    sign = load_traffic_sign_response(args.sign_results_dir, args.sample_idx)

    # Pre-instantiated pairs (deterministic seed matches answer_generator's)
    pre_pairs = pre_instantiate_pairs(
        template=tmpl,
        scene_data=scene,
        max_pairs=args.max_pairs,
        seed=args.sample_idx * 10000,
    )

    # Build user prompt
    prompt = build_verification_prompt(
        driving_command=driving_command,
        object_positions_text=object_positions_text,
        closest_objects_text=closest_objects_text,
        template=tmpl,
        pre_pairs=pre_pairs,
        template_num=1,
        total_templates=1,
        answer_mode=args.answer_mode,
    )

    # Apply prior prepends in the same order as answer_generator
    if cat == "Dynamic_Agents_and_Risk_Assessment" and risk:
        prompt = (
            "\n\n=== RISK ASSESSMENT ANALYSIS (from prior analysis) ===\n"
            f"{risk}\n"
            "=== END OF RISK ASSESSMENT ANALYSIS ===\n\n"
        ) + prompt
    if traffic:
        prompt = (
            "\n\n=== TRAFFIC SIGNAL ANALYSIS (from prior analysis) ===\n"
            f"{traffic}\n"
            "=== END OF TRAFFIC SIGNAL ANALYSIS ===\n\n"
        ) + prompt
    if sign:
        prompt = (
            "\n\n=== TRAFFIC SIGN EXTRACTION (from prior analysis) ===\n"
            f"{sign}\n"
            "=== END OF TRAFFIC SIGN EXTRACTION ===\n\n"
        ) + prompt

    sep = "=" * 80
    out_lines = [
        sep,
        f"SAMPLE {args.sample_idx} | category={cat} | q_key={qkey} | answer_type={tmpl.get('answer_type')} | answer_mode={args.answer_mode}",
        f"template: {tmpl.get('template')}",
        sep,
        "",
        sep,
        "SYSTEM PROMPT",
        sep,
        SYSTEM_PROMPT,
        "",
        sep,
        "USER PROMPT",
        sep,
        "[6 camera images would be inserted here]",
        "",
        prompt,
    ]
    out = "\n".join(out_lines)

    if args.output:
        with open(args.output, "w") as fh:
            fh.write(out)
        print(f"Wrote {len(out)} bytes to {args.output}")
    else:
        print(out)


if __name__ == "__main__":
    main()
