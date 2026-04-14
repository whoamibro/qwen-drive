#!/usr/bin/env python3
"""
Prompt Preview for Question Selector

Prints the full system prompt and user-side text prompt (batch 1 only) for each
category, using a single sample from the dataset. No inference is performed.

Usage:
    # Show prompts for all categories using sample 0
    python -m nuscenes_pipeline.scripts.show_prompts

    # Show prompt for a specific category
    python -m nuscenes_pipeline.scripts.show_prompts --category Observation

    # Use a different sample index
    python -m nuscenes_pipeline.scripts.show_prompts --sample_idx 42
"""

import os
import argparse
from typing import Optional

from nuscenes_pipeline.core.nuscenes_data_loader import NuScenesDataLoader
from nuscenes_pipeline.core.qa_utils import (
    VALID_CATEGORIES,
    SceneAnalyzer,
    VQAResultsLoader,
    PRIOR_BUILDERS,
    load_question_bank,
    get_category_templates,
    build_batch_validation_prompt,
)
from nuscenes_pipeline.modules.question_selector import (
    SYSTEM_PROMPT,
    load_risk_assessment_response,
    load_traffic_analysis_response,
)


def main():
    parser = argparse.ArgumentParser(description="Preview question selector prompts")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Sample index to use for building prompts (default: 0)")
    parser.add_argument("--category", type=str, default="all",
                        choices=VALID_CATEGORIES + ["all"],
                        help='Category to preview, or "all" (default: all)')
    parser.add_argument("--batch_size", type=int, default=5,
                        help="Number of templates per batch (default: 5)")
    parser.add_argument("--filter_distance", type=float, default=50.0,
                        help="Max distance (m) for objects (default: 50)")
    parser.add_argument("--rear_filter", type=float, default=20.0,
                        help="Max distance (m) for rear non-vehicles (default: 20)")
    parser.add_argument("--pkl_path", type=str,
                        default=os.environ.get("NUSCENES_PKL_PATH",
                                               "nuscenes2d_ego_temporal_infos_val.pkl"),
                        help="Path to nuScenes pkl file")
    parser.add_argument("--question_bank", type=str,
                        default=os.environ.get("QUESTION_BANK_PATH",
                                               "question_bank.json"),
                        help="Path to question bank JSON")
    parser.add_argument("--vqa_results_dir", type=str, default="vqa_results",
                        help="Directory containing VQA results")
    parser.add_argument("--risk_results_dir", type=str,
                        default="risk_assessment_results",
                        help="Directory containing risk assessment results")
    parser.add_argument("--traffic_results_dir", type=str,
                        default="traffic_analysis_results",
                        help="Directory containing traffic analysis results")
    args = parser.parse_args()

    # Load data
    loader = NuScenesDataLoader(args.pkl_path)
    analyzer = SceneAnalyzer(loader)
    vqa_loader = VQAResultsLoader(args.vqa_results_dir)
    question_bank = load_question_bank(args.question_bank)

    # Analyze sample
    scene_data = analyzer.analyze_sample(
        args.sample_idx,
        max_distance=args.filter_distance,
        rear_filter_distance=args.rear_filter,
    )
    vqa_result = vqa_loader.extract_risk_summary(args.sample_idx)

    # Load prior analysis responses
    risk_response = load_risk_assessment_response(args.risk_results_dir, args.sample_idx)
    traffic_response = load_traffic_analysis_response(args.traffic_results_dir, args.sample_idx)

    categories = VALID_CATEGORIES if args.category == "all" else [args.category]

    # Print system prompt once
    sep = "=" * 80
    print(sep)
    print("SYSTEM PROMPT")
    print(sep)
    print(SYSTEM_PROMPT)
    print()

    # Print user prompt per category (batch 1 only)
    for cat in categories:
        builder = PRIOR_BUILDERS[cat]
        if cat in ["Dynamic_Agents_and_Risk_Assessment", "Causal_and_Hypothetical_Reasoning"]:
            prior_knowledge = builder(scene_data, vqa_result)
        else:
            prior_knowledge = builder(scene_data)

        templates = get_category_templates(question_bank, cat)
        num_templates = len(templates)
        num_batches = (num_templates + args.batch_size - 1) // args.batch_size

        # Build batch-1 prompt
        batch_templates = templates[:args.batch_size]
        prompt = build_batch_validation_prompt(
            category=cat,
            prior_knowledge=prior_knowledge,
            templates=batch_templates,
            start_idx=0,
            batch_num=1,
            total_batches=num_batches,
        )

        # Prepend prior analysis sections (same logic as question_selector)
        if cat == "Dynamic_Agents_and_Risk_Assessment" and risk_response:
            prompt = (
                "\n\n=== RISK ASSESSMENT ANALYSIS (from prior analysis) ===\n"
                f"{risk_response}\n"
                "=== END OF RISK ASSESSMENT ANALYSIS ===\n\n"
            ) + prompt

        if cat == "Traffic_Signs_and_Signals" and traffic_response:
            prompt = (
                "\n\n=== TRAFFIC SIGNAL ANALYSIS (from prior analysis) ===\n"
                f"{traffic_response}\n"
                "=== END OF TRAFFIC SIGNAL ANALYSIS ===\n\n"
            ) + prompt

        print(sep)
        print(f"USER PROMPT — {cat}  (batch 1/{num_batches}, {num_templates} templates total)")
        print(sep)
        print("[6 camera images would be inserted here]")
        print()
        print(prompt)
        print()


if __name__ == "__main__":
    main()
