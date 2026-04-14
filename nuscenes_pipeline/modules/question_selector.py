#!/usr/bin/env python3
"""
Question Selector Module for nuScenes QA Dataset

Selects applicable question templates from a question bank for each nuScenes sample.
Uses scene context (3D objects, ego state, risk/traffic analysis results) to validate
which templates can be grounded in the current scene.

Prerequisites:
    # Start vLLM server first:
    vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct --tensor-parallel-size 8

Usage:
    # Process samples 0 to 100 for all categories with 4 workers
    python -m nuscenes_pipeline.modules.question_selector \\
        --start_idx 0 --end_idx 100 --category all --num_workers 4

    # Process specific category
    python -m nuscenes_pipeline.modules.question_selector \\
        --start_idx 0 --end_idx 100 --category Observation --num_workers 8
"""

import os
import io
import sys
import json
import re
import time
import base64
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from multiprocessing import Pool
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

from nuscenes_pipeline.core.nuscenes_data_loader import NuScenesDataLoader
from nuscenes_pipeline.core.qa_utils import (
    VALID_CATEGORIES,
    CAMERA_NAMES,
    CAMERA_NAME_MAP,
    VEHICLE_TYPES,
    SceneAnalyzer,
    VQAResultsLoader,
    PRIOR_BUILDERS,
    load_question_bank,
    get_category_templates,
    build_batch_validation_prompt
)


def _load_prior_analysis_response(results_dir: str, sample_idx: int, file_suffix: str) -> Optional[str]:
    """
    Load a prior analysis response for a given sample index.
    Searches for files matching: {sample_idx:04d}_*_{file_suffix}.json
    Returns the 'response' field text, or None if not found.
    """
    if not results_dir or not os.path.isdir(results_dir):
        return None

    import glob as glob_mod
    pattern = os.path.join(results_dir, f"{sample_idx:04d}_*_{file_suffix}.json")
    matches = glob_mod.glob(pattern)
    if not matches:
        return None

    try:
        with open(matches[0], 'r') as f:
            data = json.load(f)
        response = data.get('response', '')
        if response:
            return response.strip()
    except (json.JSONDecodeError, IOError):
        pass

    return None


def load_risk_assessment_response(risk_results_dir: str, sample_idx: int) -> Optional[str]:
    """Load risk assessment response. Files: {idx:04d}_*_single_frame.json"""
    return _load_prior_analysis_response(risk_results_dir, sample_idx, "single_frame")


def load_traffic_analysis_response(traffic_results_dir: str, sample_idx: int) -> Optional[str]:
    """Load traffic analysis response. Files: {idx:04d}_*_traffic.json"""
    return _load_prior_analysis_response(traffic_results_dir, sample_idx, "traffic")


def image_to_base64_data_uri(img_pil: Image.Image, format: str = "JPEG") -> str:
    """Convert a PIL Image to a base64 data URI string."""
    buffer = io.BytesIO()
    img_pil.save(buffer, format=format)
    base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime_type = "image/jpeg" if format.upper() == "JPEG" else f"image/{format.lower()}"
    return f"data:{mime_type};base64,{base64_str}"


def parse_json_response(response: str) -> Optional[Dict]:
    """
    Parse JSON from the model response.
    Same logic as Qwen3VLInference.parse_json_response.
    """
    # Try to parse the entire response as JSON
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code blocks
    json_patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'\{[^{}]*"category"[^{}]*"selected_templates"[^{}]*\[.*?\][^{}]*\}'
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

    # Try to find any JSON object in the response
    brace_count = 0
    start_idx = None
    for i, char in enumerate(response):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx is not None:
                try:
                    json_str = response[start_idx:i+1]
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    start_idx = None

    return None


# Egocentric camera order and labels (matching risk assessment pipeline)
EGOCENTRIC_CAMERA_NAMES = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
]
REAR_CAMERAS = {'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'}
CAM_LABELS = {
    "CAM_FRONT_LEFT": "Image 1: Front-left Camera",
    "CAM_FRONT": "Image 2: Front Camera",
    "CAM_FRONT_RIGHT": "Image 3: Front-right Camera",
    "CAM_BACK_LEFT": "Image 4: Rear-left Camera",
    "CAM_BACK": "Image 5: Rear Camera",
    "CAM_BACK_RIGHT": "Image 6: Rear-right Camera",
}

# System prompt matching egocentric camera order (same as risk assessment pipeline)
SYSTEM_PROMPT = """You are an expert autonomous driving vision-language model assistant performing ego-centric scene analysis.
You are analyzing 6 camera views from an autonomous vehicle (the "ego-vehicle") in the following order:
1. Front-left camera (CAM_FRONT_LEFT)
2. Front camera (CAM_FRONT)
3. Front-right camera (CAM_FRONT_RIGHT)
4. Rear-left camera (CAM_BACK_LEFT)
5. Rear camera (CAM_BACK)
6. Rear-right camera (CAM_BACK_RIGHT)

Rear camera images are horizontally flipped for egocentric consistency (left stays left, right stays right).

IMPORTANT — Ego-Centric Perspective:
All observations and reasoning must be anchored to the ego-vehicle's position, heading, and driving context.
- Spatial references (e.g., "ahead", "left lane", "behind") are relative to the ego-vehicle, NOT absolute coordinates.
- "Relevance" means relevance to the ego-vehicle's current driving situation — objects, signals, and road features matter only insofar as they affect the ego-vehicle's path, decisions, or safety.
- When a template mentions the ego-vehicle's lane, path, or vicinity, ground those terms using the camera views and prior knowledge from the ego-vehicle's perspective.

Your task is to evaluate question templates and select the ones that are applicable to the current scene, judging applicability strictly from the ego-vehicle's perspective."""


# ============================================================================
# Worker process initializer and global state
# ============================================================================

# Per-process globals (initialized once per worker via _worker_init)
_worker_client = None
_worker_loader = None
_worker_analyzer = None
_worker_vqa_loader = None
_worker_question_bank = None
_worker_risk_results_dir = None
_worker_traffic_results_dir = None


def _worker_init(
    model_name: str, api_base: str, api_key: str,
    pkl_path: str, question_bank_path: str, vqa_results_dir: str,
    risk_results_dir: str = "",
    traffic_results_dir: str = ""
):
    """
    Initializer for each worker process.
    Creates the OpenAI client, NuScenesDataLoader, SceneAnalyzer, VQAResultsLoader,
    and loads the question bank ONCE per worker.
    """
    global _worker_client, _worker_loader, _worker_analyzer
    global _worker_vqa_loader, _worker_question_bank
    global _worker_risk_results_dir, _worker_traffic_results_dir

    from openai import OpenAI as _OpenAI
    _worker_client = _OpenAI(
        base_url=api_base,
        api_key=api_key,
        timeout=120.0,
        max_retries=3,
    )
    _worker_loader = NuScenesDataLoader(pkl_path)
    _worker_analyzer = SceneAnalyzer(_worker_loader)
    _worker_vqa_loader = VQAResultsLoader(vqa_results_dir)
    _worker_question_bank = load_question_bank(question_bank_path)
    _worker_risk_results_dir = risk_results_dir
    _worker_traffic_results_dir = traffic_results_dir


# ============================================================================
# Worker function for multiprocessing
# ============================================================================

def _worker_process_sample(
    sample_idx: int,
    model_name: str,
    category: str,
    resize_factor: int,
    max_new_tokens: int,
    filter_distance: float,
    rear_filter: Optional[float],
    batch_size: int,
    output_dir: str,
) -> Dict:
    """
    Worker function that processes a single sample for QA generation using the vLLM API.
    Uses per-process client/loader/analyzer initialized by _worker_init.
    Produces identical prompts to qa_generator_continuous.py.
    """
    global _worker_client, _worker_loader, _worker_analyzer
    global _worker_vqa_loader, _worker_question_bank
    global _worker_risk_results_dir, _worker_traffic_results_dir

    try:
        client = _worker_client
        loader = _worker_loader
        analyzer = _worker_analyzer
        vqa_loader = _worker_vqa_loader
        question_bank = _worker_question_bank
        risk_results_dir = _worker_risk_results_dir
        traffic_results_dir = _worker_traffic_results_dir

        # Analyze sample
        scene_data = analyzer.analyze_sample(
            sample_idx,
            max_distance=filter_distance,
            rear_filter_distance=rear_filter
        )

        # Get sample object
        sample = loader.get_sample(sample_idx)

        # Load VQA result if available
        vqa_result = vqa_loader.extract_risk_summary(sample_idx)

        # Process categories
        categories = VALID_CATEGORIES if category == "all" else [category]

        inference_results = {}
        results = {}

        # Prepare base64-encoded images once (shared across all category batches)
        # Uses egocentric order (FL, F, FR, RL, R, RR) with labels and rear-flip
        image_content = []
        for cam_name in EGOCENTRIC_CAMERA_NAMES:
            label = CAM_LABELS[cam_name]
            image_content.append({"type": "text", "text": f"=== {label} ==="})
            img_path = sample.cameras[cam_name].image_path
            flip = cam_name in REAR_CAMERAS
            img_array = loader.load_image(
                img_path, resize_factor=resize_factor, flip_horizontal=flip
            )
            img_pil = Image.fromarray(img_array)
            data_uri = image_to_base64_data_uri(img_pil)
            image_content.append({
                "type": "image_url",
                "image_url": {"url": data_uri}
            })

        # Load prior analysis responses once (used for specific categories)
        risk_response = load_risk_assessment_response(risk_results_dir, sample_idx)
        traffic_response = load_traffic_analysis_response(traffic_results_dir, sample_idx)

        for cat in categories:
            # Build prior knowledge
            builder = PRIOR_BUILDERS[cat]
            if cat in ["Dynamic_Agents_and_Risk_Assessment", "Causal_and_Hypothetical_Reasoning"]:
                prior_knowledge = builder(scene_data, vqa_result)
            else:
                prior_knowledge = builder(scene_data)

            # Get templates
            templates = get_category_templates(question_bank, cat)
            num_templates = len(templates)
            num_batches = (num_templates + batch_size - 1) // batch_size

            results[cat] = {
                'category': cat,
                'num_templates': num_templates,
                'num_batches': num_batches
            }

            # Run inference for each batch
            all_applicable = []
            all_batch_responses = []

            for batch_idx in range(num_batches):
                start_idx_batch = batch_idx * batch_size
                end_idx_batch = min(start_idx_batch + batch_size, num_templates)
                batch_templates = templates[start_idx_batch:end_idx_batch]

                # Build batch validation prompt (egocentric order from qa_generator.py)
                prompt = build_batch_validation_prompt(
                    category=cat,
                    prior_knowledge=prior_knowledge,
                    templates=batch_templates,
                    start_idx=start_idx_batch,
                    batch_num=batch_idx + 1,
                    total_batches=num_batches
                )

                # For Risk Assessment category, prepend the risk analysis response
                if cat == "Dynamic_Agents_and_Risk_Assessment" and risk_response:
                    risk_section = (
                        "\n\n=== RISK ASSESSMENT ANALYSIS (from prior analysis) ===\n"
                        f"{risk_response}\n"
                        "=== END OF RISK ASSESSMENT ANALYSIS ===\n\n"
                    )
                    prompt = risk_section + prompt

                # For Traffic Signs and Signals category, prepend the traffic analysis response
                if cat == "Traffic_Signs_and_Signals" and traffic_response:
                    traffic_section = (
                        "\n\n=== TRAFFIC SIGNAL ANALYSIS (from prior analysis) ===\n"
                        f"{traffic_response}\n"
                        "=== END OF TRAFFIC SIGNAL ANALYSIS ===\n\n"
                    )
                    prompt = traffic_section + prompt

                # Build user content: images + prompt text
                user_content = list(image_content) + [
                    {"type": "text", "text": prompt}
                ]

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ]

                # Run inference via vLLM API
                response_obj = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=0.0,
                )
                response = response_obj.choices[0].message.content

                all_batch_responses.append({
                    'batch_idx': batch_idx + 1,
                    'template_range': f"{start_idx_batch + 1}-{end_idx_batch}",
                    'response': response
                })

                # Parse the JSON response
                parsed = parse_json_response(response)

                if parsed and 'batch_results' in parsed:
                    batch_results = parsed['batch_results']

                    for result in batch_results:
                        if result.get('applicable', False):
                            template_idx = int(result.get('template_idx', 0)) - 1
                            if 0 <= template_idx < num_templates:
                                original_template = templates[template_idx]
                                all_applicable.append({
                                    'template_idx': template_idx + 1,
                                    'template': original_template['template'],
                                    'answer_type': original_template.get('answer_type', 'open_ended'),
                                    'original_placeholders': original_template.get('placeholders', {}),
                                    'valid_placeholders': result.get('valid_placeholders', {}),
                                    'reason': result.get('reason', '')
                                })

            # Format applicable templates for output
            category_applicable = {}
            for i, item in enumerate(all_applicable, 1):
                q_key = f"q_{i:02d}"
                category_applicable[q_key] = {
                    'template_idx': item['template_idx'],
                    'template': item['template'],
                    'answer_type': item['answer_type'],
                    'valid_placeholders': item['valid_placeholders'],
                    'reason': item['reason']
                }

            inference_results[cat] = category_applicable
            results[cat]['applicable_templates'] = all_applicable
            results[cat]['num_applicable'] = len(all_applicable)
            results[cat]['batch_responses'] = all_batch_responses

        # Save results
        os.makedirs(output_dir, exist_ok=True)

        # Save summary
        summary = {
            'sample_idx': sample_idx,
            'scene_meta': scene_data['scene_meta'],
            'ego_info': scene_data['ego_info'],
            'class_counts': scene_data['class_counts'],
            'total_objects': scene_data['total_objects'],
            'categories': {
                cat: {
                    'num_templates': data['num_templates'],
                    'num_batches': data.get('num_batches', 0),
                    'num_applicable': data.get('num_applicable', 0)
                }
                for cat, data in results.items()
            }
        }

        summary_file = os.path.join(output_dir, f"sample_{sample_idx}_qa_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)

        # Save applicable questions
        if inference_results:
            final_output = {sample_idx: inference_results}
            applicable_file = os.path.join(output_dir, f"sample_{sample_idx}_applicable_questions.json")
            with open(applicable_file, 'w') as f:
                json.dump(final_output, f, indent=4, ensure_ascii=False)

            # Save detailed results
            detailed_output = {
                'sample_idx': sample_idx,
                'scene_meta': scene_data['scene_meta'],
                'timestamp': datetime.now().isoformat(),
                'filter_distance': filter_distance,
                'rear_filter': rear_filter,
                'batch_size': batch_size,
                'categories': {}
            }

            for cat, data in results.items():
                detailed_output['categories'][cat] = {
                    'num_templates': data['num_templates'],
                    'num_batches': data.get('num_batches', 0),
                    'num_applicable': data.get('num_applicable', 0),
                    'applicable_templates': data.get('applicable_templates', []),
                    'batch_responses': data.get('batch_responses', [])
                }

            detailed_file = os.path.join(output_dir, f"sample_{sample_idx}_inference_detailed.json")
            with open(detailed_file, 'w') as f:
                json.dump(detailed_output, f, indent=4, ensure_ascii=False)

        return {
            'sample_idx': sample_idx,
            'total_applicable': sum(len(v) for v in inference_results.values()),
            'categories': {cat: len(v) for cat, v in inference_results.items()},
            'timestamp': datetime.now().isoformat(),
        }

    except Exception as e:
        return {
            'sample_idx': sample_idx,
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
        }


# ============================================================================
# Wrapper to unpack arguments for Pool.imap_unordered
# ============================================================================

def _worker_wrapper(args):
    """Unpack tuple arguments and call _worker_process_sample."""
    return _worker_process_sample(*args)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Continuous QA Generation using vLLM (Multiprocessing)"
    )

    # Model and API settings
    parser.add_argument(
        "--model_name", type=str,
        default="Qwen/Qwen3-VL-235B-A22B-Instruct",
        help="Model name served by vLLM",
    )
    parser.add_argument(
        "--api_base", type=str,
        default="http://localhost:8000/v1",
        help="vLLM OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--api_key", type=str, default="EMPTY",
        help='API key (default "EMPTY" for local vLLM)',
    )

    # Data paths
    parser.add_argument(
        "--pkl_path", type=str,
        default=os.environ.get("NUSCENES_PKL_PATH", "nuscenes2d_ego_temporal_infos_val.pkl"),
        help="Path to nuScenes pkl file",
    )
    parser.add_argument(
        "--question_bank", type=str,
        default=os.environ.get("QUESTION_BANK_PATH", "question_bank.json"),
        help="Path to question bank JSON",
    )
    parser.add_argument(
        "--vqa_results_dir", type=str,
        default="vqa_results",
        help="Directory containing VQA results files",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="qa_outputs",
        help="Output directory for generated results",
    )
    parser.add_argument(
        "--risk_results_dir", type=str,
        default="risk_assessment_results",
        help="Directory containing risk assessment result JSONs (for Dynamic_Agents_and_Risk_Assessment category)",
    )
    parser.add_argument(
        "--traffic_results_dir", type=str,
        default="traffic_analysis_results",
        help="Directory containing traffic analysis result JSONs (for Traffic_Signs_and_Signals category)",
    )

    # Sample range
    parser.add_argument("--start_idx", type=int, default=0, help="Starting sample index")
    parser.add_argument("--end_idx", type=int, default=6018, help="Ending sample index")

    # Category selection
    parser.add_argument(
        "--category", type=str, required=True,
        choices=VALID_CATEGORIES + ["all"],
        help='Question category to process, or "all" for all categories',
    )

    # Filtering options
    parser.add_argument("--filter_distance", type=float, default=20.0,
                        help="Maximum distance (m) from ego to include objects")
    parser.add_argument("--rear_filter", type=float, default=None,
                        help="Maximum distance (m) for non-vehicle objects behind ego")

    # Inference options
    parser.add_argument("--batch_size", type=int, default=5,
                        help="Number of templates per validation batch")
    parser.add_argument("--resize_factor", type=int, default=2,
                        help="Image resize factor (1/n of original size)")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Maximum tokens to generate")

    # Multiprocessing
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of parallel worker processes")

    # Sampling
    parser.add_argument("--sampling_ratio", type=float, default=None,
                        help="Percentage of samples to randomly select (e.g., 5 for 5%%). "
                             "When set, randomly samples this ratio from the [start_idx, end_idx] range.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible sampling (default: 42)")

    # Logging
    parser.add_argument("--log_dir", type=str, default="continuous_qa_logs",
                        help="Directory for log files")

    args = parser.parse_args()

    # Determine sample indices
    all_indices = list(range(args.start_idx, args.end_idx + 1))

    if args.sampling_ratio is not None:
        ratio = args.sampling_ratio / 100.0
        n_samples = max(1, int(len(all_indices) * ratio))
        rng = np.random.RandomState(args.seed)
        sample_indices = sorted(rng.choice(all_indices, size=n_samples, replace=False).tolist())
    else:
        sample_indices = all_indices

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.log_dir, f"continuous_qa_vllm_{timestamp}.log")

    def log_and_print(message):
        print(message)
        with open(log_file, "a") as f:
            f.write(message + "\n")

    # Print configuration
    log_and_print("=" * 80)
    log_and_print("Continuous QA Generation (vLLM Multiprocessing)")
    log_and_print("=" * 80)
    log_and_print(f"  API base: {args.api_base}")
    log_and_print(f"  Model: {args.model_name}")
    log_and_print(f"  Num workers: {args.num_workers}")
    log_and_print(f"  Sample range: {args.start_idx} to {args.end_idx} ({len(all_indices)} total)")
    if args.sampling_ratio is not None:
        log_and_print(f"  Sampling ratio: {args.sampling_ratio}% -> {len(sample_indices)} samples (seed={args.seed})")
    else:
        log_and_print(f"  Processing all {len(sample_indices)} samples")
    log_and_print(f"  Category: {args.category}")
    log_and_print(f"  Filter distance: {args.filter_distance}m")
    if args.rear_filter is not None:
        log_and_print(f"  Rear filter (non-vehicles): {args.rear_filter}m")
    log_and_print(f"  Batch size: {args.batch_size}")
    log_and_print(f"  Resize factor: {args.resize_factor}")
    log_and_print(f"  Max new tokens: {args.max_new_tokens}")
    log_and_print(f"  Output dir: {args.output_dir}")
    log_and_print(f"  Log file: {log_file}")
    log_and_print(f"  Camera order: FL, F, FR, RL, R, RR (egocentric)")
    log_and_print(f"  Rear cameras: horizontally flipped")
    log_and_print(f"  Risk results dir: {args.risk_results_dir}")
    log_and_print(f"  Traffic results dir: {args.traffic_results_dir}")
    log_and_print("=" * 80)

    # Verify vLLM server
    log_and_print(f"\nVerifying vLLM server at: {args.api_base}")
    try:
        test_client = OpenAI(base_url=args.api_base, api_key=args.api_key)
        models = test_client.models.list()
        log_and_print(f"  Server reachable. Available models: {[m.id for m in models.data]}")
    except Exception as e:
        log_and_print(f"  WARNING: Could not connect to vLLM server: {e}")
        log_and_print(f"  Proceeding anyway -- workers will retry on their own.")

    # Build argument tuples for each sample
    worker_args = [
        (
            idx,
            args.model_name,
            args.category,
            args.resize_factor,
            args.max_new_tokens,
            args.filter_distance,
            args.rear_filter,
            args.batch_size,
            args.output_dir,
        )
        for idx in sample_indices
    ]

    num_workers = min(args.num_workers, len(sample_indices))
    log_and_print(f"\nStarting multiprocessing pool with {num_workers} workers...")
    log_and_print(f"Processing {len(sample_indices)} samples...\n")

    success_count = 0
    failure_count = 0
    failed_indices = []
    total_applicable_all = 0

    total_start = time.time()

    with Pool(
        processes=num_workers,
        initializer=_worker_init,
        initargs=(
            args.model_name, args.api_base, args.api_key,
            args.pkl_path, args.question_bank, args.vqa_results_dir,
            args.risk_results_dir, args.traffic_results_dir,
        ),
    ) as pool:
        results_iter = pool.imap_unordered(_worker_wrapper, worker_args)

        for result in tqdm(
            results_iter,
            total=len(sample_indices),
            desc="Processing samples",
            unit="sample",
            ncols=100,
        ):
            sample_idx = result.get("sample_idx", "unknown")
            if "error" in result:
                failure_count += 1
                failed_indices.append(sample_idx)
                tqdm.write(f"  x Sample {sample_idx} failed: {result['error']}")
            else:
                success_count += 1
                total_app = result.get("total_applicable", 0)
                total_applicable_all += total_app
                tqdm.write(f"  v Sample {sample_idx} done (applicable: {total_app})")

    total_elapsed = time.time() - total_start

    # Summary
    log_and_print(f"\n{'=' * 80}")
    log_and_print("Continuous QA Generation (vLLM Multiprocessing) Complete")
    log_and_print(f"{'=' * 80}")
    log_and_print(f"  Sample range: {args.start_idx} to {args.end_idx}")
    log_and_print(f"  Total samples: {len(sample_indices)}")
    log_and_print(f"  Successful: {success_count}")
    log_and_print(f"  Failed: {failure_count}")
    log_and_print(f"  Total applicable templates found: {total_applicable_all}")
    if success_count + failure_count > 0:
        log_and_print(f"  Success rate: {success_count * 100.0 / (success_count + failure_count):.2f}%")
    log_and_print(f"  Total wall time: {total_elapsed:.2f}s")
    log_and_print(f"  Effective throughput: {len(sample_indices) / total_elapsed:.2f} samples/s")
    log_and_print(f"  Workers used: {args.num_workers}")
    log_and_print(f"  Results saved to: {args.output_dir}/")

    # Save failed indices
    if failed_indices:
        log_and_print(f"\n  FAILED SAMPLE INDICES ({len(failed_indices)} samples):")
        log_and_print(f"    {sorted(failed_indices)}")
        failed_file = os.path.join(args.log_dir, f"failed_qa_indices_{timestamp}.txt")
        with open(failed_file, "w") as f:
            f.write("\n".join(map(str, sorted(failed_indices))))
        log_and_print(f"    Saved to: {failed_file}")

    log_and_print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
