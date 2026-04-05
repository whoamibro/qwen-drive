"""
Risk Assessment Module for nuScenes Dataset

Analyzes driving risks, hazards, and time-to-collision (TTC) per nuScenes sample using a
vLLM-served Qwen3-VL model via OpenAI-compatible API with multiprocessing for parallel inference.

Prerequisites:
    # Start vLLM server first:
    vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct --tensor-parallel-size 8

Usage:
    # Process samples 0 to 100 with 4 workers
    python -m nuscenes_pipeline.modules.risk_assessment \\
        --start_idx 0 --end_idx 100 --num_workers 4 \\
        --question "Your question here"

    # With 3D objects and global coords
    python -m nuscenes_pipeline.modules.risk_assessment \\
        --start_idx 0 --end_idx 100 --num_workers 4 \\
        --to_global --3dod --filter_length 40 --rear_filter 20 \\
        --question "Analyze risks and hazards..."
"""

import os
import io
import json
import time
import base64
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, List
from multiprocessing import Pool
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

from nuscenes_pipeline.core.nuscenes_data_loader import NuScenesDataLoader
from nuscenes_pipeline.core.nuscenes_prompt_generator import create_single_frame_prompt


def image_to_base64_data_uri(img_pil: Image.Image, format: str = "JPEG") -> str:
    """Convert a PIL Image to a base64 data URI string."""
    buffer = io.BytesIO()
    img_pil.save(buffer, format=format)
    base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime_type = "image/jpeg" if format.upper() == "JPEG" else f"image/{format.lower()}"
    return f"data:{mime_type};base64,{base64_str}"


# ============================================================================
# Worker process initializer and global state
# ============================================================================

# Per-process globals (initialized once per worker via _worker_init)
_worker_client = None
_worker_loader = None


def _worker_init(model_name: str, api_base: str, api_key: str, pkl_path: str):
    """
    Initializer for each worker process.
    Creates the OpenAI client and NuScenesDataLoader ONCE per worker,
    avoiding repeated pkl loading and client creation for every sample.
    """
    global _worker_client, _worker_loader
    from openai import OpenAI as _OpenAI
    _worker_client = _OpenAI(
        base_url=api_base,
        api_key=api_key,
        timeout=120.0,
        max_retries=3,
    )
    _worker_loader = NuScenesDataLoader(pkl_path)


# ============================================================================
# Worker function for multiprocessing
# ============================================================================

def _worker_analyze_sample(
    sample_idx: int,
    model_name: str,
    user_question: str,
    resize_factor: int,
    max_new_tokens: int,
    use_global_coords: bool,
    include_3d_objects: bool,
    filter_distance: float,
    proj2img: bool,
    rear_filter_distance: float,
    results_dir: str,
) -> Dict:
    """
    Worker function that processes a single sample using the vLLM API.
    Uses per-process client and loader initialized by _worker_init.
    Produces identical prompts to vqa_test_continuous_egocentric.py.
    """
    global _worker_client, _worker_loader
    try:
        client = _worker_client
        loader = _worker_loader

        # Get sample
        sample = loader.get_sample(sample_idx)

        # Generate prompt (identical to vqa_test_continuous_egocentric.py)
        system_prompt, user_question_text = create_single_frame_prompt(
            sample, loader, user_question,
            use_global_coords=use_global_coords,
            include_3d_objects=include_3d_objects,
            filter_distance=filter_distance,
            proj2img=proj2img,
            resize_factor=resize_factor,
            rear_filter_distance=rear_filter_distance,
        )

        # Prepare messages with base64-encoded images (OpenAI API format)
        user_content = []
        cam_labels = {
            "CAM_FRONT_LEFT": "Image 1: Front-left Camera",
            "CAM_FRONT": "Image 2: Front Camera",
            "CAM_FRONT_RIGHT": "Image 3: Front-right Camera",
            "CAM_BACK_LEFT": "Image 4: Rear-left Camera",
            "CAM_BACK": "Image 5: Rear Camera",
            "CAM_BACK_RIGHT": "Image 6: Rear-right Camera",
        }

        for cam_name in loader.CAMERA_NAMES:
            label = cam_labels[cam_name]
            user_content.append({"type": "text", "text": f"=== {label} ==="})
            img_path = sample.cameras[cam_name].image_path
            flip = cam_name in loader.REAR_CAMERAS
            img_array = loader.load_image(
                img_path, resize_factor=resize_factor, flip_horizontal=flip
            )
            img_pil = Image.fromarray(img_array)
            data_uri = image_to_base64_data_uri(img_pil)
            user_content.append({"type": "image_url", "image_url": {"url": data_uri}})

        user_content.append({"type": "text", "text": user_question_text})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        # Run inference via vLLM API
        start_time = time.time()
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=0.0,
        )
        elapsed_time = time.time() - start_time
        response_text = response.choices[0].message.content

        # Create result (same schema as vqa_test_continuous_egocentric.py)
        result = {
            "prompt_type": "single_frame",
            "sample_idx": sample_idx,
            "token": sample.token,
            "scene_token": sample.scene_token,
            "location": sample.location,
            "description": sample.description,
            "user_question": user_question,
            "system_prompt": system_prompt,
            "prompt": user_question_text,
            "response": response_text,
            "inference_time_sec": round(elapsed_time, 2),
            "timestamp": datetime.now().isoformat(),
        }

        # Save result
        result_filename = f"{sample_idx:04d}_{sample.scene_token[:16]}_{sample.token[:16]}_single_frame.json"
        result_path = os.path.join(results_dir, result_filename)
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        return result

    except Exception as e:
        return {
            "sample_idx": sample_idx,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


# ============================================================================
# Wrapper to unpack arguments for Pool.imap_unordered
# ============================================================================

def _worker_wrapper(args):
    """Unpack tuple arguments and call _worker_analyze_sample."""
    return _worker_analyze_sample(*args)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Continuous Egocentric VQA Testing using vLLM (Multiprocessing)"
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

    # Data path
    parser.add_argument(
        "--pkl_path", type=str,
        default=os.environ.get("NUSCENES_PKL_PATH", "nuscenes2d_ego_temporal_infos_val.pkl"),
        help="Path to nuScenes pkl file",
    )

    # Sample range
    parser.add_argument("--start_idx", type=int, default=0, help="Starting sample index")
    parser.add_argument("--end_idx", type=int, default=6018, help="Ending sample index")

    # Question and generation
    parser.add_argument("--question", type=str, required=True, help="Question for VQA")
    parser.add_argument("--resize_factor", type=int, default=4, help="Image resize factor")
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="Maximum tokens to generate")

    # Multiprocessing
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel worker processes")

    # Coordinate system and 3D objects
    parser.add_argument("--to_global", action="store_true", help="Use global ENU coordinates")
    parser.add_argument("--3dod", dest="include_3dod", action="store_true",
                        help="Include 3D object detection ground truth in the prompt")
    parser.add_argument("--filter_length", type=float, default=20.0,
                        help="Maximum distance (m) from ego to include 3D objects")
    parser.add_argument("--rear_filter", type=float, default=None,
                        help="Maximum distance (m) for non-vehicle objects behind ego")
    parser.add_argument("--proj2img", action="store_true",
                        help="Project 3D object positions to image pixel coordinates")

    # Output directories
    parser.add_argument("--results_dir", type=str, default="risk_assessment_results",
                        help="Directory to save result JSON files")

    # Logging
    parser.add_argument("--log_dir", type=str, default="risk_assessment_logs",
                        help="Directory for log files")

    args = parser.parse_args()

    # Determine sample indices
    sample_indices = list(range(args.start_idx, args.end_idx + 1))

    # Create output directories
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.log_dir, f"continuous_vqa_vllm_{timestamp}.log")

    def log_and_print(message):
        print(message)
        with open(log_file, "a") as f:
            f.write(message + "\n")

    # Print configuration
    log_and_print("=" * 80)
    log_and_print("Continuous Egocentric VQA Testing (vLLM Multiprocessing)")
    log_and_print("=" * 80)
    log_and_print(f"  API base: {args.api_base}")
    log_and_print(f"  Model: {args.model_name}")
    log_and_print(f"  Num workers: {args.num_workers}")
    log_and_print(f"  Sample range: {args.start_idx} to {args.end_idx} ({len(sample_indices)} samples)")
    log_and_print(f"  Resize factor: {args.resize_factor}")
    log_and_print(f"  Max new tokens: {args.max_new_tokens}")
    log_and_print(f"  Coordinate system: {'Global (ENU)' if args.to_global else 'Ego-relative (FLU)'}")
    log_and_print(f"  Include 3D objects: {args.include_3dod}")
    if args.include_3dod:
        log_and_print(f"    - Filter length: {args.filter_length}m")
        if args.rear_filter is not None:
            log_and_print(f"    - Rear non-vehicle filter: {args.rear_filter}m")
    log_and_print(f"  Project to image: {args.proj2img}")
    log_and_print(f"  Results dir: {args.results_dir}")
    log_and_print(f"  Log file: {log_file}")
    log_and_print("=" * 80)

    # Verify vLLM server
    log_and_print(f"\nVerifying vLLM server at: {args.api_base}")
    try:
        test_client = OpenAI(base_url=args.api_base, api_key=args.api_key)
        models = test_client.models.list()
        log_and_print(f"  Server reachable. Available models: {[m.id for m in models.data]}")
    except Exception as e:
        log_and_print(f"  WARNING: Could not connect to vLLM server: {e}")
        log_and_print(f"  Proceeding anyway — workers will retry on their own.")

    # Build argument tuples for each sample
    worker_args = [
        (
            idx,
            args.model_name,
            args.question,
            args.resize_factor,
            args.max_new_tokens,
            args.to_global,
            args.include_3dod,
            args.filter_length,
            args.proj2img,
            args.rear_filter,
            args.results_dir,
        )
        for idx in sample_indices
    ]

    num_workers = min(args.num_workers, len(sample_indices))
    log_and_print(f"\nStarting multiprocessing pool with {num_workers} workers...")
    log_and_print(f"Processing {len(sample_indices)} samples...\n")

    all_results = {}
    success_count = 0
    failure_count = 0
    failed_indices = []

    total_start = time.time()

    with Pool(
        processes=num_workers,
        initializer=_worker_init,
        initargs=(args.model_name, args.api_base, args.api_key, args.pkl_path),
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
                tqdm.write(f"  ✗ Sample {sample_idx} failed: {result['error']}")
            else:
                success_count += 1
                elapsed = result.get("inference_time_sec", "?")
                tqdm.write(f"  ✓ Sample {sample_idx} done ({elapsed}s)")
            all_results[f"sample_{sample_idx}"] = result

    total_elapsed = time.time() - total_start

    # Summary
    avg_time = np.mean([
        r["inference_time_sec"] for r in all_results.values() if "inference_time_sec" in r
    ]) if success_count > 0 else 0

    log_and_print(f"\n{'=' * 80}")
    log_and_print("Continuous Egocentric VQA Testing (vLLM Multiprocessing) Complete")
    log_and_print(f"{'=' * 80}")
    log_and_print(f"  Sample range: {args.start_idx} to {args.end_idx}")
    log_and_print(f"  Total samples: {len(sample_indices)}")
    log_and_print(f"  Successful: {success_count}")
    log_and_print(f"  Failed: {failure_count}")
    if success_count + failure_count > 0:
        log_and_print(f"  Success rate: {success_count * 100.0 / (success_count + failure_count):.2f}%")
    log_and_print(f"  Total wall time: {total_elapsed:.2f}s")
    log_and_print(f"  Avg inference time per sample: {avg_time:.2f}s")
    log_and_print(f"  Effective throughput: {len(sample_indices) / total_elapsed:.2f} samples/s")
    log_and_print(f"  Workers used: {args.num_workers}")
    log_and_print(f"  Results saved to: {args.results_dir}/")

    # Save failed indices
    if failed_indices:
        log_and_print(f"\n  FAILED SAMPLE INDICES ({len(failed_indices)} samples):")
        log_and_print(f"    {sorted(failed_indices)}")
        failed_file = os.path.join(args.log_dir, f"failed_indices_{timestamp}.txt")
        with open(failed_file, "w") as f:
            f.write("\n".join(map(str, sorted(failed_indices))))
        log_and_print(f"    Saved to: {failed_file}")

    log_and_print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
