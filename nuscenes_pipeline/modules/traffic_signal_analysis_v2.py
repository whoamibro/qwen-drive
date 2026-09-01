"""
Traffic Signal Analysis v2 — Stage 1B upgraded with detected signal boxes + VLM status.

Same prompts as v1 (`traffic_signal_analysis.create_traffic_signal_analysis_prompt`
is imported and reused verbatim), plus one extra block injected into the user
prompt right before the PRE-ANALYSIS REMINDER: the traffic-signal detections
stored in the infos pkl (tl_bboxes2d + tl_signal_type2d / tl_light_observable2d
/ tl_light_color2d / tl_lit_shape2d, from add_traffic_lights_to_infos.py +
apply_traffic_signal_status.py), formatted like the risk_assessment pipeline's
3D object list:

    =====
    DETECTED TRAFFIC SIGNAL INFORMATION
    (N signal detections across the camera views; M signal poles detected)
    =====
    -----
    TS 1: vehicle signal [Image 2 (Front) 2D bbox [x1=768, y1=428, x2=777, y2=446]]
    vehicle signal, light observable: RED (circle) | classifier confidence 0.95, detector score 0.70 | mounted on pole TS-P0 of this view
    ...
    =====
    ⚠️ IMPORTANT CONSTRAINTS: ...
    =====

Detections classified as not_a_signal are dropped by default
(--include_not_a_signal keeps them). 2D bbox coordinates are emitted in the
SAME image space the model sees: divided by --resize_factor and mirrored for
the rear cameras (whose images are sent horizontally flipped, like v1).

Prerequisites:
    # pkl must contain tl_* arrays (add_traffic_lights_to_infos + apply_traffic_signal_status)
    # vLLM server:
    vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct --tensor-parallel-size 8

Usage:
    python -m nuscenes_pipeline.modules.traffic_signal_analysis_v2 --sample_indices 0 10 20
    python -m nuscenes_pipeline.modules.traffic_signal_analysis_v2 --start_idx 0 --end_idx 100
"""

import argparse
import json
import os
import time
from datetime import datetime
from multiprocessing import Pool
from typing import Dict, Optional

import numpy as np
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

from nuscenes_pipeline.core.nuscenes_data_loader import NuScenesDataLoader
from nuscenes_pipeline.modules.traffic_signal_analysis import (
    create_traffic_signal_analysis_prompt,
    image_to_base64_data_uri,
)

IMG_W = 1600
CAM_TO_IMAGE = {
    "CAM_FRONT_LEFT": (1, "Front-left"),
    "CAM_FRONT": (2, "Front"),
    "CAM_FRONT_RIGHT": (3, "Front-right"),
    "CAM_BACK_LEFT": (4, "Rear-left"),
    "CAM_BACK": (5, "Rear"),
    "CAM_BACK_RIGHT": (6, "Rear-right"),
}
TYPE_DESC = {
    "vehicle": "vehicle signal",
    "pedestrian": "pedestrian signal",
    "other": "other signal device",
    "not_a_signal": "detector false positive (not a signal)",
}
INJECT_MARKER = "## PRE-ANALYSIS REMINDER"


def _prompt_bbox(box, is_rear: bool, resize_factor: int):
    """Map a pkl-space (unflipped 1600x900) box into the image space the model
    sees: rear views are horizontally flipped, and every view is resized by
    1/resize_factor (matching v1's image loading)."""
    x1, y1, x2, y2 = [float(v) for v in box]
    if is_rear:
        x1, x2 = IMG_W - x2, IMG_W - x1
    rf = resize_factor
    return [round(x1 / rf), round(y1 / rf), round(x2 / rf), round(y2 / rf)]


def build_traffic_signal_block(info: dict, resize_factor: int = 1,
                               min_det_score: Optional[float] = None,
                               include_not_a_signal: bool = False) -> str:
    """Format the pkl's detected+classified traffic signals like the
    risk_assessment object list. Returns "" if the pkl has no tl_* arrays."""
    if "tl_bboxes2d" not in info:
        return ""
    has_status = "tl_light_color2d" in info
    cam_names = list(info["cams"].keys())

    entries = []
    n_dropped_nas = 0
    n_poles = sum(len(p) for p in info.get("tl_pole_bboxes2d", []))
    # Entries ordered by image number for readability
    for cam in sorted(cam_names, key=lambda c: CAM_TO_IMAGE[c][0]):
        ci = cam_names.index(cam)
        img_num, cam_disp = CAM_TO_IMAGE[cam]
        is_rear = cam.startswith("CAM_BACK")
        for si in range(len(info["tl_bboxes2d"][ci])):
            det_score = float(info["tl_scores2d"][ci][si])
            if min_det_score is not None and det_score < min_det_score:
                continue
            sig_type = str(info["tl_signal_type2d"][ci][si]) if has_status else "unknown"
            if sig_type == "not_a_signal" and not include_not_a_signal:
                n_dropped_nas += 1
                continue
            observable = bool(info["tl_light_observable2d"][ci][si]) if has_status else False
            color = str(info["tl_light_color2d"][ci][si]) if has_status else "unknown"
            shape = str(info["tl_lit_shape2d"][ci][si]) if has_status else "unknown"
            conf = float(info["tl_status_conf2d"][ci][si]) if has_status else -1.0
            pole_idx = int(info["tl_pole_idx2d"][ci][si])

            x1, y1, x2, y2 = _prompt_bbox(info["tl_bboxes2d"][ci][si], is_rear, resize_factor)
            cam_info = f"Image {img_num} ({cam_disp}) 2D bbox [x1={x1}, y1={y1}, x2={x2}, y2={y2}]"

            type_desc = TYPE_DESC.get(sig_type, "signal (type unknown)")
            if observable:
                status_desc = f"light observable: {color.upper()}" + (f" ({shape})" if shape != "unknown" else "")
            else:
                status_desc = "light NOT observable from this view (side/back view, occluded, or too small)"
            pole_desc = f" | mounted on pole TS-P{pole_idx} of this view" if pole_idx >= 0 else ""
            conf_desc = (f"classifier confidence {conf:.2f}, detector score {det_score:.2f}" if has_status
                         else f"detector score {det_score:.2f} (no status classification in pkl)")

            entries.append(f"""
-----
TS {len(entries) + 1}: {type_desc} [{cam_info}]
{type_desc}, {status_desc} | {conf_desc}{pole_desc}""")

    if not entries:
        note = " (after filtering detector false positives)" if n_dropped_nas else ""
        return f"""
=====
DETECTED TRAFFIC SIGNAL INFORMATION
=====
No traffic signals were detected in any camera view by the dedicated detector{note}.
=====
"""

    header = f"""
=====
DETECTED TRAFFIC SIGNAL INFORMATION
({len(entries)} signal detections across the camera views; {n_poles} signal poles detected)
Source: dedicated traffic-signal detector + per-crop status classifier
====="""
    constraint = """
=====
⚠️ IMPORTANT CONSTRAINTS:
- Use the detections above to LOCATE signals — including small or partially occluded ones you might miss visually
- The SAME physical signal can appear as multiple TS entries (one per camera view it is visible in)
- "light NOT observable" means the lamp state could not be read from that view — the signal still physically exists there
- Status labels are automatic and may err on tiny or blurry signals; when a label disagrees with what you clearly see in the images, trust the images and say so
- These detections do NOT tell you which signal governs ego's lane — you MUST still apply the Traffic Flow Test and orientation checks to each one
- Rear-view entries (Images 4-6) are behind ego and never govern ego's current lane
====="""
    return header + "".join(entries) + constraint + "\n"


def create_traffic_signal_analysis_v2_prompt(sample, loader, info: dict,
                                             use_global_coords: bool = False,
                                             resize_factor: int = 1,
                                             min_det_score: Optional[float] = None,
                                             include_not_a_signal: bool = False) -> tuple:
    """v1 prompt (unchanged) + detected-signal block injected before the
    PRE-ANALYSIS REMINDER section of the user prompt."""
    system_prompt, user_prompt = create_traffic_signal_analysis_prompt(
        sample, loader, "", use_global_coords=use_global_coords)
    block = build_traffic_signal_block(info, resize_factor=resize_factor,
                                       min_det_score=min_det_score,
                                       include_not_a_signal=include_not_a_signal)
    if block:
        if INJECT_MARKER in user_prompt:
            user_prompt = user_prompt.replace(INJECT_MARKER, block + "\n" + INJECT_MARKER, 1)
        else:  # fallback: append at the end
            user_prompt = user_prompt + "\n" + block
    return system_prompt, user_prompt


# ============================================================================
# Multiprocessing workers (same pattern as v1)
# ============================================================================

_worker_client = None
_worker_loader = None


def _worker_init(model_name: str, api_base: str, api_key: str, pkl_path: str):
    global _worker_client, _worker_loader
    _worker_client = OpenAI(base_url=api_base, api_key=api_key)
    _worker_loader = NuScenesDataLoader(pkl_path=pkl_path)


def _worker_analyze_sample(sample_idx: int, model_name: str, resize_factor: int,
                           max_new_tokens: int, use_global_coords: bool,
                           min_det_score: Optional[float], include_not_a_signal: bool,
                           results_dir: str) -> Dict:
    global _worker_client, _worker_loader
    try:
        client = _worker_client
        loader = _worker_loader
        sample = loader.get_sample(sample_idx)
        info = loader.infos[sample_idx]

        system_prompt, user_question_text = create_traffic_signal_analysis_v2_prompt(
            sample, loader, info, use_global_coords=use_global_coords,
            resize_factor=resize_factor, min_det_score=min_det_score,
            include_not_a_signal=include_not_a_signal)

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
            user_content.append({"type": "text", "text": f"=== {cam_labels[cam_name]} ==="})
            img_array = loader.load_image(
                sample.cameras[cam_name].image_path, resize_factor=resize_factor,
                flip_horizontal=cam_name in loader.REAR_CAMERAS)
            user_content.append({"type": "image_url",
                                 "image_url": {"url": image_to_base64_data_uri(Image.fromarray(img_array))}})
        user_content.append({"type": "text", "text": user_question_text})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        start_time = time.time()
        response = client.chat.completions.create(
            model=model_name, messages=messages,
            max_tokens=max_new_tokens, temperature=0.0)
        elapsed_time = time.time() - start_time
        response_text = response.choices[0].message.content

        result = {
            "prompt_type": "traffic_signal_analysis_v2_vllm_mp",
            "sample_idx": sample_idx,
            "token": sample.token,
            "scene_token": sample.scene_token,
            "location": sample.location,
            "description": sample.description,
            "user_question": "",
            "system_prompt": system_prompt,
            "prompt": user_question_text,
            "response": response_text,
            "inference_time_sec": round(elapsed_time, 2),
            "timestamp": datetime.now().isoformat(),
        }
        result_filename = (f"{sample_idx:04d}_{sample.scene_token[:16]}_"
                           f"{sample.token[:16]}_traffic_signal.json")
        with open(os.path.join(results_dir, result_filename), "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        return result
    except Exception as e:  # noqa: BLE001 — report per-sample failure
        return {"sample_idx": sample_idx, "error": str(e),
                "timestamp": datetime.now().isoformat()}


def _worker_wrapper(args):
    return _worker_analyze_sample(*args)


def main():
    parser = argparse.ArgumentParser(
        description="Traffic Signal Analysis v2 (v1 prompts + detected signal boxes/status from pkl)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-235B-A22B-Instruct")
    parser.add_argument("--api_base", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--pkl_path", type=str,
                        default=os.environ.get("NUSCENES_PKL_PATH", "nuscenes2d_ego_temporal_infos_val.pkl"))
    parser.add_argument("--sample_indices", type=int, nargs="+")
    parser.add_argument("--start_idx", type=int, default=None)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--resize_factor", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--to_global", action="store_true")
    parser.add_argument("--min_det_score", type=float, default=None,
                        help="Drop detections below this detector score")
    parser.add_argument("--include_not_a_signal", action="store_true",
                        help="Keep detections the classifier marked not_a_signal")
    parser.add_argument("--results_dir", type=str, default="traffic_signal_analysis_v2_results")
    args = parser.parse_args()

    if args.sample_indices:
        sample_indices = args.sample_indices
    elif args.start_idx is not None and args.end_idx is not None:
        sample_indices = list(range(args.start_idx, args.end_idx + 1))
    elif args.start_idx is not None:
        tmp_loader = NuScenesDataLoader(args.pkl_path)
        total = len(tmp_loader)
        del tmp_loader
        sample_indices = list(range(args.start_idx, total))
        print(f"Auto-detected {total} total samples. Processing {args.start_idx} ~ {total - 1}")
    else:
        sample_indices = [0]

    os.makedirs(args.results_dir, exist_ok=True)
    print(f"\nTraffic Signal Analysis v2: {len(sample_indices)} samples "
          f"(vLLM x{args.num_workers} workers, results -> {args.results_dir}/)")

    worker_args = [
        (idx, args.model_name, args.resize_factor, args.max_new_tokens,
         args.to_global, args.min_det_score, args.include_not_a_signal, args.results_dir)
        for idx in sample_indices
    ]
    total_start = time.time()
    results = {}
    with Pool(args.num_workers, initializer=_worker_init,
              initargs=(args.model_name, args.api_base, args.api_key, args.pkl_path)) as pool:
        for res in tqdm(pool.imap_unordered(_worker_wrapper, worker_args),
                        total=len(worker_args), desc="Analyzing"):
            results[res.get("sample_idx")] = res
    total_elapsed = time.time() - total_start

    successful = sum(1 for r in results.values() if "error" not in r)
    failed = len(results) - successful
    times = [r["inference_time_sec"] for r in results.values() if "inference_time_sec" in r]
    print(f"\n{'=' * 80}")
    print("Traffic Signal Analysis v2 Complete")
    print(f"{'=' * 80}")
    print(f"  Total samples: {len(sample_indices)}")
    print(f"  Successful: {successful}   Failed: {failed}")
    if failed:
        first_err = next(r for r in results.values() if "error" in r)
        print(f"  First error: sample {first_err['sample_idx']}: {first_err['error'][:200]}")
    print(f"  Total wall time: {total_elapsed:.2f}s")
    if times:
        print(f"  Avg inference time per sample: {np.mean(times):.2f}s")
    print(f"  Results saved to: {args.results_dir}/")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
