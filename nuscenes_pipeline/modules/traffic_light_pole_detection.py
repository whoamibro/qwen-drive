"""
Traffic Light & Pole 3D Detection Module for nuScenes Dataset

Detects traffic signal light housings and their supporting poles/mast arms in the
6-view camera images per nuScenes sample using a vLLM-served Qwen3-VL model via
OpenAI-compatible API, then lifts the 2D detections to 3D using camera intrinsics
and sensor2ego extrinsics.

Differences from the other Stage-1 seed modules (risk_assessment,
traffic_signal_analysis, traffic_sign_extraction):
  - One API call PER CAMERA VIEW (not one call with all 6 images) so the model's
    pixel-coordinate grounding is unambiguous and per-request memory stays low
    (6 sequential inferences per timestamp).
  - Images are UPSCALED x2 per side (x4 area, 1600x900 -> 3200x1800, LANCZOS) by
    default before being sent to the model, so small distant traffic lights and
    poles survive the vision-encoder patch embedding.
  - COORDINATE CONVENTION: Qwen3-VL grounds in a 0-1000 normalized space per
    axis and does NOT honor absolute-pixel instructions (verified by calibration
    against the served model). The prompt therefore requests 0-1000 normalized
    coordinates, and normalize_coords() converts them to the original full-res
    pixel space before 3D lifting — alignment is thus independent of both the
    client-side upscale and any server-side smart_resize. The Qwen3-VL-235B
    default pixel budget (size.longest_edge = 16.78M px) passes an x2 (5.8M) or
    x3 (13M) upscale through without server-side downscaling.
  - Rear camera images are NOT flipped horizontally — flipping would corrupt the
    pixel coordinate space needed for 3D lifting.
  - The model must return strict JSON (2D bboxes for light housings, 2D line
    segments for poles), which is parsed and geometrically lifted to 3D:
      * Traffic lights  -> 3D bounding boxes [x, y, z, length, width, height, yaw]
                           in ego FLU frame. Depth from pinhole geometry using the
                           known physical housing size (~0.35 m per lens section),
                           falling back to the model's distance estimate.
      * Poles/mast arms -> 3D line segments {bottom: [x,y,z], top: [x,y,z]} in ego
                           FLU frame. Vertical poles with a visible ground contact
                           are lifted by ray/ground-plane intersection (ego z=0);
                           others use an attached light's depth or the model's
                           distance estimate.
  - Results are written to one JSON per sample named {sample_idx:04d}_{token}.json.

Prerequisites:
    # Start vLLM server first:
    vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct --tensor-parallel-size 8 \
        --mm-processor-kwargs '{"max_pixels": 5760000}'

Usage:
    # Single sample detection
    python -m nuscenes_pipeline.modules.traffic_light_pole_detection --sample_indices 0 10 20

    # Process range of samples
    python -m nuscenes_pipeline.modules.traffic_light_pole_detection --start_idx 0 --end_idx 100

    # Disable the x2 upscale (send original 1600x900)
    python -m nuscenes_pipeline.modules.traffic_light_pole_detection --start_idx 0 --end_idx 100 --upscale_factor 1

    # Front cameras only (faster; rear traffic lights are rarely useful)
    python -m nuscenes_pipeline.modules.traffic_light_pole_detection --start_idx 0 --end_idx 100 --front_only

    # Custom number of workers (default: 8)
    python -m nuscenes_pipeline.modules.traffic_light_pole_detection --start_idx 0 --end_idx 100 --num_workers 4
"""

import os
import io
import re
import json
import time
import base64
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from multiprocessing import Pool
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

from nuscenes_pipeline.core.nuscenes_data_loader import (
    NuScenesDataLoader,
    quaternion_to_rotation_matrix,
)


# Physical priors used for pinhole depth estimation (meters)
LENS_SECTION_SIZE_M = 0.35      # one lens section of a vehicle signal housing
HOUSING_THICKNESS_M = 0.30      # depth of a signal housing (box "length" along facing)
PEDESTRIAN_HOUSING_M = 0.45     # pedestrian signal housing height
MIN_PIXEL_EXTENT = 4.0          # below this the pinhole estimate is unreliable
MAX_PLAUSIBLE_DEPTH_M = 120.0   # reject absurd depth estimates

CAM_LABELS = {
    "CAM_FRONT_LEFT": "Front-left Camera",
    "CAM_FRONT": "Front Camera",
    "CAM_FRONT_RIGHT": "Front-right Camera",
    "CAM_BACK_LEFT": "Rear-left Camera",
    "CAM_BACK": "Rear Camera",
    "CAM_BACK_RIGHT": "Rear-right Camera",
}

FRONT_CAMERAS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]


# ============================================================================
# Prompt
# ============================================================================

def create_detection_prompt(cam_name: str) -> Tuple[str, str]:
    """
    Create (system_prompt, user_prompt) for single-view traffic light + pole detection.

    The model is asked for strict JSON with coordinates in Qwen3-VL's NATIVE
    grounding convention: 0-1000 normalized per axis. Calibration showed the model
    always grounds in this space and does NOT honor absolute-pixel instructions
    (declaring large pixel ranges even destabilizes its x coordinates). Pixel
    dimensions are deliberately NOT mentioned to avoid biasing it away from the
    native convention. normalize_coords() converts 0-1000 -> full-res pixels.
    """
    system_prompt = f"""You are a precise traffic infrastructure detector for autonomous driving.

You will be shown ONE camera image from a driving scene. Detect:
1. TRAFFIC LIGHT HOUSINGS — every signal light box (vehicle signals AND pedestrian signals). One entry per physical housing. Do NOT include street lamps, signs, or vehicle lights.
2. POLES — every pole structure that carries traffic signal equipment: vertical shafts, mast arms (horizontal/diagonal arms extending over the road), and gantry supports. One entry per straight segment (a mast-arm pole = one vertical segment + one arm segment). Include a signal pole/mast arm even if its light housings are cropped out of this image. Do NOT include street-lamp or utility poles that carry no traffic signal equipment.

All coordinates MUST use the normalized 0-1000 coordinate system: origin at the top-left corner of the image, x increases to the right from 0 (left edge) to 1000 (right edge), y increases downward from 0 (top edge) to 1000 (bottom edge). Integers only.

Respond with ONLY a JSON object (no prose, no markdown fences) in exactly this schema:
{{
  "traffic_lights": [
    {{
      "bbox_2d": [x1, y1, x2, y2],
      "signal_type": "vehicle" | "pedestrian" | "unknown",
      "orientation": "vertical" | "horizontal",
      "num_sections": <int, number of lens sections; if not clearly countable, use 3>,
      "state": "red" | "yellow" | "green" | "green_arrow" | "red_arrow" | "off" | "unknown",
      "facing": "toward_camera" | "away" | "side" | "unknown",
      "est_distance_m": <float, your best estimate of distance from the camera in meters>
    }}
  ],
  "poles": [
    {{
      "line_2d": [[x1, y1], [x2, y2]],
      "segment_type": "vertical_pole" | "mast_arm" | "gantry" | "span_wire",
      "base_on_ground": <true if the segment's lower endpoint is the visible ground contact point of the pole, else false>,
      "attached_light_indices": [<indices into the traffic_lights array of housings mounted on this segment>],
      "est_distance_m": <float, distance from the camera to the segment's base/nearest point in meters>
    }}
  ]
}}

Rules:
- Only REAL physical signal heads: do NOT include traffic lights depicted on billboards/posters/signs, reflections in windows or mirrors, or images on vehicle screens.
- Exactly one entry per physical housing: do not emit separate entries for a housing and its backplate/visor, and do not report the same housing twice.
- bbox_2d is a tight box around the housing only (exclude visor shadows, pole, backplate edges beyond the housing).
- line_2d runs along the segment's centerline. For vertical poles put the GROUND endpoint first ([x1,y1] = bottom). For mast arms put the pole-junction endpoint first.
- If the pole base is occluded or outside the image, set base_on_ground to false and give the lowest visible point.
- Report ALL visible traffic lights, including small/distant ones, cross-traffic ones, and back-facing housings.
- If none are visible, return {{"traffic_lights": [], "poles": []}}.
- Output valid JSON only."""

    user_prompt = ("Detect all traffic light housings (2D bounding boxes) and their supporting "
                   "poles/mast arms (2D line segments) in this image. Return the JSON object only.")
    return system_prompt, user_prompt


# ============================================================================
# Response parsing
# ============================================================================

def extract_json_object(text: str) -> Optional[dict]:
    """Extract the first valid JSON object from a model response (tolerates fences/prose)."""
    if not text:
        return None
    # Strip a fenced block if present
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = [fence.group(1)] if fence else []
    # First balanced {...} in raw text
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start:i + 1])
                    break
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def normalize_coords(pts: np.ndarray, full_w: int, full_h: int) -> np.ndarray:
    """
    Map model-emitted coordinates to FULL-RESOLUTION pixel space.

    Qwen3-VL grounds in a 0-1000 normalized space per axis (verified by
    calibration against the served model; it does not honor absolute-pixel
    instructions). A 0-1 normalized fallback is kept for robustness.
    pts: (N, 2) array of [x, y].
    """
    pts = pts.astype(np.float64)
    max_val = pts.max() if pts.size else 0.0
    if max_val <= 1.5:  # 0-1 normalized fallback
        pts *= 1000.0
    pts[:, 0] = np.clip(pts[:, 0] / 1000.0 * full_w, 0, full_w - 1)
    pts[:, 1] = np.clip(pts[:, 1] / 1000.0 * full_h, 0, full_h - 1)
    return pts


# ============================================================================
# 3D lifting
# ============================================================================

def pixel_ray_ego(u: float, v: float, K_inv: np.ndarray,
                  R_c2e: np.ndarray) -> np.ndarray:
    """Direction (not normalized) of the camera ray through pixel (u,v), in ego frame."""
    d_cam = K_inv @ np.array([u, v, 1.0])
    return R_c2e @ d_cam


def cam_point_to_ego(p_cam: np.ndarray, R_c2e: np.ndarray, t_c2e: np.ndarray) -> np.ndarray:
    return R_c2e @ p_cam + t_c2e


def lift_traffic_light(det: dict, K: np.ndarray, K_inv: np.ndarray,
                       R_c2e: np.ndarray, t_c2e: np.ndarray) -> Optional[dict]:
    """
    Lift a 2D traffic light bbox to a 3D box in ego FLU frame.

    Depth: pinhole geometry from assumed physical housing size; falls back to the
    model's est_distance_m when the pixel extent is too small or the estimate is
    implausible. Returns dict with center/size/yaw + depth bookkeeping, or None.
    """
    x1, y1, x2, y2 = det["bbox_2d"]
    w_px, h_px = x2 - x1, y2 - y1
    if w_px <= 0 or h_px <= 0:
        return None

    fx, fy = K[0, 0], K[1, 1]
    num_sections = max(1, int(det.get("num_sections") or 3))
    orientation = det.get("orientation", "vertical")
    signal_type = det.get("signal_type", "vehicle")

    # Assumed physical extent of the housing along its long axis
    if signal_type == "pedestrian":
        long_extent_m = PEDESTRIAN_HOUSING_M
    else:
        long_extent_m = LENS_SECTION_SIZE_M * num_sections

    # Pinhole depth from the long axis (more pixels -> more reliable)
    depth = None
    depth_source = None
    if orientation == "horizontal":
        if w_px >= MIN_PIXEL_EXTENT:
            depth = fx * long_extent_m / w_px
            depth_source = "pinhole_width"
    else:
        if h_px >= MIN_PIXEL_EXTENT:
            depth = fy * long_extent_m / h_px
            depth_source = "pinhole_height"

    est_dist = det.get("est_distance_m")
    if depth is None or depth <= 0 or depth > MAX_PLAUSIBLE_DEPTH_M:
        if est_dist and 0 < float(est_dist) <= MAX_PLAUSIBLE_DEPTH_M:
            depth = float(est_dist)
            depth_source = "model_estimate"
        else:
            return None

    # Center of bbox -> 3D point at that depth (K_inv @ [u,v,1] has z == 1)
    uc, vc = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    center_cam = depth * (K_inv @ np.array([uc, vc, 1.0]))
    center_ego = cam_point_to_ego(center_cam, R_c2e, t_c2e)

    # Physical box size from pixel extents at this depth
    width_m = w_px * depth / fx
    height_m = h_px * depth / fy

    # Yaw: housing front normal assumed to point from the box toward the ego origin
    facing_vec = -center_ego[:2]
    norm = np.linalg.norm(facing_vec)
    yaw = float(np.arctan2(facing_vec[1], facing_vec[0])) if norm > 1e-6 else 0.0

    return {
        "center": [round(float(c), 3) for c in center_ego],
        # [length (thickness along facing), width, height] — matches gt_boxes l/w/h order
        "size": [HOUSING_THICKNESS_M, round(float(width_m), 3), round(float(height_m), 3)],
        "yaw": round(yaw, 4),
        "depth_m": round(float(depth), 2),
        "depth_source": depth_source,
        "model_est_distance_m": est_dist,
    }


def lift_pole_segment(det: dict, K_inv: np.ndarray, R_c2e: np.ndarray,
                      t_c2e: np.ndarray, attached_depth: Optional[float]) -> Optional[dict]:
    """
    Lift a 2D pole line segment to a 3D line segment in ego FLU frame.

    Vertical segments with a ground-contact endpoint: intersect the bottom-pixel
    ray with the ego ground plane (z=0), then place the top endpoint on the
    vertical line above the base. Other segments: use the depth of an attached
    traffic light if available, else the model's distance estimate, applied along
    both endpoint rays.
    """
    (u1, v1), (u2, v2) = det["line_2d"]
    base_on_ground = bool(det.get("base_on_ground"))
    segment_type = det.get("segment_type", "vertical_pole")
    est_dist = det.get("est_distance_m")

    d1 = pixel_ray_ego(u1, v1, R_c2e=R_c2e, K_inv=K_inv)  # first endpoint (bottom/junction)
    d2 = pixel_ray_ego(u2, v2, R_c2e=R_c2e, K_inv=K_inv)

    if base_on_ground and segment_type in ("vertical_pole", "gantry"):
        # Ray/ground-plane intersection for the base (ego frame: ground at z=0)
        if d1[2] >= -1e-6:  # ray does not descend toward the ground
            return None
        s = -t_c2e[2] / d1[2]
        if s <= 0:
            return None
        base = t_c2e + s * d1
        if np.linalg.norm(base[:2]) > MAX_PLAUSIBLE_DEPTH_M:
            return None
        # Top endpoint: scale its ray so the point sits vertically above the base
        horiz_base = np.linalg.norm(base[:2] - t_c2e[:2])
        horiz_d2 = np.linalg.norm(d2[:2])
        if horiz_d2 < 1e-6:
            return None
        top = t_c2e + (horiz_base / horiz_d2) * d2
        top[:2] = base[:2]  # enforce verticality
        depth_source = "ground_plane"
        depth = float(horiz_base)
    else:
        depth = attached_depth
        depth_source = "attached_light"
        if depth is None:
            if est_dist and 0 < float(est_dist) <= MAX_PLAUSIBLE_DEPTH_M:
                depth = float(est_dist)
                depth_source = "model_estimate"
            else:
                return None
        n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
        if n1 < 1e-6 or n2 < 1e-6:
            return None
        base = t_c2e + depth * d1 / n1
        top = t_c2e + depth * d2 / n2

    return {
        "bottom": [round(float(c), 3) for c in base],
        "top": [round(float(c), 3) for c in top],
        "depth_m": round(float(depth), 2),
        "depth_source": depth_source,
        "model_est_distance_m": est_dist,
    }


def lift_view_detections(parsed: dict, cam_data, full_w: int,
                         full_h: int) -> Tuple[List[dict], List[dict]]:
    """
    Lift all parsed 2D detections of one camera view to 3D (ego FLU frame).

    Coordinates from the model are 0-1000 normalized (Qwen3-VL native grounding
    convention, independent of any client- or server-side resizing); they are
    converted to the full-resolution pixel space that the intrinsic matrix refers
    to before lifting.

    Returns (traffic_lights, poles) with both 2D and 3D fields per detection.
    """
    K = np.array(cam_data.intrinsic, dtype=np.float64)
    K_inv = np.linalg.inv(K)
    R_c2e = quaternion_to_rotation_matrix(cam_data.sensor2ego_rotation)
    t_c2e = np.array(cam_data.sensor2ego_translation, dtype=np.float64)

    lights_out = []
    raw_lights = parsed.get("traffic_lights") or []
    for det in raw_lights:
        try:
            bbox = np.array(det["bbox_2d"], dtype=np.float64).reshape(2, 2)
        except (KeyError, ValueError, TypeError):
            continue
        bbox = normalize_coords(bbox, full_w, full_h)
        x1, y1 = bbox.min(axis=0)
        x2, y2 = bbox.max(axis=0)
        det2 = dict(det)
        det2["bbox_2d"] = [round(float(v), 1) for v in (x1, y1, x2, y2)]
        bbox_3d = lift_traffic_light(det2, K, K_inv, R_c2e, t_c2e)
        det2["bbox_3d"] = bbox_3d
        lights_out.append(det2)

    poles_out = []
    raw_poles = parsed.get("poles") or []
    for det in raw_poles:
        try:
            line = np.array(det["line_2d"], dtype=np.float64).reshape(2, 2)
        except (KeyError, ValueError, TypeError):
            continue
        line = normalize_coords(line, full_w, full_h)
        det2 = dict(det)
        det2["line_2d"] = [[round(float(v), 1) for v in pt] for pt in line]

        # Depth hint from an attached, successfully-lifted traffic light
        attached_depth = None
        for li in det.get("attached_light_indices") or []:
            if isinstance(li, int) and 0 <= li < len(lights_out):
                b3d = lights_out[li].get("bbox_3d")
                if b3d:
                    attached_depth = b3d["depth_m"]
                    break

        det2["line_3d"] = lift_pole_segment(det2, K_inv, R_c2e, t_c2e, attached_depth)
        poles_out.append(det2)

    return lights_out, poles_out


# ============================================================================
# Misc helpers
# ============================================================================

def image_to_base64_data_uri(img_pil: Image.Image, format: str = "JPEG") -> str:
    """Convert a PIL Image to a base64 data URI string."""
    buffer = io.BytesIO()
    save_kwargs = {"format": format}
    if format.upper() == "JPEG":
        save_kwargs["quality"] = 95
    img_pil.save(buffer, **save_kwargs)
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
        timeout=120.0,  # 120s timeout to handle slow vLLM responses
        max_retries=3,  # Retry on transient connection errors
    )
    _worker_loader = NuScenesDataLoader(pkl_path)


# ============================================================================
# Worker function for multiprocessing
# ============================================================================

def _worker_detect_sample(
    sample_idx: int,
    model_name: str,
    upscale_factor: float,
    upscale_size: Optional[Tuple[int, int]],
    max_new_tokens: int,
    front_only: bool,
    results_dir: str,
    visualize: bool,
    viz_output_dir: str,
) -> Dict:
    """
    Worker that processes a single sample: one detection API call per camera view
    (6 sequential inferences per timestamp), then 3D lifting, then one merged JSON
    written as {sample_idx:04d}_{token}.json.

    Each view is upscaled x`upscale_factor` per side before inference so small
    distant lights/poles survive the vision embedding; detected coordinates are
    mapped back to the original full-resolution space.
    """
    global _worker_client, _worker_loader
    try:
        client = _worker_client
        loader = _worker_loader

        sample = loader.get_sample(sample_idx)
        camera_names = FRONT_CAMERAS if front_only else loader.CAMERA_NAMES

        per_view = {}
        all_lights, all_poles = [], []
        total_infer_time = 0.0

        for cam_name in camera_names:
            cam_data = sample.cameras[cam_name]
            # IMPORTANT: no horizontal flip — pixel coords must map to the real image
            img_pil = Image.open(cam_data.image_path).convert('RGB')
            full_w, full_h = img_pil.size
            if upscale_size is not None:
                # Exact target size (may distort aspect ratio, e.g. 4000x4000).
                # Coordinates are unaffected: the model grounds 0-1000 per axis,
                # which maps back to full-res regardless of anisotropic scaling.
                img_pil = img_pil.resize(tuple(upscale_size), Image.LANCZOS)
            elif upscale_factor > 1:
                img_pil = img_pil.resize(
                    (round(full_w * upscale_factor), round(full_h * upscale_factor)),
                    Image.LANCZOS,
                )
            img_w, img_h = img_pil.size

            system_prompt, user_prompt = create_detection_prompt(cam_name)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": image_to_base64_data_uri(img_pil)}},
                    {"type": "text", "text": user_prompt},
                ]},
            ]

            start_time = time.time()
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=0.0,
            )
            elapsed = time.time() - start_time
            total_infer_time += elapsed
            response_text = response.choices[0].message.content

            parsed = extract_json_object(response_text)
            view_record = {
                "camera": cam_name,
                "label": CAM_LABELS[cam_name],
                "image_size": [img_w, img_h],
                "inference_time_sec": round(elapsed, 2),
                "parse_ok": parsed is not None,
                "raw_response": response_text,
            }

            if parsed is not None:
                lights, poles = lift_view_detections(
                    parsed, cam_data, full_w, full_h
                )
                # Re-key pole->light references into the global light list
                light_base = len(all_lights)
                for li, light in enumerate(lights):
                    light["id"] = f"TL_{light_base + li}"
                    light["camera"] = cam_name
                    all_lights.append(light)
                pole_base = len(all_poles)
                for pi, pole in enumerate(poles):
                    pole["id"] = f"POLE_{pole_base + pi}"
                    pole["camera"] = cam_name
                    pole["attached_light_ids"] = [
                        f"TL_{light_base + li}"
                        for li in (pole.pop("attached_light_indices", None) or [])
                        if isinstance(li, int) and 0 <= li < len(lights)
                    ]
                    all_poles.append(pole)
                view_record["num_traffic_lights"] = len(lights)
                view_record["num_poles"] = len(poles)

            per_view[cam_name] = view_record

        first_cam = sample.cameras[loader.CAMERA_NAMES[0]]
        result = {
            "prompt_type": "traffic_light_pole_3d_detection_vllm_mp",
            "sample_idx": sample_idx,
            "token": sample.token,
            "scene_token": sample.scene_token,
            "location": sample.location,
            "description": sample.description,
            "model_name": model_name,
            "upscale_factor": upscale_factor,
            "upscale_size": list(upscale_size) if upscale_size else None,
            "coordinate_frame": "ego_FLU",
            "ego2global_rotation": list(first_cam.ego2global_rotation) if first_cam.ego2global_rotation is not None else None,
            "ego2global_translation": list(first_cam.ego2global_translation) if first_cam.ego2global_translation is not None else None,
            "traffic_lights": all_lights,
            "poles": all_poles,
            "per_view": per_view,
            "inference_time_sec": round(total_infer_time, 2),
            "timestamp": datetime.now().isoformat(),
        }

        result_filename = f"{sample_idx:04d}_{sample.token}.json"
        result_path = os.path.join(results_dir, result_filename)
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        if visualize:
            # Lazy import: pulls in matplotlib (via qa_visualizer) only when needed
            from nuscenes_pipeline.visualization.detection_drawing import render_detection_composite
            composite = render_detection_composite(sample, loader, all_lights, all_poles)
            viz_path = os.path.join(viz_output_dir, f"{sample_idx:04d}_{sample.token}.jpg")
            composite.save(viz_path, quality=90)
            result["viz_path"] = viz_path

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
    """Unpack tuple arguments and call _worker_detect_sample."""
    return _worker_detect_sample(*args)


class TrafficLightPoleDetectorVLLMMultiprocessing:
    """
    Traffic light & pole 3D detector for nuScenes dataset using a vLLM-served
    Qwen3-VL model. Uses multiprocessing with tqdm progress bar to send multiple
    API calls in parallel; each sample issues one detection call per camera view.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-235B-A22B-Instruct",
        api_base: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        pkl_path: str = "nuscenes2d_ego_temporal_infos_val.pkl",
        upscale_factor: float = 2.0,
        upscale_size: Optional[Tuple[int, int]] = None,
        max_new_tokens: int = 2048,
        front_only: bool = False,
        num_workers: int = 8,
        results_dir: str = "traffic_light_pole_3d_results",
        visualize: bool = False,
        viz_output_dir: str = "traffic_light_pole_3d_vis",
    ):
        print("=" * 80)
        print("Initializing Traffic Light & Pole 3D Detector (vLLM Multiprocessing)")
        print("=" * 80)

        # Store parameters for workers
        self.model_name = model_name
        self.api_base = api_base
        self.api_key = api_key
        self.pkl_path = pkl_path
        self.upscale_factor = upscale_factor
        self.upscale_size = upscale_size
        self.max_new_tokens = max_new_tokens
        self.front_only = front_only
        self.num_workers = num_workers
        self.results_dir = results_dir
        self.visualize = visualize
        self.viz_output_dir = viz_output_dir

        # Verify connection to vLLM server
        print(f"\n[1/3] Verifying vLLM server at: {api_base}")
        print(f"  Model: {model_name}")
        try:
            test_client = OpenAI(base_url=api_base, api_key=api_key)
            models = test_client.models.list()
            print(f"  Server reachable. Available models: {[m.id for m in models.data]}")
        except Exception as e:
            print(f"  WARNING: Could not connect to vLLM server: {e}")
            print(f"  Proceeding anyway — workers will retry on their own.")

        # Verify pkl file exists
        print(f"\n[2/3] Checking nuScenes data: {pkl_path}")
        if os.path.exists(pkl_path):
            loader = NuScenesDataLoader(pkl_path)
            print(f"  Found {len(loader)} samples")
            del loader
        else:
            print(f"  WARNING: pkl file not found at {pkl_path}")

        os.makedirs(self.results_dir, exist_ok=True)
        if self.visualize:
            os.makedirs(self.viz_output_dir, exist_ok=True)

        print(f"\n[3/3] Configuration:")
        print(f"  - API base: {api_base}")
        print(f"  - Model: {model_name}")
        print(f"  - Num workers: {num_workers}")
        if upscale_size is not None:
            ar_note = "aspect DISTORTED" if abs(upscale_size[0]/upscale_size[1] - 16/9) > 0.01 else "aspect preserved"
            print(f"  - Upscale size: exact {upscale_size[0]}x{upscale_size[1]} ({ar_note}; "
                  f"coords unaffected — 0-1000 normalized)")
        else:
            print(f"  - Upscale factor: x{upscale_factor:g} per side "
                  f"(x{upscale_factor * upscale_factor:g} area; 1600x900 -> "
                  f"{round(1600 * upscale_factor)}x{round(900 * upscale_factor)}, "
                  f"aspect ratio preserved)")
        print(f"  - Inference granularity: 1 image per API call "
              f"({'3' if front_only else '6'} sequential calls per timestamp)")
        print(f"  - Max new tokens: {max_new_tokens}")
        print(f"  - Cameras: {'front 3 only' if front_only else 'all 6 views'}")
        print(f"  - Rear-camera flip: DISABLED (pixel coords must match real image)")
        print(f"  - Visualize: {visualize}")
        if visualize:
            print(f"  - Visualization output: {viz_output_dir} (annotated 6-view + BEV composites)")
        print(f"  - Results output: {self.results_dir}")
        print("=" * 80 + "\n")

    def detect_batch(self, sample_indices: List[int]) -> Dict:
        """Detect traffic lights & poles for multiple samples using multiprocessing."""
        all_results = {}

        worker_args = [
            (
                idx,
                self.model_name,
                self.upscale_factor,
                self.upscale_size,
                self.max_new_tokens,
                self.front_only,
                self.results_dir,
                self.visualize,
                self.viz_output_dir,
            )
            for idx in sample_indices
        ]

        num_workers = min(self.num_workers, len(sample_indices))
        print(f"Starting multiprocessing pool with {num_workers} workers...")
        print(f"Processing {len(sample_indices)} samples...\n")

        with Pool(
            processes=num_workers,
            initializer=_worker_init,
            initargs=(self.model_name, self.api_base, self.api_key, self.pkl_path),
        ) as pool:
            results_iter = pool.imap_unordered(_worker_wrapper, worker_args)

            for result in tqdm(
                results_iter,
                total=len(sample_indices),
                desc="Detecting traffic lights & poles",
                unit="sample",
                ncols=100,
            ):
                sample_idx = result.get("sample_idx", "unknown")
                if "error" in result:
                    print(f"\n  ✗ Sample {sample_idx} failed: {result['error']}")
                else:
                    elapsed = result.get("inference_time_sec", "?")
                    n_tl = len(result.get("traffic_lights", []))
                    n_pole = len(result.get("poles", []))
                    print(f"\n  ✓ Sample {sample_idx} done ({elapsed}s) — {n_tl} lights, {n_pole} pole segments")
                all_results[f"sample_{sample_idx}"] = result

        return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Traffic Light & Pole 3D Detection using vLLM-served Qwen3-VL (Multiprocessing)"
    )

    # Model and API settings
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-VL-235B-A22B-Instruct",
        help="Model name served by vLLM",
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="EMPTY",
        help='API key (default "EMPTY" for local vLLM)',
    )

    # Data path
    parser.add_argument(
        "--pkl_path",
        type=str,
        default=os.environ.get("NUSCENES_PKL_PATH", "nuscenes2d_ego_temporal_infos_val.pkl"),
        help="Path to nuScenes pkl file",
    )

    # Sample selection
    parser.add_argument(
        "--sample_indices",
        type=int,
        nargs="+",
        help="Specific sample indices to analyze",
    )
    parser.add_argument(
        "--start_idx", type=int, default=None, help="Start index for range processing"
    )
    parser.add_argument(
        "--end_idx", type=int, default=None, help="End index for range processing"
    )

    # Processing options
    parser.add_argument(
        "--upscale_factor",
        type=float,
        default=2.0,
        help="Upscale each view x N per side (x N^2 area, aspect-preserving) before "
             "inference so small distant traffic lights survive the vision embedding. "
             "E.g. 2.0 -> 3200x1800, 2.5 -> 4000x2250, 3.0 -> 4800x2700. Set 1 to "
             "disable. Keep total pixels under the model's 16.78M budget (x3.4 max) "
             "or the server downscales it back (harmless for coordinates, wasteful "
             "for compute). Coordinates are unaffected by this setting (0-1000 "
             "normalized convention).",
    )
    parser.add_argument(
        "--upscale_size",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        default=None,
        help="Resize each view to an EXACT WxH before inference (e.g. 4000 4000), "
             "overriding --upscale_factor. A non-16:9 size distorts the aspect "
             "ratio; coordinates still map back correctly (0-1000 normalized per "
             "axis), but recognition quality may suffer from the distortion.",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=8192,
        help="Maximum tokens to generate per view (cap, not cost; 8192 leaves room "
             "for reasoning tokens if a Thinking model is served)",
    )
    parser.add_argument(
        "--front_only",
        action="store_true",
        help="Only process the 3 front-facing cameras",
    )

    # Multiprocessing
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of parallel worker processes (default: 8)",
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        default="traffic_light_pole_3d_results",
        help="Directory to save result JSON files",
    )

    # Visualization
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Also save an annotated composite image per sample (6-view grid with "
             "state-colored light bboxes + pole segments, and BEV with lifted 3D detections)",
    )
    parser.add_argument(
        "--viz_output_dir",
        type=str,
        default="traffic_light_pole_3d_vis",
        help="Directory to save visualization composites",
    )

    args = parser.parse_args()

    # Determine sample indices
    if args.sample_indices:
        sample_indices = args.sample_indices
    elif args.start_idx is not None and args.end_idx is not None:
        sample_indices = list(range(args.start_idx, args.end_idx + 1))
    elif args.start_idx is not None:
        # end_idx not given → load total sample count from pkl
        tmp_loader = NuScenesDataLoader(args.pkl_path)
        total_samples = len(tmp_loader)
        del tmp_loader
        sample_indices = list(range(args.start_idx, total_samples))
        print(f"Auto-detected {total_samples} total samples. Processing {args.start_idx} ~ {total_samples - 1}")
    else:
        # Default: analyze first sample
        sample_indices = [0]

    detector = TrafficLightPoleDetectorVLLMMultiprocessing(
        model_name=args.model_name,
        api_base=args.api_base,
        api_key=args.api_key,
        pkl_path=args.pkl_path,
        upscale_factor=args.upscale_factor,
        upscale_size=tuple(args.upscale_size) if args.upscale_size else None,
        max_new_tokens=args.max_new_tokens,
        front_only=args.front_only,
        num_workers=args.num_workers,
        results_dir=args.results_dir,
        visualize=args.visualize,
        viz_output_dir=args.viz_output_dir,
    )

    print(f"\nDetecting traffic lights & poles for {len(sample_indices)} samples "
          f"(vLLM, Multiprocessing x{args.num_workers})...\n")

    total_start = time.time()
    results = detector.detect_batch(sample_indices)
    total_elapsed = time.time() - total_start

    # Summary
    successful = sum(1 for r in results.values() if "error" not in r)
    failed = sum(1 for r in results.values() if "error" in r)
    avg_time = np.mean([
        r["inference_time_sec"] for r in results.values() if "inference_time_sec" in r
    ]) if successful > 0 else 0
    total_lights = sum(len(r.get("traffic_lights", [])) for r in results.values())
    total_poles = sum(len(r.get("poles", [])) for r in results.values())

    print(f"\n{'='*80}")
    print("Traffic Light & Pole 3D Detection (vLLM Multiprocessing) Complete")
    print(f"{'='*80}")
    print(f"  Total samples: {len(sample_indices)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Traffic lights detected: {total_lights}")
    print(f"  Pole segments detected: {total_poles}")
    print(f"  Total wall time: {total_elapsed:.2f}s")
    print(f"  Avg inference time per sample: {avg_time:.2f}s")
    print(f"  Effective throughput: {len(sample_indices)/total_elapsed:.2f} samples/s")
    print(f"  Workers used: {args.num_workers}")
    print(f"  Results saved to: {detector.results_dir}/")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
