"""
Traffic Signal Analysis v2 — Stage 1B upgraded with detected signal boxes + VLM status.

v1's SYSTEM prompt (Traffic Signal System Guide etc.) is reused verbatim via
`traffic_signal_analysis.create_traffic_signal_analysis_prompt`; the USER
prompt is new: since the traffic signals are already detected and their lamp
states identified (pkl tl_* arrays from add_traffic_lights_to_infos.py +
apply_traffic_signal_status.py), the model's task is NOT signal detection but
REFERENCE SIGNAL SELECTION — analyze the panoramic scene formed by the 6 views
and determine the signal governing ego's driving, considering ego's driving
command, heading direction, and velocity (ego-status INPUT values are extracted
from v1's prompt so they stay identical).

The detections are listed like the risk_assessment pipeline's 3D object list:

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
import re
import time
from datetime import datetime
from multiprocessing import Pool
from typing import Dict, Optional

import numpy as np
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

from nuscenes_pipeline.core.nuscenes_data_loader import NuScenesDataLoader
from nuscenes_pipeline.core.nuscenes_prompt_generator import (
    format_camera_extrinsics_for_prompt,
    format_camera_extrinsics_global_for_prompt,
)
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


SYSTEM_INJECT_MARKER = "# CORE PRINCIPLE"

# --- v2 revisions of the v1 system-prompt guide (v1 module stays untouched) ---

# Removed in v2: the large-vehicle paragraph of the CORE PRINCIPLE block.
LARGE_VEHICLE_PARAGRAPH = """
**Large vehicles (buses, trucks) on the crossing road do NOT change signal selection.** A bus in Image 1 or Image 3 is on the CROSSING road. The signal near that bus governs the crossing road. Do NOT let a large vehicle's visual prominence draw your attention toward that image's signals.
"""

# Revised in v2: B1 — leftmost/rightmost signal position does NOT always mean a
# turn lane; ego's lane position + driving command must be considered together.
B1_ORIGINAL = """### B1. GENERAL PRINCIPLES
Traffic signals are positioned above or aligned with their corresponding lanes:
| Signal Position | Governs |
|-----------------|---------|
| Leftmost | Left-turn lane |
| Center | Through/straight lanes |
| Rightmost | Right-turn lane |
**Identification Steps:**
1. Determine ego lane from road markings (arrows, lane lines)
2. Find signal aligned with that lane
3. Consider intersection geometry, not just signal position in image"""

B1_REVISED = """### B1. GENERAL PRINCIPLES
Traffic signals are positioned above or aligned with their corresponding lanes:
| Signal Position | Typically Governs (NOT a fixed rule) |
|-----------------|--------------------------------------|
| Leftmost | Often the left-turn lane — but at many intersections it also (or only) governs through/straight traffic |
| Center | Through/straight lanes |
| Rightmost | Often the right-turn lane — but at many intersections it also (or only) governs through/straight traffic |
⚠️ Leftmost/Rightmost position does NOT always indicate a turn lane. At some intersections every signal head, including the outermost ones, shows the same through-phase. Lamp SHAPE decides: a dedicated turn signal shows an ARROW; a circular lamp governs its lane's movement whatever its position.
**Identification Steps:**
1. Determine ego's lane position from road markings (arrows, lane lines)
2. Combine ego's LANE POSITION with ego's DRIVING COMMAND to fix which movement (through / left / right) needs authorization
3. Find the signal aligned with that lane AND governing that movement (arrow vs circular lamp)
4. Consider intersection geometry, not just signal position in image"""

EXPECTED_VIEWS_BLOCK = """EXPECTED CAMERA VIEWS (Egocentric order - left to right):
- Image 1 (Front-left): Should show left-front scene, possibly left side mirror.
- Image 2 (Front): Should show straight-ahead road, traffic lights, pedestrians ahead.
- Image 3 (Front-right): Should show right-front scene, possibly right side mirror.
- Image 4 (Rear-left): Should show left-rear scene (horizontally flipped for egocentric view).
- Image 5 (Rear): Should show straight-behind scene (horizontally flipped for egocentric view).
- Image 6 (Rear-right): Should show right-rear scene (horizontally flipped for egocentric view)."""

def build_camera_setup_block(sample, loader, use_global_coords: bool) -> str:
    """Risk-assessment-style camera setup: per-camera extrinsics computed from
    calibration + the expected-views description (same content as
    create_single_frame_prompt's system prompt)."""
    fmt = (format_camera_extrinsics_global_for_prompt if use_global_coords
           else format_camera_extrinsics_for_prompt)
    cam_extrinsics = []
    for i, cam_name in enumerate(loader.CAMERA_NAMES, 1):
        cam_extrinsics.append(f"{i}. {fmt(sample.cameras[cam_name], cam_name)}")
    return ("# CAMERA SETUP\nThe 6 cameras are mounted as follows:\n\n"
            + "\n\n".join(cam_extrinsics) + "\n\n" + EXPECTED_VIEWS_BLOCK + "\n")


V2_STEPS_OUTPUT_RULES = """
## ANALYSIS STEPS

**Step 0: SIGNAL REFERENCE CHECK**
- Determine if signal reference is required based on System Prompt C7 (open road, roundabout, non-signalized intersection → NO)
- If the TS list is empty AND no signalized intersection is visible → signal reference NOT required
- If NO → Skip signal analysis, report reason
- If YES → Proceed to Step 1

**Step 1: LANE POSITION** - Identify ego lane from road markings in Front/Front-Left/Front-Right views.

**Step 2: DRIVING STATE & TURN TYPE**
2-1. Determine driving state from command + velocity:
| State | Condition |
|-------|-----------|
| Waiting | At stop line, speed ≈ 0 |
| Straight | vx > 0, vy ≈ 0, command = straight |
| Turn in progress | Command = turn, speed > 0 |
2-2. If turning, determine turn type:
| Location | Command | Turn Type | Key Rule |
|----------|---------|-----------|----------|
| US | Turn Left | **Complex** | Mid-turn: maintain departure signal |
| US | Turn Right | **Simple** | Yield to pedestrians; Turn-on-Red may apply |
| SG | Turn Right | **Complex** | Mid-turn: maintain departure signal |
| SG | Turn Left | **Simple** | Yield to pedestrians; Turn-on-Red if signed |
2-3. If turning, determine turn phase:
| Phase | Speed | Visual Cues | Signal Reference |
|-------|-------|-------------|------------------|
| Waiting | ≈ 0 | At stop line | Departure lane signal |
| Entering | > 0 | Crossing stop line | Departure lane signal |
| Mid-turn | > 0 | Inside intersection, heading changing | **Departure authorization (maintain)** |
| Exiting | > 0 | Aligning with new lane | New lane signal |

**Step 3: PANORAMIC SCENE UNDERSTANDING**
- Combine the 6 views into ONE coherent scene: where does ego's road run, where do the crossing roads run, and where is the intersection (if any)?
- Place each TS entry into that scene using its Image number and 2D bbox
- GROUP TS entries that are the SAME physical signal seen from different views (adjacent cameras overlap; e.g., one mast-arm signal can appear in Image 1 AND Image 2)
- Rear-view entries (Images 4-6) are behind ego: use them only for scene understanding (e.g., confirming ego is inside an intersection); they are NEVER the reference signal

**Step 4: PER-SIGNAL RELEVANCE CHECK (Traffic Flow Test + orientation)**
For EACH front-view TS entry (Images 1-3) — do NOT search for additional signals:
- Type: skip pedestrian-signal entries as reference candidates (still note them for Step 7)
- Orientation: does it face ego's approach direction? The identified status is a strong hint — "light observable" usually means the lamp face is toward the camera; "light NOT observable" usually means a side/back view
- **Traffic Flow Test:** Are vehicles near this signal traveling ACROSS ego's path?
  → YES → **CROSS-TRAFFIC** (these vehicles obey this signal on the crossing road)
  → NO → Check if it faces ego's approach → possible CANDIDATE
- **Large vehicle check:** A bus/truck near a signal on the crossing road CONFIRMS it is CROSS-TRAFFIC; do not let its visual size influence selection
- Verdict per TS entry: CROSS-TRAFFIC / CANDIDATE / UNCERTAIN

**Step 5: REFERENCE SIGNAL SELECTION**
Among CANDIDATE entries, determine THE reference signal for ego's driving:
- It must govern ego's CURRENT lane and ego's INTENDED path (driving command)
- Use ego's heading direction and velocity to fix the driving phase (Step 2-3) and apply the Complex/Simple-turn rules:
  - Complex turn (US Left / SG Right): Waiting/Entering → departure-lane signal; Mid-turn → maintain departure authorization, do NOT re-select; Exiting → new lane signal
  - Simple turn (US Right / SG Left): departure-lane signal; RED Arrow = no turn; Circular RED = Turn-on-Red may apply (US default)
```
IF exactly 1 CANDIDATE group → SELECT it. DONE.
IF multiple CANDIDATE groups → SELECT the one best aligned with ego's lane and intended path. DONE.
IF 0 CANDIDATEs → Report UNCERTAIN.
```
- A physical signal seen in several views counts ONCE — report all TS ids of the selected group
- If you are confident a signal governing ego exists in the images but is NOT in the TS list, report MISSING and describe it (do not silently invent a TS id)

**Step 6: REFERENCE SIGNAL STATE**
- Start from the identified status of the selected TS entry(ies) and cross-check against the images
- If the identified color conflicts with what is clearly visible, TRUST THE IMAGES and state the override
- If duplicates of the same signal carry conflicting identified colors, resolve visually; prefer the larger box / higher classifier confidence
- Position > Color when reading the lamps yourself: TOP=RED, MID=YELLOW, BOT=GREEN

**Step 7: PEDESTRIAN CHECK** - Critical for turns:
| Turn Type | Critical Crosswalk | Camera View |
|-----------|-------------------|-------------|
| Complex (US Left / SG Right) | Target lane side (ego will cross after turn) | Front (after turn begins) |
| Simple (US Right / SG Left) | **Immediate crosswalk** (turn path crosses it) | Front-Right (US) / Front-Left (SG) |
⚠️ **Pedestrian "WALK" signal ≠ Vehicle signal** — Always YIELD to pedestrians regardless of vehicle signal state.
---
## OUTPUT FORMAT
```
=== TRAFFIC ANALYSIS (V2) ===

[SITUATION] Type: {{Signalized intersection / Mid-turn / Roundabout / Open road / Non-signalized intersection}}
[LANE] Position: {{lane}} | Evidence: {{markings}}
[STATE] Driving: {{state}} | Turn Type: {{Complex/Simple/N/A}} | Phase: {{phase}}

[SIGNAL-REQUIRED] {{YES / NO}} | Reason: {{if NO, explain}}

--- IF SIGNAL REQUIRED ---
[SCENE] {{one-line panoramic summary: ego road direction, crossing roads, intersection layout}}
[TS-EVALUATION]
  TS {{id}} (Image {{#}}): group {{G#}} | Orientation: {{faces ego/side/back}} | Traffic Flow Test: {{CROSS-TRAFFIC / CANDIDATE / UNCERTAIN}} | Note: {{...}}
  (one line per front-view TS entry; same physical signal → same group G#)
[CANDIDATES] {{TS ids grouped by physical signal}}
[REFERENCE-SIGNAL] TS {{id(s)}} (Image {{#}}) | Reason: {{why this governs ego's command + heading + velocity}}
  (or: UNCERTAIN | or: MISSING: {{description of the ungoverned signal you see}})
[SIGNAL-STATE] State: {{color/arrow}} | Source: {{identified status / visual override}} | Confidence: {{H/M/L}}

--- IF SIGNAL NOT REQUIRED ---
[TRAFFIC-ELEMENTS] {{Oncoming:, Pedestrians:, Yield to:, Obstacles:}}
[APPLICABLE-RULES] {{relevant rules for this situation}}

--- COMMON ---
[PEDESTRIAN] Crosswalk: {{loc}} | Present: {{y/n}} | Yield Required: {{y/n}}
[RESULT] Authorization: {{PROCEED/STOP/YIELD/UNCERTAIN}} | Based on: {{Signal/Rule/Clear}} | Action: {{recommendation}}

=== END ===
```
---
## KEY RULES
1. **Do NOT search for new signals** — evaluate and select ONLY among the given TS entries (report MISSING if one is clearly absent)
2. **Traffic Flow Test determines signal relevance** — Crossing-road vehicles near a signal → that signal is cross-traffic
3. **Any front camera can contain the reference signal** — Image 1, 2, or 3; the Traffic Flow Test decides
4. **"Turn Left/Right" ≠ "Front-Left/Right camera"** — Turn command = intended path, not camera
5. **Same physical signal, multiple TS entries** — group them; select the GROUP, report all its TS ids
6. **Rear entries (Images 4-6) are never the reference signal** — scene context only
7. **Exclude pedestrian-signal entries as reference candidates** — but use them for the pedestrian check
8. **Identified status is a prior, not ground truth** — on clear visual conflict, trust the images and say so
9. **Complex Turn mid-turn** — Maintain departure authorization; do NOT re-select
10. **Position > Color** — TOP=RED, MID=YELLOW, BOT=GREEN
11. **Report UNCERTAIN** — if no TS entry passes the Traffic Flow Test + orientation check
---
## TASK
Determine the reference traffic signal for the ego vehicle's driving from the given detections and the panoramic scene, and report its state and the resulting authorization.
"""


def create_traffic_signal_analysis_v2_prompt(sample, loader, info: dict,
                                             use_global_coords: bool = False,
                                             resize_factor: int = 1,
                                             min_det_score: Optional[float] = None,
                                             include_not_a_signal: bool = False) -> tuple:
    """v2 prompt: v1 SYSTEM prompt + risk-assessment-style CAMERA SETUP
    (per-camera extrinsics + expected views, injected before the CORE PRINCIPLE
    block); new USER prompt whose task is reference-signal SELECTION over the
    given detections (not signal search).

    The ego-status INPUT lines (region, driving command, FLU velocity, speed)
    are extracted from v1's user prompt so the values stay identical to v1."""
    system_prompt, v1_user_prompt = create_traffic_signal_analysis_prompt(
        sample, loader, "", use_global_coords=use_global_coords)
    camera_setup = build_camera_setup_block(sample, loader, use_global_coords)
    if SYSTEM_INJECT_MARKER in system_prompt:
        system_prompt = system_prompt.replace(
            SYSTEM_INJECT_MARKER, camera_setup + "\n" + SYSTEM_INJECT_MARKER, 1)
    else:  # fallback: append
        system_prompt = system_prompt + "\n" + camera_setup
    # v2 guide revisions (see constants above)
    system_prompt = system_prompt.replace(LARGE_VEHICLE_PARAGRAPH, "\n", 1)
    system_prompt = system_prompt.replace(B1_ORIGINAL, B1_REVISED, 1)
    m = re.search(r"\*\*Ego Vehicle Status:\*\*\n(.*?)\n---", v1_user_prompt, re.DOTALL)
    ego_status = m.group(1) if m else "(ego status unavailable)"
    block = build_traffic_signal_block(info, resize_factor=resize_factor,
                                       min_det_score=min_det_score,
                                       include_not_a_signal=include_not_a_signal)

    user_prompt = f"""# TRAFFIC SIGNAL ANALYSIS TASK (V2 — REFERENCE SIGNAL SELECTION)
---
## INPUT
**Ego Vehicle Status:**
{ego_status}
---
{block}
## YOUR TASK
All traffic signals in this scene have ALREADY been detected and their lamp states identified — they are listed in DETECTED TRAFFIC SIGNAL INFORMATION above. You do NOT need to find traffic signals in the images.

Analyze the panoramic scene formed by the 6 camera views and determine THE REFERENCE TRAFFIC SIGNAL that governs the ego vehicle's driving, considering the ego vehicle's driving command, its heading direction, and its velocity. Then report that signal's state and the resulting driving authorization.

## PRE-ANALYSIS REMINDER
- The signals are GIVEN — your job is scene understanding and SELECTION, not detection
- **"Turn Left/Right" does NOT mean "look at the Front-Left/Right camera for your signal"**
- Ego's reference signal is typically ahead, governing the lane ego is currently in
- **Use the Traffic Flow Test on each given TS entry** to separate cross-traffic signals from ego's
- Vehicles on the crossing road (moving perpendicular to ego) identify cross-traffic signals
- **Large vehicles (buses, trucks) on the crossing road do NOT change signal selection**
---
{V2_STEPS_OUTPUT_RULES}"""
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
