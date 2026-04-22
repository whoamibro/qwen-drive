"""
Traffic Sign Extraction Module for nuScenes Dataset

Extracts and classifies traffic signs visible in 6-view camera images per nuScenes
sample using a vLLM-served Qwen3-VL model via OpenAI-compatible API. Uses
multiprocessing for parallel inference.

Scans ALL 6 views (front 3 + rear 3) for regulatory, warning, and guide signs.
For each sign, determines category, text/symbol, orientation, and whether it
applies to the ego-vehicle via the Sign Ego-Relevance Test.

Prerequisites:
    # Start vLLM server first:
    vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct

Usage:
    # Single sample sign extraction
    python -m nuscenes_pipeline.modules.traffic_sign_extraction --sample_indices 0 10 20

    # With image resizing
    python -m nuscenes_pipeline.modules.traffic_sign_extraction --sample_indices 0 --resize_factor 4

    # Process range of samples
    python -m nuscenes_pipeline.modules.traffic_sign_extraction --start_idx 0 --end_idx 100

    # Custom number of workers (default: 8)
    python -m nuscenes_pipeline.modules.traffic_sign_extraction --start_idx 0 --end_idx 100 --num_workers 4
"""

import os
import io
import json
import time
import base64
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from multiprocessing import Pool
from functools import partial
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

from nuscenes_pipeline.core.nuscenes_data_loader import NuScenesDataLoader


def quaternion_to_rotation_matrix(q):
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    q = np.array(q)
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])


def get_camera_viewing_yaw(sensor2ego_rotation):
    """
    Get camera viewing direction yaw angle in ego frame (FLU).

    Camera frame: RDF (Right-Down-Forward) - +Z is viewing direction
    Ego frame: FLU (Forward-Left-Up) - +X is forward

    Returns:
        Yaw angle in degrees (0° = forward, +90° = left, -90° = right, ±180° = backward)
    """
    R_sensor2ego = quaternion_to_rotation_matrix(sensor2ego_rotation)
    # Camera forward direction is +Z in RDF frame
    cam_forward_sensor = np.array([0, 0, 1])
    cam_forward_ego = R_sensor2ego @ cam_forward_sensor
    yaw_rad = np.arctan2(cam_forward_ego[1], cam_forward_ego[0])
    return np.degrees(yaw_rad)


def get_camera_heading_info(sample, loader: NuScenesDataLoader) -> str:
    """
    Get camera heading information table with yaw angles computed from sensor2ego_rotation.

    Args:
        sample: NuScenesSample object
        loader: NuScenesDataLoader instance

    Returns:
        Formatted markdown table with camera heading information
    """
    # Camera name to label (must match exactly with prepare_messages_for_single_frame labels)
    cam_label_map = {
        'CAM_FRONT_LEFT': 'Image 1: Front-left Camera',
        'CAM_FRONT': 'Image 2: Front Camera',
        'CAM_FRONT_RIGHT': 'Image 3: Front-right Camera',
        'CAM_BACK_LEFT': 'Image 4: Rear-left Camera',
        'CAM_BACK': 'Image 5: Rear Camera',
        'CAM_BACK_RIGHT': 'Image 6: Rear-right Camera'
    }

    lines = []
    lines.append("| Label | Yaw | Viewing Direction | Note |")
    lines.append("|-------|-----|-------------------|------|")

    for cam_name in loader.CAMERA_NAMES:
        cam_data = sample.cameras[cam_name]
        label = cam_label_map[cam_name]

        # Compute viewing yaw from sensor2ego_rotation
        yaw_deg = get_camera_viewing_yaw(cam_data.sensor2ego_rotation)

        # Determine direction description based on yaw angle
        if -22.5 <= yaw_deg <= 22.5:
            direction = "Forward"
        elif 22.5 < yaw_deg <= 67.5:
            direction = "Forward-Left"
        elif 67.5 < yaw_deg <= 112.5:
            direction = "Left"
        elif 112.5 < yaw_deg <= 157.5:
            direction = "Backward-Left"
        elif yaw_deg > 157.5 or yaw_deg < -157.5:
            direction = "Backward"
        elif -157.5 <= yaw_deg < -112.5:
            direction = "Backward-Right"
        elif -112.5 <= yaw_deg < -67.5:
            direction = "Right"
        elif -67.5 <= yaw_deg < -22.5:
            direction = "Forward-Right"
        else:
            direction = f"({yaw_deg:.0f}°)"

        # Note for rear cameras (images are flipped for egocentric view)
        note = "Image flipped" if cam_name in loader.REAR_CAMERAS else "-"

        lines.append(f"| {label} | {yaw_deg:.0f}° | {direction} | {note} |")

    return "\n".join(lines)


def create_traffic_sign_extraction_prompt(sample, loader: NuScenesDataLoader, user_question: str,
                                           use_global_coords: bool = False) -> tuple:
    """
    Create the prompt for traffic sign extraction across all 6 camera views.

    Scans every view for regulatory, warning, and guide signs. For each sign,
    determines category (by shape + color), text/symbol, orientation, and applies
    the Sign Ego-Relevance Test to decide whether the sign governs the ego-vehicle.
    Output includes [SIGN-SCAN] per-view results, [SIGN-APPLICABLE] list, and
    [SIGN-ACTION] derived ego obligation.

    Args:
        sample: NuScenesSample object
        loader: NuScenesDataLoader instance
        user_question: Custom question/task from user
        use_global_coords: If True, use global ENU coordinates; if False, use ego-relative FLU coordinates

    Returns:
        Tuple of (system_prompt, user_question)
    """
    # Get camera heading information from sensor2ego_rotation
    camera_heading_table = get_camera_heading_info(sample, loader)

    # Base context about coordinate system (simplified - no extrinsics)
    if use_global_coords:
        coord_system_desc = """You are analyzing 6 camera images from an ego vehicle. The camera positions are specified in global coordinates using a right-handed coordinate system (ENU - East-North-Up):
- X: East
- Y: North
- Z: Up"""
    else:
        coord_system_desc = """You are analyzing 6 camera images from an ego vehicle. The vehicle uses a right-handed coordinate system (FLU - Forward-Left-Up):
- X: forward
- Y: left
- Z: up"""

    # Determine location (US or Singapore)
    location_str = sample.location.lower() if sample.location else ""
    if "singapore" in location_str:
        region = "Singapore"
    else:
        region = "US"

    # Compute ego velocity in FLU (ego-relative) coordinates
    first_cam = sample.cameras[loader.CAMERA_NAMES[0]]
    ego_vel_global = np.array([0.0, 0.0])
    try:
        next_sample = loader.get_sample(sample.sample_idx + 1)
        first_cam_next = next_sample.cameras[list(next_sample.cameras.keys())[0]]
        pos_current = np.array(first_cam.ego2global_translation[:2])
        pos_next = np.array(first_cam_next.ego2global_translation[:2])
        dt = (next_sample.timestamp - sample.timestamp) / 1e6
        if dt > 0:
            ego_vel_global = (pos_next - pos_current) / dt
    except:
        pass

    # Transform global velocity to ego-relative FLU
    ego_quat = first_cam.ego2global_rotation
    qw, qx, qy, qz = ego_quat[0], ego_quat[1], ego_quat[2], ego_quat[3]
    ego_yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
    cos_yaw = np.cos(-ego_yaw)
    sin_yaw = np.sin(-ego_yaw)
    vx_ego = ego_vel_global[0] * cos_yaw - ego_vel_global[1] * sin_yaw
    vy_ego = ego_vel_global[0] * sin_yaw + ego_vel_global[1] * cos_yaw
    speed = np.sqrt(vx_ego**2 + vy_ego**2)

    # Determine driving command from gt_navigation_command
    driving_command_map = {0: "Turn left", 1: "Turn right", 2: "Go straight", 3: "Follow lane", 4: "Change lane to left", 5: "Change lane to right", 6: "U-Turn"}
    if hasattr(sample, 'gt_navigation_command') and sample.gt_navigation_command is not None:
        cmd_idx = int(sample.gt_navigation_command)
        driving_command = driving_command_map.get(cmd_idx, "Go straight")
    else:
        driving_command = "Go straight"

    # Simplified system prompt header
    system_prompt_header = "You are a driving expert agent, and should answer the question at the viewpoint of a driver."

    # Core principle block (Sign Ego-Relevance Test based)
    core_principle = """
# CORE PRINCIPLE — READ BEFORE ALL ANALYSIS
**A traffic sign applies to the ego-vehicle ONLY if ALL three conditions hold:**
1. The sign **faces ego's approach direction** (text/symbol readable head-on — not back-of-sign, not side-profile).
2. The sign is **on ego's road** (roadside, median, gantry, or overpass of ego's lane or entry).
3. The sign is **NOT obviously intended for cross-traffic or a different lane**.

Signs can appear in ANY of the 6 camera views — front 3 (Images 1-3) AND rear 3 (Images 4-6).
- **Front views**: Signs ego is approaching (most important for ego's upcoming action).
- **Rear views**: Signs ego has already passed (continuing speed limits, one-way confirmation, "do not enter" for the direction ego came from).

**KEY TEST — Sign Ego-Relevance:**
For each candidate sign, ask:
- **Orientation**: Can I read the front face (text/symbol clearly visible)? If I see only the back (blank metal panel) → NOT ego-relevant.
- **Road Alignment**: Is the sign mounted on ego's road, or on a cross street / opposite side of a divided road?
- **Lane Specificity**: Does the sign target a specific lane different from ego's?

**"Back-of-sign" panels are NOT signs for ego** — a blank metal rectangle means the sign faces the OTHER direction.

**Cross-street signs are NOT ego's signs** — a "No Right Turn" sign on the perpendicular cross street does NOT restrict the ego-vehicle.

**Multiple signs? Apply priority: Regulatory > Warning > Guide.** A stop sign outranks a warning sign, which outranks a directional guide sign, for deciding ego's action.
"""

    # Traffic Signal System Guide (Traffic Flow Test based)
    traffic_sign_guide = """
---

# TRAFFIC SIGN SYSTEM GUIDE
---
## PART A: SIGN CATEGORIES (by shape + color)
---

| Shape              | Color           | Category    | Meaning                                    | Ego-Relevance Priority |
|--------------------|-----------------|-------------|--------------------------------------------|------------------------|
| Octagon            | Red + white     | Regulatory  | STOP                                        | HIGH — mandatory stop  |
| Inverted triangle  | Red + white     | Regulatory  | YIELD                                       | HIGH — yield required  |
| Rectangle (vert.)  | Black on white  | Regulatory  | Speed limit, No turn, No entry, One-way     | HIGH — legal obligation|
| Rectangle (vert.)  | Red circle+slash| Regulatory  | Prohibition (no U-turn, no parking, etc.)   | HIGH                   |
| Circle             | Red border      | Regulatory  | Prohibition (SG/EU style)                   | HIGH                   |
| Diamond            | Yellow/orange   | Warning     | Curve, merge, pedestrian, school, workzone  | MEDIUM — advisory      |
| Pentagon           | Yellow          | Warning     | School zone                                 | MEDIUM                 |
| Rectangle (horiz.) | Green/blue/brown| Guide       | Route, direction, POI, parking, services    | LOW — informational    |

---
## PART B: SIGN MOUNTING LOCATIONS
---

| Mount Type        | Description                                     | Typical Location                   |
|-------------------|-------------------------------------------------|------------------------------------|
| Roadside post     | Short pole on shoulder                          | Right shoulder (US) / Left (SG/UK) |
| Median post       | Pole on divider between opposing lanes          | Center median                      |
| Gantry / overhead | Structure spanning lanes (signs face approach)  | Highway / major intersection       |
| Overpass / bridge | Mounted on bridge face                          | Over ego's lane                    |
| Cantilever arm    | Arm extending over roadway                      | Multi-lane main roads              |

**Regional convention:** US/SG/EU signs are typically mounted on the RIGHT (ego's near side for right-lane driving). Singapore uses LEFT-side driving, so regulatory signs are more often on the LEFT.

---
## PART C: SIGN EGO-RELEVANCE TEST
---

### C1. THREE CONDITIONS (ALL MUST HOLD)

A sign applies to the ego-vehicle only if:
1. **Faces ego's approach direction** — sign text/symbol readable head-on.
2. **On ego's road** — mounted on ego's lane, adjacent shoulder, median, or overhead gantry.
3. **Not for cross-traffic / different lane** — not on the perpendicular cross street, not on the opposite side of a divided road.

### C2. ORIENTATION CHECK

Signboard geometry (not vehicle proximity) is the primary cue for which approach
a sign governs. Use the projected shape of the signboard as follows:

| Observation                                                   | Ego-Relevance                         |
|---------------------------------------------------------------|---------------------------------------|
| Sign in correct proportions (octagon/rectangle/diamond face-on), text fully readable | Faces ego; passes orientation check |
| Sign appears as a skewed/stretched trapezoid (partially facing ego) | Partially facing ego; defer to C5 (signpost direction) + C3 (road alignment) |
| Sign edge-on (thin slit, tall narrow profile)                 | Signboard perpendicular to camera; faces a different stream; FAIL |
| Rear visible (blank metal panel, no text/symbol)              | Faces opposite direction; FAIL         |

### C3. ROAD ALIGNMENT CHECK

| Sign Location (relative to ego)                   | Ego-Relevance           |
|---------------------------------------------------|-------------------------|
| Right shoulder / left shoulder of ego's road      | Likely ego              |
| Median between ego and oncoming (facing ego)      | Likely ego              |
| Gantry spanning ego's lane (facing ego)           | Likely ego              |
| Cross-street shoulder / perpendicular roadside    | Cross-traffic; FAIL     |
| Opposite side of divided road (facing oncoming)   | For oncoming traffic; FAIL |

### C4. REAR-VIEW SPECIAL CASE

Signs in Images 4-6 (rear cameras) ego has ALREADY passed:
- Readable from rear-view = sign faced ego when passed = still applies (continuing speed limits, one-way confirmation) unless superseded by a front-view sign.
- Back of sign visible from rear = sign faces oncoming/cross traffic, NOT ego.

### C5. SIGNPOST DIRECTIONAL INFERENCE

A sign is a directional device: the signboard physically faces ONE specific
approach. Vehicle proximity to the sign is NOT a reliable indicator (a cross-traffic
vehicle driving past a sign that faces ego does not make the sign belong to
cross-traffic). Instead, use these cues:

1. **Signboard angle (most reliable — also covered in C2)**:
   - Face-on proportions (octagon looks like an octagon, rectangle looks rectangular) → faces the camera / ego
   - Foreshortened / skewed trapezoid → angled — combine with post position below
   - Edge-on slit → faces perpendicular stream, NOT ego
   - Blank rear → faces opposite direction, NOT ego

2. **Post position relative to road geometry**:
   - Post on the shoulder of ego's road, near an intersection, signboard facing ego's approach → **for ego**
   - Post at the far corner of an intersection, signboard extending toward a cross-street → **for cross-traffic on that street**
   - Post on a median between ego and oncoming lanes, signboard facing ego → **for ego**
   - Post on the opposite side of a divided road, signboard facing away from ego → **for oncoming traffic**

3. **Related road markings (strong corroborating evidence)**:
   - STOP sign + stop line painted on ego's lane/approach → sign is **for ego**
   - STOP sign + stop line painted on the cross-street → sign is **for cross-traffic**
   - Crosswalk painted in front of a sign → sign pertains to that crosswalk
   - Lane arrows painted directly below an arrow sign → sign is lane-specific
   - No visible road markings → defer to signboard angle + post position

**Key reminder**: physical proximity of vehicles to a sign does NOT indicate
which stream the sign governs. A stop sign on the NE corner facing ego may have
cross-traffic vehicles driving past it closer to the sign than ego's approach
vehicles — the sign still belongs to ego if the signboard faces ego's approach.

### C6. LANE-SPECIFIC APPLICABILITY

A sign mounted on ego's road may still not apply to ego if it targets a
specific lane that ego is not in. Lane scope:

| Sign type                                                | Lane scope                           | Applies to ego?                                    |
|----------------------------------------------------------|--------------------------------------|----------------------------------------------------|
| Speed Limit (shoulder-mounted)                           | All lanes of ego's road              | Yes                                                |
| Speed Limit (lane-specific overhead gantry)              | Only the lane directly under it      | Only if ego is in that lane                        |
| "Straight Only" arrow (painted on lane or overhead)      | The specific lane it governs         | Only if ego is in that lane                        |
| "Right Turn Only" arrow                                  | The right-turn lane                  | Only if ego is in the right-turn lane              |
| "Left Turn Only" arrow                                   | The left-turn lane                   | Only if ego is in the left-turn lane               |
| "No Right Turn" (intersection approach)                  | All lanes on ego's approach          | Yes if ego intends to turn right; otherwise N/A    |
| "No Left Turn"                                           | All lanes on ego's approach          | Yes if ego intends to turn left; otherwise N/A     |
| STOP / YIELD (intersection approach)                     | All lanes of ego's approach          | Yes                                                |
| "Keep Right / Keep Left"                                 | All lanes (directional rule)         | Yes                                                |
| "Do Not Enter" / "No Entry"                              | The entry it blocks                  | Yes if ego's intended route enters that segment    |

### C7. DRIVING-COMMAND CROSS-CHECK

After passing C1-C6, cross-check the sign against ego's current driving command
(provided in the INPUT block of this prompt). A sign should only be marked
"Applies to ego: y" if it affects ego's intended action.

| Ego Driving Command | Sign                               | Applies to ego? |
|---------------------|------------------------------------|-----------------|
| Go straight         | STOP / YIELD                       | Yes             |
| Go straight         | Speed Limit                        | Yes (universal) |
| Go straight         | "No Right Turn"                    | No (ego isn't turning right) |
| Go straight         | "No Left Turn"                     | No              |
| Go straight         | "Right Turn Only" arrow above ego's lane | No (ego wouldn't be in that lane if going straight) |
| Turn Left           | "No Left Turn"                     | **Yes — CRITICAL** (prohibits ego's action) |
| Turn Left           | "Left Turn Only" arrow in ego's lane | Yes (confirms lane) |
| Turn Left           | "Straight Only" arrow in ego's lane | Flag as WARNING (ego is in wrong lane for the intended turn) |
| Turn Left           | "No Right Turn"                    | No              |
| Turn Right          | "No Right Turn"                    | **Yes — CRITICAL** |
| Turn Right          | "Right Turn Only" arrow in ego's lane | Yes (confirms lane) |
| Turn Right          | "No Left Turn"                     | No              |
| U-Turn              | "No U-Turn"                        | **Yes — CRITICAL** |
| Any                 | Speed Limit                        | Yes             |
| Any                 | Pedestrian Crossing (warning)      | Yes (applies to any maneuver) |
| Any                 | Curve Ahead / School Zone / Construction | Yes (applies along the path) |
| Any                 | Do Not Enter                       | Yes if ego's route enters that segment |

**Rule summary**:
- A sign that prohibits or restricts ego's intended action is HIGHLY relevant (CRITICAL).
- A sign that describes a maneuver ego is NOT making (e.g., "No Right Turn" when ego is going straight) is NOT applicable to ego.
- Universal signs (speed limit, pedestrian crossing, curve ahead) apply regardless of maneuver.

---
## PART D: COMMON SIGNS AND EGO ACTIONS
---

### D1. REGULATORY SIGNS (HIGHEST PRIORITY)

| Sign (text/symbol)            | Ego Action                                         |
|-------------------------------|----------------------------------------------------|
| STOP (octagon)                | Full stop at stop line; proceed when clear         |
| YIELD (inv. triangle)         | Slow, give way; stop only if conflict              |
| Speed Limit XX                | Do not exceed XX in ego's lane                     |
| No Right Turn / No Left Turn  | Turn prohibited from ego's lane                    |
| No U-Turn                     | U-turn prohibited                                  |
| One-Way (arrow)               | Travel only in arrow direction                     |
| Do Not Enter / No Entry       | Ego must not enter that segment                    |
| No Parking / No Stopping      | Parking/stopping prohibited in this zone           |
| Keep Right / Keep Left        | Stay in indicated lane                             |

### D2. WARNING SIGNS (MEDIUM PRIORITY)

| Sign (text/symbol)            | Ego Action                                         |
|-------------------------------|----------------------------------------------------|
| Pedestrian Crossing (diamond) | Reduce speed; yield to pedestrians at crosswalk    |
| School Zone (pentagon)        | Reduced speed during school hours                  |
| Merge / Lane Ends             | Prepare to merge or change lane                    |
| Curve Ahead / Sharp Turn      | Reduce speed for curve                             |
| Construction / Workzone       | Reduce speed, watch for workers/cones              |
| Slippery When Wet             | Reduce speed, watch for skid                       |
| Deer / Animal Crossing        | Scan for animals                                   |

### D3. GUIDE SIGNS (LOW PRIORITY)

| Sign (text/symbol)            | Ego Action                                         |
|-------------------------------|----------------------------------------------------|
| Route Marker / Highway shield | Informational; confirms ego is on the road         |
| Direction arrow (city names)  | Informational; supports navigation                 |
| Parking symbol (P)            | Informational                                      |
| Services / Gas / Food         | Informational                                      |

---
## PART E: SIGN PRIORITY RESOLUTION
---

When multiple signs are applicable to ego:
1. **Regulatory > Warning > Guide** — regulatory sign dictates ego action.
2. If multiple regulatory signs, the most restrictive applies (e.g., STOP outranks YIELD).
3. Warning signs modify how ego executes the regulatory obligation (e.g., YIELD + Pedestrian Crossing = slow AND check pedestrians).
4. Guide signs never override regulatory or warning signs.

---
## PART F: CONFIDENCE LEVELS FOR SIGNS
---

| Confidence | Criteria                                                                        |
|------------|----------------------------------------------------------------------------------|
| HIGH       | Sign text/symbol clearly readable; orientation confirms it faces ego             |
| MEDIUM     | Sign partially occluded or angled but type inferable from shape/color/partial text |
| LOW        | Sign visible but text unreadable or orientation ambiguous                        |
| NONE       | No ego-applicable signs found in any view                                        |
"""

    system_context = f"""{system_prompt_header}

{coord_system_desc}
{core_principle}
{traffic_sign_guide}
---
CRITICAL: Before reporting any sign, VERIFY which image number it appears in AND that its front face is readable (not back-of-sign).

Response in English."""

    # Build pre-analysis reminder
    pre_analysis_reminder = """
## PRE-ANALYSIS REMINDER
- **Scan ALL 6 views** — signs can appear in Images 1-6, not just front cameras
- **Back-of-sign (blank metal panel) ≠ a sign for ego** — it faces the other direction
- **Cross-street signs do NOT apply to ego** — a "No Right Turn" sign on a perpendicular street is for that street, not ego
- **Signboard orientation is the primary cue** — face-on proportions = faces ego; skewed/edge-on/rear = faces a different stream (C2 + C5)
- **Vehicle proximity to a sign is NOT reliable** — a cross-traffic vehicle driving past a sign that faces ego does not make the sign belong to cross-traffic
- **Road markings corroborate** — stop lines, crosswalks, lane arrows tell you which approach the sign pertains to (C5)
- **Lane-specific signs apply only to targeted lanes** — arrow signs, lane-specific speed limits, turn-only lanes (C6)
- **Cross-check against driving command** — a "No Left Turn" sign is CRITICAL when ego is turning left, N/A when ego is going straight (C7)
- **Apply ALL tests in order**: C1 → C2 → C3 → C4 → C5 → C6 → C7 before declaring Applies to ego: y
- **Priority: Regulatory > Warning > Guide** when multiple applicable signs exist

---"""

    # User prompt - Traffic Sign Extraction Task
    user_prompt = f"""# TRAFFIC SIGN EXTRACTION TASK
---
## INPUT
**Ego Vehicle Status:**
- Location: {region}
- Driving Command: {driving_command}
- Velocity (FLU): [vx={vx_ego:.2f}, vy={vy_ego:.2f}] m/s (+vx=forward, +vy=left)
- Speed: {speed:.2f} m/s
---
{pre_analysis_reminder}
## ANALYSIS STEPS

**Step 1: LANE POSITION** — Identify ego lane from road markings in Front/Front-Left/Front-Right views (Images 1-3).

**Step 2: PER-VIEW SIGN SCAN (ALL 6 VIEWS)**

Scan EVERY camera view. For each sign found, record:
- **Category:** Regulatory / Warning / Guide (determined by shape + color per PART A)
- **Text/Symbol:** Exact text or symbol visible (e.g., "STOP", "50", "No Right Turn", pedestrian glyph)
- **Orientation:** Faces ego / side-profile / back (blank metal)
- **Mount:** Roadside / median / gantry / overpass / pole

| View | What to scan for |
|------|-----------------|
| Image 1 (Front-Left)  | Left-shoulder signs, median signs facing ego, overpass left panels |
| Image 2 (Front)       | Gantry signs, main regulatory signs ahead, lane-assignment signs |
| Image 3 (Front-Right) | Right-shoulder signs (primary placement in US/SG), gantry right panels |
| Image 4 (Rear-Left)   | Signs already passed on the left (rear-facing speed limits, one-way confirm) |
| Image 5 (Rear)        | Signs behind ego (continuing limits, no-entry for reverse direction) |
| Image 6 (Rear-Right)  | Signs already passed on the right shoulder |

**Exclude:** back-of-sign (blank metal panels), signs clearly on a perpendicular cross street, informational signs that are not legally binding (unless they convey lane/direction obligation).

**Step 3: EGO-RELEVANCE TEST (PER SIGN)**

For EACH sign recorded in Step 2, run the full C1-C7 pipeline:
1. **C1+C2 Orientation**: signboard face-on proportions (not edge-on / rear / skewed beyond recognition)?
2. **C3 Road Alignment**: mounted on ego's road (shoulder/median/gantry/overpass — not cross-street / opposite divided road)?
3. **C4 Rear-view special case**: if in Images 4-6, is the front face still visible (already-passed sign) or only the back (not for ego)?
4. **C5 Signpost direction**: signboard angle + post position + road markings (stop line, crosswalk, lane arrows) all agree that the sign faces ego's approach? Ignore vehicle-proximity evidence.
5. **C6 Lane scope**: is the sign lane-specific? If so, is ego in the targeted lane?
6. **C7 Driving-command cross-check**: does ego's current command (Go straight / Turn Left / Turn Right / U-Turn / etc.) match the action the sign governs? A sign prohibiting ego's intended action is CRITICAL; a sign describing a maneuver ego is NOT making is N/A.

**Decision**: A sign is marked "Applies to ego: y" ONLY if ALL of C1-C7 pass.
Any failure → "Applies to ego: n" with the failing check cited in the Reason field.

**Step 4: SIGN PRIORITY RESOLUTION**

Among applicable signs:
- If Regulatory + Warning + Guide coexist → Regulatory dictates ego action; Warning modifies how; Guide is informational.
- If multiple Regulatory signs → Most restrictive applies (e.g., STOP outranks YIELD).
- If no applicable signs → Report `SIGN-APPLICABLE: None`.

**Step 5: DERIVE SIGN ACTION**

Based on the highest-priority applicable sign, state the ego obligation:
- Regulatory: "Stop required", "Max speed 50", "No right turn permitted", "Do not enter", "Yield required", "One-way eastbound", etc.
- Warning: "Reduce speed for pedestrian crossing", "Prepare to merge", "School zone speed", etc.
- Guide only: "None (informational)"
- No applicable signs: "None"

**Step 6: CONFIDENCE**
| Condition | Confidence |
|-----------|------------|
| Sign text/symbol clearly readable; orientation confirms ego | HIGH |
| Sign partially occluded or angled but type inferable from shape/color | MEDIUM |
| Sign visible but text unreadable or orientation ambiguous | LOW |
| No ego-applicable signs found in any view | NONE |

---
## OUTPUT FORMAT
```
=== TRAFFIC SIGN EXTRACTION ===

[SITUATION] Type: {{Intersection / Highway / Urban road / Rural / Parking area / Residential}}
[LANE] Position: {{lane}} | Evidence: {{markings}}

[SIGN-SCAN]
  Image 1: {{sign type + text/symbol, or "No sign"}} | Mount: {{roadside/median/gantry/overpass/post}} | Orientation: {{faces ego/skewed/edge-on/rear/N/A}} | Stream: {{ego-direction/oncoming/cross-traffic/unknown}} | Lane scope: {{all lanes/ego's lane/right-turn lane/left-turn lane/through lane/other}} | Applies to ego: {{y/n/uncertain}} | Reason: {{brief explanation referencing C2-C7 that led to the y/n decision, including ego's driving command when relevant}}
  Image 2: {{...}}
  Image 3: {{...}}
  Image 4: {{...}}
  Image 5: {{...}}
  Image 6: {{...}}

[SIGN-APPLICABLE] {{comma-separated list of ego-applicable signs with brief context, e.g. "STOP (Image 2, ego approach)", or "None"}}
[SIGN-PRIORITY] {{highest-priority category: Regulatory / Warning / Guide / None}}
[SIGN-ACTION] {{derived ego obligation — e.g., "Stop required", "Max speed 50", "No right turn permitted (ego intends to turn right — CRITICAL)", "None"}}
[SIGN-CONFIDENCE] {{HIGH / MEDIUM / LOW / NONE}}

=== END ===
```
---
## KEY RULES
1. **Scan ALL 6 views** — signs can appear in front OR rear cameras
2. **Back-of-sign ≠ sign for ego** — a blank metal panel means the sign faces the other direction
3. **Cross-street signs ≠ ego's signs** — perpendicular-road signs do NOT apply to ego
4. **Signboard angle > vehicle proximity** — a face-on signboard is for the approach it faces, regardless of which vehicles happen to drive near it
5. **Lane-specific signs apply only to the targeted lane** — arrow signs, lane-specific speed limits on gantries, turn-only lanes (see C6)
6. **Cross-check against driving command** — "No Left Turn" is irrelevant when ego goes straight; "Straight Only" arrow is irrelevant (and may warn wrong-lane) when ego is turning (see C7)
7. **Road markings corroborate** — stop lines, crosswalks, lane arrows indicate which approach a sign pertains to
8. **Regulatory > Warning > Guide** — the regulatory sign dictates action when multiple applicable signs exist
9. **Ignore pedestrian signals** — they are covered by the signal analysis module, not here
10. **Report "None"** — when no ego-applicable signs exist, do NOT fabricate
---
## TASK
Extract traffic signs and report per the output format above.
"""

    return (system_context, user_prompt)


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

def _worker_analyze_sample(
    sample_idx: int,
    model_name: str,
    resize_factor: int,
    max_new_tokens: int,
    use_global_coords: bool,
    results_dir: str,
) -> Dict:
    """
    Worker function that processes a single sample in its own process.
    Uses per-process client and loader initialized by _worker_init.

    Args:
        sample_idx: Sample index to analyze
        model_name: Model name served by vLLM
        resize_factor: Image resize factor
        max_new_tokens: Maximum tokens to generate
        use_global_coords: If True, use global ENU coordinates
        results_dir: Directory to save results

    Returns:
        Result dictionary
    """
    global _worker_client, _worker_loader
    try:
        client = _worker_client
        loader = _worker_loader

        # Get sample
        sample = loader.get_sample(sample_idx)

        # Generate prompt
        system_prompt, user_question_text = create_traffic_sign_extraction_prompt(
            sample, loader, "", use_global_coords=use_global_coords
        )

        # Prepare messages with images
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

        # Run inference
        start_time = time.time()
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=0.0,
        )
        elapsed_time = time.time() - start_time
        response_text = response.choices[0].message.content

        # Create result
        result = {
            "prompt_type": "traffic_sign_extraction_vllm_mp",
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

        # Save result
        result_filename = f"{sample_idx:04d}_{sample.scene_token[:16]}_{sample.token[:16]}_sign.json"
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


class TrafficSignExtractorVLLMMultiprocessing:
    """
    Traffic Signal Analyzer for nuScenes dataset using vLLM-served Qwen3-VL model.
    Uses multiprocessing with tqdm progress bar to send multiple API calls in parallel.
    Uses CONDITIONAL output format with signal-required branching.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-235B-A22B-Instruct",
        api_base: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        pkl_path: str = "nuscenes2d_ego_temporal_infos_val.pkl",
        resize_factor: int = 1,
        max_new_tokens: int = 4096,
        visualize: bool = False,
        viz_output_dir: str = "traffic_visualizations",
        use_global_coords: bool = False,
        num_workers: int = 8,
        results_dir: str = "traffic_sign_results",
    ):
        """
        Initialize the Traffic Signal Analyzer (vLLM Multiprocessing version).

        Args:
            model_name: Model name served by vLLM
            api_base: vLLM OpenAI-compatible API base URL
            api_key: API key (default "EMPTY" for local vLLM)
            pkl_path: Path to nuScenes pkl file
            resize_factor: Image resize factor (1/m of original size)
            max_new_tokens: Maximum tokens to generate
            visualize: Whether to generate visualizations
            viz_output_dir: Directory to save visualizations
            use_global_coords: If True, use global ENU coordinates; if False, use ego-relative FLU coordinates
            num_workers: Number of parallel worker processes (default: 8)
        """
        print("=" * 80)
        print(
            "Initializing Traffic Sign Extractor (vLLM Multiprocessing - 6-view sign scan)"
        )
        print("=" * 80)

        # Store parameters for workers
        self.model_name = model_name
        self.api_base = api_base
        self.api_key = api_key
        self.pkl_path = pkl_path
        self.resize_factor = resize_factor
        self.max_new_tokens = max_new_tokens
        self.visualize = visualize
        self.viz_output_dir = viz_output_dir
        self.use_global_coords = use_global_coords
        self.num_workers = num_workers
        self.results_dir = results_dir

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
            # Quick load to get count
            loader = NuScenesDataLoader(pkl_path)
            print(f"  Found {len(loader)} samples")
            del loader
        else:
            print(f"  WARNING: pkl file not found at {pkl_path}")

        # Create output directories
        os.makedirs(self.results_dir, exist_ok=True)
        if self.visualize:
            os.makedirs(self.viz_output_dir, exist_ok=True)

        print(f"\n[3/3] Configuration:")
        print(f"  - API base: {api_base}")
        print(f"  - Model: {model_name}")
        print(f"  - Num workers: {num_workers}")
        print(f"  - Resize factor: {resize_factor} (1/{resize_factor} of original)")
        print(f"  - Max new tokens: {max_new_tokens}")
        print(f"  - Visualize: {visualize}")
        print(f"  - Use global coords: {use_global_coords}")
        if visualize:
            print(f"  - Visualization output: {viz_output_dir}")
        print(f"  - Results output: {self.results_dir}")
        print("=" * 80 + "\n")

    def analyze_batch(self, sample_indices: List[int], user_question: str) -> Dict:
        """
        Analyze traffic signals for multiple samples using multiprocessing.

        Args:
            sample_indices: List of sample indices to analyze
            user_question: Custom question/task from user

        Returns:
            Dictionary of all results
        """
        all_results = {}

        # Build argument tuples for each sample (no pkl_path/api params — handled by initializer)
        worker_args = [
            (
                idx,
                self.model_name,
                self.resize_factor,
                self.max_new_tokens,
                self.use_global_coords,
                self.results_dir,
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

            # Wrap with tqdm for progress bar
            for result in tqdm(
                results_iter,
                total=len(sample_indices),
                desc="Analyzing traffic signals",
                unit="sample",
                ncols=100,
            ):
                sample_idx = result.get("sample_idx", "unknown")
                if "error" in result:
                    print(f"\n  ✗ Sample {sample_idx} failed: {result['error']}")
                else:
                    elapsed = result.get("inference_time_sec", "?")
                    print(f"\n  ✓ Sample {sample_idx} done ({elapsed}s)")
                all_results[f"sample_{sample_idx}"] = result

        return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Traffic Sign Extraction using vLLM-served Qwen3-VL (Multiprocessing, 6-view sign scan)"
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

    # Question
    parser.add_argument(
        "--question",
        type=str,
        default="Analyze the traffic signals visible in the images. Identify which signal governs the ego vehicle's lane and report its current state.",
        help="Custom question for traffic analysis",
    )

    # Processing options
    parser.add_argument(
        "--resize_factor",
        type=int,
        default=4,
        help="Image resize factor (1/m of original size). Default: 4",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=4096, help="Maximum tokens to generate"
    )

    # Multiprocessing
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of parallel worker processes (default: 8)",
    )

    # Coordinate system
    parser.add_argument(
        "--to_global",
        action="store_true",
        help="Use global ENU coordinates instead of ego-relative FLU coordinates",
    )

    # Visualization
    parser.add_argument(
        "--visualize", action="store_true", help="Generate visualizations"
    )
    parser.add_argument(
        "--viz_output_dir",
        type=str,
        default="traffic_visualizations",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="traffic_sign_results",
        help="Directory to save result JSON files",
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

    # Initialize analyzer
    analyzer = TrafficSignExtractorVLLMMultiprocessing(
        model_name=args.model_name,
        api_base=args.api_base,
        api_key=args.api_key,
        pkl_path=args.pkl_path,
        resize_factor=args.resize_factor,
        max_new_tokens=args.max_new_tokens,
        visualize=args.visualize,
        viz_output_dir=args.viz_output_dir,
        use_global_coords=args.to_global,
        num_workers=args.num_workers,
        results_dir=args.results_dir,
    )

    # Run analysis
    print(f"\nAnalyzing {len(sample_indices)} samples (Traffic Flow Test, vLLM, Multiprocessing x{args.num_workers})...")
    print(f"Question: {args.question}\n")

    total_start = time.time()
    results = analyzer.analyze_batch(sample_indices, args.question)
    total_elapsed = time.time() - total_start

    # Summary
    successful = sum(1 for r in results.values() if "error" not in r)
    failed = sum(1 for r in results.values() if "error" in r)
    avg_time = np.mean([
        r["inference_time_sec"] for r in results.values() if "inference_time_sec" in r
    ]) if successful > 0 else 0

    print(f"\n{'='*80}")
    print("Traffic Sign Extraction (vLLM Multiprocessing) Complete")
    print(f"{'='*80}")
    print(f"  Total samples: {len(sample_indices)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total wall time: {total_elapsed:.2f}s")
    print(f"  Avg inference time per sample: {avg_time:.2f}s")
    print(f"  Effective throughput: {len(sample_indices)/total_elapsed:.2f} samples/s")
    print(f"  Workers used: {args.num_workers}")
    print(f"  Results saved to: {analyzer.results_dir}/")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
