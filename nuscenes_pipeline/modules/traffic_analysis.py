"""
Traffic Signal Analysis Module for nuScenes Dataset

Analyzes traffic signal states (red/yellow/green) per nuScenes sample using a vLLM-served
Qwen3-VL model via OpenAI-compatible API. Uses multiprocessing for parallel inference.

Uses a Traffic Flow Test approach as the primary method for signal identification,
with orientation and road alignment analysis per signal.

Prerequisites:
    # Start vLLM server first:
    vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct

Usage:
    # Single sample traffic analysis
    python -m nuscenes_pipeline.modules.traffic_analysis --sample_indices 0 10 20

    # With image resizing
    python -m nuscenes_pipeline.modules.traffic_analysis --sample_indices 0 --resize_factor 4

    # Process range of samples
    python -m nuscenes_pipeline.modules.traffic_analysis --start_idx 0 --end_idx 100

    # With global coordinates
    python -m nuscenes_pipeline.modules.traffic_analysis --sample_indices 0 --to_global

    # Custom number of workers (default: 8)
    python -m nuscenes_pipeline.modules.traffic_analysis --start_idx 0 --end_idx 100 --num_workers 4
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


def create_traffic_analysis_prompt_v8(sample, loader: NuScenesDataLoader, user_question: str,
                                       use_global_coords: bool = False) -> tuple:
    """
    Create an ENHANCED prompt for traffic signal analysis (V8).

    Key improvements over v7:
    - Replaced Image 2 priority rule with CORE PRINCIPLE: orientation + road alignment
    - Enhanced B2 with Road Alignment Check (KEY TEST)
    - Replaced MANDATORY RULE with CORE PRINCIPLE (orientation + traffic flow) for all phases
    - Step 3 applies Traffic Flow Test per signal with large vehicle check per signal
    - Step 5 uses CANDIDATE collection instead of decision tree
    - Output uses [SIGNAL-SCAN] with Traffic Flow Test results per signal
    - PRE-ANALYSIS REMINDER updated for Traffic Flow Test approach
    - D1 restructured around Traffic Flow Test
    - D2 restructured: 13 rules with Traffic Flow Test as primary
    - Confidence levels updated for Traffic Flow Test

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

    # Core principle block (v8 - Traffic Flow Test based)
    core_principle = """
# CORE PRINCIPLE — READ BEFORE ALL ANALYSIS
**A traffic signal governs ego's lane ONLY if it faces ego's approach direction AND is NOT aligned with the crossing road's traffic flow.**

At any intersection, there are signals for MULTIPLE traffic streams. Your job is to find the one that belongs to EGO's road — not the crossing road's.

**How to tell them apart:**
```
EGO'S SIGNAL:
  Faces ego's approach direction (light panels visible head-on)
  Vehicles on EGO's road (same direction as ego, or oncoming) obey this signal
  No crossing-road vehicles are flowing past this signal

CROSS-TRAFFIC SIGNAL:
  Faces vehicles on the CROSSING road (perpendicular to ego)
  Vehicles on the crossing road (moving left-to-right or right-to-left across ego's path) obey this signal
  Even if you can see its light panels clearly — the camera may be angled toward the crossing road
```

**KEY TEST — Traffic Flow Alignment:**
Look at the vehicles near a signal. Which direction are they traveling?
- Vehicles near the signal are traveling **ACROSS** ego's path (perpendicular) → **CROSS-TRAFFIC signal**
- Vehicles near the signal are traveling **ALONG** ego's road (same or opposite direction) → Possibly **EGO's signal**
- No vehicles near the signal → Check orientation and position relative to ego's road

**"Turn Left/Right" does NOT mean "look at the Front-Left/Right camera."** The turn command describes ego's intended path, not which camera contains ego's signal. Ego's departure signal is typically ahead of ego, governing the lane ego is currently in.

**Large vehicles (buses, trucks) on the crossing road do NOT change signal selection.** A bus in Image 1 or Image 3 is on the CROSSING road. The signal near that bus governs the crossing road. Do NOT let a large vehicle's visual prominence draw your attention toward that image's signals.
"""

    # Traffic Signal System Guide (V8 - Traffic Flow Test based)
    traffic_signal_guide = """
---

# TRAFFIC SIGNAL SYSTEM GUIDE
---
## PART A: SIGNAL BASICS
---
### POLE MOUNTING TYPES
| Type | Description | Typical Height | Usage |
|------|-------------|----------------|-------|
| Pedestal/Post-mounted | Standalone pole from ground | 3-5m | Small intersections, median, auxiliary |
| Mast arm | Arm extending over roadway | 6-10m | Multi-lane main signals |
| Span wire | Suspended on wires | 5-8m | Some US regions |
| Gantry | Structure spanning full road | 6-10m | Highways, large intersections |
| Cantilever | Single-side supported arm | 6-10m | Medium-large intersections |
---
### A1. REGIONAL TRAFFIC SYSTEM
| Attribute | United States (US) | Singapore (SG) |
|-----------|-------------------|----------------|
| Traffic Flow | RIGHT-hand (drive on right) | LEFT-hand (drive on left) |
| Complex Turn | LEFT turn (crosses oncoming) | RIGHT turn (crosses oncoming) |
| Simple Turn | RIGHT turn | LEFT turn |
| Turn on Red | RIGHT turn permitted (default) | LEFT turn only if signed |
---
### A2. SIGNAL STATE BY POSITION (PRIMARY METHOD)
**Position > Color** — Determine state by illuminated light POSITION, not by color recognition.
**Vertical (Most Common):**
| Position | State |
|----------|-------|
| TOP | RED (Stop) |
| MIDDLE | YELLOW (Caution) |
| BOTTOM | GREEN (Proceed) |
**Horizontal (Some US regions):**
| Position | State |
|----------|-------|
| LEFT | RED (Stop) |
| CENTER | YELLOW (Caution) |
| RIGHT | GREEN (Proceed) |
⚠️ Position is reliable. Color recognition may fail due to lighting/glare.
---
### A3. SIGNAL TYPES
| Type | Shape | Meaning |
|------|-------|---------|
| Circular RED | ● | STOP (turn on red may be allowed) |
| Circular YELLOW | ● | CAUTION - prepare to stop |
| Circular GREEN | ● | PROCEED - all directions including turns |
| Arrow GREEN | ← ↑ → | PROTECTED - arrow direction only |
| Arrow RED | ← ↑ → | PROHIBITED - arrow direction forbidden |
| RED + GREEN Arrow | ● + → | PARTIAL - only arrow direction permitted |
**Key Rule:** Circular GREEN permits turns even WITHOUT a dedicated arrow (permissive turn).
---
### A4. VEHICLE vs PEDESTRIAN SIGNALS
| Feature | Vehicle Signal | Pedestrian Signal |
|---------|---------------|-------------------|
| **Lights** | RED/YELLOW/GREEN circles or arrows | Walk figure / Stop hand symbols |
| Symbol | Circles (●) or arrows (←↑→) | Walking person / Raised hand |
| Height | HIGH (above roadway, on signal mast) | LOWER (near crosswalk, pole-mounted) |
| Faces | Vehicle travel direction | Pedestrian waiting area |
⚠️ **EXCLUDE pedestrian signals when identifying ego vehicle's signal.**
---
### A5. FLASHING SIGNALS
| Signal | Meaning |
|--------|---------|
| Flashing RED | Treat as STOP sign |
| Flashing YELLOW | Proceed with caution (NOT a right-of-way signal) |
| Flashing GREEN (SG) | About to turn amber |
⚠️ **Flashing yellow beacons are NOT traffic signals** — they are warnings only.
---
## PART B: SIGNAL-LANE PAIRING
---
### B1. GENERAL PRINCIPLES
Traffic signals are positioned above or aligned with their corresponding lanes:
| Signal Position | Governs |
|-----------------|---------|
| Leftmost | Left-turn lane |
| Center | Through/straight lanes |
| Rightmost | Right-turn lane |
**Identification Steps:**
1. Determine ego lane from road markings (arrows, lane lines)
2. Find signal aligned with that lane
3. Consider intersection geometry, not just signal position in image
---
### B2. SIGNAL ORIENTATION VERIFICATION
A signal governs ego's lane ONLY if:
1. Signal **faces ego's approach direction** (light panels visible, not side/back view)
2. Signal is **NOT aligned with the crossing road's traffic flow**
**Visual Cues:**
| Cue | Faces Ego (Front View) | Faces Cross Traffic (Side View) |
|-----|------------------------|--------------------------------|
| Housing shape | Circular lights visible | Narrow/rectangular profile |
| Light appearance | Distinct circles | Thin slit or edge glow |
| Hood/visor | Ring around lights | Protruding edge |
⚠️ **Seeing light from a signal ≠ Signal faces you.** Side-viewed signals show edge glow.
---
## PART C: CAMERA SYSTEM
---
### C1. CAMERA HEADING INFORMATION
| Label | Yaw | Viewing Direction |
|-------|-----|-------------------|
| Image 1: Front-left | 55° | Forward-Left |
| Image 2: Front | 1° | Forward |
| Image 3: Front-right | -58° | Forward-Right |
| Image 4: Rear-left | 109° | Left (flipped) |
| Image 5: Rear | 179° | Backward (flipped) |
| Image 6: Rear-right | -113° | Backward-Right (flipped) |

**Understanding what each camera captures at intersections:**
- Image 2 (yaw ~0°) points along ego's travel direction → Often captures ego's signal, but not always
- Image 1 (yaw ~55°) points left of ego → Often captures signals facing LEFT-APPROACHING cross-traffic
- Image 3 (yaw ~-58°) points right of ego → Often captures signals facing RIGHT-APPROACHING cross-traffic
- **However, ego's signal CAN appear in Image 1 or Image 3** depending on intersection geometry and signal placement
- **Do NOT assume any camera is always correct or always wrong** — use the Traffic Flow Test (see C4) to determine each signal's relevance
---
### C2. SIGNAL CAMERA PRIORITY BY DRIVING PHASE

| Phase | Ego Position | Speed | Signal Action |
|-------|--------------|-------|---------------|
| **Waiting** | Before stop line | ≈ 0 | Scan all front views; apply Traffic Flow Test to each signal |
| **Entering** | At/crossing stop line | > 0 | Confirm departure signal |
| **Crossing** | Inside intersection | > 0 | Maintain departure authorization — do NOT re-select |
| **Exiting** | Leaving intersection | > 0 | Scan for new lane signal |

**No single camera has automatic priority.** The correct signal is determined by the Traffic Flow Test applied to each signal individually.
---
### C3. SIGNAL ORIENTATION BY CAMERA VIEW
| Signal Orientation | What You See | Confidence |
|-------------------|--------------|------------|
| Faces camera (head-on) | Clear view of lights | HIGH — but still apply Traffic Flow Test |
| Slight angle (<30°) | Lights visible, less clear | MEDIUM |
| Side view (>45°) | Edge/profile of housing | LOW or NOT DETERMINABLE |
| Facing away | Back of housing | NOT DETERMINABLE |
---
### C4. TRAFFIC FLOW TEST — CROSS-TRAFFIC SIGNAL IDENTIFICATION

**This is the PRIMARY method for determining if a signal is ego's or cross-traffic.**

For each signal you find, look at the vehicles and road near that signal:

```
TRAFFIC FLOW TEST:
  1. Are there vehicles near this signal that are traveling ACROSS ego's path?
     (i.e., moving left-to-right or right-to-left relative to ego's forward direction)
     → YES: This signal governs the CROSSING road. Mark as CROSS-TRAFFIC.

  2. Is this signal mounted above/beside a road that runs PERPENDICULAR to ego's road?
     → YES: This signal governs the CROSSING road. Mark as CROSS-TRAFFIC.

  3. Is this signal facing ego's approach direction, with no crossing-road vehicles flowing past it?
     → YES: This signal may govern ego's lane. Mark as CANDIDATE.

  4. Cannot determine traffic flow or road alignment?
     → Mark as UNCERTAIN.
```

**LARGE-VEHICLE SALIENCY WARNING:**
Large vehicles (buses, trucks, trailers) on the crossing road can visually dominate Image 1 or Image 3 and create a false sense of importance. The rule is:

| Situation in Image 1/3 | Correct interpretation |
|-------------------------|----------------------|
| Bus/truck stopped near a signal | That signal governs the crossing road (the bus obeys it) → CROSS-TRAFFIC |
| Bus/truck passing through intersection | That signal allowed the bus to proceed on the crossing road → CROSS-TRAFFIC |
| Large vehicle visually dominates the image | Ignore its size — apply the same Traffic Flow Test as for any small car |

**Why this matters:** When a large vehicle appears near a signal, it is tempting to reason: "There is a bus here → this is important → this signal must be relevant to me." This reasoning is WRONG. The bus and the signal both belong to the CROSSING road.

---
### C7. SIGNAL REFERENCE REQUIREMENT

| Situation | Signal Required? | Action |
|-----------|-----------------|--------|
| Approaching/Waiting at intersection | YES | Identify signal using Traffic Flow Test |
| Entering intersection | YES | Confirm signal |
| Mid-turn / Crossing intersection | NO (re-evaluation) | Departure authorization applies |
| Exiting intersection | YES (if new signal visible) | Identify new lane signal |
| Open road (no intersection) | NO | Not applicable |
| Roundabout | NO | YIELD rules only |
---
## PART D: QUICK REFERENCE
---
### D1. SIGNAL IDENTIFICATION STEPS
```
1. SCAN all front-facing views (Images 1, 2, 3) for traffic signals
2. EXCLUDE pedestrian signals (walking figure/hand symbols)
3. For EACH signal, apply the TRAFFIC FLOW TEST:
   - Are crossing-road vehicles near this signal? → CROSS-TRAFFIC
   - Is this signal above a perpendicular road? → CROSS-TRAFFIC
   - Does this signal face ego with no crossing traffic flow? → CANDIDATE
4. IGNORE vehicle size — buses/trucks on crossing road do not change signal selection
5. Among CANDIDATE signals, select the one aligned with ego's lane
6. DETERMINE state by POSITION (top/mid/bottom), not color
7. If no CANDIDATE found → Report UNCERTAIN
```
---
### D2. KEY RULES SUMMARY
1. **Traffic Flow Test determines signal relevance** — Not camera number, not signal size
2. **Crossing-road vehicles identify cross-traffic signals** — If vehicles near a signal travel across ego's path, that signal is cross-traffic
3. **Large vehicles on crossing road ≠ signal relevance** — A bus near a signal does NOT make it ego's signal
4. **Any camera can contain ego's signal** — Image 1, 2, or 3; the Traffic Flow Test decides
5. **"Turn Left/Right" ≠ "Front-Left/Right camera"** — Turn command = intended path, not camera selection
6. **Size and brightness ≠ Relevance** — Large bright signal may be cross-traffic
7. **Position > Color** — TOP=RED, MID=YELLOW, BOT=GREEN
8. **Circular GREEN permits ALL directions** — turns allowed without arrow
9. **Pedestrian signals ≠ Vehicle signals** — exclude walking figure/hand
10. **Complex Turn mid-turn** — Maintain departure authorization; do NOT re-select signal
11. **Simple Turn** — Check for RED Arrow (prohibits turn); YIELD to pedestrians
12. **Turn-on-Red** — Allowed on Circular RED (US Right default); NOT on RED Arrow
13. **Report UNCERTAIN when** — no signal passes the Traffic Flow Test as CANDIDATE
---
### D3. CONFIDENCE LEVELS
| Confidence | Criteria |
|------------|----------|
| HIGH | Signal passes Traffic Flow Test as CANDIDATE; faces ego; state clearly identifiable |
| MEDIUM | Signal likely CANDIDATE but traffic flow partially ambiguous; state distinguishable |
| LOW | Traffic flow unclear or signal orientation ambiguous |
| UNCERTAIN | Cannot confirm any signal governs ego's lane |
"""

    system_context = f"""{system_prompt_header}

{coord_system_desc}
{core_principle}
{traffic_signal_guide}
---
CRITICAL: Before reporting any visual feature (signs, lights, objects), VERIFY which image number it appears in.

Response in English."""

    # Build pre-analysis reminder based on driving command and speed
    pre_analysis_reminder = """
## PRE-ANALYSIS REMINDER
- **"Turn Left/Right" does NOT mean "look at the Front-Left/Right camera for your signal"**
- Ego's departure signal is typically ahead, governing the lane ego is currently in
- **Use the Traffic Flow Test** to determine which signals are cross-traffic and which are ego's
- Vehicles on the crossing road (moving perpendicular to ego) identify cross-traffic signals
- **Large vehicles (buses, trucks) on the crossing road do NOT change signal selection**

---"""

    # User prompt - Traffic Signal Analysis Task V8 (Traffic Flow Test)
    user_prompt = f"""# TRAFFIC SIGNAL ANALYSIS TASK
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

**Step 0: SIGNAL REFERENCE CHECK**
- Determine if signal reference is required based on System Prompt C7
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

**Step 3: PER-VIEW SIGNAL SCAN WITH TRAFFIC FLOW TEST**

Scan ALL three front-facing views. For EACH signal found, apply the Traffic Flow Test.

**Image 1 (Front-Left):**
For each signal:
- Type: Vehicle signal or pedestrian signal?
- Orientation: Can you see the front face of light panels?
- **Traffic Flow Test:** Are there vehicles near this signal traveling ACROSS ego's path?
  → YES → **CROSS-TRAFFIC** (these vehicles obey this signal on the crossing road)
  → NO → Check if signal faces ego's approach → possible CANDIDATE
- **Large vehicle check:** Is there a bus/truck/trailer near this signal on the crossing road?
  → YES → This CONFIRMS the signal is CROSS-TRAFFIC (the large vehicle obeys this crossing road signal)
  → Do NOT let the large vehicle's visual size influence your signal selection

**Image 2 (Front):**
For each signal:
- Type: Vehicle signal or pedestrian signal?
- Orientation: Can you see the front face of light panels?
- **Traffic Flow Test:** Same as above
- Note: Even small or partially occluded signals should be evaluated
- Pedestal signals (short poles, 3-5m) may appear small — they are valid

**Image 3 (Front-Right):**
For each signal:
- Type: Vehicle signal or pedestrian signal?
- Orientation: Can you see the front face of light panels?
- **Traffic Flow Test:** Same as above
- **Large vehicle check:** Same as Image 1

**Step 4: SIGNAL ORIENTATION CHECK** - For each signal NOT marked as CROSS-TRAFFIC:
- Housing shape: Circular lights (front view) or narrow profile (side view)?
- Light appearance: Distinct circles or thin slits/edge glow?
- Visor/hood: Ring around lights (front) or protruding edge (side)?
- Conclusion: Faces ego / Faces cross traffic / Uncertain

**Step 5: SIGNAL SELECTION**

Collect all CANDIDATE signals (passed Traffic Flow Test + faces ego):
```
IF exactly 1 CANDIDATE → SELECT it. DONE.
IF multiple CANDIDATEs → SELECT the one best aligned with ego's lane. DONE.
IF 0 CANDIDATEs → Report UNCERTAIN.
```

**FOR COMPLEX TURN (Turn Left in US / Turn Right in SG):**
- Waiting/Entering phase: Select ego's DEPARTURE lane signal (the one governing ego's current lane)
- Check for: GREEN Arrow (protected) or Circular GREEN (permissive, yield to oncoming)
- Mid-turn phase: Maintain departure authorization — do NOT re-select signals
- Exiting phase: New lane signal applies

**FOR SIMPLE TURN (Turn Right in US / Turn Left in SG):**
- Waiting phase: Select ego's departure lane signal
- Check for: RED Arrow (turn prohibited) vs Circular RED (turn-on-red may apply)
- Turn-on-Red: Allowed on Circular RED after full stop (US default; SG only if signed)
- **RED Arrow = Turn NOT allowed** — must wait for green

**Step 6: SIGNAL STATE** - Report based on selected signal:
| Condition | Confidence |
|-----------|------------|
| CANDIDATE, front face visible, state clearly identifiable | HIGH |
| CANDIDATE, slight angle, state distinguishable | MEDIUM |
| Traffic flow test ambiguous or orientation unclear | LOW / Not determinable |

**Step 7: PEDESTRIAN CHECK** - Critical for turns:
| Turn Type | Critical Crosswalk | Camera View |
|-----------|-------------------|-------------|
| Complex (US Left / SG Right) | Target lane side (ego will cross after turn) | Front (after turn begins) |
| Simple (US Right / SG Left) | **Immediate crosswalk** (turn path crosses it) | Front-Right (US) / Front-Left (SG) |
⚠️ **Pedestrian "WALK" signal ≠ Vehicle signal** — Always YIELD to pedestrians regardless of vehicle signal state.
---
## OUTPUT FORMAT
```
=== TRAFFIC ANALYSIS ===

[SITUATION] Type: {{Signalized intersection / Mid-turn / Roundabout / Open road / Non-signalized intersection}}
[LANE] Position: {{lane}} | Evidence: {{markings}}
[STATE] Driving: {{state}} | Turn Type: {{Complex/Simple/N/A}} | Phase: {{phase}}

[SIGNAL-REQUIRED] {{YES / NO}} | Reason: {{if NO, explain}}

--- IF SIGNAL REQUIRED ---
[SIGNAL-SCAN]
  Image 1: {{signal desc}} | Orientation: {{faces ego/side/back}} | Traffic Flow Test: {{crossing vehicles? → CROSS-TRAFFIC / no crossing vehicles → CANDIDATE}} | Large vehicle nearby: {{y/n}}
  Image 2: {{signal desc}} | Orientation: {{faces ego/side/back}} | Traffic Flow Test: {{result}}
  Image 3: {{signal desc}} | Orientation: {{faces ego/side/back}} | Traffic Flow Test: {{result}} | Large vehicle nearby: {{y/n}}
[CANDIDATES] {{list of signals that passed Traffic Flow Test + face ego}}
[SELECTED] Image: {{#}} | Signal: {{desc}} | Reason: {{why — based on Traffic Flow Test result}}
[SIGNAL-STATE] State: {{color/arrow}} | Confidence: {{H/M/L}}

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
1. **Traffic Flow Test determines signal relevance** — Crossing-road vehicles near a signal → that signal is cross-traffic
2. **Large vehicles on crossing road ≠ signal relevance** — A bus near a signal does NOT make it ego's signal
3. **Any camera can contain ego's signal** — Image 1, 2, or 3; the Traffic Flow Test decides
4. **"Turn Left/Right" ≠ "Front-Left/Right camera"** — Turn command = intended path, not camera
5. **Size and brightness ≠ Relevance** — Large bright signal may be cross-traffic
6. **Exclude pedestrian signals** — Walking figure / hand ≠ vehicle signal
7. **Verify orientation** — Signal must FACE ego, not just be visible
8. **Complex Turn mid-turn** — Maintain departure authorization; do NOT re-select
9. **Simple Turn** — RED Arrow = no turn; Circular RED = Turn-on-Red may apply (US Right)
10. **Turn-on-Red** — Allowed on Circular RED (US Right default); NOT on RED Arrow
11. **Position > Color** — TOP=RED, MID=YELLOW, BOT=GREEN
12. **Report UNCERTAIN** — If no signal passes Traffic Flow Test + orientation check
---
## TASK
Analyze the traffic information provided above.
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
        system_prompt, user_question_text = create_traffic_analysis_prompt_v8(
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
            "prompt_type": "traffic_analysis_v8_enhanced_vllm_mp",
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
        result_filename = f"{sample_idx:04d}_{sample.scene_token[:16]}_{sample.token[:16]}_traffic_v8.json"
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


class TrafficSignalAnalyzerV8VLLMMultiprocessing:
    """
    Traffic Signal Analyzer V8 for nuScenes dataset using vLLM-served Qwen3-VL model.
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
        results_dir: str = "traffic_analysis_results",
    ):
        """
        Initialize the Traffic Signal Analyzer V8 (vLLM Multiprocessing version).

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
            "Initializing Traffic Signal Analyzer V8 (vLLM Multiprocessing - Traffic Flow Test)"
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
        description="Traffic Signal Analysis V8 using vLLM-served Qwen3-VL (Multiprocessing, Traffic Flow Test)"
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
        default="traffic_analysis_results",
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
    analyzer = TrafficSignalAnalyzerV8VLLMMultiprocessing(
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
    print(f"\nAnalyzing {len(sample_indices)} samples (V8 - Traffic Flow Test, vLLM, Multiprocessing x{args.num_workers})...")
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
    print("Traffic Signal Analysis V8 (vLLM Multiprocessing) Complete")
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
