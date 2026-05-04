"""
SFT Prompt Builder for nuScenes Driving VQA

Constructs structured prompts for VLM fine-tuning from raw nuScenes data.
Designed as a modular, scalable pipeline that:
  - Builds system prompts defining the VLM's driving expert role
  - Constructs user prompts with ego status, 3D object lists, and task queries
  - Supports variable numbers of detected objects
  - Produces prompts directly consumable by Qwen3-VL

This module does NOT copy reference prompts; it reconstructs the structure
from first principles using the nuScenes data API.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from nuscenes_pipeline.core.nuscenes_data_loader import (
    NuScenesDataLoader,
    NuScenesSample,
    CameraData,
    quaternion_to_rotation_matrix,
    check_bbox_in_camera,
)
from nuscenes_pipeline.core.nuscenes_prompt_generator import (
    transform_bbox_to_global as ref_bbox_to_global,
    transform_velocity_to_global as ref_vel_to_global,
    compute_ttc_obb_global,
    get_bbox_2d_projection,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CAMERA_ORDER = [
    'CAM_FRONT_LEFT',
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT',
    'CAM_BACK',
    'CAM_BACK_RIGHT',
]

CAMERA_DISPLAY_NAMES = {
    'CAM_FRONT_LEFT':  'Front-left',
    'CAM_FRONT':       'Front',
    'CAM_FRONT_RIGHT': 'Front-right',
    'CAM_BACK_LEFT':   'Rear-left',
    'CAM_BACK':        'Rear',
    'CAM_BACK_RIGHT':  'Rear-right',
}

# 1-indexed image numbers matching the egocentric panoramic layout
CAMERA_IMAGE_NUM = {cam: i for i, cam in enumerate(CAMERA_ORDER, 1)}

REAR_CAMERAS = {'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'}

DRIVING_COMMANDS = {
    0: "Turn left",
    1: "Turn right",
    2: "Go straight",
    3: "Follow lane",
    4: "Change lane to left",
    5: "Change lane to right",
    6: "U-Turn",
}

# Vehicle categories (not filtered by rear distance)
VEHICLE_TYPES = {
    'car', 'truck', 'bus', 'trailer',
    'construction_vehicle', 'motorcycle', 'bicycle',
}

# Ego vehicle physical dimensions
EGO_LENGTH = 4.7
EGO_WIDTH = 1.9
EGO_REAR_TO_CENTER = 1.65


# ---------------------------------------------------------------------------
# Helper: Quaternion → yaw
# ---------------------------------------------------------------------------

def quaternion_to_yaw(q: List[float]) -> float:
    """Extract yaw from quaternion [w, x, y, z]."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def velocity_global_to_ego(vel_global: np.ndarray, ego2global_rot: List[float]) -> np.ndarray:
    """Rotate a 2D velocity from global ENU to ego FLU frame."""
    R = quaternion_to_rotation_matrix(ego2global_rot)
    v3 = np.array([vel_global[0], vel_global[1], 0.0])
    v_ego = R.T @ v3
    return v_ego[:2]


# ---------------------------------------------------------------------------
# Helper: Compute ego velocity / acceleration from adjacent frames
# ---------------------------------------------------------------------------

def compute_ego_velocity(sample: NuScenesSample, loader: NuScenesDataLoader) -> np.ndarray:
    """Compute ego velocity in global ENU from position difference to next frame."""
    try:
        nxt = loader.get_sample(sample.sample_idx + 1)
        if nxt.scene_token != sample.scene_token:
            return np.zeros(2)
        cam_cur = sample.cameras[CAMERA_ORDER[0]]
        cam_nxt = nxt.cameras[CAMERA_ORDER[0]]
        dt = (nxt.timestamp - sample.timestamp) / 1e6
        if dt <= 0:
            return np.zeros(2)
        dp = np.array(cam_nxt.ego2global_translation[:2]) - np.array(cam_cur.ego2global_translation[:2])
        return dp / dt
    except Exception:
        return np.zeros(2)


def compute_ego_acceleration(sample: NuScenesSample, loader: NuScenesDataLoader) -> Optional[np.ndarray]:
    """Centered-difference ego acceleration in global ENU."""
    try:
        idx = sample.sample_idx
        prev = loader.get_sample(idx - 1)
        nxt = loader.get_sample(idx + 1)
        if prev.scene_token != sample.scene_token or nxt.scene_token != sample.scene_token:
            return None
        cam_p = prev.cameras[CAMERA_ORDER[0]]
        cam_c = sample.cameras[CAMERA_ORDER[0]]
        cam_n = nxt.cameras[CAMERA_ORDER[0]]
        dt_f = (nxt.timestamp - sample.timestamp) / 1e6
        dt_b = (sample.timestamp - prev.timestamp) / 1e6
        if dt_f <= 0 or dt_b <= 0:
            return None
        v_f = (np.array(cam_n.ego2global_translation[:2]) - np.array(cam_c.ego2global_translation[:2])) / dt_f
        v_b = (np.array(cam_c.ego2global_translation[:2]) - np.array(cam_p.ego2global_translation[:2])) / dt_b
        return (v_f - v_b) / ((dt_f + dt_b) / 2.0)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Structured data extraction from sample
# ---------------------------------------------------------------------------

@dataclass
class EgoState:
    """Extracted ego vehicle state."""
    command: str
    position: np.ndarray      # [x, y, z] global ENU
    yaw: float                 # radians
    velocity_global: np.ndarray  # [vx, vy] global ENU
    velocity_ego: np.ndarray     # [vx, vy] FLU
    speed: float
    acceleration_global: Optional[np.ndarray] = None


@dataclass
class DetectedObject:
    """Single detected 3D object with computed attributes."""
    obj_id: int
    category: str
    distance: float
    velocity: np.ndarray       # [vx, vy] global
    camera_views: Dict[str, Tuple[int, int, int, int]]  # cam_name → 2D bbox
    ttc: float
    risk_level: str
    delta_p_x: float
    delta_p_y: float
    closing_speed: float


def extract_ego_state(sample: NuScenesSample, loader: NuScenesDataLoader) -> EgoState:
    """Extract all ego vehicle state from a sample."""
    cam0 = sample.cameras[CAMERA_ORDER[0]]
    pos = np.array(cam0.ego2global_translation)
    rot = cam0.ego2global_rotation
    yaw = quaternion_to_yaw(rot)

    cmd_idx = (sample.gt_navigation_command
               if hasattr(sample, 'gt_navigation_command') and sample.gt_navigation_command is not None
               else sample.gt_planning_command)
    command = DRIVING_COMMANDS.get(cmd_idx, f"Unknown ({cmd_idx})")

    vel_g = compute_ego_velocity(sample, loader)
    vel_e = velocity_global_to_ego(vel_g, rot)
    speed = float(np.linalg.norm(vel_g))
    accel = compute_ego_acceleration(sample, loader)

    return EgoState(
        command=command,
        position=pos,
        yaw=yaw,
        velocity_global=vel_g,
        velocity_ego=vel_e,
        speed=speed,
        acceleration_global=accel,
    )


def extract_objects(
    sample: NuScenesSample,
    ego_state: EgoState,
    max_distance: float = 50.0,
    rear_filter: float = 20.0,
    resize_factor: int = 2,
) -> List[DetectedObject]:
    """
    Extract and filter 3D detected objects with TTC computation.

    Follows the exact same pipeline as risk_assessment.py:
      - Visibility: check_bbox_in_camera() from nuscenes_data_loader
      - Transform: ref_bbox_to_global / ref_vel_to_global from nuscenes_prompt_generator
      - TTC: compute_ttc_obb_global() (OBB + SAT, acceleration-aware)
      - 2D bbox: get_bbox_2d_projection() for display coordinates

    gt_boxes and gt_velocity are in ego-relative FLU coordinates.
    """
    if len(sample.gt_boxes) == 0:
        return []

    # Ego→global transform
    cam0 = sample.cameras[CAMERA_ORDER[0]]
    ego2global_rot = cam0.ego2global_rotation
    ego2global_trans = cam0.ego2global_translation
    ego_pos = np.array(ego2global_trans[:2])

    # Ego acceleration (for acceleration-aware TTC)
    ego_accel = ego_state.acceleration_global

    objects = []
    obj_id = 0

    for i in range(len(sample.gt_boxes)):
        bbox = sample.gt_boxes[i]  # ego-relative FLU [x,y,z,L,W,H,yaw]
        name = sample.gt_names[i] if sample.gt_names is not None else "object"

        # Distance from ego origin in ego frame
        obj_x, obj_y = bbox[0], bbox[1]
        dist = float(np.sqrt(obj_x**2 + obj_y**2))

        if dist > max_distance:
            continue

        # Rear filter: x < 0 means behind ego in FLU frame
        if rear_filter is not None and obj_x < 0:
            if name.lower() not in VEHICLE_TYPES and dist > rear_filter:
                continue

        # Camera visibility — same function as reference pipeline
        visible_cameras = []
        for cam_name in CAMERA_ORDER:
            camera_data = sample.cameras[cam_name]
            if check_bbox_in_camera(bbox, camera_data):
                visible_cameras.append(cam_name)

        if not visible_cameras:
            continue

        # 2D bbox projection for each visible camera (for prompt display)
        cam_views = {}
        for cam_name in visible_cameras:
            camera_data = sample.cameras[cam_name]
            bbox_2d = get_bbox_2d_projection(bbox, camera_data, resize_factor)
            if bbox_2d is not None:
                x1, y1, x2, y2 = bbox_2d
                cam_views[cam_name] = (x1, y1, x2, y2)

        # Velocity (ego-frame FLU)
        vel_ego = sample.gt_velocity[i] if sample.gt_velocity is not None else np.zeros(2)
        if np.any(np.isnan(vel_ego)):
            vel_ego = np.zeros(2)

        # Transform to global — same functions as reference pipeline
        bbox_global = ref_bbox_to_global(bbox, ego2global_rot, ego2global_trans)
        vel_global = ref_vel_to_global(vel_ego, ego2global_rot)

        # TTC — same OBB function as reference pipeline
        ttc_result = compute_ttc_obb_global(
            p_ego_ref=ego_pos,
            v_ego_global=ego_state.velocity_global,
            yaw_ego=ego_state.yaw,
            p_obj_global=np.array(bbox_global[:2]),
            v_obj_global=np.array(vel_global),
            yaw_obj_global=bbox_global[6],
            L_obj=bbox_global[3],
            W_obj=bbox_global[4],
            a_ego_global=ego_accel,
        )

        obj_id += 1
        objects.append(DetectedObject(
            obj_id=obj_id,
            category=name,
            distance=round(dist, 1),
            velocity=vel_global,
            camera_views=cam_views,
            ttc=ttc_result['ttc'],
            risk_level=ttc_result['risk_level'],
            delta_p_x=ttc_result['intermediate']['delta_p_x'],
            delta_p_y=ttc_result['intermediate']['delta_p_y'],
            closing_speed=ttc_result['intermediate']['closing_speed'],
        ))

    return objects


# ===================================================================
# PROMPT CONSTRUCTION
# ===================================================================

# -------------------------------------------------------------------
# 1. SYSTEM PROMPT
# -------------------------------------------------------------------

def build_system_prompt() -> str:
    """
    Construct the system-level instruction prompt.

    Defines:
      - VLM role as a driving expert agent
      - Multi-view camera layout with egocentric ordering
      - Interleaved multi-modal reasoning expectations
      - Response language requirement
    """

    # Role definition
    role_block = (
        "You are an autonomous driving analysis agent with expertise in "
        "visual scene understanding, spatial reasoning, and real-time risk assessment.\n"
        "\n"
        "Your task is to interpret six surround-view camera images captured simultaneously "
        "from an ego vehicle, together with structured sensor data (3D detected objects and "
        "ego vehicle telemetry), in order to answer driving-related questions accurately."
    )

    # Camera layout specification
    camera_block_lines = [
        "MULTI-VIEW CAMERA CONFIGURATION (Egocentric order — left to right):",
    ]
    view_descriptions = {
        1: ("Front-left",  "Covers the left-front quadrant; may include the left side mirror area."),
        2: ("Front",       "Covers the forward driving direction; shows road ahead, traffic signals, and pedestrians."),
        3: ("Front-right", "Covers the right-front quadrant; may include the right side mirror area."),
        4: ("Rear-left",   "Covers the left-rear quadrant (horizontally flipped to preserve egocentric consistency)."),
        5: ("Rear",        "Covers the straight-behind view (horizontally flipped to preserve egocentric consistency)."),
        6: ("Rear-right",  "Covers the right-rear quadrant (horizontally flipped to preserve egocentric consistency)."),
    }
    for num, (name, desc) in view_descriptions.items():
        camera_block_lines.append(f"  Image {num} ({name}): {desc}")
    camera_block = "\n".join(camera_block_lines)

    # Interleaved reasoning requirement
    reasoning_block = (
        "INTERLEAVED MULTI-MODAL REASONING:\n"
        "When analyzing the scene you must jointly reason over:\n"
        "  (a) Visual content from each camera image\n"
        "  (b) Structured 3D object detections — each object is linked to specific camera\n"
        "      view(s) via 2D bounding box coordinates\n"
        "  (c) Ego vehicle telemetry — driving command, velocity, and acceleration\n"
        "\n"
        "For every object you reference, specify:\n"
        "  - Which Image number(s) it appears in\n"
        "  - Its spatial relationship to the ego vehicle (distance, direction)\n"
        "  - Its motion state and associated risk level\n"
        "\n"
        "IMAGE-DIRECTION MAPPING (use this to verify visual references):\n"
        "  Image 1 → Front-left | Image 2 → Front | Image 3 → Front-right\n"
        "  Image 4 → Rear-left  | Image 5 → Rear  | Image 6 → Rear-right\n"
        "Do NOT state a direction that conflicts with the Image number."
    )

    return f"{role_block}\n\n{camera_block}\n\n{reasoning_block}\n\nRespond in English."


# -------------------------------------------------------------------
# 2. USER PROMPT — Ego State Block
# -------------------------------------------------------------------

def format_ego_state(ego: EgoState) -> str:
    """Format the ego vehicle status block."""
    lines = [
        "** EGO VEHICLE's DRIVING STATUS **",
        f"  - Driving Command: {ego.command}",
        f"  - Position (global ENU): [x={ego.position[0]:.2f}, y={ego.position[1]:.2f}, z={ego.position[2]:.2f}] m",
        f"  - Yaw (global): {ego.yaw:.4f} rad ({np.degrees(ego.yaw):.2f}\u00b0)",
        f"  - Velocity (global ENU): [vx={ego.velocity_global[0]:.2f}, vy={ego.velocity_global[1]:.2f}] m/s",
        f"  - Velocity (ego-relative FLU): [vx={ego.velocity_ego[0]:.2f}, vy={ego.velocity_ego[1]:.2f}] m/s",
        f"  - Speed: {ego.speed:.2f} m/s",
    ]
    if ego.acceleration_global is not None:
        lines.append(
            f"  - Acceleration (global ENU): [ax={ego.acceleration_global[0]:.2f}, ay={ego.acceleration_global[1]:.2f}] m/s\u00b2"
        )
    lines.append("  - [FLU Velocity Guide: +vx=forward, -vx=backward, +vy=left, -vy=right]")
    return "\n".join(lines)


# -------------------------------------------------------------------
# 2. USER PROMPT — Object List Block
# -------------------------------------------------------------------

def format_object_list(objects: List[DetectedObject], max_distance: float) -> str:
    """Format the 3D detected objects block with per-object TTC."""
    if not objects:
        return "No 3D objects detected within the filter range."

    header = (
        f"=====\n"
        f"3D OBJECT INFORMATION WITH PRE-COMPUTED TTC\n"
        f"({len(objects)} objects within {max_distance:.0f}m)\n"
        f"====="
    )

    entries = []
    for obj in objects:
        # Build camera visibility string
        vis_parts = []
        for cam_name, (x1, y1, x2, y2) in obj.camera_views.items():
            img_num = CAMERA_IMAGE_NUM[cam_name]
            cam_label = CAMERA_DISPLAY_NAMES[cam_name]
            vis_parts.append(f"Image {img_num} ({cam_label}) 2D bbox [x1={x1}, y1={y1}, x2={x2}, y2={y2}]")
        vis_str = ", ".join(vis_parts)

        # TTC display
        ttc_str = f"{obj.ttc:.2f}s" if not np.isinf(obj.ttc) else "\u221e"

        entry = (
            f"-----\n"
            f"OBJ {obj.obj_id}: {obj.category} [{vis_str}]\n"
            f"  {obj.category} at distance={obj.distance}m from ego, "
            f"velocity [vx={obj.velocity[0]:.2f}, vy={obj.velocity[1]:.2f}] m/s\n"
            f"  >> TTC = {ttc_str}, Risk = {obj.risk_level} | "
            f"\u0394p_x={obj.delta_p_x:.2f}m, \u0394p_y={obj.delta_p_y:.2f}m, "
            f"closing_speed={obj.closing_speed:.2f}m/s"
        )
        entries.append(entry)

    footer = (
        "-----\n"
        "=====\n"
        "CONSTRAINTS:\n"
        "- TTC and Risk Level are pre-computed — use them directly\n"
        "- Velocity [vx=0.00, vy=0.00] indicates a stationary object\n"
        "====="
    )

    return "\n".join([header] + entries + [footer])


# -------------------------------------------------------------------
# 2. USER PROMPT — Image Interleaving Block
# -------------------------------------------------------------------

def format_image_interleave() -> str:
    """Build the interleaved image labels with <image> placeholders."""
    parts = []
    for cam_name in CAMERA_ORDER:
        img_num = CAMERA_IMAGE_NUM[cam_name]
        label = CAMERA_DISPLAY_NAMES[cam_name]
        parts.append(f"=== Image {img_num}: {label} Camera ===")
        parts.append("<image>")
    return "\n".join(parts)


# -------------------------------------------------------------------
# 2. USER PROMPT — Full assembly
# -------------------------------------------------------------------

def build_user_prompt(
    ego: EgoState,
    objects: List[DetectedObject],
    question: str,
    max_distance: float = 50.0,
) -> str:
    """
    Assemble the complete user prompt.

    Structure:
      1. Interleaved camera images with labels
      2. Ego vehicle driving status
      3. 3D detected object list with TTC
      4. Task question
    """
    images_block = format_image_interleave()
    ego_block = format_ego_state(ego)
    objects_block = format_object_list(objects, max_distance)

    return (
        f"{images_block}\n"
        f"\n"
        f"Given the six surround-view images above and the structured sensor data below, "
        f"answer the following task.\n"
        f"\n"
        f"{ego_block}\n"
        f"\n"
        f"{objects_block}\n"
        f"\n"
        f"TASK:\n"
        f"{question}"
    )


# ===================================================================
# PUBLIC API
# ===================================================================

def build_sft_prompts(
    sample: NuScenesSample,
    loader: NuScenesDataLoader,
    question: str,
    max_distance: float = 50.0,
    rear_filter: float = 20.0,
    resize_factor: int = 2,
) -> Tuple[str, str]:
    """
    Top-level API: build (system_prompt, user_prompt) for one QA pair.

    Args:
        sample: NuScenesSample loaded via the data loader
        loader: NuScenesDataLoader (needed for adjacent-frame velocity)
        question: The driving QA question text
        max_distance: Max object distance in meters
        rear_filter: Max distance for non-vehicle objects behind ego
        resize_factor: Image resize factor for 2D bbox projection

    Returns:
        (system_prompt, user_prompt) tuple of strings
    """
    ego = extract_ego_state(sample, loader)
    objects = extract_objects(sample, ego, max_distance, rear_filter, resize_factor)
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(ego, objects, question, max_distance)
    return system_prompt, user_prompt


def build_sft_conversations(
    sample: NuScenesSample,
    loader: NuScenesDataLoader,
    question: str,
    answer: str,
    reasoning: str = None,
    answer_type: str = None,
    max_distance: float = 50.0,
    rear_filter: float = 20.0,
    resize_factor: int = 2,
) -> List[Dict]:
    """
    Build a complete conversation in Qwen-VL SFT format.

    Returns list of conversation turns:
        [
            {"from": "system", "value": "..."},
            {"from": "human",  "value": "... <image> ... data ... question"},
            {"from": "gpt",    "value": "... answer + reasoning"},
        ]
    """
    system_prompt, user_prompt = build_sft_prompts(
        sample, loader, question, max_distance, rear_filter, resize_factor,
    )

    if answer_type == "y_or_n":
        # "Yes. <reasoning>" or "No. <reasoning>"
        ans_cap = answer.strip().capitalize()
        gpt_value = f"{ans_cap}. {reasoning}" if reasoning else ans_cap
    elif answer_type == "num_count":
        # "15. <reasoning>"
        gpt_value = f"{answer}. {reasoning}" if reasoning else str(answer)
    elif answer_type == "list":
        # Flatten list answer to a readable string, then append reasoning
        if isinstance(answer, list):
            items = ", ".join(str(a) for a in answer)
        else:
            items = str(answer)
        gpt_value = f"{items}. {reasoning}" if reasoning else items
    else:
        gpt_value = f"{answer}. {reasoning}" if reasoning else str(answer)

    return [
        {"from": "system", "value": system_prompt},
        {"from": "human",  "value": user_prompt},
        {"from": "gpt",    "value": gpt_value},
    ]


# ===================================================================
# NO-OBJECT-LIST VARIANT (for OBJ-removed training data)
# ===================================================================

def build_system_prompt_no_objects() -> str:
    """
    System prompt for the no-object-list variant.
    The model must rely on visual perception from images + ego telemetry only.
    """
    role_block = (
        "You are an autonomous driving analysis agent with expertise in "
        "visual scene understanding, spatial reasoning, and real-time risk assessment.\n"
        "\n"
        "Your task is to interpret six surround-view camera images captured simultaneously "
        "from an ego vehicle, together with the ego vehicle's driving telemetry, "
        "in order to answer driving-related questions accurately.\n"
        "\n"
        "You must identify and reason about objects, hazards, and scene conditions "
        "directly from the camera images — no pre-detected object list is provided."
    )

    camera_block_lines = [
        "MULTI-VIEW CAMERA CONFIGURATION (Egocentric order — left to right):",
    ]
    view_descriptions = {
        1: ("Front-left",  "Covers the left-front quadrant; may include the left side mirror area."),
        2: ("Front",       "Covers the forward driving direction; shows road ahead, traffic signals, and pedestrians."),
        3: ("Front-right", "Covers the right-front quadrant; may include the right side mirror area."),
        4: ("Rear-left",   "Covers the left-rear quadrant (horizontally flipped to preserve egocentric consistency)."),
        5: ("Rear",        "Covers the straight-behind view (horizontally flipped to preserve egocentric consistency)."),
        6: ("Rear-right",  "Covers the right-rear quadrant (horizontally flipped to preserve egocentric consistency)."),
    }
    for num, (name, desc) in view_descriptions.items():
        camera_block_lines.append(f"  Image {num} ({name}): {desc}")
    camera_block = "\n".join(camera_block_lines)

    reasoning_block = (
        "VISUAL REASONING:\n"
        "When analyzing the scene you must reason over:\n"
        "  (a) Visual content from each camera image — identify objects, their types,\n"
        "      positions, and motion states directly from the images\n"
        "  (b) Ego vehicle telemetry — driving command, velocity, and acceleration\n"
        "\n"
        "For every object you reference, specify:\n"
        "  - Which Image number(s) it appears in\n"
        "  - Its approximate spatial relationship to the ego vehicle\n"
        "  - Its apparent motion state\n"
        "\n"
        "IMAGE-DIRECTION MAPPING (use this to verify visual references):\n"
        "  Image 1 → Front-left | Image 2 → Front | Image 3 → Front-right\n"
        "  Image 4 → Rear-left  | Image 5 → Rear  | Image 6 → Rear-right\n"
        "Do NOT state a direction that conflicts with the Image number."
    )

    return f"{role_block}\n\n{camera_block}\n\n{reasoning_block}\n\nRespond in English."


def format_mcq_options(mcq_options: Optional[Dict[str, str]]) -> str:
    """Render an MCQ options dict as a labeled list suitable for the TASK block.

    Returns an empty string if no options are provided. Output looks like:

        Options:
        (A) vehicles
        (B) pedestrians
        ...
    """
    if not mcq_options:
        return ""
    lines = ["Options:"]
    for letter in sorted(mcq_options.keys()):
        lines.append(f"({letter}) {mcq_options[letter]}")
    return "\n".join(lines)


def build_user_prompt_no_objects(
    ego: EgoState,
    question: str,
    mcq_options: Optional[Dict[str, str]] = None,
) -> str:
    """
    User prompt without 3D object list — images + ego status + question only.
    For MCQ questions, the labeled options are appended under the TASK block
    so the model sees what each letter (A)-(E) refers to.
    """
    images_block = format_image_interleave()
    ego_block = format_ego_state(ego)
    options_block = format_mcq_options(mcq_options)
    task_block = f"TASK:\n{question}"
    if options_block:
        task_block = f"{task_block}\n\n{options_block}"

    return (
        f"{images_block}\n"
        f"\n"
        f"Given the six surround-view images above and the ego vehicle status below, "
        f"answer the following task.\n"
        f"\n"
        f"{ego_block}\n"
        f"\n"
        f"{task_block}"
    )


def build_sft_prompts_no_objects(
    sample: NuScenesSample,
    loader: NuScenesDataLoader,
    question: str,
    mcq_options: Optional[Dict[str, str]] = None,
) -> Tuple[str, str]:
    """
    Build (system_prompt, user_prompt) WITHOUT object list.
    Only images + ego telemetry are provided.
    """
    ego = extract_ego_state(sample, loader)
    system_prompt = build_system_prompt_no_objects()
    user_prompt = build_user_prompt_no_objects(ego, question, mcq_options=mcq_options)
    return system_prompt, user_prompt


def build_sft_conversations_no_objects(
    sample: NuScenesSample,
    loader: NuScenesDataLoader,
    question: str,
    answer: str,
    reasoning: str = None,
    answer_type: str = None,
    mcq_options: Optional[Dict[str, str]] = None,
) -> List[Dict]:
    """
    Build SFT conversation WITHOUT object list in the prompt.
    Used with OBJ-removed training data.
    """
    system_prompt, user_prompt = build_sft_prompts_no_objects(
        sample, loader, question, mcq_options=mcq_options,
    )

    if answer_type == "y_or_n":
        ans_cap = answer.strip().capitalize()
        gpt_value = f"{ans_cap}. {reasoning}" if reasoning else ans_cap
    elif answer_type == "num_count":
        gpt_value = f"{answer}. {reasoning}" if reasoning else str(answer)
    elif answer_type == "list":
        if isinstance(answer, list):
            items = ", ".join(str(a) for a in answer)
        else:
            items = str(answer)
        gpt_value = f"{items}. {reasoning}" if reasoning else items
    else:
        gpt_value = f"{answer}. {reasoning}" if reasoning else str(answer)

    return [
        {"from": "system", "value": system_prompt},
        {"from": "human",  "value": user_prompt},
        {"from": "gpt",    "value": gpt_value},
    ]
