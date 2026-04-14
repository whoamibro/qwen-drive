import json
import numpy as np
from typing import Dict, List, Optional
from nuscenes_pipeline.core.nuscenes_data_loader import (
    NuScenesDataLoader,
    quaternion_to_rotation_matrix,
    compute_sensor2global_transform,
    check_bbox_in_camera,
    project_points_to_camera,
    get_bbox_corners_3d
)
import os


def quaternion_to_euler(q: List[float]) -> tuple:
    """
    Convert quaternion [w, x, y, z] to Euler angles (roll, pitch, yaw) in degrees.

    Coordinate conventions:
    - Roll: rotation around X-axis (forward)
    - Pitch: rotation around Y-axis (left)
    - Yaw: rotation around Z-axis (up)

    Args:
        q: Quaternion in [w, x, y, z] format

    Returns:
        Tuple of (roll, pitch, yaw) in degrees
    """
    w, x, y, z = q[0], q[1], q[2], q[3]

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    # Convert to degrees
    roll_deg = np.degrees(roll)
    pitch_deg = np.degrees(pitch)
    yaw_deg = np.degrees(yaw)

    return roll_deg, pitch_deg, yaw_deg


def format_camera_extrinsics_for_prompt(camera_data, cam_name: str) -> str:
    """
    Format camera extrinsic parameters for the VQA prompt.

    Args:
        camera_data: CameraData object
        cam_name: Camera name (e.g., 'CAM_FRONT')

    Returns:
        Formatted string describing the camera extrinsics
    """
    # Get translation (position relative to ego vehicle center)
    trans = camera_data.sensor2ego_translation

    # Convert quaternion to Euler angles
    roll, pitch, yaw = quaternion_to_euler(camera_data.sensor2ego_rotation)

    # Map camera name to short description
    cam_descriptions = {
        'CAM_FRONT': 'Front (F)',
        'CAM_FRONT_LEFT': 'Front-left (FL)',
        'CAM_FRONT_RIGHT': 'Front-right (FR)',
        'CAM_BACK': 'Rear (R)',
        'CAM_BACK_LEFT': 'Rear-left (RL)',
        'CAM_BACK_RIGHT': 'Rear-right (RR)'
    }

    cam_desc = cam_descriptions.get(cam_name, cam_name)

    # Format position with 2 decimal places
    pos_str = f"[{trans[0]:.2f}, {trans[1]:.2f}, {trans[2]:.2f}] m"

    # Format rotation angles
    rot_str = f"Roll = {roll:.1f}°, Pitch = {pitch:.1f}°, Yaw = {yaw:.1f}°"

    return f"{cam_desc}:\n   - Position (X, Y, Z in FLU): {pos_str}\n   - Rotation: {rot_str}"


def transform_bbox_to_global(bbox: np.ndarray, ego2global_rotation: List[float],
                            ego2global_translation: List[float]) -> np.ndarray:
    """
    Transform a 3D bounding box from ego-relative FLU to global ENU coordinates.

    Args:
        bbox: Bounding box in ego frame [x, y, z, l, w, h, yaw]
        ego2global_rotation: Quaternion [w, x, y, z] from ego to global
        ego2global_translation: Translation [x, y, z] from ego to global

    Returns:
        Transformed bbox in global frame [x, y, z, l, w, h, yaw]
    """
    from nuscenes_pipeline.core.nuscenes_data_loader import quaternion_to_rotation_matrix

    # Extract position and yaw
    pos_ego = np.array([bbox[0], bbox[1], bbox[2]])
    yaw_ego = bbox[6]

    # Transform position to global
    R_ego2global = quaternion_to_rotation_matrix(ego2global_rotation)
    pos_global = R_ego2global @ pos_ego + np.array(ego2global_translation)

    # Transform yaw to global
    # Yaw is rotation around Z-axis, extract yaw from ego2global rotation
    # For a rotation matrix, yaw (rotation around z) can be extracted using atan2
    yaw_ego2global = np.arctan2(R_ego2global[1, 0], R_ego2global[0, 0])
    yaw_global = yaw_ego + yaw_ego2global

    # Normalize yaw to [-pi, pi]
    yaw_global = np.arctan2(np.sin(yaw_global), np.cos(yaw_global))

    # Return transformed bbox with same dimensions
    return np.array([pos_global[0], pos_global[1], pos_global[2],
                    bbox[3], bbox[4], bbox[5], yaw_global])


def transform_velocity_to_global(velocity: np.ndarray, ego2global_rotation: List[float]) -> np.ndarray:
    """
    Transform velocity from ego-relative FLU to global ENU coordinates.

    VERIFIED: In nuScenes dataset (this pkl file), surrounding object velocities (gt_velocity)
    are stored in the EGO frame (FLU: Forward-Left-Up). This was verified by checking that
    velocity angles align with object yaw angles in ego frame, not global frame.

    This function transforms velocities FROM ego TO global.

    Args:
        velocity: Velocity in ego frame [vx, vy] (FLU)
        ego2global_rotation: Quaternion [w, x, y, z] from ego to global

    Returns:
        Transformed velocity in global frame [vx, vy] (ENU)
    """
    from nuscenes_pipeline.core.nuscenes_data_loader import quaternion_to_rotation_matrix

    # Transform velocity vector (only 2D horizontal component)
    # Apply forward rotation: ego to global
    vel_ego_3d = np.array([velocity[0], velocity[1], 0.0])
    R_ego2global = quaternion_to_rotation_matrix(ego2global_rotation)
    vel_global_3d = R_ego2global @ vel_ego_3d

    return np.array([vel_global_3d[0], vel_global_3d[1]])


def transform_velocity_to_ego(velocity: np.ndarray, ego2global_rotation: List[float]) -> np.ndarray:
    """
    Transform velocity from global ENU to ego-relative FLU coordinates.

    This function transforms velocities FROM global TO ego.
    Used for ego vehicle velocity which is computed from global position differences.

    Args:
        velocity: Velocity in global frame [vx, vy] (ENU)
        ego2global_rotation: Quaternion [w, x, y, z] from ego to global

    Returns:
        Transformed velocity in ego frame [vx, vy] (FLU)
    """
    from nuscenes_pipeline.core.nuscenes_data_loader import quaternion_to_rotation_matrix

    # Transform velocity vector (only 2D horizontal component)
    # Use inverse rotation: global to ego is the transpose of ego to global
    vel_global_3d = np.array([velocity[0], velocity[1], 0.0])
    R_ego2global = quaternion_to_rotation_matrix(ego2global_rotation)
    # R_global2ego = R_ego2global^T (transpose for rotation matrices)
    vel_ego_3d = R_ego2global.T @ vel_global_3d

    return np.array([vel_ego_3d[0], vel_ego_3d[1]])


def get_bbox_2d_projection(bbox: np.ndarray, camera_data, resize_factor: int = 1,
                            original_width: int = 1600, original_height: int = 900) -> Optional[tuple]:
    """
    Project a 3D bounding box to 2D image plane and return the 2D bounding box.

    Args:
        bbox: 3D bounding box [x, y, z, l, w, h, yaw] in ego frame
        camera_data: CameraData object
        resize_factor: Image resize factor (images are 1/resize_factor of original)
        original_width: Original image width in pixels (nuScenes default: 1600)
        original_height: Original image height in pixels (nuScenes default: 900)

    Returns:
        Tuple (x_min, y_min, x_max, y_max) in resized image coordinates, or None if not visible
    """
    # Get 8 corners of the 3D bbox
    corners_3d = get_bbox_corners_3d(bbox)  # Shape: (8, 3)

    # Project all corners to camera
    image_points, depths = project_points_to_camera(corners_3d, camera_data)

    # Filter corners that are in front of the camera (positive depth)
    valid_mask = depths > 0.1
    if not np.any(valid_mask):
        return None

    valid_points = image_points[valid_mask]

    # Clip points to image bounds and find 2D bbox
    valid_points[:, 0] = np.clip(valid_points[:, 0], 0, original_width - 1)
    valid_points[:, 1] = np.clip(valid_points[:, 1], 0, original_height - 1)

    x_min_orig = np.min(valid_points[:, 0])
    y_min_orig = np.min(valid_points[:, 1])
    x_max_orig = np.max(valid_points[:, 0])
    y_max_orig = np.max(valid_points[:, 1])

    # Check if bbox has reasonable size (not completely outside image)
    if x_max_orig <= 0 or y_max_orig <= 0 or x_min_orig >= original_width or y_min_orig >= original_height:
        return None

    # Scale to resized image coordinates
    x_min = int(x_min_orig / resize_factor)
    y_min = int(y_min_orig / resize_factor)
    x_max = int(x_max_orig / resize_factor)
    y_max = int(y_max_orig / resize_factor)

    return (x_min, y_min, x_max, y_max)


def obb_overlap_sat(c1: np.ndarray, yaw1: float, hx1: float, hy1: float,
                     c2: np.ndarray, yaw2: float, hx2: float, hy2: float) -> bool:
    """
    Check if two Oriented Bounding Boxes (OBB) overlap using Separating Axis Theorem (SAT).

    WHY OBB vs OBB with SAT:
    - AABB (Axis-Aligned Bounding Box) ignores object orientation, leading to incorrect collision detection
    - OBB properly accounts for the actual shape and orientation of both vehicles
    - SAT provides exact collision detection for convex polygons

    Args:
        c1: Center of OBB1 (x, y) in global coordinates
        yaw1: Yaw angle of OBB1 in global frame (radians)
        hx1: Half-length of OBB1 along its local x-axis (longitudinal)
        hy1: Half-width of OBB1 along its local y-axis (lateral)
        c2: Center of OBB2 (x, y) in global coordinates
        yaw2: Yaw angle of OBB2 in global frame (radians)
        hx2: Half-length of OBB2 along its local x-axis
        hy2: Half-width of OBB2 along its local y-axis

    Returns:
        True if OBBs overlap, False otherwise
    """
    # Local axes for OBB1
    cos1, sin1 = np.cos(yaw1), np.sin(yaw1)
    a0 = np.array([cos1, sin1])    # OBB1 local x-axis (forward)
    a1 = np.array([-sin1, cos1])   # OBB1 local y-axis (left)

    # Local axes for OBB2
    cos2, sin2 = np.cos(yaw2), np.sin(yaw2)
    b0 = np.array([cos2, sin2])    # OBB2 local x-axis (forward)
    b1 = np.array([-sin2, cos2])   # OBB2 local y-axis (left)

    # Vector from center1 to center2
    d = c2 - c1

    # Test all 4 separating axes (2 from each OBB)
    axes = [a0, a1, b0, b1]

    for u in axes:
        # Project centers onto axis
        proj_d = abs(np.dot(d, u))

        # Compute projected half-extents for each OBB
        # r = hx * |dot(local_x, u)| + hy * |dot(local_y, u)|
        r1 = hx1 * abs(np.dot(a0, u)) + hy1 * abs(np.dot(a1, u))
        r2 = hx2 * abs(np.dot(b0, u)) + hy2 * abs(np.dot(b1, u))

        # If projection of d exceeds sum of radii, no overlap
        if proj_d > r1 + r2:
            return False

    # All axes show overlap -> collision
    return True


def compute_ttc_obb_global(
    p_ego_ref: np.ndarray,
    v_ego_global: np.ndarray,
    yaw_ego: float,
    p_obj_global: np.ndarray,
    v_obj_global: np.ndarray,
    yaw_obj_global: float,
    L_obj: float,
    W_obj: float,
    T_max: float = 10.0,
    dt: float = 0.05,
    eps: float = 1e-3,
    a_ego_global: np.ndarray = None,
    a_obj_global: np.ndarray = None
) -> Dict:
    """
    Compute Time-To-Collision (TTC) using OBB collision detection in GLOBAL coordinates.

    Motion model: p(t) = p_0 + v*t + 0.5*a*t²
    When acceleration is provided, uses quadratic motion model to reduce false positives
    from braking scenarios. Falls back to constant-velocity (linear) when acceleration is None.

    ==================== CRITICAL REQUIREMENTS ====================

    WHY GLOBAL COORDINATES ARE REQUIRED:
    - TTC computation requires knowing the ACTUAL positions of both vehicles in space
    - Distance-only TTC (e.g., "25m ahead") is INVALID because it ignores lateral position and orientation
    - Ego-centric assumptions fail when objects have different headings

    WHY BOTH EGO AND OBJECT MOTION MUST BE CONSIDERED:
    - TTC depends on RELATIVE motion between ego and object
    - Ignoring object velocity leads to incorrect TTC (object may be moving away)
    - Ignoring ego velocity leads to incorrect TTC (ego may be approaching faster)

    WHY EGO REFERENCE POINT CONVERSION IS MANDATORY:
    - Ego pose is given at a REFERENCE POINT, NOT the geometric center
    - The reference point is 4.0m behind the front face of the ego
    - Collision detection must use the actual geometric center of the ego OBB
    - Skipping this conversion causes TTC errors up to several seconds

    ================================================================

    Args:
        p_ego_ref: Ego REFERENCE POINT position (x, y) in GLOBAL coordinates
                   This is 4.0m behind the ego's front face, NOT the geometric center
        v_ego_global: Ego velocity (vx, vy) in GLOBAL coordinates (m/s)
        yaw_ego: Ego yaw angle in GLOBAL frame (radians)
        p_obj_global: Object center position (x, y) in GLOBAL coordinates
                      This is the object's GEOMETRIC CENTER
        v_obj_global: Object velocity (vx, vy) in GLOBAL coordinates (m/s)
        yaw_obj_global: Object yaw angle in GLOBAL frame (radians)
        L_obj: Object length (m)
        W_obj: Object width (m)
        T_max: Maximum time horizon to search (seconds)
        dt: Time step for initial search (seconds)
        eps: Precision for bisection (seconds)
        a_ego_global: Ego acceleration (ax, ay) in GLOBAL coordinates (m/s²), or None
        a_obj_global: Object acceleration (ax, ay) in GLOBAL coordinates (m/s²), or None

    Returns:
        Dictionary containing:
        - ttc: Final TTC value (float or np.inf)
        - risk_level: "CRITICAL" / "HIGH" / "MODERATE" / "LOW" / "NONE"
        - intermediate: Dict with all intermediate computation values
        - reason: Human-readable explanation string
    """
    # ==================== CONSTANTS ====================
    L_EGO = 4.7  # Ego vehicle length (m)
    W_EGO = 1.9  # Ego vehicle width (m)
    REF_TO_FRONT = 4.0  # Distance from reference point to ego front face (m)

    # ==================== STEP 1: EGO REFERENCE POINT → GEOMETRIC CENTER ====================
    #
    # CRITICAL: The ego pose (p_ego_ref) is at a reference point that is:
    #   - On the longitudinal centerline of the ego vehicle
    #   - 4.0 meters behind the FRONT FACE of the ego
    #
    # To get the geometric center:
    #   1. Compute forward unit vector: f = (cos(yaw), sin(yaw))
    #   2. Front face position: p_front = p_ref + REF_TO_FRONT * f
    #   3. Geometric center: p_center = p_front - (L_EGO / 2) * f
    #
    # Equivalently: p_center = p_ref + (REF_TO_FRONT - L_EGO / 2) * f

    f_ego = np.array([np.cos(yaw_ego), np.sin(yaw_ego)])  # Ego forward unit vector
    ego_center_offset = REF_TO_FRONT - L_EGO / 2  # = 4.0 - 2.35 = 1.65m ahead of reference
    p_ego_center = p_ego_ref + ego_center_offset * f_ego

    # ==================== STEP 2: RELATIVE MOTION FORMULATION ====================
    #
    # With acceleration: p(t) = p_0 + v*t + 0.5*a*t²
    # Relative position at time t: p_rel(t) = p_obj(t) - p_ego(t)
    #   = (p_obj_0 + v_obj*t + 0.5*a_obj*t²) - (p_ego_0 + v_ego*t + 0.5*a_ego*t²)
    #   = (p_obj_0 - p_ego_0) + (v_obj - v_ego)*t + 0.5*(a_obj - a_ego)*t²
    #   = p_rel_0 + v_rel*t + 0.5*a_rel*t²
    #
    # We fix ego at origin and move object with relative velocity and acceleration

    v_rel = v_obj_global - v_ego_global
    p_rel_0 = p_obj_global - p_ego_center  # Initial relative position

    # Compute relative acceleration (default to zero if not provided)
    a_ego = np.array(a_ego_global[:2], dtype=float) if a_ego_global is not None else np.array([0.0, 0.0])
    a_obj = np.array(a_obj_global[:2], dtype=float) if a_obj_global is not None else np.array([0.0, 0.0])
    a_rel = a_obj - a_ego

    # ==================== STEP 3: OBB PARAMETERS ====================

    # Ego OBB (fixed at origin in relative frame)
    ego_center_rel = np.array([0.0, 0.0])  # Ego at origin
    hx_ego = L_EGO / 2  # 2.35m
    hy_ego = W_EGO / 2  # 0.95m

    # Object OBB
    hx_obj = L_obj / 2
    hy_obj = W_obj / 2

    # ==================== STEP 4: COLLISION CHECK FUNCTION ====================

    def check_collision_at_time(t: float) -> bool:
        """Check if OBBs overlap at time t."""
        # Object position at time t (relative to ego at origin)
        # p_rel(t) = p_rel_0 + v_rel * t + 0.5 * a_rel * t²
        p_obj_t = p_rel_0 + v_rel * t + 0.5 * a_rel * t * t

        # Check OBB overlap using SAT
        return obb_overlap_sat(
            ego_center_rel, yaw_ego, hx_ego, hy_ego,
            p_obj_t, yaw_obj_global, hx_obj, hy_obj
        )

    # ==================== STEP 5: CHECK IMMEDIATE COLLISION (t=0) ====================

    if check_collision_at_time(0.0):
        ttc = 0.0
        risk_level = "CRITICAL"
        reason = "Currently overlapping. Immediate collision."

        # Compute relative position in ego frame for debug output
        cos_ego, sin_ego = np.cos(yaw_ego), np.sin(yaw_ego)
        delta_p_ego_frame_x = np.dot(p_rel_0, np.array([cos_ego, sin_ego]))
        delta_p_ego_frame_y = np.dot(p_rel_0, np.array([-sin_ego, cos_ego]))

        intermediate = {
            'p_ego_ref': p_ego_ref.tolist(),
            'p_ego_center': p_ego_center.tolist(),
            'p_obj_global': p_obj_global.tolist(),
            'v_ego_global': v_ego_global.tolist(),
            'v_obj_global': v_obj_global.tolist(),
            'v_rel': v_rel.tolist(),
            'a_ego_global': a_ego.tolist(),
            'a_rel': a_rel.tolist(),
            'yaw_ego': yaw_ego,
            'yaw_obj_global': yaw_obj_global,
            'delta_p_x': delta_p_ego_frame_x,
            'delta_p_y': delta_p_ego_frame_y,
            'delta_v_x': np.dot(v_rel, np.array([cos_ego, sin_ego])),
            'delta_v_y': np.dot(v_rel, np.array([-sin_ego, cos_ego])),
            'closing_speed': -np.dot(v_rel, p_rel_0 / (np.linalg.norm(p_rel_0) + 1e-6)),
            'extent_front': hx_ego + hx_obj,
            'extent_y': hy_ego + hy_obj,
            'ttc_x': 0.0,
            'ttc_y': 0.0
        }
        return {'ttc': ttc, 'risk_level': risk_level, 'intermediate': intermediate, 'reason': reason}

    # ==================== STEP 6: CHECK FOR RELATIVE MOTION ====================

    rel_speed = np.linalg.norm(v_rel)
    rel_accel = np.linalg.norm(a_rel)
    if rel_speed < 0.1 and rel_accel < 0.1:  # Stationary threshold (also check acceleration)
        ttc = np.inf
        risk_level = "NONE"
        reason = "No significant relative motion. TTC = ∞."

        cos_ego, sin_ego = np.cos(yaw_ego), np.sin(yaw_ego)
        delta_p_ego_frame_x = np.dot(p_rel_0, np.array([cos_ego, sin_ego]))
        delta_p_ego_frame_y = np.dot(p_rel_0, np.array([-sin_ego, cos_ego]))

        intermediate = {
            'p_ego_ref': p_ego_ref.tolist(),
            'p_ego_center': p_ego_center.tolist(),
            'p_obj_global': p_obj_global.tolist(),
            'v_ego_global': v_ego_global.tolist(),
            'v_obj_global': v_obj_global.tolist(),
            'v_rel': v_rel.tolist(),
            'a_ego_global': a_ego.tolist(),
            'a_rel': a_rel.tolist(),
            'yaw_ego': yaw_ego,
            'yaw_obj_global': yaw_obj_global,
            'delta_p_x': delta_p_ego_frame_x,
            'delta_p_y': delta_p_ego_frame_y,
            'delta_v_x': np.dot(v_rel, np.array([cos_ego, sin_ego])),
            'delta_v_y': np.dot(v_rel, np.array([-sin_ego, cos_ego])),
            'closing_speed': 0.0,
            'extent_front': hx_ego + hx_obj,
            'extent_y': hy_ego + hy_obj,
            'ttc_x': np.inf,
            'ttc_y': np.inf
        }
        return {'ttc': ttc, 'risk_level': risk_level, 'intermediate': intermediate, 'reason': reason}

    # ==================== STEP 7: TIME SEARCH WITH BISECTION ====================
    #
    # Sample t ∈ [0, T_max] with step dt to find first collision interval
    # Then use bisection to refine

    ttc = np.inf
    collision_found = False

    # Coarse search
    t_prev = 0.0
    collide_prev = False

    t = dt
    while t <= T_max:
        collide_curr = check_collision_at_time(t)

        if not collide_prev and collide_curr:
            # Transition from no-collision to collision between t_prev and t
            # Use bisection to find exact collision time
            t_lo, t_hi = t_prev, t

            while t_hi - t_lo > eps:
                t_mid = (t_lo + t_hi) / 2
                if check_collision_at_time(t_mid):
                    t_hi = t_mid
                else:
                    t_lo = t_mid

            ttc = t_hi
            collision_found = True
            break

        t_prev = t
        collide_prev = collide_curr
        t += dt

    # ==================== STEP 8: COMPUTE ADDITIONAL INFO ====================

    # Compute relative position in ego frame for output
    cos_ego, sin_ego = np.cos(yaw_ego), np.sin(yaw_ego)
    ego_forward = np.array([cos_ego, sin_ego])
    ego_left = np.array([-sin_ego, cos_ego])

    delta_p_ego_frame_x = np.dot(p_rel_0, ego_forward)  # Forward distance
    delta_p_ego_frame_y = np.dot(p_rel_0, ego_left)     # Left distance
    delta_v_ego_frame_x = np.dot(v_rel, ego_forward)
    delta_v_ego_frame_y = np.dot(v_rel, ego_left)

    # Closing speed along line-of-sight
    dist = np.linalg.norm(p_rel_0)
    if dist > 1e-6:
        closing_speed = -np.dot(v_rel, p_rel_0 / dist)
    else:
        closing_speed = 0.0

    # ==================== STEP 9: RISK LEVEL ====================

    if ttc == 0:
        risk_level = "CRITICAL"
    elif ttc < 2:
        risk_level = "HIGH"
    elif ttc < 5:
        risk_level = "MODERATE"
    elif not np.isinf(ttc):
        risk_level = "LOW"
    else:
        risk_level = "NONE"

    # ==================== STEP 10: GENERATE REASON ====================

    x_pos = "front" if delta_p_ego_frame_x > 0 else "behind" if delta_p_ego_frame_x < 0 else "aligned"
    y_pos = "left" if delta_p_ego_frame_y > 0 else "right" if delta_p_ego_frame_y < 0 else "centered"

    if np.isinf(ttc):
        if closing_speed <= 0:
            reason = f"Object is {x_pos}-{y_pos}, diverging (closing_speed={closing_speed:.2f}m/s). TTC = ∞."
        else:
            reason = f"Object is {x_pos}-{y_pos}, approaching but paths don't intersect within {T_max}s. TTC = ∞."
    else:
        reason = f"Object is {x_pos}-{y_pos}, approaching at {closing_speed:.2f}m/s. TTC = {ttc:.2f}s (OBB collision)."

    # ==================== BUILD OUTPUT ====================

    intermediate = {
        'p_ego_ref': p_ego_ref.tolist(),
        'p_ego_center': p_ego_center.tolist(),
        'p_obj_global': p_obj_global.tolist(),
        'v_ego_global': v_ego_global.tolist(),
        'v_obj_global': v_obj_global.tolist(),
        'v_rel': v_rel.tolist(),
        'a_ego_global': a_ego.tolist(),
        'a_rel': a_rel.tolist(),
        'yaw_ego': yaw_ego,
        'yaw_obj_global': yaw_obj_global,
        'delta_p_x': delta_p_ego_frame_x,
        'delta_p_y': delta_p_ego_frame_y,
        'delta_v_x': delta_v_ego_frame_x,
        'delta_v_y': delta_v_ego_frame_y,
        'closing_speed': closing_speed,
        'extent_front': hx_ego + hx_obj,
        'extent_y': hy_ego + hy_obj,
        'ttc_x': ttc,  # Not axis-specific in OBB method
        'ttc_y': ttc
    }

    return {
        'ttc': ttc,
        'risk_level': risk_level,
        'intermediate': intermediate,
        'reason': reason
    }


# Keep old function for backward compatibility (deprecated)
def compute_ttc_for_object_ego_frame(p_obj_ego: np.ndarray, v_obj_ego: np.ndarray,
                                      L_obj: float, W_obj: float,
                                      v_ego_ego: np.ndarray, yaw_obj: float = 0.0) -> Dict:
    """
    DEPRECATED: This function uses AABB and ego-centric coordinates which is incorrect.
    Use compute_ttc_obb_global() instead with proper global coordinates.

    This function is kept for backward compatibility but should not be used.
    """
    # Return infinity TTC to indicate this method is deprecated
    return {
        'ttc': np.inf,
        'risk_level': "NONE",
        'intermediate': {
            'delta_p_x': p_obj_ego[0],
            'delta_p_y': p_obj_ego[1],
            'delta_v_x': 0.0,
            'delta_v_y': 0.0,
            'extent_front': 0.0,
            'extent_y': 0.0,
            'closing_speed': 0.0,
            'ttc_x': np.inf,
            'ttc_y': np.inf
        },
        'reason': "DEPRECATED: Use compute_ttc_obb_global() with global coordinates."
    }


def compute_ego_acceleration(sample, loader: 'NuScenesDataLoader') -> Optional[np.ndarray]:
    """
    Compute ego vehicle acceleration in global (ENU) coordinates using centered difference.

    Uses adjacent frames (prev and next) to compute acceleration:
        v_forward  = (pos_next - pos_curr) / dt_forward
        v_backward = (pos_curr - pos_prev) / dt_backward
        accel      = (v_forward - v_backward) / dt_avg

    This gives a centered difference estimate of acceleration at the current frame,
    which is more accurate than a one-sided difference.

    Args:
        sample: Current NuScenesSample object
        loader: NuScenesDataLoader instance

    Returns:
        np.ndarray of shape (2,) with [ax, ay] in global ENU (m/s²), or None if
        adjacent frames are unavailable or belong to different scenes.
    """
    try:
        idx = sample.sample_idx

        # Load previous and next samples
        prev_sample = loader.get_sample(idx - 1)
        next_sample = loader.get_sample(idx + 1)

        # Get first camera key for accessing ego pose
        cam_keys_curr = list(sample.cameras.keys())
        cam_keys_prev = list(prev_sample.cameras.keys())
        cam_keys_next = list(next_sample.cameras.keys())

        cam_curr = sample.cameras[cam_keys_curr[0]]
        cam_prev = prev_sample.cameras[cam_keys_prev[0]]
        cam_next = next_sample.cameras[cam_keys_next[0]]

        # Verify all samples are from the same scene
        if prev_sample.scene_token != sample.scene_token or next_sample.scene_token != sample.scene_token:
            return None

        # Get positions (2D, global ENU)
        pos_prev = np.array(cam_prev.ego2global_translation[:2])
        pos_curr = np.array(cam_curr.ego2global_translation[:2])
        pos_next = np.array(cam_next.ego2global_translation[:2])

        # Compute time differences (timestamps are in microseconds)
        dt_forward = (next_sample.timestamp - sample.timestamp) / 1e6
        dt_backward = (sample.timestamp - prev_sample.timestamp) / 1e6

        if dt_forward <= 0 or dt_backward <= 0:
            return None

        # Compute velocities
        v_forward = (pos_next - pos_curr) / dt_forward
        v_backward = (pos_curr - pos_prev) / dt_backward

        # Centered difference acceleration
        dt_avg = (dt_forward + dt_backward) / 2.0
        accel_global = (v_forward - v_backward) / dt_avg

        return accel_global

    except Exception:
        return None


def compute_ttc_for_object(p_obj: np.ndarray, v_obj: np.ndarray, L_obj: float, W_obj: float,
                           p_ego: np.ndarray, v_ego: np.ndarray, theta_ego: float) -> Dict:
    """
    Compute Time-To-Collision (TTC) for a single object using rectangular collision model.

    Uses axis-aligned bounding box collision detection with independent longitudinal
    and lateral axis checks.

    Args:
        p_obj: Object center position (x, y) in global ENU
        v_obj: Object velocity (vx, vy) in global ENU
        L_obj: Object length (m)
        W_obj: Object width (m)
        p_ego: Ego rear-center position (x, y) in global ENU
        v_ego: Ego velocity (vx, vy) in global ENU
        theta_ego: Ego yaw angle (radians)

    Returns:
        Dictionary containing:
        - ttc: Final TTC value (float or np.inf)
        - risk_level: "CRITICAL" / "HIGH" / "MODERATE" / "LOW" / "NONE"
        - intermediate: Dict with all intermediate computation values
        - reason: Human-readable explanation string
    """
    # Constants
    L_EGO = 4.7  # Ego vehicle length (m)
    W_EGO = 1.9  # Ego vehicle width (m)
    EGO_REAR_TO_CENTER = 2.35  # Distance from rear-center to geometric center (m)
    STATIONARY_THRESHOLD = 0.1  # Speed below this is considered stationary (m/s)

    # 1. Ego orientation vectors
    forward = np.array([np.cos(theta_ego), np.sin(theta_ego)])
    right = np.array([np.sin(theta_ego), -np.cos(theta_ego)])

    # 2. Ego geometric center
    p_ego_center = p_ego + EGO_REAR_TO_CENTER * forward

    # 3. Relative position (x=forward, y=left in FLU frame)
    delta_p = p_obj - p_ego_center
    delta_p_x = np.dot(delta_p, forward)  # + front, - behind
    delta_p_y = -np.dot(delta_p, right)   # + left, - right (FLU convention)

    # 4. Relative velocity
    delta_v = v_obj - v_ego
    delta_v_x = np.dot(delta_v, forward)
    delta_v_y = -np.dot(delta_v, right)   # + left, - right (FLU convention)

    # 5. Collision extents
    extent_x = (L_EGO / 2) + (L_obj / 2)  # = 2.35 + L_obj/2
    extent_y = (W_EGO / 2) + (W_obj / 2)  # = 0.95 + W_obj/2

    # 6. Overlap check
    overlap_x = abs(delta_p_x) < extent_x
    overlap_y = abs(delta_p_y) < extent_y

    # 7. Approach check
    # Object approaches if relative velocity reduces the gap
    # sign(delta_p) * delta_v < 0 means approaching
    if delta_p_x != 0:
        approach_x = np.sign(delta_p_x) * delta_v_x < 0
    else:
        approach_x = delta_v_x != 0  # At same position, any velocity is concerning

    if delta_p_y != 0:
        approach_y = np.sign(delta_p_y) * delta_v_y < 0
    else:
        approach_y = delta_v_y != 0

    # 8. TTC computation per axis
    # X-axis (longitudinal) TTC
    if overlap_x:
        ttc_x = 0.0
    elif approach_x and abs(delta_v_x) > 1e-6:
        ttc_x = (abs(delta_p_x) - extent_x) / abs(delta_v_x)
    else:
        ttc_x = np.inf

    # Y-axis (lateral) TTC
    if overlap_y:
        ttc_y = 0.0
    elif approach_y and abs(delta_v_y) > 1e-6:
        ttc_y = (abs(delta_p_y) - extent_y) / abs(delta_v_y)
    else:
        ttc_y = np.inf

    # 9. Combined TTC
    # Collision requires both axes to overlap, so TTC is the max (last axis to reach overlap)
    # If either is inf, no collision on that axis, so overall TTC is inf
    if np.isinf(ttc_x) or np.isinf(ttc_y):
        ttc = np.inf
    else:
        ttc = max(ttc_x, ttc_y)

    # 9.5 Compute closing speed (rate of approach along line connecting ego to object)
    distance = np.sqrt(delta_p_x**2 + delta_p_y**2)
    if distance > 1e-6:
        # Unit vector from ego to object in ego frame
        dir_to_obj = np.array([delta_p_x, delta_p_y]) / distance
        # Closing speed = negative of relative velocity component along direction to object
        # Positive closing_speed means object is approaching
        closing_speed = -(delta_v_x * dir_to_obj[0] + delta_v_y * dir_to_obj[1])
    else:
        closing_speed = 0.0

    # 10. Risk level
    if ttc == 0:
        risk_level = "CRITICAL"
    elif ttc < 2:
        risk_level = "HIGH"
    elif ttc < 5:
        risk_level = "MODERATE"
    elif not np.isinf(ttc):
        risk_level = "LOW"
    else:
        risk_level = "NONE"

    # 11. Generate reason string
    ego_speed = np.sqrt(v_ego[0]**2 + v_ego[1]**2)
    obj_speed = np.sqrt(v_obj[0]**2 + v_obj[1]**2)
    ego_stationary = ego_speed < STATIONARY_THRESHOLD
    obj_stationary = obj_speed < STATIONARY_THRESHOLD

    # Position description (x=forward, y=left in FLU frame)
    if delta_p_x > 0:
        x_pos = "front"
    elif delta_p_x < 0:
        x_pos = "behind"
    else:
        x_pos = "aligned"

    if delta_p_y > 0:
        y_pos = "left"
    elif delta_p_y < 0:
        y_pos = "right"
    else:
        y_pos = "centered"

    # Generate reason based on conditions
    if ego_stationary and obj_stationary:
        reason = "Both ego and object are stationary. No relative motion, TTC = ∞."
    elif overlap_x and overlap_y:
        reason = "Currently overlapping in both axes. Immediate collision risk."
    elif np.isinf(ttc_x) and np.isinf(ttc_y):
        if not approach_x and not approach_y:
            reason = f"Object is {x_pos}-{y_pos} and diverging on both axes. No collision risk."
        elif not approach_x:
            reason = f"Object is {x_pos} and not approaching in x-axis. TTC_x = ∞."
        else:
            reason = f"Object is to the {y_pos} and not approaching in y-axis. TTC_y = ∞."
    elif np.isinf(ttc_x):
        reason = f"Object is {x_pos} and not approaching in x-axis. TTC_x = ∞, so TTC = ∞."
    elif np.isinf(ttc_y):
        reason = f"Object is to the {y_pos} and not approaching in y-axis. TTC_y = ∞, so TTC = ∞."
    else:
        if ttc_x >= ttc_y:
            reason = f"Approaching from {x_pos}. TTC_x ({ttc_x:.2f}s) >= TTC_y ({ttc_y:.2f}s), so TTC = {ttc:.2f}s."
        else:
            reason = f"Approaching from {y_pos}. TTC_y ({ttc_y:.2f}s) > TTC_x ({ttc_x:.2f}s), so TTC = {ttc:.2f}s."

    # Build intermediate values dict (x=forward, y=left in FLU frame)
    intermediate = {
        'p_ego_center': p_ego_center,
        'forward': forward,
        'right': right,
        'delta_p': delta_p,
        'delta_p_x': delta_p_x,
        'delta_p_y': delta_p_y,
        'delta_v': delta_v,
        'delta_v_x': delta_v_x,
        'delta_v_y': delta_v_y,
        'extent_x': extent_x,
        'extent_y': extent_y,
        'overlap_x': overlap_x,
        'overlap_y': overlap_y,
        'approach_x': approach_x,
        'approach_y': approach_y,
        'ttc_x': ttc_x,
        'ttc_y': ttc_y,
        'closing_speed': closing_speed
    }

    return {
        'ttc': ttc,
        'risk_level': risk_level,
        'intermediate': intermediate,
        'reason': reason
    }


def format_ego_velocity_for_prompt(sample, loader: NuScenesDataLoader, use_global_coords: bool = False,
                                    ego_accel_global: np.ndarray = None) -> str:
    """
    Format ego vehicle velocity, acceleration, and driving command for the VQA prompt.

    Ego velocity is computed from ego2global_translation differences between
    consecutive frames: v = (pos_next - pos_current) / dt

    Args:
        sample: Current NuScenesSample object
        loader: NuScenesDataLoader instance (needed to load next frame)
        use_global_coords: If True, keep velocity in global ENU; if False, transform to ego-relative FLU
        ego_accel_global: Ego acceleration in global ENU [ax, ay] m/s², or None

    Returns:
        Formatted string describing ego vehicle velocity, acceleration, and driving command
    """
    # Driving command mapping (0-6 for gt_navigation_command)
    DRIVING_COMMAND_MAP = {
        0: "Turn left",
        1: "Turn right",
        2: "Go straight",
        3: "Follow lane",
        4: "Change lane to left",
        5: "Change lane to right",
        6: "U-Turn"
    }

    # Get driving command (use gt_navigation_command if available, fallback to gt_planning_command)
    cmd_idx = sample.gt_navigation_command if hasattr(sample, 'gt_navigation_command') and sample.gt_navigation_command is not None else sample.gt_planning_command
    driving_command = DRIVING_COMMAND_MAP.get(cmd_idx, f"Unknown command ({cmd_idx})")

    # Get ego global pose from first camera
    first_cam_current = sample.cameras[list(sample.cameras.keys())[0]]
    ego_pos = first_cam_current.ego2global_translation  # [x, y, z] in global frame
    ego_quat = first_cam_current.ego2global_rotation  # [qw, qx, qy, qz]

    # Compute yaw from quaternion
    qw, qx, qy, qz = ego_quat[0], ego_quat[1], ego_quat[2], ego_quat[3]
    ego_yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))

    # Try to get next frame to calculate velocity
    try:
        # Get current and next samples
        next_sample = loader.get_sample(sample.sample_idx + 1)

        # Get ego positions (ego2global_translation) from first camera
        first_cam_next = next_sample.cameras[list(next_sample.cameras.keys())[0]]

        pos_current = np.array(ego_pos)  # [x, y, z] in global frame
        pos_next = np.array(first_cam_next.ego2global_translation)

        # Calculate time difference (convert from microseconds to seconds)
        dt = (next_sample.timestamp - sample.timestamp) / 1e6

        if dt == 0:
            raise ValueError("Zero time difference between frames")

        # Calculate velocity in global frame
        dx = pos_next[0] - pos_current[0]  # x difference
        dy = pos_next[1] - pos_current[1]  # y difference

        ego_velocity_global = np.array([dx / dt, dy / dt])  # [vx, vy] in global frame

        # Transform based on coordinate system
        ego2global_rot = first_cam_current.ego2global_rotation

        if use_global_coords:
            # Keep as-is (already in global frame)
            velocity_transformed = ego_velocity_global
            coord_frame = "global ENU"
            # Include global pose
            pose_str = f"  - Position (global ENU): [x={ego_pos[0]:.2f}, y={ego_pos[1]:.2f}, z={ego_pos[2]:.2f}] m\n  - Yaw (global): {ego_yaw:.4f} rad ({np.degrees(ego_yaw):.2f}°)"
            # Also compute ego-relative FLU velocity
            velocity_ego_flu = transform_velocity_to_ego(ego_velocity_global, ego2global_rot)
            vel_str_flu = f"[vx={velocity_ego_flu[0]:.2f}, vy={velocity_ego_flu[1]:.2f}] m/s"
        else:
            # Transform FROM global TO ego
            velocity_transformed = transform_velocity_to_ego(ego_velocity_global, ego2global_rot)
            coord_frame = "ego-relative FLU"
            pose_str = ""

        # Calculate speed (magnitude)
        speed = np.sqrt(velocity_transformed[0]**2 + velocity_transformed[1]**2)

        # Format velocity string
        vel_str = f"[vx={velocity_transformed[0]:.2f}, vy={velocity_transformed[1]:.2f}] m/s"

        # Velocity direction explanation for FLU
        vel_explanation = "  - [FLU Velocity Guide: +vx=forward, -vx=backward, +vy=left, -vy=right]"

        # Format acceleration string if available
        accel_str = ""
        if ego_accel_global is not None:
            if use_global_coords:
                accel_str = f"\n  - Acceleration (global ENU): [ax={ego_accel_global[0]:.2f}, ay={ego_accel_global[1]:.2f}] m/s²"
            else:
                # Transform acceleration to ego frame
                accel_ego = transform_velocity_to_ego(ego_accel_global, ego2global_rot)
                accel_str = f"\n  - Acceleration (ego-relative FLU): [ax={accel_ego[0]:.2f}, ay={accel_ego[1]:.2f}] m/s²"

        if use_global_coords:
            return f"** EGO VEHICLE's DRIVING STATUS **\n  - Driving Command: {driving_command}\n{pose_str}\n  - Velocity ({coord_frame}): {vel_str}\n  - Velocity (ego-relative FLU): {vel_str_flu}\n  - Speed: {speed:.2f} m/s{accel_str}\n{vel_explanation}"
        else:
            return f"** EGO VEHICLE's DRIVING STATUS **\n  - Driving Command: {driving_command}\n  - Velocity ({coord_frame}): {vel_str}\n  - Speed: {speed:.2f} m/s{accel_str}\n{vel_explanation}"

    except Exception as e:
        # If we can't get next frame or calculate velocity, return without velocity info
        if use_global_coords:
            pose_str = f"  - Position (global ENU): [x={ego_pos[0]:.2f}, y={ego_pos[1]:.2f}, z={ego_pos[2]:.2f}] m\n  - Yaw (global): {ego_yaw:.4f} rad ({np.degrees(ego_yaw):.2f}°)"
            return f"** EGO VEHICLE's DRIVING STATUS **\n  - Driving Command: {driving_command}\n{pose_str}\n  - Velocity: Unable to compute (next frame not available)"
        else:
            return f"** EGO VEHICLE's DRIVING STATUS **\n  - Driving Command: {driving_command}\n  - Velocity: Unable to compute (next frame not available)"


def format_3d_objects_for_prompt(sample, loader: NuScenesDataLoader, max_distance: float = 20.0,
                                 use_global_coords: bool = False,
                                 ego_vel_global: np.ndarray = None, ego_yaw: float = None,
                                 proj2img: bool = False, resize_factor: int = 1,
                                 rear_filter_distance: float = None,
                                 ego_accel_global: np.ndarray = None) -> str:
    """
    Format 3D object ground truth bounding boxes and velocities for the VQA prompt.

    Args:
        sample: NuScenesSample object
        loader: NuScenesDataLoader instance
        max_distance: Maximum distance (in meters) from ego vehicle to include objects. Default 20m.
        use_global_coords: If True, transform bbox and velocity to global ENU; if False, use ego-relative FLU
        ego_vel_global: Ego velocity in global frame (required for TTC computation when use_global_coords=True)
        ego_yaw: Ego yaw angle in radians (required for TTC computation when use_global_coords=True)
        proj2img: If True, include projected pixel coordinates for each visible camera
        resize_factor: Image resize factor (images are 1/resize_factor of original)
        rear_filter_distance: Maximum distance for non-vehicle objects behind ego. If None, uses max_distance for all.
        ego_accel_global: Ego acceleration in global ENU [ax, ay] m/s², or None

    Returns:
        Formatted string describing 3D objects with their locations and velocities
    """
    if len(sample.gt_boxes) == 0:
        return "No 3D objects detected in this scene."

    # Camera name mapping for display
    cam_name_map = {
        'CAM_FRONT': 'Front',
        'CAM_FRONT_LEFT': 'Front-left',
        'CAM_FRONT_RIGHT': 'Front-right',
        'CAM_BACK': 'Rear',
        'CAM_BACK_LEFT': 'Rear-left',
        'CAM_BACK_RIGHT': 'Rear-right'
    }

    # Image number mapping (1-indexed for prompt)
    # Egocentric order: FL, F, FR, RL, R, RR
    cam_to_image_num = {
        'CAM_FRONT_LEFT': 1,
        'CAM_FRONT': 2,
        'CAM_FRONT_RIGHT': 3,
        'CAM_BACK_LEFT': 4,
        'CAM_BACK': 5,
        'CAM_BACK_RIGHT': 6
    }

    # Define vehicle types (these are NOT filtered by rear_filter_distance)
    VEHICLE_TYPES = {'car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'motorcycle', 'bicycle'}

    # Define rear cameras for rear_filter logic
    REAR_CAMERAS = {'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'}

    # Map each object to cameras where it's visible, with distance filtering
    object_camera_map = {}  # obj_idx -> (list of (cam_name, image_num), distance)

    for obj_idx in range(len(sample.gt_boxes)):
        bbox = sample.gt_boxes[obj_idx]
        obj_name = sample.gt_names[obj_idx] if sample.gt_names is not None else "object"

        # Calculate distance from ego vehicle (at origin in ego-relative coords)
        # Use 2D horizontal distance (x, y) as is standard in autonomous driving
        obj_x, obj_y = bbox[0], bbox[1]
        distance = np.sqrt(obj_x**2 + obj_y**2)

        # Filter by distance
        if distance > max_distance:
            continue

        # === ORIGINAL rear_filter logic (x < 0 based) - ACTIVE ===
        # Apply rear filter for non-vehicle objects (x < 0 means behind ego in FLU frame)
        if rear_filter_distance is not None and obj_x < 0:
            # Check if this is a non-vehicle object
            if obj_name.lower() not in VEHICLE_TYPES:
                # Filter by rear_filter_distance for non-vehicles behind ego
                if distance > rear_filter_distance:
                    continue

        # === NEW rear_filter logic (visibility-based) - COMMENTED OUT ===
        # # First, determine which cameras can see this object (needed for rear_filter)
        # visible_cameras_check = []
        # for cam_name in loader.CAMERA_NAMES:
        #     camera_data = sample.cameras[cam_name]
        #     if check_bbox_in_camera(bbox, camera_data):
        #         visible_cameras_check.append(cam_name)
        #
        # # Apply rear filter for non-vehicle objects visible in ANY rear camera
        # if rear_filter_distance is not None:
        #     # Check if this is a non-vehicle object
        #     if obj_name.lower() not in VEHICLE_TYPES:
        #         # Check if object is visible in ANY rear camera
        #         if len(visible_cameras_check) > 0:
        #             visible_in_any_rear = any(cam in REAR_CAMERAS for cam in visible_cameras_check)
        #             if visible_in_any_rear and distance > rear_filter_distance:
        #                 continue

        visible_cameras = []

        for cam_name in loader.CAMERA_NAMES:
            camera_data = sample.cameras[cam_name]
            if check_bbox_in_camera(bbox, camera_data):
                visible_cameras.append((cam_name, cam_to_image_num[cam_name]))

        if visible_cameras:
            object_camera_map[obj_idx] = (visible_cameras, distance)

    # Get ego2global transformation (needed for both ego and global coord modes)
    # Use first camera's ego2global transformation (same for all cameras)
    first_cam = sample.cameras[loader.CAMERA_NAMES[0]]
    ego2global_rot = first_cam.ego2global_rotation
    ego2global_trans = first_cam.ego2global_translation

    # Format the information
    object_descriptions = []

    for obj_idx, (visible_cameras, distance) in object_camera_map.items():
        bbox = sample.gt_boxes[obj_idx]
        velocity = sample.gt_velocity[obj_idx]
        obj_name = sample.gt_names[obj_idx] if sample.gt_names is not None else "object"

        # VERIFIED: In this nuScenes pkl file, both bbox AND velocity are in EGO frame (FLU).
        # This was confirmed by checking that velocity angles align with object yaw in ego frame.
        if use_global_coords:
            # Transform both bbox and velocity FROM ego TO global
            bbox_transformed = transform_bbox_to_global(bbox, ego2global_rot, ego2global_trans)
            velocity_transformed = transform_velocity_to_global(velocity, ego2global_rot)
        else:
            # Keep both as-is (already in ego frame)
            bbox_transformed = bbox
            velocity_transformed = velocity

        # Get camera names and optionally 2D bounding box coordinates
        if proj2img:
            # Include 2D bbox for VLM matching
            camera_info_parts = []
            for cam_name, img_num in visible_cameras:
                cam_display_name = cam_name_map[cam_name]
                camera_data = sample.cameras[cam_name]
                bbox_2d = get_bbox_2d_projection(bbox, camera_data, resize_factor)
                if bbox_2d is not None:
                    x_min, y_min, x_max, y_max = bbox_2d
                    camera_info_parts.append(f"Image {img_num} ({cam_display_name}) 2D bbox [x1={x_min}, y1={y_min}, x2={x_max}, y2={y_max}]")
                else:
                    camera_info_parts.append(f"Image {img_num} ({cam_display_name})")
            camera_info = ", ".join(camera_info_parts)
        else:
            # Original format without 2D bbox
            camera_names = [cam_name_map[cam] for cam, _ in visible_cameras]
            image_nums = [str(img_num) for _, img_num in visible_cameras]
            if len(visible_cameras) == 1:
                camera_info = f"Image {image_nums[0]} ({camera_names[0]})"
            else:
                camera_info = f"Images {', '.join(image_nums)} ({', '.join(camera_names)})"

        # Format bbox coordinates [x, y, z, l, w, h, yaw]
        bbox_str = f"[x={bbox_transformed[0]:.2f}, y={bbox_transformed[1]:.2f}, z={bbox_transformed[2]:.2f}, l={bbox_transformed[3]:.2f}, w={bbox_transformed[4]:.2f}, h={bbox_transformed[5]:.2f}, yaw={bbox_transformed[6]:.2f}]"

        # Format velocity [vx, vy]
        vel_str = f"[vx={velocity_transformed[0]:.2f}, vy={velocity_transformed[1]:.2f}] m/s"

        if use_global_coords and ego_vel_global is not None and ego_yaw is not None:
            # ==================== NEW OBB-BASED TTC COMPUTATION ====================
            # Uses GLOBAL coordinates with proper OBB collision detection (SAT)
            #
            # CRITICAL REQUIREMENTS:
            # 1. All positions and velocities must be in GLOBAL coordinates
            # 2. Ego reference point must be converted to geometric center
            # 3. Object yaw must be in GLOBAL frame
            # 4. OBB collision detection with SAT for accuracy

            # Ego reference point position in GLOBAL coordinates
            p_ego_ref = np.array([ego2global_trans[0], ego2global_trans[1]])

            # Ego velocity in GLOBAL coordinates (already available)
            v_ego_global_2d = np.array([ego_vel_global[0], ego_vel_global[1]])

            # Object position in GLOBAL coordinates (from transformed bbox)
            p_obj_global = np.array([bbox_transformed[0], bbox_transformed[1]])

            # Object velocity in GLOBAL coordinates (from transformed velocity)
            v_obj_global_2d = np.array([velocity_transformed[0], velocity_transformed[1]])

            # Object dimensions
            L_obj = bbox_transformed[3]
            W_obj = bbox_transformed[4]

            # Object yaw in GLOBAL frame (from transformed bbox)
            yaw_obj_global = bbox_transformed[6]

            # Compute TTC using OBB collision detection in GLOBAL coordinates
            # Pass ego acceleration for quadratic motion model (reduces braking false positives)
            ttc_result = compute_ttc_obb_global(
                p_ego_ref=p_ego_ref,
                v_ego_global=v_ego_global_2d,
                yaw_ego=ego_yaw,
                p_obj_global=p_obj_global,
                v_obj_global=v_obj_global_2d,
                yaw_obj_global=yaw_obj_global,
                L_obj=L_obj,
                W_obj=W_obj,
                T_max=10.0,  # Search up to 10 seconds
                dt=0.05,     # 50ms time step for coarse search
                eps=1e-3,    # 1ms precision for bisection
                a_ego_global=ego_accel_global  # Ego acceleration (None if unavailable)
            )

            # Format TTC value
            ttc_val = ttc_result['ttc']
            if np.isinf(ttc_val):
                ttc_str = "∞"
            else:
                ttc_str = f"{ttc_val:.2f}s"

            # Get intermediate values
            inter = ttc_result['intermediate']

            # Format TTC_x and TTC_y
            ttc_x_str = "∞" if np.isinf(inter['ttc_x']) else f"{inter['ttc_x']:.2f}s"
            ttc_y_str = "∞" if np.isinf(inter['ttc_y']) else f"{inter['ttc_y']:.2f}s"

            # Build object description with OBB-based TTC
            closing_speed = inter.get('closing_speed', 0.0)
            obj_desc = f"""
-----
OBJ {len(object_descriptions) + 1}: {obj_name} [{camera_info}]
{obj_name} at distance={distance:.1f}m from the ego, moving with velocity {vel_str} visible in {camera_info}
>> TTC = {ttc_str}, Risk = {ttc_result['risk_level']} | Δp_x={inter['delta_p_x']:.2f}m, Δp_y={inter['delta_p_y']:.2f}m, closing_speed={closing_speed:.2f}m/s"""
            # | TTC_x={ttc_x_str}, TTC_y={ttc_y_str}  # Commented out to reduce token usage
            # >> Reason: {ttc_result['reason']}  # Commented out to reduce token usage
        else:
            # Original format for ego-relative coordinates (no TTC)
            obj_desc = f"  - {obj_name} at {bbox_str}, distance={distance:.1f}m, moving with velocity {vel_str} visible in {camera_info}"

        object_descriptions.append(obj_desc)

    if object_descriptions:
        if use_global_coords:
            header = f"""=====
3D OBJECT INFORMATION WITH PRE-COMPUTED TTC
({len(object_descriptions)} objects within {max_distance}m)
====="""
            constraint = """
=====
⚠️ IMPORTANT CONSTRAINTS:
- ONLY consider objects listed above for risk analysis
- TTC and Risk Level are PRE-COMPUTED - use these values directly
- If velocity is [vx=0.00, vy=0.00], the object is STATIONARY
- Incorporate TTC-based risk into your final assessment
====="""
        else:
            header = f"3D OBJECT INFORMATION ({len(object_descriptions)} objects within {max_distance}m):"
            constraint = "\n\n⚠️ IMPORTANT CONSTRAINT: While analyzing risks with surrounding vehicles and pedestrians, ONLY consider the objects listed above. If you detect a vehicle or pedestrian in the images that is NOT in the given list, IGNORE it. Base your analysis EXCLUSIVELY on the provided object information. Also must consider the velocity of the objects and match the numerically given information with the visual features in the images. If an object's velocities along both the x- and y-axes (vx, vy) are 0 m/s, it must be considered stationary, and if it has either positive or negative vx or vy values, it must be considered in motion. When an object is stationary, represent it as a stationary object not parked or stopped."

        return header + "".join(object_descriptions) + constraint
    else:
        return f"No 3D objects are visible in the camera views within {max_distance}m of ego vehicle."


def format_camera_extrinsics_global_for_prompt(camera_data, cam_name: str) -> str:
    """
    Format camera extrinsic parameters in global coordinates for the VQA prompt.

    Args:
        camera_data: CameraData object
        cam_name: Camera name (e.g., 'CAM_FRONT')

    Returns:
        Formatted string describing the camera extrinsics in global coordinates
    """
    # Compute sensor2global transformation
    sensor2global_rot, sensor2global_trans = compute_sensor2global_transform(
        camera_data.sensor2ego_rotation,
        camera_data.sensor2ego_translation,
        camera_data.ego2global_rotation,
        camera_data.ego2global_translation
    )

    # Convert quaternion to Euler angles
    roll, pitch, yaw = quaternion_to_euler(sensor2global_rot)

    # Map camera name to short description
    cam_descriptions = {
        'CAM_FRONT': 'Front (F)',
        'CAM_FRONT_LEFT': 'Front-left (FL)',
        'CAM_FRONT_RIGHT': 'Front-right (FR)',
        'CAM_BACK': 'Rear (R)',
        'CAM_BACK_LEFT': 'Rear-left (RL)',
        'CAM_BACK_RIGHT': 'Rear-right (RR)'
    }

    cam_desc = cam_descriptions.get(cam_name, cam_name)

    # Format position with 2 decimal places
    pos_str = f"[{sensor2global_trans[0]:.2f}, {sensor2global_trans[1]:.2f}, {sensor2global_trans[2]:.2f}] m"

    # Format rotation angles
    rot_str = f"Roll = {roll:.1f}°, Pitch = {pitch:.1f}°, Yaw = {yaw:.1f}°"

    return f"{cam_desc}:\n   - Position (X, Y, Z in ENU): {pos_str}\n   - Rotation: {rot_str}"


def create_single_frame_prompt(sample, loader: NuScenesDataLoader, user_question: str, use_global_coords: bool = False, include_3d_objects: bool = False, filter_distance: float = 20.0, concise_text: bool = False, proj2img: bool = False, resize_factor: int = 1, rear_filter_distance: float = None) -> tuple:
    """
    Create a prompt for single-frame analysis with custom user question.

    Args:
        sample: NuScenesSample object
        loader: NuScenesDataLoader instance
        user_question: Custom question/task from user
        use_global_coords: If True, use global ENU coordinates; if False, use ego-relative FLU coordinates
        include_3d_objects: If True, include 3D object bounding box and velocity information
        filter_distance: Maximum distance (in meters) from ego vehicle to include objects. Default 20.0m
        concise_text: If True, add instructions for concise responses. Default False.
        proj2img: If True, include projected pixel coordinates for each visible camera
        resize_factor: Image resize factor (images are 1/resize_factor of original)
        rear_filter_distance: Maximum distance for non-vehicle objects behind ego. If None, uses filter_distance for all.

    Returns:
        Tuple of (system_prompt, user_question) where:
        - system_prompt: Base context from "You are..." to ego vehicle's driving status
        - user_question: The user's question/task
    """
    # Build camera extrinsics section
    cam_extrinsics = []
    for i, cam_name in enumerate(loader.CAMERA_NAMES, 1):
        cam_data = sample.cameras[cam_name]
        if use_global_coords:
            extr_str = format_camera_extrinsics_global_for_prompt(cam_data, cam_name)
        else:
            extr_str = format_camera_extrinsics_for_prompt(cam_data, cam_name)
        cam_extrinsics.append(f"{i}. {extr_str}")

    extrinsics_text = "\n\n".join(cam_extrinsics)

    # Base context about coordinate system
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

    # Get ego vehicle position and yaw for collision risk info
    first_cam = sample.cameras[loader.CAMERA_NAMES[0]]
    ego_pos = first_cam.ego2global_translation
    ego_quat = first_cam.ego2global_rotation  # [qw, qx, qy, qz]
    # Compute yaw from quaternion: yaw = atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
    qw, qx, qy, qz = ego_quat[0], ego_quat[1], ego_quat[2], ego_quat[3]
    ego_yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))

    # Compute ego velocity in global frame for collision risk section
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
        pass  # Keep default [0, 0] if unable to compute

    # Compute ego speed for display
    ego_speed = np.sqrt(ego_vel_global[0]**2 + ego_vel_global[1]**2)

    # Compute ego acceleration using centered difference (prev + next frames)
    ego_accel_global = compute_ego_acceleration(sample, loader)

    # Build 3D object section if requested (pass ego info for TTC computation)
    object_info_text = ""
    if include_3d_objects:
        object_info_text = format_3d_objects_for_prompt(
            sample, loader, max_distance=filter_distance, use_global_coords=use_global_coords,
            ego_vel_global=ego_vel_global, ego_yaw=ego_yaw,
            proj2img=proj2img, resize_factor=resize_factor,
            rear_filter_distance=rear_filter_distance,
            ego_accel_global=ego_accel_global
        )

    # Build ego velocity section
    ego_velocity_text = format_ego_velocity_for_prompt(sample, loader, use_global_coords=use_global_coords,
                                                        ego_accel_global=ego_accel_global)

    # Collision risk guidance section (only for global coords with 3D objects)
    collision_risk_section = ""
    if use_global_coords and include_3d_objects:
        collision_risk_section = """
=====
COLLISION RISK ESTIMATION MODULE
(Time-To-Collision Based Assessment)
=====

Time-To-Collision (TTC) is a fundamental metric for quantifying the imminence
of a potential collision between the ego vehicle and surrounding dynamic or
static objects.

TTC is defined as the estimated time remaining until a collision occurs,
using an acceleration-aware motion model: p(t) = p₀ + v·t + ½·a·t².
When ego acceleration is available, the model accounts for braking/acceleration
to reduce false positives. Otherwise, it falls back to constant-velocity prediction.

When performing risk assessment, the model must explicitly evaluate TTC for all
relevant objects in the scene and treat it as a primary indicator of collision risk.

Objects with lower TTC values represent imminent threats, even when their absolute
distance is large. Conversely, objects with large distances but rapidly decreasing
TTC should be prioritized over nearby but non-converging objects.

-----
TTC-BASED RISK REASONING
-----
Incorporate TTC into final risk reasoning, emphasizing:
- Imminent front-collision risk
- Cross-traffic collision risk
- Sudden pedestrian/vehicle emergence with rapidly decreasing TTC
- Dynamic vs. static hazards

Treat TTC as a core metric alongside object category, relative motion, road
geometry, and traffic context.

-----
RISK LEVEL CRITERIA
-----
TTC = 0      → CRITICAL (currently overlapping)
0 < TTC < 2  → HIGH
2 ≤ TTC < 5  → MODERATE
TTC ≥ 5      → LOW
TTC = ∞      → NONE (not approaching or diverging)

-----
SUPPLEMENTARY ASSESSMENT
-----
Assign an appropriate risk level based on the collision risk estimation.
Additionally, when available, incorporate:
- Object appearance cues (e.g., vehicle color, type, brake lights)
- Contextual scene elements (e.g., traffic signals, road markings, signs)
- Road-surface objects that may pose potential hazards (e.g., debris, potholes)

Provide this supplementary information alongside the TTC-based risk assessment.

"""

    # System prompt (ends at the ⚠️ IMPORTANT line)
    system_prompt_header = "You are a driving expert agent, and should answer the question at the viewpoint of a driver."

    system_context = f"""{system_prompt_header}

{coord_system_desc}

{extrinsics_text}

EXPECTED CAMERA VIEWS (Egocentric order - left to right):
- Image 1 (Front-left): Should show left-front scene, possibly left side mirror.
- Image 2 (Front): Should show straight-ahead road, traffic lights, pedestrians ahead.
- Image 3 (Front-right): Should show right-front scene, possibly right side mirror.
- Image 4 (Rear-left): Should show left-rear scene (horizontally flipped for egocentric view).
- Image 5 (Rear): Should show straight-behind scene (horizontally flipped for egocentric view).
- Image 6 (Rear-right): Should show right-rear scene (horizontally flipped for egocentric view).
{collision_risk_section}
-----
EGO POSE-COMMAND-LANE INTERPRETATION GUIDE
-----
The driving command, ego velocity, and current lane position together define
the ego vehicle's intended maneuver and expected behavior:

COMMAND-LANE CONSISTENCY:
| Command       | Expected Lane Position           | Expected Behavior                    |
|---------------|----------------------------------|--------------------------------------|
| Go straight   | Center/through lane              | Stay in lane, no heading change      |
| Turn left     | Left-turn lane or leftmost lane  | Heading rotating left (vy > 0)       |
| Turn right    | Right-turn lane or rightmost lane| Heading rotating right (vy < 0)      |
| U-Turn        | Left-turn lane or leftmost lane  | Full 180° heading reversal (vy > 0)  |
| Follow lane   | Any lane                         | Stay in current lane, follow curve   |
| Change lane L | Adjacent to target lane (right of target) | Lateral movement left (vy > 0) |
| Change lane R | Adjacent to target lane (left of target)  | Lateral movement right (vy < 0)|

LANE IDENTIFICATION FROM VISUAL CUES:
- Road markings: Lane arrows, solid/dashed lines, turn-only markings
- Position relative to median/curb: Leftmost, center, rightmost
- Number of lanes: Count from road edges or lane markings

COMMAND vs LANE MISMATCH:
- If command says "Turn left" but ego is in a through lane → ego may be
  preparing to change lanes or approaching turn from a shared lane
- If command says "Go straight" but ego is in a turn lane → ego may be
  in a shared straight+turn lane
- Trust the COMMAND for intended behavior; use LANE for spatial context

VELOCITY CONFIRMS MANEUVER PHASE:
| Speed  | Lateral (vy) | Interpretation                           |
|--------|-------------|-------------------------------------------|
| ≈ 0    | ≈ 0         | Stopped/waiting (at intersection or signal)|
| > 0    | ≈ 0         | Moving straight ahead                     |
| > 0    | > 0 (left)  | Turning left or changing lane left         |
| > 0    | < 0 (right) | Turning right or changing lane right       |

⚠️ CRITICAL: Before reporting any visual feature (signs, lights, objects), VERIFY which image number it appears in by carefully checking each image. Image 1=Front-left, Image 2=Front, Image 3=Front-right, Image 4=Rear-left, Image 5=Rear, Image 6=Rear-right. Do not write the validation process in the answer.

Response in English."""

    # User prompt with given information and task
    image_layout_reminder = """
[IMAGE-DIRECTION MAPPING - Egocentric order]
When reporting visual features, ALWAYS:
1. First identify the Image number (1-6)
2. Then derive the direction ONLY from this mapping:
   Image 1→Front-left, Image 2→Front, Image 3→Front-right,
   Image 4→Rear-left, Image 5→Rear, Image 6→Rear-right
3. NEVER write a direction that contradicts the image number
4. NOTE: Rear camera images (4,5,6) are horizontally flipped for egocentric consistency
"""

    user_prompt = f"""Here is the given information (3D Object's information and EGO vehicle's driving status) of the scene.

{object_info_text}

{ego_velocity_text}

The task you have to conduct is:

{user_question}
{image_layout_reminder}"""

    # Return system context and user prompt separately
    return (system_context, user_prompt)


def create_multi_frame_prompt(samples: List, loader: NuScenesDataLoader, user_question: str, use_global_coords: bool = False, include_3d_objects: bool = False, filter_distance: float = 20.0, proj2img: bool = False, resize_factor: int = 1, rear_filter_distance: float = None) -> tuple:
    """
    Create a prompt for multi-frame temporal sequence analysis with custom user question.

    Args:
        samples: List of NuScenesSample objects (temporal sequence)
        loader: NuScenesDataLoader instance
        user_question: Custom question/task from user
        use_global_coords: If True, use global ENU coordinates; if False, use ego-relative FLU coordinates
        include_3d_objects: If True, include 3D object bounding box and velocity information
        filter_distance: Maximum distance (in meters) from ego vehicle to include objects. Default 20.0m
        proj2img: If True, include projected pixel coordinates for each visible camera
        resize_factor: Image resize factor (images are 1/resize_factor of original)
        rear_filter_distance: Maximum distance for non-vehicle objects behind ego. If None, uses filter_distance for all.

    Returns:
        Tuple of (system_prompt, user_question) where:
        - system_prompt: Base context from "You are..." to ego vehicle's driving status
        - user_question: The user's question/task
    """
    n_frames = len(samples)

    # Use first sample's extrinsics (same for all frames)
    first_sample = samples[0]

    # Build camera extrinsics section
    cam_extrinsics = []
    for i, cam_name in enumerate(loader.CAMERA_NAMES, 1):
        cam_data = first_sample.cameras[cam_name]
        if use_global_coords:
            extr_str = format_camera_extrinsics_global_for_prompt(cam_data, cam_name)
        else:
            extr_str = format_camera_extrinsics_for_prompt(cam_data, cam_name)
        cam_extrinsics.append(f"{i}. {extr_str}")

    extrinsics_text = "\n\n".join(cam_extrinsics)

    # Add temporal information
    time_deltas = []
    for i in range(1, n_frames):
        delta_t = (samples[i].timestamp - samples[0].timestamp) / 1e6  # Convert to seconds
        time_deltas.append(f"Frame {i}: +{delta_t:.2f}s")

    temporal_info = "\n".join(time_deltas)

    # Base context about coordinate system
    if use_global_coords:
        coord_system_desc = f"""You are analyzing a temporal sequence of {n_frames} frames from 6 cameras on an ego vehicle.

COORDINATE SYSTEM (ENU - East-North-Up):
- X: East
- Y: North
- Z: Up

CAMERA EXTRINSIC PARAMETERS (in global coordinates):"""
    else:
        coord_system_desc = f"""You are analyzing a temporal sequence of {n_frames} frames from 6 cameras on an ego vehicle.

COORDINATE SYSTEM (FLU - Forward-Left-Up):
- X: forward
- Y: left
- Z: up

CAMERA EXTRINSIC PARAMETERS (same for all frames):"""

    # Get ego vehicle position and yaw for collision risk info
    first_cam = first_sample.cameras[loader.CAMERA_NAMES[0]]
    ego_pos = first_cam.ego2global_translation
    ego_quat = first_cam.ego2global_rotation  # [qw, qx, qy, qz]
    # Compute yaw from quaternion: yaw = atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
    qw, qx, qy, qz = ego_quat[0], ego_quat[1], ego_quat[2], ego_quat[3]
    ego_yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))

    # Compute ego velocity in global frame for collision risk section
    ego_vel_global = np.array([0.0, 0.0])
    try:
        next_sample = loader.get_sample(first_sample.sample_idx + 1)
        first_cam_next = next_sample.cameras[list(next_sample.cameras.keys())[0]]
        pos_current = np.array(first_cam.ego2global_translation[:2])
        pos_next = np.array(first_cam_next.ego2global_translation[:2])
        dt = (next_sample.timestamp - first_sample.timestamp) / 1e6
        if dt > 0:
            ego_vel_global = (pos_next - pos_current) / dt
    except:
        pass  # Keep default [0, 0] if unable to compute

    # Compute ego speed for display
    ego_speed = np.sqrt(ego_vel_global[0]**2 + ego_vel_global[1]**2)

    # Compute ego acceleration using centered difference (prev + next frames)
    ego_accel_global = compute_ego_acceleration(first_sample, loader)

    # Build 3D object section if requested (using first frame, pass ego info for TTC computation)
    object_info_text = ""
    if include_3d_objects:
        object_info_text = format_3d_objects_for_prompt(
            first_sample, loader, max_distance=filter_distance, use_global_coords=use_global_coords,
            ego_vel_global=ego_vel_global, ego_yaw=ego_yaw,
            proj2img=proj2img, resize_factor=resize_factor,
            rear_filter_distance=rear_filter_distance,
            ego_accel_global=ego_accel_global
        )

    # Build ego velocity section (using first frame)
    ego_velocity_text = format_ego_velocity_for_prompt(first_sample, loader, use_global_coords=use_global_coords,
                                                        ego_accel_global=ego_accel_global)

    # System prompt (ends at the ⚠️ IMPORTANT line)
    system_prompt_header = "You are a driving expert agent, and should answer the question at the viewpoint of a driver."

    system_context = f"""{system_prompt_header}

{coord_system_desc}

{extrinsics_text}

EXPECTED CAMERA VIEWS (Egocentric order - left to right):
- Image 1 (Front-left): Should show left-front scene, possibly left side mirror.
- Image 2 (Front): Should show straight-ahead road, traffic lights, vehicles, pedestrians ahead.
- Image 3 (Front-right): Should show right-front scene, possibly right side mirror.
- Image 4 (Rear-left): Should show left-rear scene (horizontally flipped for egocentric view).
- Image 5 (Rear): Should show straight-behind scene (horizontally flipped for egocentric view).
- Image 6 (Rear-right): Should show right-rear scene (horizontally flipped for egocentric view).

TEMPORAL INFORMATION:
Frame 0: Reference (t=0.00s)
{temporal_info}

EGO VEHICLE CONTEXT:
- Planning command: {loader.COMMAND_DESCRIPTIONS.get(samples[0].gt_navigation_command if hasattr(samples[0], 'gt_navigation_command') and samples[0].gt_navigation_command is not None else samples[0].gt_planning_command, 'unknown')}
- Location: {first_sample.location}
- Scene: {first_sample.description}
{collision_risk_section}
-----
EGO POSE-COMMAND-LANE INTERPRETATION GUIDE
-----
The driving command, ego velocity, and current lane position together define
the ego vehicle's intended maneuver and expected behavior:

COMMAND-LANE CONSISTENCY:
| Command       | Expected Lane Position           | Expected Behavior                    |
|---------------|----------------------------------|--------------------------------------|
| Go straight   | Center/through lane              | Stay in lane, no heading change      |
| Turn left     | Left-turn lane or leftmost lane  | Heading rotating left (vy > 0)       |
| Turn right    | Right-turn lane or rightmost lane| Heading rotating right (vy < 0)      |
| U-Turn        | Left-turn lane or leftmost lane  | Full 180° heading reversal (vy > 0)  |
| Follow lane   | Any lane                         | Stay in current lane, follow curve   |
| Change lane L | Adjacent to target lane (right of target) | Lateral movement left (vy > 0) |
| Change lane R | Adjacent to target lane (left of target)  | Lateral movement right (vy < 0)|

LANE IDENTIFICATION FROM VISUAL CUES:
- Road markings: Lane arrows, solid/dashed lines, turn-only markings
- Position relative to median/curb: Leftmost, center, rightmost
- Number of lanes: Count from road edges or lane markings

COMMAND vs LANE MISMATCH:
- If command says "Turn left" but ego is in a through lane → ego may be
  preparing to change lanes or approaching turn from a shared lane
- If command says "Go straight" but ego is in a turn lane → ego may be
  in a shared straight+turn lane
- Trust the COMMAND for intended behavior; use LANE for spatial context

VELOCITY CONFIRMS MANEUVER PHASE:
| Speed  | Lateral (vy) | Interpretation                           |
|--------|-------------|-------------------------------------------|
| ≈ 0    | ≈ 0         | Stopped/waiting (at intersection or signal)|
| > 0    | ≈ 0         | Moving straight ahead                     |
| > 0    | > 0 (left)  | Turning left or changing lane left         |
| > 0    | < 0 (right) | Turning right or changing lane right       |

⚠️ CRITICAL: Before reporting any visual feature (signs, lights, objects), VERIFY which image number it appears in by carefully checking each image. Image 1=Front-left, Image 2=Front, Image 3=Front-right, Image 4=Rear-left, Image 5=Rear, Image 6=Rear-right. Do not write the validation process in the answer.

Response in English."""

    # User prompt with given information and task
    image_layout_reminder = """
[IMAGE-DIRECTION MAPPING - Egocentric order]
When reporting visual features, ALWAYS:
1. First identify the Image number (1-6)
2. Then derive the direction ONLY from this mapping:
   Image 1→Front-left, Image 2→Front, Image 3→Front-right,
   Image 4→Rear-left, Image 5→Rear, Image 6→Rear-right
3. NEVER write a direction that contradicts the image number
4. NOTE: Rear camera images (4,5,6) are horizontally flipped for egocentric consistency
"""

    user_prompt = f"""Here is the given information (3D Object's information and EGO vehicle's driving status) of the scene.

{object_info_text}

{ego_velocity_text}

The task you have to conduct is:

{user_question}
{image_layout_reminder}"""

    # Return system context and user prompt separately
    return (system_context, user_prompt)


# Removed create_object_detection_prompt - use create_single_frame_prompt with custom question instead

'''
def generate_prompts_for_dataset(loader: NuScenesDataLoader,
                                sample_indices: List[int],
                                n_temporal_frames: int = 5,
                                output_path: str = 'nuscenes_vqa_prompts.json') -> Dict:
    """
    Generate VQA prompts for nuScenes dataset samples.

    Args:
        loader: NuScenesDataLoader instance
        sample_indices: List of sample indices to generate prompts for
        n_temporal_frames: Number of frames for temporal prompts
        output_path: Path to save JSON file

    Returns:
        Dictionary of prompts
    """
    prompts_dict = {}

    for idx in sample_indices:
        print(f"Generating prompts for sample {idx}...")

        # Single frame prompts
        sample = loader.get_sample(idx)

        single_frame_prompt = create_single_frame_prompt(sample, loader)
        prompts_dict[f'single_frame_{idx}'] = {
            'sample_idx': idx,
            'token': sample.token,
            'scene_token': sample.scene_token,
            'prompt_type': 'single_frame_camera_identification',
            'prompt': single_frame_prompt,
            'metadata': {
                'location': sample.location,
                'description': sample.description,
                'num_objects': len(sample.gt_boxes)
            }
        }

        # Object detection prompt
        object_prompt = create_object_detection_prompt(sample, loader)

        # Convert gt_names to list if it's a numpy array
        gt_names_list = sample.gt_names
        if gt_names_list is not None:
            if isinstance(gt_names_list, np.ndarray):
                gt_names_list = gt_names_list.tolist()
            elif not isinstance(gt_names_list, list):
                gt_names_list = list(gt_names_list)

        prompts_dict[f'object_detection_{idx}'] = {
            'sample_idx': idx,
            'token': sample.token,
            'scene_token': sample.scene_token,
            'prompt_type': 'object_detection',
            'prompt': object_prompt,
            'metadata': {
                'num_objects': len(sample.gt_boxes),
                'gt_boxes': sample.gt_boxes.tolist(),
                'gt_velocity': sample.gt_velocity.tolist(),
                'gt_names': gt_names_list
            }
        }

        # Multi-frame temporal prompt
        temporal_samples = loader.get_temporal_samples(idx, n_temporal_frames, load_images=False)

        if temporal_samples is not None:
            multi_frame_prompt = create_multi_frame_prompt(temporal_samples, loader)
            prompts_dict[f'multi_frame_{idx}'] = {
                'start_idx': idx,
                'n_frames': n_temporal_frames,
                'scene_token': sample.scene_token,
                'prompt_type': 'multi_frame_temporal_analysis',
                'prompt': multi_frame_prompt,
                'metadata': {
                    'sample_indices': [s.sample_idx for s in temporal_samples],
                    'tokens': [s.token for s in temporal_samples],
                    'planning_command': loader.COMMAND_DESCRIPTIONS[sample.gt_planning_command]
                }
            }
        else:
            print(f"  Warning: Could not create temporal sequence for sample {idx} (different scenes)")

    # Save to JSON
    print(f"\nSaving prompts to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(prompts_dict, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved {len(prompts_dict)} prompts to {output_path}")

    return prompts_dict
'''

def main():
    """Example usage of the prompt generator"""

    # Initialize loader
    pkl_path = os.environ.get("NUSCENES_PKL_PATH", "nuscenes2d_ego_temporal_infos_val.pkl")
    loader = NuScenesDataLoader(pkl_path)

    print(f"Loaded {len(loader)} samples from nuScenes dataset")

    # Generate prompts for a few samples
    sample_indices = [0, 10, 20, 30, 40]  # Generate for 5 samples

    prompts = generate_prompts_for_dataset(
        loader=loader,
        sample_indices=sample_indices,
        n_temporal_frames=5,
        output_path='nuscenes_vqa_prompts.json'
    )

    # Print example prompt
    print("\n" + "="*80)
    print("EXAMPLE SINGLE FRAME PROMPT:")
    print("="*80)
    print(prompts['single_frame_0']['prompt'])

    print("\n" + "="*80)
    print("EXAMPLE MULTI FRAME PROMPT:")
    print("="*80)
    if 'multi_frame_0' in prompts:
        print(prompts['multi_frame_0']['prompt'][:1000] + "...")

    print("\n" + "="*80)
    print(f"Generated {len(prompts)} prompts in total")
    print("="*80)


if __name__ == "__main__":
    main()
