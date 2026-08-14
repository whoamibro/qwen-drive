"""
Export the FROZEN virtual-camera rig — intrinsics + extrinsics of the six
virtual views as finally tuned in the crop verifier — to
custom_dataset/virtual_rig.json.

The batch image converter and the pkl builder (Phase 3) read this file, so
the adopted geometry lives in exactly one place. The crop verifier also
auto-loads it at startup when present, making the tuned setting persistent.

What is saved per camera:
  - source intrinsics/extrinsics (from calib_tuned, rig-constant over clips)
  - the virtual target camera (width/height/focal)
  - the warp: source footprint quad + full 3x3 output→source homography
  - the EFFECTIVE virtual intrinsic matrix and rotation, recovered by RQ
    decomposition of the homography (H = K_src · R · K_eff^-1). For the
    pure-rotation cameras K_eff equals the target K exactly; for CAM_BACK's
    keystone warp (pin_top + bottom_scale) K_eff is anamorphic with a
    shifted principal point — the honest camera model of that view.
  - the virtual optical axis in the road frame (azimuth/elevation/roll)

FROZEN CONFIG (decided 2026-08-13 in the crop verifier session):
  base f_t=1100; CAM_BACK f_t = base * 809.22/1266.42 (nuScenes wide-rear);
  auto pitch-leveling from camera_orientation.json; extra pitch
  CAM_BACK_LEFT/RIGHT = -5 deg; CAM_BACK keystone: pin_top + bottom_scale
  0.85; rear views flipped at composition/inference time.

Usage:
    python -m custom_dataset.export_virtual_rig \\
        --dataset_root /yjj_rnd_home/data/vlm_dataset/demo_dataset
"""

import argparse
import json
import math
import os
from datetime import datetime

import numpy as np

from custom_dataset.virtual_camera import (
    CAMERA_ORDER,
    DEFAULT_TARGET,
    REAR_CAMERAS,
    _perspective_coeffs_for_quad,
    axis_elevation_after,
    load_camera_elevations,
    load_clip_calibs,
    source_quad,
    target_for_camera,
)

# ---------------------------------------------------------------------------
# The frozen warp configuration (single source of truth from here on).
# ---------------------------------------------------------------------------
FROZEN_CONFIG = {
    'base_focal': DEFAULT_TARGET.focal,          # 1100.0
    'back_focal': None,                          # None = nuScenes ratio (702.9)
    'pitch_offsets_deg': {'CAM_BACK_LEFT': -5.0, 'CAM_BACK_RIGHT': -5.0},
    'bottom_scales': {'CAM_BACK': 0.85},
    'pin_tops': ['CAM_BACK'],
    'rear_flip_cameras': sorted(REAR_CAMERAS),
}


def _cfg_pitch(cam):
    return float(FROZEN_CONFIG['pitch_offsets_deg'].get(cam, 0.0))


def _cfg_bscale(cam):
    return float(FROZEN_CONFIG['bottom_scales'].get(cam, 1.0))


def _cfg_pintop(cam):
    return cam in FROZEN_CONFIG['pin_tops']


def homography_from_quad(quad, target):
    """3x3 H mapping output pixels -> source pixels (row-major ndarray)."""
    a, b, c, d, e, f, g, h = _perspective_coeffs_for_quad(quad, target)
    return np.array([[a, b, c], [d, e, f], [g, h, 1.0]])


def rq_decompose(B):
    """RQ decomposition: B = K (upper triangular, positive diagonal) @ Q
    (rotation). Standard flip/QR trick."""
    P = np.fliplr(np.eye(3))
    Q_, R_ = np.linalg.qr((P @ B).T)
    K = P @ R_.T @ P
    Q = P @ Q_.T
    # Force positive diagonal on K
    S = np.diag(np.sign(np.diag(K)))
    K, Q = K @ S, S @ Q
    if np.linalg.det(Q) < 0:
        K, Q = -K, -Q  # keep B = K@Q while making Q a proper rotation
    return K, Q


def decompose_effective_camera(H_out_to_src, calib):
    """Recover (K_eff, R_src_from_virtual) with H = K_src · R · K_eff^-1.

    M = K_src^-1 H = R K_eff^-1  →  M^-1 = K_eff R^T = RQ-decomposable.
    Returns (K_eff normalized to [2,2]=1, R as ndarray)."""
    K_src = np.array([[calib.fx, 0, calib.cx],
                      [0, calib.fy, calib.cy],
                      [0, 0, 1.0]])
    M = np.linalg.inv(K_src) @ H_out_to_src
    K_eff, Q = rq_decompose(np.linalg.inv(M))
    K_eff = K_eff / K_eff[2, 2]
    R = Q.T   # M^-1 = K_eff @ R^T  →  R^T = K_eff^-1 M^-1... Q = R^T
    # Verify round trip
    H_chk = K_src @ R @ np.linalg.inv(K_eff)
    H_chk /= H_chk[2, 2]
    H_ref = H_out_to_src / H_out_to_src[2, 2]
    err = float(np.abs(H_chk - H_ref).max())
    return K_eff, R, err


def quat_to_R(qw, qx, qy, qz):
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
    ])


def R_to_quat(R):
    """Rotation matrix -> quaternion [w, x, y, z] (nuScenes order)."""
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return [s / 4, (R[2, 1] - R[1, 2]) / s,
                (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(max(1e-12, 1.0 + R[i, i] - R[j, j] - R[k, k])) * 2
    q = [0.0] * 4
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = s / 4
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q


def road_frame_orientation(cam_meta, R_virtual_in_cam):
    """Virtual optical axis + roll in the road frame, using the measured
    per-camera road orientation (azimuth/elevation/roll of the SOURCE axis
    from camera_orientation.json's per-camera measurement)."""
    az = math.radians(cam_meta['azimuth'])
    el = math.radians(cam_meta['elevation'])
    roll = math.radians(cam_meta['roll'])
    # Reconstruct R_road_from_cam of the SOURCE camera: z_cam -> (az, el),
    # x_cam (image right) horizontal-ish with the measured roll.
    z = np.array([math.cos(el) * math.cos(az),
                  math.cos(el) * math.sin(az),
                  math.sin(el)])
    up = np.array([0.0, 0.0, 1.0])
    x0 = np.cross(z, up)
    x0 /= np.linalg.norm(x0)
    y0 = np.cross(z, x0)   # image-down direction with zero roll
    x = math.cos(roll) * x0 + math.sin(roll) * y0
    y = np.cross(z, x)
    R_road_cam = np.column_stack([x, y, z])
    # Virtual camera axes in road frame
    R_road_virtual = R_road_cam @ R_virtual_in_cam
    zv = R_road_virtual[:, 2]
    xv = R_road_virtual[:, 0]
    return {
        'azimuth_deg': round(math.degrees(math.atan2(zv[1], zv[0])), 2),
        'elevation_deg': round(math.degrees(math.asin(np.clip(zv[2], -1, 1))), 2),
        'roll_deg': round(math.degrees(math.asin(np.clip(xv[2], -1, 1))), 2),
        'quaternion_road_from_virtual_wxyz':
            [round(v, 6) for v in R_to_quat(R_road_virtual)],
    }


def main():
    parser = argparse.ArgumentParser(
        description='Export the frozen virtual-camera rig to virtual_rig.json')
    parser.add_argument('--dataset_root', type=str,
                        default='/yjj_rnd_home/data/vlm_dataset/demo_dataset')
    parser.add_argument('--output', type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             'virtual_rig.json'))
    parser.add_argument('--orientation_json', type=str, default=None)
    args = parser.parse_args()

    # Calibs are rig-constant; verify across clips and use the first.
    clip_dirs = sorted(
        os.path.join(args.dataset_root, d) for d in os.listdir(args.dataset_root)
        if os.path.isdir(os.path.join(args.dataset_root, d, 'maps', 'calib_tuned')))
    all_calibs = [load_clip_calibs(d) for d in clip_dirs]
    calibs = all_calibs[0]
    for other, d in zip(all_calibs[1:], clip_dirs[1:]):
        for cam in CAMERA_ORDER:
            a, b = calibs[cam], other[cam]
            if any(abs(getattr(a, k) - getattr(b, k)) > 1e-6
                   for k in ('fx', 'fy', 'cx', 'cy')):
                print(f'WARNING: {cam} intrinsics differ in {d} — '
                      f'exporting the first clip\'s values')

    elevations = (load_camera_elevations(args.orientation_json)
                  if args.orientation_json else load_camera_elevations())
    orient_path = args.orientation_json or os.path.join(
        os.path.dirname(__file__), 'camera_orientation.json')
    with open(orient_path) as f:
        orientation = json.load(f)

    out = {
        'generated': datetime.now().isoformat(timespec='seconds'),
        'description': 'Frozen virtual-camera rig for the custom demo '
                       'dataset -> nuScenes-format conversion (see manifest '
                       'report section 9).',
        'frozen_config': FROZEN_CONFIG,
        'source_elevations_deg': elevations,
        'cameras': {},
    }

    for cam in CAMERA_ORDER:
        calib = calibs[cam]
        target = target_for_camera(cam, DEFAULT_TARGET,
                                   FROZEN_CONFIG['back_focal'])
        e = elevations.get(cam, 0.0)
        quad = source_quad(calib, target, e, _cfg_pitch(cam),
                           _cfg_bscale(cam), _cfg_pintop(cam))
        H = homography_from_quad(quad, target)
        K_eff, R, err = decompose_effective_camera(H, calib)
        axis_after = axis_elevation_after(calib, target, e, _cfg_pitch(cam))
        cam_orient = orientation['per_clip'][
            sorted(orientation['per_clip'])[0]][cam]

        out['cameras'][cam] = {
            'channel': calib.channel,
            'source': {
                'width': calib.width, 'height': calib.height,
                'fx': calib.fx, 'fy': calib.fy,
                'cx': calib.cx, 'cy': calib.cy,
                'vcs_extrinsic_nominal': calib.vcs_extrinsic,
                'lcs_extrinsic_tuned': calib.lcs_extrinsic,
                'road_orientation_measured_deg': cam_orient,
            },
            'target': {'width': target.width, 'height': target.height,
                       'focal': round(target.focal, 3)},
            'warp': {
                'source_quad_tl_tr_br_bl':
                    [[round(x, 2), round(y, 2)] for x, y in quad],
                'homography_out_to_src':
                    [[round(v, 8) for v in row] for row in H.tolist()],
                'pure_rotation': not (_cfg_bscale(cam) != 1.0 or _cfg_pintop(cam)),
                'decomposition_residual': round(err, 8),
            },
            'effective_intrinsic':
                [[round(v, 4) for v in row] for row in K_eff.tolist()],
            'rotation_src_from_virtual':
                [[round(v, 8) for v in row] for row in R.tolist()],
            'virtual_axis_road_frame':
                road_frame_orientation(cam_orient, R),
            'axis_elevation_after_deg': round(axis_after, 2),
            'flip_at_inference': cam in REAR_CAMERAS,
        }

    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)

    print(f'Virtual rig written: {args.output}')
    for cam in CAMERA_ORDER:
        c = out['cameras'][cam]
        K = c['effective_intrinsic']
        v = c['virtual_axis_road_frame']
        print(f"  {cam:<16s} K_eff fx={K[0][0]:8.1f} fy={K[1][1]:8.1f} "
              f"cx={K[0][2]:7.1f} cy={K[1][2]:7.1f} | axis az={v['azimuth_deg']:8.2f} "
              f"el={v['elevation_deg']:7.2f} roll={v['roll_deg']:6.2f} | "
              f"H residual={c['warp']['decomposition_residual']:.1e}")


if __name__ == '__main__':
    main()
