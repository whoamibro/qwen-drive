"""
Measure each camera's TRUE optical-axis orientation relative to the road
plane, and persist it for the virtual-camera pitch correction.

Why: the calib yamls' `vcs_extrinsic` blocks are NOMINAL design values (all
round numbers; CAM_BACK's block is plainly wrong — it claims the rear camera
faces forward). The tuned extrinsics are the `lcs_extrinsic` blocks
(camera→lidar), but they only relate cameras to the LIDAR, not to the road.
The missing anchor comes from the SLAM trajectory: on straight, real-motion
segments the ego displacement direction is road-parallel, so expressing it
in the lidar body frame gives a road-level forward axis. A road frame
(forward / left / up) is built from that, and every camera's optical axis is
measured against it.

Output: custom_dataset/camera_orientation.json
    {
      "elevation_deg": {cam: mean elevation, + = pointing up},
      "azimuth_deg":   {cam: mean azimuth, + = to the left},
      "roll_deg":      {cam: mean roll},
      "per_clip":      {clip: {cam: {azimuth, elevation, roll}}, ...}
    }

`virtual_camera.load_camera_elevations()` reads this file; the crop math
then levels each virtual camera by shifting the crop center vertically.

Usage:
    python -m custom_dataset.measure_camera_orientation \\
        --dataset_root /yjj_rnd_home/data/vlm_dataset/demo_dataset
"""

import argparse
import json
import math
import os
from datetime import datetime

import numpy as np
import yaml

from custom_dataset.virtual_camera import CHANNEL_TO_NUSCENES, CAMERA_ORDER

MIN_SPEED_MPS = 5.0        # ignore near-stationary frames
MAX_TURN_DEG = 0.5         # per-step heading change allowed on "straight"


def quat_to_R(qw, qx, qy, qz):
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
    ])


def road_frame_in_lidar(traj_path: str):
    """(forward, left, up) unit vectors of the road frame, in lidar coords.

    Averages the ego motion direction (expressed in the lidar body frame)
    over straight segments with real speed, then orthogonalizes with the
    lidar z axis. Returns (frame_3x3_rows, n_samples_used).
    """
    rows = [l.split() for l in open(traj_path) if l.strip()]
    T = [(float(r[0]),
          np.array([float(r[1]), float(r[2]), float(r[3])]),
          quat_to_R(float(r[7]), float(r[4]), float(r[5]), float(r[6])))
         for r in rows]
    fwd = []
    for i in range(1, len(T) - 1):
        t0, p0, _ = T[i - 1]
        _, p1, R1 = T[i]
        t2, p2, _ = T[i + 1]
        v = (p2 - p0) / (t2 - t0)
        if np.linalg.norm(v) < MIN_SPEED_MPS:
            continue
        d1 = p1 - p0
        d2 = p2 - p1
        if min(np.linalg.norm(d1), np.linalg.norm(d2)) < 1e-6:
            continue
        d1, d2 = d1 / np.linalg.norm(d1), d2 / np.linalg.norm(d2)
        if math.degrees(math.acos(np.clip(d1 @ d2, -1, 1))) > MAX_TURN_DEG:
            continue
        fwd.append(R1.T @ (v / np.linalg.norm(v)))
    if not fwd:
        raise RuntimeError(f'no straight moving segments in {traj_path}')
    f = np.mean(fwd, axis=0)
    f /= np.linalg.norm(f)
    left = np.cross([0.0, 0.0, 1.0], f)
    left /= np.linalg.norm(left)
    up = np.cross(f, left)
    return (f, left, up), len(fwd)


def measure_clip(clip_dir: str) -> dict:
    """Per-camera {azimuth, elevation, roll} (deg, road-relative)."""
    (f, left, up), n = road_frame_in_lidar(
        os.path.join(clip_dir, 'maps', 'map', 'traj_lcs.txt'))
    out = {'_n_straight_samples': n}
    for channel, cam in CHANNEL_TO_NUSCENES.items():
        d = yaml.safe_load(
            open(os.path.join(clip_dir, 'maps', 'calib_tuned', f'{channel}.yaml')))
        e = d['lcs_extrinsic']   # tuned: camera frame -> lidar frame
        R = quat_to_R(e['qw'], e['qx'], e['qy'], e['qz'])
        axis = R @ np.array([0.0, 0.0, 1.0])     # optical axis in lidar frame
        x_img = R @ np.array([1.0, 0.0, 0.0])    # image +x in lidar frame
        out[cam] = {
            'azimuth': round(math.degrees(math.atan2(axis @ left, axis @ f)), 2),
            'elevation': round(math.degrees(math.asin(np.clip(axis @ up, -1, 1))), 2),
            'roll': round(math.degrees(math.asin(np.clip(x_img @ up, -1, 1))), 2),
        }
    return out


def main():
    parser = argparse.ArgumentParser(
        description='Measure camera orientations vs the road plane and write '
                    'camera_orientation.json for the virtual-camera pitch '
                    'correction')
    parser.add_argument('--dataset_root', type=str,
                        default='/yjj_rnd_home/data/vlm_dataset/demo_dataset')
    parser.add_argument('--output', type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             'camera_orientation.json'))
    args = parser.parse_args()

    per_clip = {}
    for name in sorted(os.listdir(args.dataset_root)):
        clip_dir = os.path.join(args.dataset_root, name)
        traj = os.path.join(clip_dir, 'maps', 'map', 'traj_lcs.txt')
        if not os.path.isfile(traj):
            continue
        per_clip[name] = measure_clip(clip_dir)
        n = per_clip[name].pop('_n_straight_samples')
        print(f'[{name}] measured from {n} straight samples')
        for cam in CAMERA_ORDER:
            m = per_clip[name][cam]
            print(f"  {cam:<16s} az={m['azimuth']:>8.2f}  "
                  f"el={m['elevation']:>7.2f}  roll={m['roll']:>6.2f}")

    if not per_clip:
        raise SystemExit(f'no clips with maps/map/traj_lcs.txt under '
                         f'{args.dataset_root}')

    mean = {}
    for key in ('azimuth', 'elevation', 'roll'):
        mean[key] = {
            cam: round(float(np.mean([per_clip[c][cam][key] for c in per_clip])), 2)
            for cam in CAMERA_ORDER
        }
        spread = {
            cam: max(per_clip[c][cam][key] for c in per_clip)
                 - min(per_clip[c][cam][key] for c in per_clip)
            for cam in CAMERA_ORDER
        }
        worst = max(spread.values())
        if worst > 1.0:
            print(f'WARNING: {key} varies {worst:.2f} deg across clips — '
                  f'check per_clip values before trusting the mean')

    out = {
        'generated': datetime.now().isoformat(timespec='seconds'),
        'method': 'trajectory-anchored road frame + tuned lcs_extrinsic '
                  '(see module docstring)',
        'azimuth_deg': mean['azimuth'],
        'elevation_deg': mean['elevation'],
        'roll_deg': mean['roll'],
        'per_clip': per_clip,
    }
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nMean elevations (deg, + = up): {mean["elevation"]}')
    print(f'Written: {args.output}')


if __name__ == '__main__':
    main()
