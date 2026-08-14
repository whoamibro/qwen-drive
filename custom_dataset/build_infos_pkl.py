"""
Build nuScenes-format info pkls for the custom demo dataset (Phases 2+3).

For each clip:
  - selects 2 Hz keyframes (every 5th 10 Hz frame → 120 per clip),
  - computes per-keyframe EGO (VCS) poses from the SLAM trajectory
    (maps/map/traj_lcs.txt, lidar poses in a clip-local map frame) chained
    through the tuned lidar↔camera extrinsics:
        T_map←vcs = T_map←lidar · T_lidar←front(tuned) · T_front←vcs(nominal)
    The clip-local map frame serves as "global" — the pipeline only uses
    relative displacement and yaw.
  - cuts the keyframes into consecutive subclips (default 15 samples =
    7.5 s each; 120/15 = 8 subclips) and writes ONE pkl per subclip:
        <out_root>/<clip>_<NN>.pkl        (NN = 01, 02, ...)
    Each subclip gets its own scene_token; samples carry prev/next links.

The pkl schema mirrors nuscenes2d_ego_temporal_infos_val.pkl. Unlabeled
fields are dummies: gt_navigation_command / gt_planning_command = -1 (to be
filled by the Phase-4 command labeler), object/label arrays empty, planning
tensors zero. Camera entries use the FROZEN virtual rig's effective
intrinsics and derived virtual extrinsics (custom_dataset/virtual_rig.json).

QA: reconstructed speed is cross-checked against the CAN CSV's
Cluster_Display_Speed; the per-clip mean absolute difference is printed.

Usage:
    python -m custom_dataset.build_infos_pkl \\
        --dataset_root /yjj_rnd_home/data/vlm_dataset/demo_dataset \\
        --out_root /yjj_rnd_home/data/vlm_dataset/demo_dataset_nusc
"""

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import re

import numpy as np

from custom_dataset.export_virtual_rig import R_to_quat, quat_to_R
from custom_dataset.virtual_camera import (
    CAMERA_ORDER,
    NUSCENES_TO_CHANNEL,
    load_clip_calibs,
)

_TS_RE = re.compile(r'_(\d{13,17})_')
RIG_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'virtual_rig.json')
DUMMY_COMMAND = -1          # unlabeled; Phase-4 labeler fills the real value
TRAJ_MATCH_TOL_US = 20000   # image↔trajectory timestamp tolerance (20 ms)


def md5_token(*parts) -> str:
    return hashlib.md5('|'.join(str(p) for p in parts).encode()).hexdigest()


def T_from(qwxyz, t):
    """4x4 from quaternion dict/list [w,x,y,z] + translation."""
    T = np.eye(4)
    T[:3, :3] = quat_to_R(*qwxyz)
    T[:3, 3] = t
    return T


def T_inv(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def load_trajectory(clip_dir):
    """{timestamp_us: 4x4 T_map<-lidar} from traj_lcs.txt (TUM format)."""
    out = {}
    with open(os.path.join(clip_dir, 'maps', 'map', 'traj_lcs.txt')) as f:
        for line in f:
            p = line.split()
            if len(p) != 8:
                continue
            ts_us = round(float(p[0]) * 1e6)
            T = np.eye(4)
            T[:3, :3] = quat_to_R(float(p[7]), float(p[4]),
                                  float(p[5]), float(p[6]))
            T[:3, 3] = [float(p[1]), float(p[2]), float(p[3])]
            out[ts_us] = T
    return out


def match_traj(traj, ts_us):
    """Nearest trajectory pose within tolerance; None if absent."""
    if ts_us in traj:
        return traj[ts_us]
    keys = match_traj._sorted.setdefault(id(traj), sorted(traj))
    import bisect
    i = bisect.bisect_left(keys, ts_us)
    best = None
    for j in (i - 1, i):
        if 0 <= j < len(keys) and abs(keys[j] - ts_us) <= TRAJ_MATCH_TOL_US:
            if best is None or abs(keys[j] - ts_us) < abs(best - ts_us):
                best = keys[j]
    return traj[best] if best is not None else None


match_traj._sorted = {}


def can_speed_series(clip_dir, clip_name):
    """[(time_s, speed_mps)] from the CAN CSV's Cluster_Display_Speed (km/h);
    [] when the file/column can't be parsed."""
    path = os.path.join(clip_dir, f'GER_{clip_name.split("demo_1_")[-1]}')
    cands = [f for f in os.listdir(clip_dir) if f.endswith('_ccan_3_0.csv')]
    if not cands:
        return []
    out = []
    with open(os.path.join(clip_dir, cands[0])) as f:
        reader = csv.reader(f)
        header = None
        for row in reader:
            if header is None:
                if row and row[0].strip() == 'Time':
                    header = row
                    i_spd = header.index('Cluster_Display_Speed')
                continue
            try:
                out.append((float(row[0]), float(row[i_spd]) / 3.6))
            except (ValueError, IndexError):
                continue
    return out


def build_cam_entries(rig, calibs, T_vcs_lidar, ego2global_q, ego2global_t,
                      clip_name, ts):
    """Per-camera dicts in nuScenes cams{} schema, for the VIRTUAL cameras."""
    cams = {}
    T_lidar_vcs = T_inv(T_vcs_lidar)
    for cam in CAMERA_ORDER:
        rc = rig['cameras'][cam]
        calib = calibs[cam]
        # Virtual camera orientation: R maps virtual-cam coords -> source-cam
        # coords; chain into lidar then ego (VCS). Optical center unchanged.
        R_v = np.array(rc['rotation_src_from_virtual'])
        lcs = calib.lcs_extrinsic     # tuned: camera -> lidar
        T_lidar_cam = T_from([lcs['qw'], lcs['qx'], lcs['qy'], lcs['qz']],
                             [lcs['tx'], lcs['ty'], lcs['tz']])
        R_lidar_virtual = T_lidar_cam[:3, :3] @ R_v
        t_lidar_cam = T_lidar_cam[:3, 3]
        R_vcs_virtual = T_vcs_lidar[:3, :3] @ R_lidar_virtual
        t_vcs_cam = T_vcs_lidar[:3, :3] @ t_lidar_cam + T_vcs_lidar[:3, 3]
        cams[cam] = {
            'data_path': f'samples/{cam}/{clip_name}__{cam}__{ts}.jpg',
            'type': cam,
            'sample_data_token': md5_token(clip_name, ts, cam),
            'sensor2ego_translation': [round(v, 6) for v in t_vcs_cam],
            'sensor2ego_rotation': [round(v, 8) for v in R_to_quat(R_vcs_virtual)],
            'ego2global_translation': ego2global_t,
            'ego2global_rotation': ego2global_q,
            'timestamp': int(ts),
            'sensor2lidar_rotation': R_lidar_virtual,
            'sensor2lidar_translation': t_lidar_cam.copy(),
            'cam_intrinsic': np.array(rc['effective_intrinsic']),
        }
    return cams


def main():
    parser = argparse.ArgumentParser(
        description='Build nuScenes-format subclip pkls for the custom demo '
                    'dataset')
    parser.add_argument('--dataset_root', type=str,
                        default='/yjj_rnd_home/data/vlm_dataset/demo_dataset')
    parser.add_argument('--out_root', type=str,
                        default='/yjj_rnd_home/data/vlm_dataset/demo_dataset_nusc')
    parser.add_argument('--keyframe_stride', type=int, default=5,
                        help='10 Hz frames per keyframe (5 -> 2 Hz)')
    parser.add_argument('--chunk_size', type=int, default=15,
                        help='Keyframes per subclip pkl (15 -> 7.5 s scenes)')
    args = parser.parse_args()

    with open(RIG_JSON) as f:
        rig = json.load(f)
    os.makedirs(args.out_root, exist_ok=True)

    clips = sorted(
        d for d in os.listdir(args.dataset_root)
        if os.path.isdir(os.path.join(args.dataset_root, d, 'images')))

    total_pkls = 0
    for clip_name in clips:
        clip_dir = os.path.join(args.dataset_root, clip_name)
        calibs = load_clip_calibs(clip_dir)
        traj = load_trajectory(clip_dir)

        # T_lidar<-vcs via the front camera: tuned lcs (front->lidar) chained
        # with nominal vcs (front->vcs; the front block is the canonical one).
        f_lcs = calibs['CAM_FRONT'].lcs_extrinsic
        f_vcs = calibs['CAM_FRONT'].vcs_extrinsic
        T_lidar_front = T_from([f_lcs['qw'], f_lcs['qx'], f_lcs['qy'], f_lcs['qz']],
                               [f_lcs['tx'], f_lcs['ty'], f_lcs['tz']])
        T_vcs_front = T_from([f_vcs['qw'], f_vcs['qx'], f_vcs['qy'], f_vcs['qz']],
                             [f_vcs['tx'], f_vcs['ty'], f_vcs['tz']])
        T_lidar_vcs = T_lidar_front @ T_inv(T_vcs_front)
        T_vcs_lidar = T_inv(T_lidar_vcs)

        # Keyframes: common timestamps across channels, strided to 2 Hz.
        img_root = os.path.join(clip_dir, 'images')
        ts_sets = []
        for cam in CAMERA_ORDER:
            ch_dir = os.path.join(img_root, NUSCENES_TO_CHANNEL[cam])
            ts_sets.append({m.group(1) for f in os.listdir(ch_dir)
                            if (m := _TS_RE.search(f)) and f.endswith('.jpg')})
        keyframes = sorted(set.intersection(*ts_sets), key=int)
        keyframes = keyframes[::args.keyframe_stride]

        # Ego pose per keyframe (drop keyframes without a trajectory match).
        samples = []
        n_dropped = 0
        for ts in keyframes:
            T_map_lidar = match_traj(traj, int(ts))
            if T_map_lidar is None:
                n_dropped += 1
                continue
            T_map_vcs = T_map_lidar @ T_lidar_vcs
            samples.append((ts, T_map_vcs))
        if n_dropped:
            print(f'[{clip_name}] dropped {n_dropped} keyframes without '
                  f'trajectory pose')

        # QA: reconstructed speed vs CAN cluster speed.
        can = can_speed_series(clip_dir, clip_name)
        if can and len(samples) > 2:
            t0 = int(samples[0][0]) / 1e6
            diffs = []
            can_t = np.array([c[0] for c in can])
            can_v = np.array([c[1] for c in can])
            can_t0 = can_t[0]
            for i in range(1, len(samples) - 1):
                tsp, Tp = samples[i - 1]
                tsn, Tn = samples[i + 1]
                dt = (int(tsn) - int(tsp)) / 1e6
                v = np.linalg.norm(Tn[:2, 3] - Tp[:2, 3]) / dt
                trel = int(samples[i][0]) / 1e6 - t0
                j = int(np.argmin(np.abs((can_t - can_t0) - trel)))
                diffs.append(abs(v - can_v[j]))
            print(f'[{clip_name}] speed QA vs CAN: mean |Δv| = '
                  f'{np.mean(diffs):.2f} m/s over {len(diffs)} keyframes')

        # Cut into subclips and write one pkl per subclip.
        chunks = [samples[i:i + args.chunk_size]
                  for i in range(0, len(samples), args.chunk_size)]
        for ci, chunk in enumerate(chunks, 1):
            scene_name = f'{clip_name}_{ci:02d}'
            scene_token = md5_token(scene_name)
            tokens = [md5_token(clip_name, ts) for ts, _ in chunk]
            infos = []
            for i, (ts, T_map_vcs) in enumerate(chunk):
                e2g_q = [round(v, 8) for v in R_to_quat(T_map_vcs[:3, :3])]
                e2g_t = [round(float(v), 6) for v in T_map_vcs[:3, 3]]
                cams = build_cam_entries(rig, calibs, T_vcs_lidar,
                                         e2g_q, e2g_t, clip_name, ts)
                # lidar2ego from the same chain (lidar -> vcs)
                l2e_q = [round(v, 8) for v in R_to_quat(T_vcs_lidar[:3, :3])]
                l2e_t = [round(float(v), 6) for v in T_vcs_lidar[:3, 3]]
                infos.append({
                    'lidar_path': f'lidar/{clip_name}_{ts}.pcd',  # placeholder
                    'token': tokens[i],
                    'prev': tokens[i - 1] if i > 0 else '',
                    'next': tokens[i + 1] if i < len(chunk) - 1 else '',
                    'can_bus': np.zeros(13, dtype=np.float64),
                    'sweeps': [],
                    'frame_idx': i,
                    'cams': cams,
                    'scene_token': scene_token,
                    'lidar2ego_translation': l2e_t,
                    'lidar2ego_rotation': l2e_q,
                    'ego2global_translation': e2g_t,
                    'ego2global_rotation': e2g_q,
                    'timestamp': int(ts),
                    'gt_boxes': np.zeros((0, 7), dtype=np.float32),
                    'gt_names': np.array([], dtype='<U32'),
                    'gt_velocity': np.zeros((0, 2), dtype=np.float32),
                    'num_lidar_pts': np.zeros(0, dtype=np.int64),
                    'num_radar_pts': np.zeros(0, dtype=np.int64),
                    'valid_flag': np.zeros(0, dtype=bool),
                    'bboxes2d': [],
                    'bboxes3d_cams': [],
                    'labels2d': [],
                    'centers2d': [],
                    'depths': [],
                    'bboxes_ignore': [],
                    'visibilities': [],
                    'lane_info': None,
                    'gt_planning': np.zeros((1, 6, 3), dtype=np.float32),
                    'gt_planning_mask': np.zeros((1, 6, 2), dtype=bool),
                    'gt_planning_command': DUMMY_COMMAND,
                    'description': f'custom demo clip {clip_name} '
                                   f'subclip {ci:02d}',
                    'location': 'custom-demo-GER',
                    'scene_name': scene_name,
                    'map_geoms': {},
                    'gt_fut_traj': np.zeros((0, 12, 2), dtype=np.float32),
                    'gt_fut_traj_mask': np.zeros((0, 12), dtype=np.float32),
                    'gt_fullnames': [],
                    'gt_attrs': [],
                    'gt_fut_yaw': np.zeros((0, 12), dtype=np.float32),
                    'gt_fut_idx': np.zeros((0, 1), dtype=np.int64),
                    'gt_navigation_command': DUMMY_COMMAND,
                })
            out_path = os.path.join(args.out_root, f'{scene_name}.pkl')
            with open(out_path, 'wb') as f:
                pickle.dump({'metadata': {'version': 'custom-demo-v1'},
                             'infos': infos}, f)
            total_pkls += 1
            print(f'  wrote {out_path}  ({len(infos)} samples, '
                  f'scene_token={scene_token[:16]})')

    print(f'\n{total_pkls} subclip pkls written to {args.out_root}')
    print('NOTE: gt_navigation_command / gt_planning_command are DUMMY (-1) '
          'until the Phase-4 command labeler fills them.')


if __name__ == '__main__':
    main()
