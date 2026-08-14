"""
Batch image converter — custom demo dataset → nuScenes-style samples/ tree.

Applies the FROZEN virtual-camera rig (custom_dataset/virtual_rig.json — see
manifest report section 9) to every frame of every clip:

    <out_root>/samples/<CAM_X>/<clip>__<CAM_X>__<timestamp>.jpg   (1600x900)

Notes:
  - The rear-view horizontal flip is NOT baked into the files — the demo
    tester applies it at inference time by camera name, matching nuScenes.
  - Idempotent: existing outputs are skipped, so re-runs only fill gaps.
  - Multiprocessed; a full 3-clip run (10,782 images) takes a few minutes.

Usage:
    python -m custom_dataset.convert_images \\
        --dataset_root /yjj_rnd_home/data/vlm_dataset/demo_dataset \\
        --out_root /yjj_rnd_home/data/vlm_dataset/demo_dataset_nusc
"""

import argparse
import json
import multiprocessing as mp
import os
import re
import time

from PIL import Image

from custom_dataset.virtual_camera import (
    CAMERA_ORDER,
    DEFAULT_TARGET,
    NUSCENES_TO_CHANNEL,
    convert_image,
    load_camera_elevations,
    load_clip_calibs,
    target_for_camera,
)

_TS_RE = re.compile(r'_(\d{13,17})_')

RIG_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'virtual_rig.json')

# Worker globals (initialized once per process)
_W = {}


def _init_worker(rig_config, elevations, calib_by_clip):
    _W['cfg'] = rig_config
    _W['elev'] = elevations
    _W['calibs'] = calib_by_clip


def _convert_one(job):
    clip, cam, ts, src_path, dst_path = job
    if os.path.exists(dst_path):
        return 0
    cfg = _W['cfg']
    calib = _W['calibs'][clip][cam]
    target = target_for_camera(cam, DEFAULT_TARGET, cfg.get('back_focal'))
    img = Image.open(src_path).convert('RGB')
    out = convert_image(
        img, calib, target,
        flip=False,  # rear flip happens at inference, not on disk
        elevation_deg=_W['elev'].get(cam, 0.0),
        extra_pitch_deg=float(cfg.get('pitch_offsets_deg', {}).get(cam, 0.0)),
        bottom_scale=float(cfg.get('bottom_scales', {}).get(cam, 1.0)),
        pin_top=cam in cfg.get('pin_tops', []),
    )
    tmp = dst_path + '.tmp'
    out.save(tmp, format='JPEG', quality=95)
    os.replace(tmp, dst_path)
    return 1


def main():
    parser = argparse.ArgumentParser(
        description='Convert all clip images through the frozen virtual rig')
    parser.add_argument('--dataset_root', type=str,
                        default='/yjj_rnd_home/data/vlm_dataset/demo_dataset')
    parser.add_argument('--out_root', type=str,
                        default='/yjj_rnd_home/data/vlm_dataset/demo_dataset_nusc')
    parser.add_argument('--stride', type=int, default=1,
                        help='Convert every Nth frame (1 = all frames)')
    parser.add_argument('--workers', type=int, default=max(4, mp.cpu_count() // 2))
    args = parser.parse_args()

    with open(RIG_JSON) as f:
        rig = json.load(f)
    cfg = rig['frozen_config']
    elevations = load_camera_elevations()
    print(f'Rig: base_focal={cfg["base_focal"]}  '
          f'pitch={cfg["pitch_offsets_deg"]}  bscale={cfg["bottom_scales"]}  '
          f'pin_tops={cfg["pin_tops"]}')

    clips = sorted(
        d for d in os.listdir(args.dataset_root)
        if os.path.isdir(os.path.join(args.dataset_root, d, 'images')))
    calib_by_clip = {c: load_clip_calibs(os.path.join(args.dataset_root, c))
                     for c in clips}

    jobs = []
    for clip in clips:
        img_root = os.path.join(args.dataset_root, clip, 'images')
        # Common timestamps across the 6 used channels, temporally sorted
        per_cam = {}
        for cam in CAMERA_ORDER:
            ch_dir = os.path.join(img_root, NUSCENES_TO_CHANNEL[cam])
            per_cam[cam] = {m.group(1): os.path.join(ch_dir, f)
                            for f in os.listdir(ch_dir)
                            if (m := _TS_RE.search(f)) and f.endswith('.jpg')}
        common = sorted(set.intersection(*(set(v) for v in per_cam.values())),
                        key=int)[::args.stride]
        for cam in CAMERA_ORDER:
            dst_dir = os.path.join(args.out_root, 'samples', cam)
            os.makedirs(dst_dir, exist_ok=True)
            for ts in common:
                dst = os.path.join(dst_dir, f'{clip}__{cam}__{ts}.jpg')
                jobs.append((clip, cam, ts, per_cam[cam][ts], dst))

    print(f'{len(jobs)} images over {len(clips)} clips '
          f'(stride={args.stride}, workers={args.workers})')
    t0 = time.time()
    n_done = 0
    n_new = 0
    with mp.Pool(args.workers, initializer=_init_worker,
                 initargs=(cfg, elevations, calib_by_clip)) as pool:
        for r in pool.imap_unordered(_convert_one, jobs, chunksize=16):
            n_done += 1
            n_new += r
            if n_done % 1000 == 0 or n_done == len(jobs):
                dt = time.time() - t0
                print(f'  [{n_done}/{len(jobs)}] new={n_new} '
                      f'({n_done / dt:.0f} img/s, {dt:.0f}s elapsed)')
    print(f'Done: {n_new} converted, {n_done - n_new} already existed.')
    print(f'Output: {args.out_root}/samples/CAM_*/')


if __name__ == '__main__':
    main()
