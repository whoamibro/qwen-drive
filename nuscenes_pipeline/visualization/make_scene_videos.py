"""
Per-scene Video Generator for nuScenes Visualizations

Reads pre-rendered panoramic and BEV PNGs, groups them by scene_token
(parsed from filenames produced by pan_generator.py / bev_generator.py:
    {idx:04d}_{scene_token[:16]}_{sample_token[:16]}_{pan|bev}.png),
and emits one MP4 per scene with panoramic on the left and BEV on the right.

Usage:
    python -m nuscenes_pipeline.visualization.make_scene_videos \\
        --vis_dir pan_vis_results_train \\
        --bev_dir bev_vis_results_train \\
        --output_dir scene_videos_train \\
        --framerate 2

Output: scene_videos_train/scene_<scene_token>.mp4 (one per scene)
"""

import os
import cv2
import argparse
import subprocess
from glob import glob
from collections import defaultdict

import numpy as np


def parse_filename(path):
    """Return (idx, scene_token, sample_token) from a pan/bev PNG filename."""
    base = os.path.basename(path).replace('.png', '')
    parts = base.split('_')
    if len(parts) < 3:
        return None
    return parts[0], parts[1], parts[2]


def index_by_scene(png_dir):
    """Return {scene_token: {idx: path}} from all PNGs under png_dir."""
    out = defaultdict(dict)
    for p in glob(os.path.join(png_dir, '*.png')):
        parsed = parse_filename(p)
        if parsed is None:
            continue
        idx, scene_tok, _ = parsed
        out[scene_tok][idx] = p
    return out


def compute_target_dims(vis_path, bev_path):
    """Match heights for side-by-side; force even dims for H.264."""
    vis = cv2.imread(vis_path)
    bev = cv2.imread(bev_path)
    if vis is None or bev is None:
        return None
    vis_h, vis_w = vis.shape[:2]
    bev_h, bev_w = bev.shape[:2]

    target_h = max(vis_h, bev_h)
    target_h = target_h if target_h % 2 == 0 else target_h + 1
    new_vis_w = int(vis_w * target_h / vis_h)
    new_bev_w = int(bev_w * target_h / bev_h)
    frame_w = new_vis_w + new_bev_w
    frame_w = frame_w if frame_w % 2 == 0 else frame_w + 1
    return target_h, new_vis_w, new_bev_w, frame_w


def render_scene(scene_tok, frames, dims, temp_dir, output_path, framerate):
    """Write one MP4 for a single scene. `frames` is a sorted list of (idx, vis, bev)."""
    target_h, new_vis_w, new_bev_w, frame_w = dims
    os.makedirs(temp_dir, exist_ok=True)

    for i, (_idx, vis_path, bev_path) in enumerate(frames):
        vis = cv2.imread(vis_path)
        bev = cv2.imread(bev_path)
        if vis is None or bev is None:
            print(f"  warn: missing image in scene {scene_tok} idx {_idx}, skipping frame")
            continue
        vis = cv2.resize(vis, (new_vis_w, target_h))
        bev = cv2.resize(bev, (new_bev_w, target_h))
        combined = np.hstack([vis, bev])
        if combined.shape[1] != frame_w:
            padded = np.zeros((target_h, frame_w, 3), dtype=np.uint8)
            padded[:, :combined.shape[1]] = combined
            combined = padded
        cv2.imwrite(os.path.join(temp_dir, f'frame_{i:04d}.png'), combined)

    subprocess.run([
        'ffmpeg', '-y', '-loglevel', 'error',
        '-framerate', str(framerate),
        '-i', os.path.join(temp_dir, 'frame_%04d.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',
        output_path,
    ], check=True)

    for f in glob(os.path.join(temp_dir, '*.png')):
        os.remove(f)


def main():
    parser = argparse.ArgumentParser(description='Per-scene MP4 videos from pan+bev PNGs')
    parser.add_argument('--vis_dir', type=str, required=True,
                        help='Directory of panoramic PNGs (pan_vis_results_train)')
    parser.add_argument('--bev_dir', type=str, required=True,
                        help='Directory of BEV PNGs (bev_vis_results_train)')
    parser.add_argument('--output_dir', type=str, default='scene_videos_train',
                        help='Directory to write scene_<token>.mp4 files into')
    parser.add_argument('--framerate', type=int, default=2,
                        help='Output video framerate (default 2, matches nuScenes annotation rate)')
    parser.add_argument('--limit_scenes', type=int, default=None,
                        help='If set, only render the first N scenes (for testing)')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip scenes whose output MP4 already exists')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    temp_dir = os.path.join(args.output_dir, '_temp_frames')

    vis_by_scene = index_by_scene(args.vis_dir)
    bev_by_scene = index_by_scene(args.bev_dir)

    common_scenes = set(vis_by_scene) & set(bev_by_scene)
    # Order scenes by their earliest sample index so scene_0001 is temporally
    # the first scene in the dataset.
    common_scenes = sorted(common_scenes,
                           key=lambda tok: min(set(vis_by_scene[tok]) & set(bev_by_scene[tok]),
                                               default='9999'))
    print(f"Found {len(common_scenes)} scenes present in both dirs "
          f"(pan only: {len(set(vis_by_scene)-set(bev_by_scene))}, "
          f"bev only: {len(set(bev_by_scene)-set(vis_by_scene))})")

    if args.limit_scenes:
        common_scenes = common_scenes[:args.limit_scenes]

    dims = None
    rendered = 0
    skipped = 0
    for scene_num, scene_tok in enumerate(common_scenes, start=1):
        filename = f'scene_{scene_num:04d}.mp4'
        output_path = os.path.join(args.output_dir, filename)
        if args.skip_existing and os.path.exists(output_path):
            skipped += 1
            continue

        vis_map = vis_by_scene[scene_tok]
        bev_map = bev_by_scene[scene_tok]
        common_idx = sorted(set(vis_map) & set(bev_map))
        if not common_idx:
            print(f"  scene {scene_tok}: no overlapping indices, skipping")
            continue
        frames = [(i, vis_map[i], bev_map[i]) for i in common_idx]

        if dims is None:
            dims = compute_target_dims(frames[0][1], frames[0][2])
            if dims is None:
                print(f"  failed to read first frame, aborting"); return
            print(f"Frame size: {dims[3]}x{dims[0]} (vis {dims[1]} + bev {dims[2]})")

        render_scene(scene_tok, frames, dims, temp_dir, output_path, args.framerate)
        rendered += 1
        print(f"  [{rendered}/{len(common_scenes)-skipped}] {filename} "
              f"({len(frames)} frames)")

    if os.path.isdir(temp_dir):
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass

    print(f"\nDone. Rendered {rendered} scene videos -> {args.output_dir}/  (skipped {skipped})")


if __name__ == '__main__':
    main()
