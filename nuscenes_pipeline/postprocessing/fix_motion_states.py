"""
Fix incorrect motion state claims in QA answers.

Cross-references bbox references in answers with ground truth velocity data.
If velocity (vx and vy both < 0.1 m/s) -> object is stationary/parked/stopped.
If velocity > 0.1 m/s -> object is in motion/moving.

Corrects claims that contradict the velocity data.

Usage:
    python fix_motion_states.py
"""

import os
import sys
import json
import re
import argparse
import numpy as np
from tqdm import tqdm

from nuscenes_pipeline.core.nuscenes_data_loader import NuScenesDataLoader
from nuscenes_pipeline.core.nuscenes_prompt_generator import get_bbox_2d_projection
from nuscenes_pipeline.core.qa_utils import SceneAnalyzer

CAMERA_ORDER = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT',
]
CAM_IDX = {c: i + 1 for i, c in enumerate(CAMERA_ORDER)}

VELOCITY_THRESHOLD = 0.1  # m/s — below this is stationary

# Phrase replacements: motion -> stationary
MOTION_TO_STATIONARY = {
    'actively riding': 'stationary',
    'actively moving': 'stationary',
    'actively driving': 'stationary',
    'in motion': 'stationary',
    'is moving': 'is stationary',
    'are moving': 'are stationary',
    'is riding': 'is stopped',
    'is driving': 'is stopped',
    'is traveling': 'is stopped',
    'is approaching': 'is stationary near',
    'appears to be moving': 'appears to be stationary',
    'appears to be driving': 'appears to be parked',
    'also in motion': 'also stationary',
    'and in motion': 'and stationary',
}

# Phrase replacements: stationary -> motion
STATIONARY_TO_MOTION = {
    'pulled over': 'in motion on the road',
    'is parked': 'is moving',
    'are parked': 'are moving',
    'is stopped': 'is moving',
    'is stationary': 'is moving',
    'are stationary': 'are moving',
    'is stalled': 'is moving',
    'is idle': 'is moving',
    'is waiting': 'is moving',
    'at rest': 'in motion',
    'standing still': 'moving',
    'not moving': 'in motion',
    'appears parked': 'appears to be moving',
    'appears stopped': 'appears to be moving',
}


def build_bbox_to_velocity(sample, scene_data, resize_factor=2):
    """Build mapping: (img_num, x1, y1, x2, y2) -> velocity info."""
    mapping = {}
    for seq_idx, obj in enumerate(scene_data['objects'], 1):
        raw_idx = obj['index']
        raw_bbox = sample.gt_boxes[raw_idx]
        vel = obj.get('velocity_ego', [0, 0])
        speed = np.sqrt(vel[0] ** 2 + vel[1] ** 2)

        for cam_name in obj['visible_cameras']:
            cam_data = sample.cameras[cam_name]
            bbox_2d = get_bbox_2d_projection(raw_bbox, cam_data, resize_factor)
            if bbox_2d:
                img_num = CAM_IDX[cam_name]
                mapping[(img_num, bbox_2d[0], bbox_2d[1], bbox_2d[2], bbox_2d[3])] = {
                    'speed': speed,
                    'is_stationary': speed < VELOCITY_THRESHOLD,
                }
    return mapping


def build_image_index(loader):
    """Pre-build index: front camera basename -> sample_idx."""
    index = {}
    for idx in range(len(loader.infos)):
        cam_path = loader.infos[idx]['cams']['CAM_FRONT']['data_path']
        index[os.path.basename(cam_path)] = idx
    return index


def fix_motion_in_text(text, bbox_vel_map):
    """
    Fix motion state claims in text based on velocity data.
    Returns (fixed_text, num_corrections).
    """
    bbox_pattern = re.compile(
        r'(\w[\w_]*)\s*\(Image\s+(\d+)\s*\([^)]*\)\s*bbox\[(\d+),(\d+),(\d+),(\d+)\]\)'
    )

    # Collect all bbox refs with their velocity info
    bbox_refs = []
    for match in bbox_pattern.finditer(text):
        img_num = int(match.group(2))
        x1, y1, x2, y2 = int(match.group(3)), int(match.group(4)), int(match.group(5)), int(match.group(6))
        key = (img_num, x1, y1, x2, y2)
        vel_info = bbox_vel_map.get(key)
        if vel_info:
            bbox_refs.append((match.start(), match.end(), vel_info))

    if not bbox_refs:
        return text, 0

    corrections = 0

    # Process each bbox ref — apply corrections in context window
    # Work backwards to preserve string positions
    for ref_start, ref_end, vel_info in reversed(bbox_refs):
        is_stationary = vel_info['is_stationary']

        ctx_start = max(0, ref_start - 150)
        ctx_end = min(len(text), ref_end + 150)
        segment = text[ctx_start:ctx_end]

        if is_stationary:
            # Object is stationary — fix "moving" claims
            for phrase, replacement in MOTION_TO_STATIONARY.items():
                pat = re.compile(re.escape(phrase), re.IGNORECASE)
                if pat.search(segment):
                    segment = pat.sub(replacement, segment)
                    corrections += 1
        else:
            # Object is moving — fix "parked/stopped" claims
            for phrase, replacement in STATIONARY_TO_MOTION.items():
                pat = re.compile(re.escape(phrase), re.IGNORECASE)
                if pat.search(segment):
                    segment = pat.sub(replacement, segment)
                    corrections += 1

        text = text[:ctx_start] + segment + text[ctx_end:]

    return text, corrections


def main():
    parser = argparse.ArgumentParser(description="Fix motion state claims against ground truth velocity")
    parser.add_argument('--data_dir', type=str, default='sft_dataset',
                        help='Directory containing SFT JSON files to fix')
    parser.add_argument('--pkl_path', type=str,
                        default=os.environ.get("NUSCENES_PKL_PATH", "nuscenes2d_ego_temporal_infos_val.pkl"))
    parser.add_argument('--data_root', type=str,
                        default=os.environ.get("NUSCENES_DATA_ROOT", "/home/yongjinjeon/datasets/nuscenes"))
    args = parser.parse_args()

    data_dir = args.data_dir

    print("Loading nuScenes data...")
    loader = NuScenesDataLoader(
        pkl_path=args.pkl_path,
        data_root=args.data_root,
    )
    analyzer = SceneAnalyzer(loader)

    print("Building image index...")
    img_index = build_image_index(loader)

    for fname in ['sft_train_no_objlist.json', 'sft_val_no_objlist.json']:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            print(f"  {fname}: not found, skipping")
            continue

        print(f"\nProcessing {fname}...")
        with open(fpath) as f:
            data = json.load(f)

        total_corrections = 0
        samples_fixed = 0
        cache = {}

        for s in tqdm(data, desc=f"  Fixing"):
            gpt = s['conversations'][2]['value']

            # Quick check
            gpt_lower = gpt.lower()
            has_claim = any(w in gpt_lower for w in [
                'motion', 'moving', 'riding', 'driving', 'traveling', 'approaching',
                'parked', 'stopped', 'stationary', 'stalled', 'idle', 'pulled over',
            ])
            if not has_claim:
                continue

            front_basename = os.path.basename(s['image'][1])
            sample_idx = img_index.get(front_basename)
            if sample_idx is None:
                continue

            if sample_idx not in cache:
                try:
                    sample = loader.get_sample(sample_idx)
                    scene = analyzer.analyze_sample(
                        sample_idx, max_distance=50.0, rear_filter_distance=20.0
                    )
                    cache[sample_idx] = build_bbox_to_velocity(sample, scene)
                except:
                    cache[sample_idx] = {}

            bbox_vel_map = cache[sample_idx]
            if not bbox_vel_map:
                continue

            fixed_text, n = fix_motion_in_text(gpt, bbox_vel_map)
            if n > 0:
                s['conversations'][2]['value'] = fixed_text
                total_corrections += n
                samples_fixed += 1

        with open(fpath, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"  {fname}: {len(data)} samples, {samples_fixed} fixed, {total_corrections} corrections")

    print("\nDone!")


if __name__ == '__main__':
    main()
