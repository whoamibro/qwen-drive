"""
Transform OBJ ID references in QA results to 2D bbox descriptions.

Reads original QA data from outputs_qa_vllm/Observation/ (with OBJ IDs),
replaces OBJ references with bbox-based descriptions, and saves to
qwen3vl_8b_sft_dataset/Observation/.

Example transformations:
  "OBJ 30" -> "pedestrian (Image 2 bbox[788,234,799,314])"
  "OBJ 5 (pedestrian, 46.6m ahead)" -> "pedestrian (46.6m ahead, Image 1 bbox[691,219,712,247])"
  "OBJ 2, OBJ 4, OBJ 5" -> "car (Image 4), car (Image 5), pedestrian (Image 1)"

Usage:
    python transform_obj_to_bbox.py
"""

import os
import sys
import json
import re
import glob
import argparse
from tqdm import tqdm

from nuscenes_pipeline.core.nuscenes_data_loader import NuScenesDataLoader
from nuscenes_pipeline.core.nuscenes_prompt_generator import get_bbox_2d_projection
from nuscenes_pipeline.core.qa_utils import SceneAnalyzer

CAMERA_ORDER = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT',
]
CAM_IDX = {c: i + 1 for i, c in enumerate(CAMERA_ORDER)}
CAM_DISPLAY = {
    'CAM_FRONT_LEFT': 'Front-left', 'CAM_FRONT': 'Front',
    'CAM_FRONT_RIGHT': 'Front-right', 'CAM_BACK_LEFT': 'Rear-left',
    'CAM_BACK': 'Rear', 'CAM_BACK_RIGHT': 'Rear-right',
}


def build_obj_mapping(sample, scene_data, resize_factor=2):
    """
    Build mapping: OBJ ID (1-based) -> bbox description string.

    Returns dict like:
        {1: "car (Image 4 bbox[0,221,108,346])",
         5: "pedestrian (Image 1 bbox[691,219,712,247])", ...}
    """
    objects = scene_data['objects']
    mapping = {}

    for seq_idx, obj in enumerate(objects, 1):
        category = obj['category']
        raw_idx = obj['index']
        raw_bbox = sample.gt_boxes[raw_idx]

        # Get primary camera (first visible) and its 2D bbox
        best_cam = None
        best_bbox = None
        for cam_name in obj['visible_cameras']:
            cam_data = sample.cameras[cam_name]
            bbox_2d = get_bbox_2d_projection(raw_bbox, cam_data, resize_factor)
            if bbox_2d:
                best_cam = cam_name
                best_bbox = bbox_2d
                break  # Use first visible camera with valid projection

        if best_cam and best_bbox:
            x1, y1, x2, y2 = best_bbox
            img_num = CAM_IDX[best_cam]
            cam_label = CAM_DISPLAY[best_cam]
            desc = f"{category} (Image {img_num} ({cam_label}) bbox[{x1},{y1},{x2},{y2}])"
        else:
            # Fallback: no valid 2D projection
            cams = obj['visible_cameras']
            if cams:
                img_num = CAM_IDX[cams[0]]
                desc = f"{category} (Image {img_num})"
            else:
                desc = category

        mapping[seq_idx] = desc

    return mapping


def replace_obj_refs(text, obj_mapping):
    """
    Replace all OBJ references in text with bbox descriptions.

    Handles patterns:
      - "OBJ 30" -> "pedestrian (Image 2 bbox[...])"
      - "OBJ 30 (pedestrian, 19.6m ahead...)" -> "pedestrian (19.6m ahead, Image 2 bbox[...])"
      - "OBJs 3, 8, 11" -> "traffic_cone (Image 2 bbox[...]), car (Image 5 bbox[...]), ..."
    """
    if not obj_mapping:
        return text

    # Step 1: "OBJ N (description)" -> bbox desc (handles parenthetical context)
    def replace_obj_with_parens(match):
        obj_id = int(match.group(1))
        if obj_id in obj_mapping:
            return obj_mapping[obj_id]
        return match.group(0)

    text = re.sub(r'OBJ\s*(\d+)\s*\([^)]*\)', replace_obj_with_parens, text)

    # Step 2: "OBJs N, M, K" (plural prefix with bare number list)
    def replace_objs_plural(match):
        nums_str = match.group(1)
        nums = re.findall(r'\d+', nums_str)
        descs = []
        for n in nums:
            obj_id = int(n)
            if obj_id in obj_mapping:
                descs.append(obj_mapping[obj_id])
        return ", ".join(descs) + " " if descs else match.group(0)

    text = re.sub(r'OBJs\s+((?:\d+\s*(?:,\s*)?)+)', replace_objs_plural, text)

    # Step 3: All remaining standalone "OBJ N" (one at a time)
    def replace_standalone(match):
        obj_id = int(match.group(1))
        if obj_id in obj_mapping:
            return obj_mapping[obj_id]
        return match.group(0)

    text = re.sub(r'OBJ\s+(\d+)', replace_standalone, text)

    # Clean up artifacts
    text = re.sub(r'\s{2,}', ' ', text)
    text = text.strip()

    return text


def transform_qa_data(qa_data, obj_mapping):
    """Transform all OBJ references in a qa_results structure."""
    for result in qa_data.get('qa_results', []):
        for pair in result.get('pairs', []):
            # Positive
            pos = pair.get('positive')
            if pos:
                if pos.get('answer'):
                    pos['answer'] = replace_obj_refs(str(pos['answer']), obj_mapping)
                if pos.get('reasoning'):
                    pos['reasoning'] = replace_obj_refs(pos['reasoning'], obj_mapping)

            # Contrastive
            cont = pair.get('contrastive')
            if cont:
                if cont.get('answer'):
                    cont['answer'] = replace_obj_refs(str(cont['answer']), obj_mapping)
                if cont.get('reasoning'):
                    cont['reasoning'] = replace_obj_refs(cont['reasoning'], obj_mapping)

            # VLM-proposed contrastives
            for vc in pair.get('vlm_proposed_contrastives', []):
                if vc:
                    if vc.get('answer'):
                        vc['answer'] = replace_obj_refs(str(vc['answer']), obj_mapping)
                    if vc.get('reasoning'):
                        vc['reasoning'] = replace_obj_refs(vc['reasoning'], obj_mapping)

    return qa_data


def main():
    parser = argparse.ArgumentParser(description="Transform OBJ ID references to 2D bbox descriptions")
    parser.add_argument('--input_dir', type=str, default='qa_results/Observation',
                        help='Input directory with QA results containing OBJ IDs')
    parser.add_argument('--output_dir', type=str, default='sft_dataset/Observation',
                        help='Output directory for bbox-transformed QA results')
    parser.add_argument('--pkl_path', type=str,
                        default=os.environ.get("NUSCENES_PKL_PATH", "nuscenes2d_ego_temporal_infos_val.pkl"))
    parser.add_argument('--data_root', type=str,
                        default=os.environ.get("NUSCENES_DATA_ROOT", "/home/yongjinjeon/datasets/nuscenes"))
    parser.add_argument('--resize_factor', type=int, default=2)
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("Loading nuScenes data...")
    loader = NuScenesDataLoader(
        pkl_path=args.pkl_path,
        data_root=args.data_root,
    )
    analyzer = SceneAnalyzer(loader)

    input_files = sorted(glob.glob(os.path.join(input_dir, 'sample_*_qa_results.json')))
    print(f"Found {len(input_files)} QA results files")

    total_transformed = 0
    total_obj_refs = 0

    for fpath in tqdm(input_files, desc="Transforming"):
        with open(fpath) as f:
            qa_data = json.load(f)

        sample_idx = qa_data['sample_idx']

        # Build OBJ ID -> bbox mapping for this sample
        try:
            sample = loader.get_sample(sample_idx)
            scene_data = analyzer.analyze_sample(
                sample_idx, max_distance=50.0, rear_filter_distance=20.0
            )
            obj_mapping = build_obj_mapping(sample, scene_data, resize_factor=2)
        except Exception as e:
            print(f"  Warning: Failed to build mapping for sample {sample_idx}: {e}")
            obj_mapping = {}

        # Count OBJ refs before
        raw_text = json.dumps(qa_data)
        refs_before = len(re.findall(r'OBJ\s*\d+', raw_text))

        # Transform
        qa_data = transform_qa_data(qa_data, obj_mapping)

        # Count OBJ refs after
        transformed_text = json.dumps(qa_data)
        refs_after = len(re.findall(r'OBJ\s*\d+', transformed_text))

        if refs_before > 0:
            total_transformed += 1
            total_obj_refs += refs_before - refs_after

        # Save to output dir (both qa_results and detailed)
        out_path = os.path.join(output_dir, os.path.basename(fpath))
        with open(out_path, 'w') as f:
            json.dump(qa_data, f, indent=4, ensure_ascii=False)

        # Also copy detailed file if exists
        detailed_name = os.path.basename(fpath).replace('_qa_results.json', '_qa_detailed.json')
        detailed_in = os.path.join(input_dir, detailed_name)
        detailed_out = os.path.join(output_dir, detailed_name)
        if os.path.exists(detailed_in):
            with open(detailed_in) as f:
                detailed_data = json.load(f)
            detailed_data = transform_qa_data(detailed_data, obj_mapping)
            with open(detailed_out, 'w') as f:
                json.dump(detailed_data, f, indent=4, ensure_ascii=False)

    print(f"\nDone!")
    print(f"  Files with OBJ refs transformed: {total_transformed}")
    print(f"  Total OBJ references replaced: {total_obj_refs}")
    print(f"  Output: {output_dir}")


if __name__ == '__main__':
    main()
