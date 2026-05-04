"""
Prepare nuScenes QA dataset for Qwen3-VL SFT — NO OBJECT LIST variant.

Uses OBJ-removed QA data from qwen3vl_8b_sft_dataset/ directory.
Prompts contain only 6 camera images + ego driving status (no 3D object list).
The model must rely on visual perception from images to answer questions.

Usage:
    python -m nuscenes_pipeline.postprocessing.prepare_sft_dataset \
        --qa_dir sft_dataset/Observation \
        --val_size 40000
"""

import os
import sys
import json
import glob
import pickle
import argparse
import random
import numpy as np
from tqdm import tqdm

from nuscenes_pipeline.core.nuscenes_data_loader import NuScenesDataLoader
from nuscenes_pipeline.modules.sft_prompt_builder import (
    CAMERA_ORDER,
    build_sft_conversations_no_objects,
)


def resolve_image_path(data_root: str, relative_path: str) -> str:
    """Resolve nuScenes image path from the pickle's relative path."""
    if relative_path.startswith('./'):
        relative_path = relative_path[2:]
    if relative_path.startswith('data/nuscenes/'):
        relative_path = relative_path[len('data/nuscenes/'):]
    return os.path.join(data_root, relative_path)


def get_image_paths_for_sample(infos: list, idx: int, data_root: str) -> list:
    """Get the 6 camera image paths for a given sample index."""
    cams = infos[idx]['cams']
    return [resolve_image_path(data_root, cams[cam]['data_path']) for cam in CAMERA_ORDER]


def process_qa_file(
    qa_path: str,
    loader: NuScenesDataLoader,
    data_root: str,
    infos: list,
    no_contrast: bool = False,
) -> list:
    """
    Process a single *_qa_results.json file (OBJ-removed) into SFT samples.
    Prompts contain NO object list — only images + ego status.
    """
    with open(qa_path, 'r') as f:
        qa_data = json.load(f)

    sample_idx = qa_data['sample_idx']

    try:
        sample = loader.get_sample(sample_idx)
    except Exception as e:
        print(f"  Warning: Failed to load sample {sample_idx}: {e}")
        return []

    # Get and verify image paths
    image_paths = get_image_paths_for_sample(infos, sample_idx, data_root)
    for p in image_paths:
        if not os.path.exists(p):
            print(f"  Warning: Image not found: {p} (sample {sample_idx})")
            return []

    sft_samples = []
    seen_qa = set()  # Deduplicate (question + answer) within this sample

    for qa_result in qa_data.get('qa_results', []):
        answer_type = qa_result.get('answer_type')

        for pair in qa_result.get('pairs', []):
            # Positive QA
            pos = pair.get('positive')
            if pos and pos.get('instantiated_question') and pos.get('answer'):
                q = pos['instantiated_question']
                a = str(pos['answer'])
                if (q, a) not in seen_qa:
                    seen_qa.add((q, a))
                    conversations = build_sft_conversations_no_objects(
                        sample=sample,
                        loader=loader,
                        question=q,
                        answer=pos['answer'],
                        reasoning=pos.get('reasoning'),
                        answer_type=answer_type,
                        mcq_options=pos.get('mcq_options'),
                    )
                    sft_samples.append({
                        "image": image_paths,
                        "conversations": conversations,
                    })

            if not no_contrast:
                # Contrastive QA
                cont = pair.get('contrastive')
                if cont and cont.get('instantiated_question') and cont.get('answer'):
                    q = cont['instantiated_question']
                    a = str(cont['answer'])
                    if (q, a) not in seen_qa:
                        seen_qa.add((q, a))
                        conversations = build_sft_conversations_no_objects(
                            sample=sample,
                            loader=loader,
                            question=q,
                            answer=cont['answer'],
                            reasoning=cont.get('reasoning'),
                            answer_type=answer_type,
                            mcq_options=cont.get('mcq_options'),
                        )
                        sft_samples.append({
                            "image": image_paths,
                            "conversations": conversations,
                        })

                # VLM-proposed additional contrastives
                for vlm_cont in pair.get('vlm_proposed_contrastives', []):
                    if vlm_cont and vlm_cont.get('instantiated_question') and vlm_cont.get('answer'):
                        q = vlm_cont['instantiated_question']
                        a = str(vlm_cont['answer'])
                        if (q, a) not in seen_qa:
                            seen_qa.add((q, a))
                            conversations = build_sft_conversations_no_objects(
                                sample=sample,
                                loader=loader,
                                question=q,
                                answer=vlm_cont['answer'],
                                reasoning=vlm_cont.get('reasoning'),
                                answer_type=answer_type,
                                mcq_options=vlm_cont.get('mcq_options'),
                            )
                            sft_samples.append({
                                "image": image_paths,
                                "conversations": conversations,
                            })

    return sft_samples


def main():
    parser = argparse.ArgumentParser(description="Prepare nuScenes QA SFT data — NO object list variant")
    parser.add_argument('--qa_dir', type=str,
                        default='sft_dataset/Observation',
                        help='Directory containing OBJ-removed *_qa_results.json files')
    parser.add_argument('--pkl_path', type=str,
                        default=os.environ.get("NUSCENES_PKL_PATH", "nuscenes2d_ego_temporal_infos_val.pkl"))
    parser.add_argument('--data_root', type=str,
                        default=os.environ.get("NUSCENES_DATA_ROOT", "/home/yongjinjeon/datasets/nuscenes"))
    parser.add_argument('--output_dir', type=str,
                        default='sft_dataset',
                        help='Output directory for SFT JSON files')
    parser.add_argument('--resize_factor', type=int, default=2)
    parser.add_argument('--val_size', type=int, default=40000,
                        help='Fixed number of validation samples (rest goes to train)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_contrast', action='store_true',
                        help='Exclude contrastive and VLM-proposed contrastive QA pairs. '
                             'Output files will have _no_contrast suffix.')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Loading nuScenes data loader...")
    loader = NuScenesDataLoader(pkl_path=args.pkl_path, data_root=args.data_root)

    with open(args.pkl_path, 'rb') as f:
        infos = pickle.load(f)['infos']

    qa_files = sorted(glob.glob(os.path.join(args.qa_dir, '*_qa_results.json')))
    print(f"Found {len(qa_files)} QA results files")

    all_samples = []
    for qa_path in tqdm(qa_files, desc="Processing QA files"):
        samples = process_qa_file(qa_path, loader, args.data_root, infos,
                                  no_contrast=args.no_contrast)
        all_samples.extend(samples)

    print(f"\nTotal SFT samples generated: {len(all_samples)}")

    random.shuffle(all_samples)
    val_size = min(args.val_size, len(all_samples) // 5)  # cap at 20% max
    val_samples = all_samples[:val_size]
    train_samples = all_samples[val_size:]

    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples:   {len(val_samples)}")

    os.makedirs(args.output_dir, exist_ok=True)
    suffix = '_no_contrast' if args.no_contrast else ''
    train_path = os.path.join(args.output_dir, f'sft_train_no_objlist{suffix}.json')
    val_path = os.path.join(args.output_dir, f'sft_val_no_objlist{suffix}.json')

    with open(train_path, 'w') as f:
        json.dump(train_samples, f, indent=2, ensure_ascii=False)
    print(f"Saved train data to: {train_path}")

    with open(val_path, 'w') as f:
        json.dump(val_samples, f, indent=2, ensure_ascii=False)
    print(f"Saved val data to: {val_path}")

    # Preview
    if train_samples:
        s = train_samples[0]
        print(f"\n--- Sample Preview ---")
        print(f"Images per sample: {len(s['image'])}")
        print(f"Conversation turns: {len(s['conversations'])}")
        print(f"System prompt: {len(s['conversations'][0]['value'])} chars")
        print(f"User message:  {len(s['conversations'][1]['value'])} chars")
        print(f"GPT response:  {len(s['conversations'][2]['value'])} chars")


if __name__ == '__main__':
    main()
