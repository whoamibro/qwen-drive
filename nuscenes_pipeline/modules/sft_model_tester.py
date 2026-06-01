"""
Qualitative Inference Test for SFT-trained Qwen3-VL LoRA Model

Loads the base model + LoRA adapter, runs inference on nuScenes samples
with the same prompt structure used during training (no-objlist variant),
and saves results in the same format as sft_train_no_objlist.json.

Output format (per sample):
{
    "image": [6 image paths],
    "conversations": [
        {"from": "system", "value": "..."},
        {"from": "human", "value": "..."},
        {"from": "gpt", "value": "<model prediction>"},
        {"from": "gt", "value": "<ground truth if available>"}
    ]
}

Usage (from qwen-drive project root):
    # Test on specific samples
    python -m nuscenes_pipeline.modules.sft_model_tester --sample_indices 0 10 20

    # Test with ground truth from val set
    python -m nuscenes_pipeline.modules.sft_model_tester --from_val --val_indices 0 1 2

    # Test a range of samples with a custom question
    python -m nuscenes_pipeline.modules.sft_model_tester \
        --start_idx 0 --end_idx 10 \
        --question "Are there any pedestrians on the sidewalk?"
"""

import os
import sys
import json
import glob
import re
import argparse
import torch
from datetime import datetime
from PIL import Image

from nuscenes_pipeline.core.nuscenes_data_loader import NuScenesDataLoader
from nuscenes_pipeline.modules.sft_prompt_builder import (
    build_system_prompt_no_objects,
    build_user_prompt_no_objects,
    extract_ego_state,
    CAMERA_ORDER,
    REAR_CAMERAS,
)

# Indices of rear cameras in the saved SFT sample's "image" list (0-indexed).
# Order matches CAMERA_ORDER: FL(0), F(1), FR(2), BL(3), B(4), BR(5).
REAR_IMAGE_INDICES = {3, 4, 5}

CATEGORY_SUBSET_RE = re.compile(r"sft_val_qwen3vl_([A-Z]+)(?:_subset)?\.json$")


CAM_LABELS = {
    'CAM_FRONT_LEFT':  'Image 1: Front-left Camera',
    'CAM_FRONT':       'Image 2: Front Camera',
    'CAM_FRONT_RIGHT': 'Image 3: Front-right Camera',
    'CAM_BACK_LEFT':   'Image 4: Rear-left Camera',
    'CAM_BACK':        'Image 5: Rear Camera',
    'CAM_BACK_RIGHT':  'Image 6: Rear-right Camera',
}


def load_model(base_model_path: str, lora_path: str):
    """Load base Qwen3-VL model with LoRA adapter merged."""
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel

    print(f"Loading base model: {base_model_path}")
    model = AutoModelForImageTextToText.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    print(f"Loading LoRA adapter: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)
    model = model.merge_and_unload()
    model.eval()
    print("Model loaded and LoRA merged.")

    processor = AutoProcessor.from_pretrained(base_model_path, use_fast=True)
    return model, processor


def get_image_paths(sample):
    """Return the 6 camera image paths in CAMERA_ORDER."""
    return [sample.cameras[cam].image_path for cam in CAMERA_ORDER]


def prepare_messages(sample, loader, question, resize_factor=2):
    """
    Prepare messages in Qwen3-VL chat format (no-objlist variant).
    Returns (messages_for_inference, system_text, user_text_for_saving).
    """
    ego = extract_ego_state(sample, loader)
    system_prompt = build_system_prompt_no_objects()
    user_prompt_text = build_user_prompt_no_objects(ego, question)

    # Build user content with PIL images for the processor
    user_content = []
    for cam_name in CAMERA_ORDER:
        user_content.append({"type": "text", "text": f"=== {CAM_LABELS[cam_name]} ==="})

        img_path = sample.cameras[cam_name].image_path
        flip = cam_name in REAR_CAMERAS
        img = Image.open(img_path).convert('RGB')

        if resize_factor > 1:
            w, h = img.size
            img = img.resize((w // resize_factor, h // resize_factor), Image.LANCZOS)
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        user_content.append({"type": "image", "image": img})

    # Keep only the text part after the image interleave (ego status + question)
    if 'Given the six surround-view' in user_prompt_text:
        text_part = user_prompt_text[user_prompt_text.index('Given the six surround-view'):]
    else:
        text_part = user_prompt_text

    user_content.append({"type": "text", "text": text_part})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content},
    ]
    return messages, system_prompt, user_prompt_text


def run_inference(model, processor, messages, max_new_tokens=2048):
    """Run greedy inference and return the generated text."""
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def build_test_cases(args, loader, img_index):
    """Assemble the list of test cases from CLI arguments."""
    test_cases = []

    if args.from_val:
        print(f"Loading val set: {args.val_path}")
        with open(args.val_path) as f:
            val_data = json.load(f)

        indices = args.val_indices if args.val_indices else list(range(min(10, len(val_data))))
        for vi in indices:
            if vi >= len(val_data):
                continue
            v = val_data[vi]
            user_msg = v['conversations'][1]['value']
            gt_answer = v['conversations'][2]['value']
            question = user_msg.split('TASK:\n')[1].strip() if 'TASK:\n' in user_msg else ''

            front_basename = os.path.basename(v['image'][1])
            sample_idx = img_index.get(front_basename)

            if sample_idx is not None:
                test_cases.append({
                    'val_idx': vi,
                    'sample_idx': sample_idx,
                    'question': args.question if args.question else question,
                    'gt_answer': gt_answer,
                    'image_paths': v['image'],
                })
    else:
        if args.sample_indices:
            indices = args.sample_indices
        elif args.start_idx is not None and args.end_idx is not None:
            indices = list(range(args.start_idx, args.end_idx + 1))
        else:
            indices = [0, 10, 50, 100, 500]

        default_q = args.question or (
            "Describe the driving scene. What objects are visible and what is the ego vehicle doing?"
        )

        for idx in indices:
            sample = loader.get_sample(idx)
            test_cases.append({
                'sample_idx': idx,
                'question': default_q,
                'gt_answer': None,
                'image_paths': get_image_paths(sample),
            })

    return test_cases


# ---------------------------------------------------------------------------
# Per-category generation eval (curriculum stage-end)
# ---------------------------------------------------------------------------
def _extract_answer(value: str):
    """Pull the `answer` field out of the Qwen3-VL unified-JSON envelope."""
    if not isinstance(value, str):
        return None
    try:
        obj = json.loads(value)
        if isinstance(obj, dict) and "answer" in obj:
            return str(obj["answer"])
    except (json.JSONDecodeError, ValueError):
        pass
    # Fallback: regex hunt — models sometimes drop trailing braces.
    m = re.search(r'"answer"\s*:\s*"([^"]*)"', value, re.DOTALL)
    if m:
        return m.group(1)
    return None


def _normalize_answer(ans):
    """Normalise for exact-match: trim, upper-case + sort MCQ letter sets."""
    if not ans:
        return ""
    s = str(ans).strip().rstrip(".").strip()
    parts = [p.strip().upper() for p in s.split(",") if p.strip()]
    if parts and all(len(p) == 1 and p.isalpha() for p in parts):
        return ",".join(sorted(parts))
    return s.lower()


def _build_messages_from_sft_sample(sample, resize_factor: int = 1):
    """Reconstruct (system, user) messages from a saved SFT val sample."""
    img_paths = sample["image"]
    convs = sample["conversations"]
    system_text = convs[0]["value"]
    human_text = convs[1]["value"]

    pil_images = []
    for i, p in enumerate(img_paths):
        img = Image.open(p).convert("RGB")
        if resize_factor > 1:
            w, h = img.size
            img = img.resize((w // resize_factor, h // resize_factor), Image.LANCZOS)
        if i in REAR_IMAGE_INDICES:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        pil_images.append(img)

    parts = human_text.split("<image>")
    if len(parts) - 1 != len(pil_images):
        raise ValueError(
            f"Image-placeholder count mismatch: {len(parts)-1} <image> tokens vs "
            f"{len(pil_images)} images in sample"
        )

    user_content = []
    for i, part in enumerate(parts):
        if part:
            user_content.append({"type": "text", "text": part})
        if i < len(pil_images):
            user_content.append({"type": "image", "image": pil_images[i]})

    return [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": user_content},
    ]


def _discover_category_subsets(eval_dir: str):
    """Return dict {CAT: path} for all sft_val_qwen3vl_{CAT}(_subset)?.json in dir."""
    out = {}
    for p in sorted(glob.glob(os.path.join(eval_dir, "sft_val_qwen3vl_*.json"))):
        m = CATEGORY_SUBSET_RE.search(os.path.basename(p))
        if m:
            out[m.group(1)] = p
    return out


def run_per_category_eval(args):
    """Evaluate a LoRA checkpoint per category and write eval_report.json."""
    model, processor = load_model(args.base_model, args.lora_path)

    cat_paths = _discover_category_subsets(args.per_category_eval_dir)
    if not cat_paths:
        raise FileNotFoundError(
            f"No sft_val_qwen3vl_*.json files in {args.per_category_eval_dir}"
        )
    print(f"Per-category eval over {len(cat_paths)} categories: {sorted(cat_paths)}")

    report = {
        "lora_path": os.path.abspath(args.lora_path),
        "eval_dir": os.path.abspath(args.per_category_eval_dir),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "max_new_tokens": args.max_new_tokens,
        "resize_factor": args.resize_factor,
        "metrics": {},
    }

    cat_predictions = {}

    for cat in sorted(cat_paths):
        path = cat_paths[cat]
        with open(path) as f:
            data = json.load(f)
        if args.n_per_cat_limit and args.n_per_cat_limit < len(data):
            data = data[: args.n_per_cat_limit]

        n_total = len(data)
        n_correct = 0
        n_parsed = 0
        per_sample = []

        print(f"\n[{cat}] {n_total} samples")
        for i, sample in enumerate(data):
            try:
                messages = _build_messages_from_sft_sample(sample, args.resize_factor)
                pred_text = run_inference(model, processor, messages, args.max_new_tokens)
            except Exception as e:
                print(f"  sample {i}: inference error: {e}")
                per_sample.append({"i": i, "error": str(e)})
                continue

            gt_text = sample["conversations"][2]["value"]
            pred_ans = _extract_answer(pred_text)
            gt_ans = _extract_answer(gt_text)

            if pred_ans is not None:
                n_parsed += 1
            is_correct = (
                pred_ans is not None
                and gt_ans is not None
                and _normalize_answer(pred_ans) == _normalize_answer(gt_ans)
            )
            if is_correct:
                n_correct += 1

            per_sample.append({
                "i": i,
                "pred_answer": pred_ans,
                "gt_answer": gt_ans,
                "correct": is_correct,
            })
            if (i + 1) % 25 == 0 or i == n_total - 1:
                running = n_correct / max(i + 1, 1)
                print(f"  [{i+1}/{n_total}] running acc={running:.3f}")

        acc = n_correct / max(n_total, 1)
        parse_rate = n_parsed / max(n_total, 1)
        report["metrics"][cat] = {
            "n": n_total,
            "n_correct": n_correct,
            "acc": acc,
            "parse_rate": parse_rate,
        }
        cat_predictions[cat] = per_sample
        print(f"[{cat}] done: acc={acc:.3f} parse_rate={parse_rate:.3f}")

    macro = (
        sum(m["acc"] for m in report["metrics"].values()) / len(report["metrics"])
        if report["metrics"] else 0.0
    )
    report["macro_acc"] = macro

    out_path = os.path.join(args.lora_path, "eval_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    if args.save_predictions:
        preds_path = os.path.join(args.lora_path, "eval_predictions.json")
        with open(preds_path, "w") as f:
            json.dump(cat_predictions, f, indent=2)
        print(f"Per-sample predictions saved to: {preds_path}")

    print(f"\nEval report saved to: {out_path}")
    print(f"Macro accuracy: {macro:.3f}")
    for cat, m in report["metrics"].items():
        print(f"  {cat:>3s}  acc={m['acc']:.3f}  (n={m['n']}, parsed={m['parse_rate']:.2f})")


def main():
    parser = argparse.ArgumentParser(description="Qualitative test for SFT-trained Qwen3-VL")

    # Model paths
    parser.add_argument('--base_model', type=str,
                        default='ckpts/qwen3_vl_8b_instruct')
    parser.add_argument('--lora_path', type=str,
                        default='output_nuscenes_lora_no_objlist_cleansing_qads/checkpoint-5500')
    parser.add_argument('--pkl_path', type=str,
                        default='/home/yongjinjeon/datasets/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl')

    # Sample selection
    parser.add_argument('--sample_indices', type=int, nargs='+', default=None)
    parser.add_argument('--start_idx', type=int, default=None)
    parser.add_argument('--end_idx', type=int, default=None)

    # Val set GT comparison
    parser.add_argument('--from_val', action='store_true')
    parser.add_argument('--val_path', type=str,
                        default='/home/yongjinjeon/workspace/qa_dataset/qwen3vl_8b_sft_dataset/sft_val_no_objlist.json')
    parser.add_argument('--val_indices', type=int, nargs='+', default=None)

    # Custom question
    parser.add_argument('--question', type=str, default=None)

    # Inference options
    parser.add_argument('--resize_factor', type=int, default=2)
    parser.add_argument('--max_new_tokens', type=int, default=2048)

    # Output
    parser.add_argument('--output_dir', type=str,
                        default='/home/yongjinjeon/workspace/qa_dataset/qwen3vl_8b_sft_dataset')
    parser.add_argument('--output_name', type=str, default=None)

    # Per-category eval mode (curriculum stage-end)
    parser.add_argument('--per_category_eval_dir', type=str, default=None,
                        help="Directory containing sft_val_qwen3vl_{CAT}(_subset)?.json files. "
                             "When set, run per-category accuracy eval and write "
                             "eval_report.json into --lora_path; the legacy qualitative-test "
                             "code path is skipped.")
    parser.add_argument('--n_per_cat_limit', type=int, default=None,
                        help="Cap samples per category (debug/smoke).")
    parser.add_argument('--save_predictions', action='store_true',
                        help="Also dump per-sample predictions to eval_predictions.json.")

    args = parser.parse_args()

    # Per-category eval branch — skips the legacy NuScenesDataLoader path.
    if args.per_category_eval_dir:
        run_per_category_eval(args)
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model and nuScenes
    model, processor = load_model(args.base_model, args.lora_path)
    print("Loading nuScenes data...")
    loader = NuScenesDataLoader(pkl_path=args.pkl_path)

    # Build basename → sample_idx index for val set mapping
    img_index = {}
    for idx in range(len(loader.infos)):
        cam_path = loader.infos[idx]['cams']['CAM_FRONT']['data_path']
        img_index[os.path.basename(cam_path)] = idx

    test_cases = build_test_cases(args, loader, img_index)
    print(f"\nRunning inference on {len(test_cases)} test cases")
    print("=" * 80)

    results = []
    for i, tc in enumerate(test_cases):
        sample_idx = tc['sample_idx']
        question = tc['question']

        print(f"\n[{i+1}/{len(test_cases)}] Sample {sample_idx}")
        print(f"  Q: {question[:80]}")

        try:
            sample = loader.get_sample(sample_idx)
            messages, system_text, user_text = prepare_messages(
                sample, loader, question, args.resize_factor
            )
            prediction = run_inference(model, processor, messages, args.max_new_tokens)

            print(f"  Pred: {prediction[:150]}")
            if tc.get('gt_answer'):
                print(f"  GT:   {tc['gt_answer'][:150]}")

            conversations = [
                {"from": "system", "value": system_text},
                {"from": "human", "value": user_text},
                {"from": "gpt", "value": prediction},
            ]
            if tc.get('gt_answer'):
                conversations.append({"from": "gt", "value": tc['gt_answer']})

            results.append({
                "image": tc['image_paths'],
                "conversations": conversations,
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    out_name = args.output_name or f"sft_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path = os.path.join(args.output_dir, out_name)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 80}")
    print(f"Results saved to: {out_path}")
    print(f"Total: {len(results)} samples")
    print(f"Format matches sft_train_no_objlist.json")
    print(f"  conversations[0] = system | [1] = human | [2] = gpt (prediction)")
    if any(tc.get('gt_answer') for tc in test_cases):
        print(f"  conversations[3] = gt (ground truth)")


if __name__ == '__main__':
    main()
