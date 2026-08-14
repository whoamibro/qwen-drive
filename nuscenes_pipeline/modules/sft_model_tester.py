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

# Curriculum iteration order (matches split_sft_by_category.py:CURRICULUM_ORDER
# and aggregate_curriculum_reports.py:CURRICULUM_ORDER). When the v2 eval
# iterates per-category, it walks categories in this order rather than
# alphabetical, so per-stage logs and merged reports read OBS -> CHR.
CURRICULUM_ORDER = [
    "OBS", "IDN", "AAS", "SRO", "TSS",
    "RML", "DRA", "RWP", "ESC", "CHR",
]


def _order_by_curriculum(cats):
    """Return `cats` (iterable of codes) in curriculum order; unknown
    categories are appended alphabetically at the tail."""
    cat_set = set(cats)
    known = [c for c in CURRICULUM_ORDER if c in cat_set]
    unknown = sorted(cat_set - set(CURRICULUM_ORDER))
    return known + unknown


CAM_LABELS = {
    'CAM_FRONT_LEFT':  'Image 1: Front-left Camera',
    'CAM_FRONT':       'Image 2: Front Camera',
    'CAM_FRONT_RIGHT': 'Image 3: Front-right Camera',
    'CAM_BACK_LEFT':   'Image 4: Rear-left Camera',
    'CAM_BACK':        'Image 5: Rear Camera',
    'CAM_BACK_RIGHT':  'Image 6: Rear-right Camera',
}


def load_model(base_model_path: str, lora_path: str = None):
    """Load base Qwen3-VL model, with the LoRA adapter merged when given.

    A lora_path of None, "", or "none" (case-insensitive) loads the base
    model only.
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"Loading base model: {base_model_path}")
    model = AutoModelForImageTextToText.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    if lora_path and lora_path.lower() != "none":
        from peft import PeftModel
        print(f"Loading LoRA adapter: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
        print("Model loaded and LoRA merged.")
    else:
        print("No LoRA adapter — using the base model as-is.")
    model.eval()

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
    """Run greedy inference and return the generated text (skip_special_tokens).

    Kept for backwards compatibility with the qualitative-test code path.
    The v2 per-category eval should use `run_inference_with_ids()` and the
    token-id-based grounding extractor so box delimiters aren't stripped.
    """
    text, _ = run_inference_with_ids(model, processor, messages, max_new_tokens)
    return text


def run_inference_with_ids(model, processor, messages, max_new_tokens=2048):
    """Greedy inference returning BOTH a human-readable text decode (special
    tokens skipped) AND the raw generated token-id list.

    The token-id list is essential for grounding extraction: Qwen3-VL emits
    boxes inside `<|box_start|> ... <|box_end|>` special-token brackets, and
    a `skip_special_tokens=True` decode strips those brackets — leaving the
    regex-based text parser unable to find any boxes (P0-1).
    """
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
    pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # Return as a plain Python list for downstream tokenizer-based parsing.
    pred_token_ids = generated_ids[0].tolist()
    return pred_text, pred_token_ids


# Matches Qwen special-token markers like <|box_start|>, <|im_end|>, ...
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^<>|]+\|>")


def run_inference_vllm(client, model_name, messages, max_new_tokens=2048):
    """Greedy inference against a vLLM OpenAI-compatible server.

    Mirrors `run_inference_with_ids` for the API backend: sends the same chat
    messages (PIL images base64-encoded as data URIs) and requests
    `skip_special_tokens: false` so `<|box_start|>`/`<|object_ref_start|>`
    markers survive in the returned text — grounding is then parsed from the
    raw text via `_parse_grounding_string` instead of token IDs.

    Returns (pred_text, raw_text): `pred_text` has the special-token markers
    stripped (equivalent to a skip_special_tokens=True decode); `raw_text`
    keeps them for grounding extraction.
    """
    import base64
    import io

    api_messages = []
    for m in messages:
        content = []
        for chunk in m["content"]:
            if chunk.get("type") == "image":
                buf = io.BytesIO()
                chunk["image"].save(buf, format="JPEG", quality=95)
                b64 = base64.b64encode(buf.getvalue()).decode()
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })
            else:
                content.append({"type": "text", "text": chunk["text"]})
        api_messages.append({"role": m["role"], "content": content})

    resp = client.chat.completions.create(
        model=model_name,
        messages=api_messages,
        max_tokens=max_new_tokens,
        temperature=0.0,
        extra_body={"skip_special_tokens": False},
    )
    raw_text = resp.choices[0].message.content or ""
    pred_text = _SPECIAL_TOKEN_RE.sub("", raw_text).strip()
    return pred_text, raw_text


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


# ---------------------------------------------------------------------------
# v2 per-category eval — grounded metrics (md Section 7)
# ---------------------------------------------------------------------------
_V2_BOX_RE = re.compile(
    r"<\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>"
)
_V2_LABEL_RE = re.compile(
    r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>"
)

# Token IDs (Qwen3-VL 8B Instruct vocab). Mirrored from
# qwen-vl-finetune/qwenvl/train/token_role_masks.py — kept duplicated here
# so this module has no training-side import dependency.
_BOX_START_ID      = 151648
_BOX_END_ID        = 151649
_OBJ_REF_START_ID  = 151646
_OBJ_REF_END_ID    = 151647
# image_idx digit landmark: 'image'(1805) '_idx'(7258) '":'(788) ' '(220)
_IMAGE_IDX_LANDMARK = (1805, 7258, 788, 220)
_VIEW_DIGIT_IDS = (16, 17, 18, 19, 20, 21)   # '1'..'6'


def _parse_grounding_from_token_ids(token_ids, tokenizer):
    """Extract grounding entries directly from generated token IDs (P0-1).

    Scans for `<|box_start|> ... <|box_end|>` spans, and for each box pairs it
    with the nearest preceding `<|object_ref_start|> ... <|object_ref_end|>`
    label and the nearest preceding `image_idx: <d>` landmark digit.

    This works regardless of `skip_special_tokens` settings at decode time,
    because we operate on raw IDs. Returns the same list-of-dict shape as
    `_parse_grounding_string()` so downstream metric code is identical.
    """
    entries = []
    last_label = None
    last_label_end = -1
    last_idx = None
    last_idx_end = -1
    i = 0
    T = len(token_ids)
    while i < T:
        tid = token_ids[i]
        # object_ref label
        if tid == _OBJ_REF_START_ID:
            j = i + 1
            while j < T and token_ids[j] != _OBJ_REF_END_ID:
                j += 1
            if j < T:
                # Decode label text from tokens between the ref delimiters.
                last_label = tokenizer.decode(token_ids[i + 1:j]).strip()
                last_label_end = j
            i = j + 1
            continue
        # image_idx landmark: locate followed digit
        if (
            i + len(_IMAGE_IDX_LANDMARK) <= T
            and tuple(token_ids[i:i + len(_IMAGE_IDX_LANDMARK)]) == _IMAGE_IDX_LANDMARK
            and i + len(_IMAGE_IDX_LANDMARK) < T
            and token_ids[i + len(_IMAGE_IDX_LANDMARK)] in _VIEW_DIGIT_IDS
        ):
            j = i + len(_IMAGE_IDX_LANDMARK)
            last_idx = token_ids[j] - _VIEW_DIGIT_IDS[0] + 1   # 1..6
            last_idx_end = j
            i = j + 1
            continue
        # box span
        if tid == _BOX_START_ID:
            j = i + 1
            while j < T and token_ids[j] != _BOX_END_ID:
                j += 1
            if j >= T:
                break  # unterminated box at tail — stop
            coord_ids = token_ids[i + 1:j]
            coord_text = tokenizer.decode(coord_ids).replace(" ", "").replace("\n", "")
            box = None
            if coord_text.startswith("(") and coord_text.endswith(")"):
                try:
                    inner = coord_text[1:-1]
                    a, b = inner.split("),(")
                    x1, y1 = (int(v) for v in a.split(","))
                    x2, y2 = (int(v) for v in b.split(","))
                    if x1 < x2 and y1 < y2 and all(0 <= v <= 1000 for v in (x1, y1, x2, y2)):
                        box = (x1, y1, x2, y2)
                except (ValueError, IndexError):
                    box = None
            entries.append({
                "image_idx": last_idx,
                "label": last_label or "",
                "box": box,
                "parsed_ok": box is not None and last_idx is not None,
            })
            # Consume label so it doesn't bind to a later box.
            last_label = None
            i = j + 1
            continue
        i += 1
    return entries


def _parse_grounding_string(value, *, drop_invalid=False):
    """Extract grounding entries from a model output string.

    Returns list of {image_idx, label, box, parsed_ok}. Robust to:
      - Top-level JSON parse failure (regex fallback finds box/ref pairs).
      - Missing image_idx (entry dropped — needed for view metric).
      - Malformed box / label tokens (entry's parsed_ok=False, retained for
        format-validity stats).

    If `drop_invalid=True`, entries that did not parse to a usable box are
    skipped entirely. Use this for GT extraction (where an unparseable GT
    box is a data-prep bug, not a metric signal); leave False for PRED
    extraction (where format-validity is the metric).
    """
    if not isinstance(value, str):
        return []

    # Path A: clean JSON
    try:
        obj = json.loads(value)
        if isinstance(obj, dict) and isinstance(obj.get("grounding"), list):
            out = []
            for g in obj["grounding"]:
                if not isinstance(g, dict):
                    continue
                ref = g.get("ref", "")
                bm = _V2_BOX_RE.search(ref)
                lm = _V2_LABEL_RE.search(ref)
                idx = g.get("image_idx")
                if isinstance(idx, int) and 1 <= idx <= 6 and bm:
                    x1, y1, x2, y2 = (int(v) for v in bm.groups())
                    ok = x1 < x2 and y1 < y2 and all(0 <= v <= 1000 for v in (x1, y1, x2, y2))
                    if drop_invalid and not ok:
                        continue
                    out.append({
                        "image_idx": idx,
                        "label": (lm.group(1).strip() if lm else ""),
                        "box": (x1, y1, x2, y2) if ok else None,
                        "parsed_ok": ok,
                    })
            return out
    except (json.JSONDecodeError, ValueError):
        pass

    # Path B: regex fallback — pair each box span with the nearest preceding
    # image_idx digit and object_ref label.
    out = []
    last_idx = None
    last_label = None
    for m in re.finditer(
        r'"image_idx"\s*:\s*(\d+)|<\|object_ref_start\|>(.*?)<\|object_ref_end\|>|'
        r'<\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>',
        value,
        re.DOTALL,
    ):
        if m.group(1) is not None:
            last_idx = int(m.group(1))
        elif m.group(2) is not None:
            last_label = m.group(2).strip()
        else:
            x1, y1, x2, y2 = (int(v) for v in m.group(3, 4, 5, 6))
            if last_idx is None or not (1 <= last_idx <= 6):
                continue
            ok = x1 < x2 and y1 < y2 and all(0 <= v <= 1000 for v in (x1, y1, x2, y2))
            if drop_invalid and not ok:
                last_label = None
                continue
            out.append({
                "image_idx": last_idx,
                "label": last_label or "",
                "box": (x1, y1, x2, y2) if ok else None,
                "parsed_ok": ok,
            })
            last_label = None  # consume label
    return out


def _iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _match_pred_to_gt(pred_entries, gt_entries):
    """Greedy matching keyed on (image_idx, normalized label). Within a group,
    pair predicted boxes to GT boxes by descending IoU. Returns list of
    (gt_idx, pred_idx_or_None, iou_or_0). Spurious predicted boxes are
    counted but not matched.
    """
    from collections import defaultdict

    def norm(s):
        return (s or "").strip().lower()

    pred_by_key = defaultdict(list)
    for pi, p in enumerate(pred_entries):
        if p["box"] is None:
            continue
        pred_by_key[(p["image_idx"], norm(p["label"]))].append(pi)

    matches = []
    used_pred = set()
    for gi, g in enumerate(gt_entries):
        # Defensive: GT entries with no parseable box can't be IoU-scored.
        # _extract_gt_grounding already drops them; this guard catches any
        # future caller that forgets the drop_invalid=True flag.
        if g.get("box") is None:
            matches.append((gi, None, 0.0))
            continue
        key = (g["image_idx"], norm(g["label"]))
        cands = [pi for pi in pred_by_key.get(key, []) if pi not in used_pred]
        if not cands:
            matches.append((gi, None, 0.0))
            continue
        best_pi = max(cands, key=lambda pi: _iou_xyxy(pred_entries[pi]["box"], g["box"]))
        best_iou = _iou_xyxy(pred_entries[best_pi]["box"], g["box"])
        used_pred.add(best_pi)
        matches.append((gi, best_pi, best_iou))
    n_spurious = sum(1 for pi, p in enumerate(pred_entries)
                     if p["box"] is not None and pi not in used_pred)
    return matches, n_spurious


def _match_by_label_only(pred_entries, gt_entries):
    """Pair GT and PRED by normalized label only (ignoring image_idx), greedy
    on IoU. Used ONLY to build T1's `view_confusion` so view errors on emitted
    boxes are visible. Distinct from `_match_pred_to_gt`, which requires
    matching view — that one drives recall/missing/spurious, this one drives
    confusion. Returns [(gt_idx, pred_idx)] pairs (only successful matches).
    """
    from collections import defaultdict

    def norm(s):
        return (s or "").strip().lower()

    pred_by_label = defaultdict(list)
    for pi, p in enumerate(pred_entries):
        if p["box"] is None:
            continue
        pred_by_label[norm(p["label"])].append(pi)

    pairs = []
    used = set()
    for gi, g in enumerate(gt_entries):
        if g.get("box") is None:
            continue
        cands = [pi for pi in pred_by_label.get(norm(g["label"]), []) if pi not in used]
        if not cands:
            continue
        best_pi = max(cands, key=lambda pi: _iou_xyxy(pred_entries[pi]["box"], g["box"]))
        used.add(best_pi)
        pairs.append((gi, best_pi))
    return pairs


_SANITY_REASONING_RE = re.compile(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', re.DOTALL)
_SANITY_ANSWER_RE    = re.compile(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"',    re.DOTALL)


def _sanity_extract_reasoning_answer(text):
    """Pull reasoning + answer strings out of a Qwen3-VL unified JSON envelope.
    Mirrors visualization/eval_sample_visualizer.py — kept duplicated here so
    this module has no viz-side import dependency. Handles clean JSON,
    markdown-fenced JSON, and truncated outputs via regex fallback."""
    if not isinstance(text, str) or not text:
        return "", ""
    cand = text.strip()
    if cand.startswith("```"):
        cand = re.sub(r"^```(?:json)?\s*", "", cand)
        cand = re.sub(r"\s*```\s*$", "", cand)
    try:
        obj = json.loads(cand)
        if isinstance(obj, dict):
            return (str(obj.get("reasoning") or ""),
                    str(obj.get("answer") or ""))
    except (json.JSONDecodeError, ValueError):
        pass
    def _unesc(s):
        try: return json.loads(f'"{s}"')
        except Exception:
            return (s.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\"))
    r = a = ""
    m = _SANITY_REASONING_RE.search(cand)
    if m: r = _unesc(m.group(1))
    m = _SANITY_ANSWER_RE.search(cand)
    if m: a = _unesc(m.group(1))
    return r, a


def _fmt_grounding_tuples(entries, limit=4):
    """One-line summary of a grounding list: `img2-'car'-[635,503,857,724]`."""
    if not entries:
        return "<none>"
    bits = []
    for e in entries[:limit]:
        box = e.get("box")
        box_s = f"[{','.join(str(v) for v in box)}]" if box else "?"
        bits.append(f"img{e.get('image_idx')}-{(e.get('label') or '')!r}-{box_s}")
    if len(entries) > limit:
        bits.append(f"... +{len(entries) - limit} more")
    return "; ".join(bits)


def _indent(text, prefix, max_chars=400):
    """Pretty-print multi-line text with a left-margin prefix; truncates
    each line and adds an overall char cap."""
    s = (text or "").strip()
    if not s:
        return f"{prefix}<empty>"
    if len(s) > max_chars:
        s = s[:max_chars] + " ..."
    return "\n".join(f"{prefix}{line}" for line in s.splitlines())


def _extract_gt_grounding(sample):
    """GT grounding from an SFT sample's `gpt` value, normalized to the same
    {image_idx, label, box} schema used by _parse_grounding_string. GT is
    authoritative — entries that don't parse to a valid box are dropped
    (otherwise downstream IoU / count code crashes on `box=None`)."""
    gt_value = sample["conversations"][2]["value"]
    return _parse_grounding_string(gt_value, drop_invalid=True)


def _describe_lora_dir(lora_path: str) -> str:
    """Build a human-readable summary of which files PeftModel will load.

    Helps the user verify they're evaluating the FINAL adapter (the
    end-of-stage save by trainer.model.save_pretrained()) and not picking
    up a stale intermediate checkpoint."""
    if not os.path.isdir(lora_path):
        return f"(dir does not exist: {lora_path})"
    lines = []
    main_adapter = os.path.join(lora_path, "adapter_model.safetensors")
    cfg = os.path.join(lora_path, "adapter_config.json")
    if os.path.exists(main_adapter):
        size_mb = os.path.getsize(main_adapter) / (1024 * 1024)
        mtime = datetime.fromtimestamp(os.path.getmtime(main_adapter)).isoformat(timespec="seconds")
        lines.append(f"adapter_model.safetensors  ({size_mb:.1f} MB, mtime={mtime})")
    else:
        lines.append("adapter_model.safetensors  MISSING  <-- end-of-stage save did not complete")
    if os.path.exists(cfg):
        lines.append(f"adapter_config.json        present")
    intermediates = sorted(
        d for d in os.listdir(lora_path)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(lora_path, d))
    )
    if intermediates:
        lines.append(f"intermediate checkpoints   {intermediates}  (NOT loaded; use --lora_checkpoint_subdir to override)")
    return "\n    ".join(lines)


def run_per_category_eval_v2(args):
    """Per-category eval producing the v2 metric set:
    answer_acc, view_acc, grounding_acc@<iou>, grounding_format_valid.
    Writes eval_report.json with extra columns alongside the v1 schema.

    Box extraction operates on raw token IDs (P0-1), so it does not depend
    on `skip_special_tokens` at decode time.
    """
    # Resolve effective LoRA dir — optionally pin to an intermediate
    # checkpoint subdir for stage-progress comparison.
    effective_lora_path = args.lora_path
    if getattr(args, "lora_checkpoint_subdir", None):
        cand = os.path.join(args.lora_path, args.lora_checkpoint_subdir)
        if not os.path.isdir(cand):
            raise FileNotFoundError(
                f"--lora_checkpoint_subdir not found: {cand}\n"
                f"Available under {args.lora_path}: "
                f"{[d for d in os.listdir(args.lora_path) if d.startswith('checkpoint-')]}"
            )
        effective_lora_path = cand

    print("=" * 60)
    print("Loading model for v2 eval")
    print(f"  base_model: {args.base_model}")
    print(f"  lora_path:  {effective_lora_path}")
    print(f"    {_describe_lora_dir(effective_lora_path)}")
    print("=" * 60)

    model, processor = load_model(args.base_model, effective_lora_path)
    tokenizer = processor.tokenizer  # for token-id based grounding parsing

    cat_paths = _discover_category_subsets(args.per_category_eval_dir)
    if not cat_paths:
        raise FileNotFoundError(
            f"No sft_val_qwen3vl_*.json files in {args.per_category_eval_dir}"
        )

    # Iterate categories in curriculum order (OBS -> CHR), not alphabetical.
    cat_iter_order = _order_by_curriculum(cat_paths.keys())
    print(f"v2 per-category eval over {len(cat_iter_order)} categories "
          f"(curriculum order): {cat_iter_order}")

    # Sample-stride args — when set, this worker only processes samples whose
    # original index modulo `sample_stride` equals `sample_offset`. Lets a
    # coordinator fan out 8 workers across 8 GPUs for a single stage's eval.
    sample_stride = max(1, int(getattr(args, "sample_stride", 1) or 1))
    sample_offset = int(getattr(args, "sample_offset", 0) or 0)
    if sample_offset < 0 or sample_offset >= sample_stride:
        raise ValueError(
            f"--sample_offset must be in [0, {sample_stride}); got {sample_offset}"
        )
    report_suffix = getattr(args, "report_suffix", "") or ""
    if report_suffix and not report_suffix.startswith(("_", ".", "-")):
        report_suffix = "_" + report_suffix
    is_partial_worker = sample_stride > 1

    iou_threshold = float(args.iou_threshold)
    report = {
        "eval_version": "v2",
        "lora_path": os.path.abspath(effective_lora_path),
        "eval_dir": os.path.abspath(args.per_category_eval_dir),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "max_new_tokens": args.max_new_tokens,
        "resize_factor": args.resize_factor,
        "iou_threshold": iou_threshold,
        "metrics": {},
    }
    if getattr(args, "report_completeness", True):
        report["field_meaning"] = {
            "per_view_recall":  ("OMISSION-side metric: of GT boxes in view v, fraction "
                                 "matched at (image_idx, label) with IoU >= threshold. "
                                 "Denominator includes views the model never emitted a box for."),
            "view_confusion":   ("VIEW-ERROR on EMITTED boxes only: count[gt_view-1][pred_view-1] "
                                 "over GT-PRED pairs matched by LABEL across all views. Says "
                                 "nothing about boxes the model failed to emit (those live in "
                                 "per_view_recall). Diagonal mass = correct view; off-diagonal "
                                 "= view confusion (e.g. front-bias / adjacent-view mixing)."),
            "n_missing_boxes":  ("OMISSION scalar: mean (per sample with GT grounding) count of "
                                 "GT boxes with no IoU-threshold match."),
            "n_spurious_boxes": ("HALLUCINATION scalar: mean (per sample with GT grounding) "
                                 "count of pred boxes not paired to any GT under (image_idx, label)."),
            "referring_completeness": ("Per-sample matched_at_threshold / n_gt, averaged across "
                                 "samples with GT grounding. 1.0 = perfect recall on this sample."),
        }
    if is_partial_worker:
        report["sample_stride"] = sample_stride
        report["sample_offset"] = sample_offset
    cat_predictions = {}

    for cat in cat_iter_order:
        path = cat_paths[cat]
        with open(path) as f:
            full_data = json.load(f)
        if args.n_per_cat_limit and args.n_per_cat_limit < len(full_data):
            full_data = full_data[: args.n_per_cat_limit]

        n_total_full = len(full_data)
        # Slice for this worker. We preserve the original index so the merger
        # can stitch results back across workers in val-subset order.
        data = [(orig_i, s) for orig_i, s in enumerate(full_data)
                if orig_i % sample_stride == sample_offset]
        n_total = len(data)
        n_ans_correct = 0
        n_parsed_answer = 0
        n_grounded = 0                      # samples whose GT has grounding
        n_grounded_with_valid_pred = 0      # P1-3 numerator: grounded samples
                                            #   whose pred grounding has >=1 valid box
        gt_box_total = 0
        gt_box_view_correct = 0
        gt_box_iou_correct = 0
        spurious_pred_boxes = 0
        sanity_printed = 0
        per_sample = []
        # T1 — completeness columns (active when args.report_completeness)
        per_view_gt_count      = [0] * 6   # GT boxes per view (denominator of per_view_recall)
        per_view_matched_count = [0] * 6   # matched at iou_threshold AND view_ok per view (numerator)
        view_confusion         = [[0] * 6 for _ in range(6)]  # gt_view x pred_view, label-only matches
        ref_completeness_sample_sum = 0.0  # per-sample (matched_thr / n_gt), summed
        n_missing_boxes_sum         = 0    # total GT boxes with no IoU-thr match

        slice_tag = (
            f" (worker stride={sample_stride} offset={sample_offset}; "
            f"{n_total}/{n_total_full} samples)"
            if is_partial_worker else f" ({n_total} samples)"
        )
        print(f"\n[{cat}]{slice_tag}")
        for slot, (i, sample) in enumerate(data):
            try:
                messages = _build_messages_from_sft_sample(sample, args.resize_factor)
                pred_text, pred_token_ids = run_inference_with_ids(
                    model, processor, messages, args.max_new_tokens
                )
            except Exception as e:
                per_sample.append({"i": i, "error": str(e)})
                continue

            gt_text = sample["conversations"][2]["value"]
            pred_ans = _extract_answer(pred_text)
            gt_ans = _extract_answer(gt_text)
            if pred_ans is not None:
                n_parsed_answer += 1
            ans_correct = (
                pred_ans is not None
                and gt_ans is not None
                and _normalize_answer(pred_ans) == _normalize_answer(gt_ans)
            )
            if ans_correct:
                n_ans_correct += 1

            # Grounding parsing: PREDICTION uses token-id parser (works with
            # special tokens stripped from text). GT uses the text parser
            # because the on-disk value field includes raw special-token
            # markup in the string. Both arrive at the same dict shape.
            gt_entries = _extract_gt_grounding(sample)
            pred_entries = _parse_grounding_from_token_ids(pred_token_ids, tokenizer)

            has_gt_grounding = len(gt_entries) > 0
            n_valid_pred_boxes = sum(1 for p in pred_entries if p["box"] is not None)
            if has_gt_grounding:
                n_grounded += 1
                if n_valid_pred_boxes > 0:
                    n_grounded_with_valid_pred += 1

            matches, n_spur = _match_pred_to_gt(pred_entries, gt_entries)
            spurious_pred_boxes += n_spur

            sample_grounding = []
            for gi, pi, iou in matches:
                gt_box_total += 1
                pred_view = pred_entries[pi]["image_idx"] if pi is not None else None
                view_ok = pred_view == gt_entries[gi]["image_idx"]
                iou_ok = iou >= iou_threshold and view_ok
                if view_ok:
                    gt_box_view_correct += 1
                if iou_ok:
                    gt_box_iou_correct += 1
                sample_grounding.append({
                    "gt": gt_entries[gi],
                    "pred_idx": pi,
                    "pred": pred_entries[pi] if pi is not None else None,
                    "iou": iou,
                    "view_ok": view_ok,
                    "iou_ok": iou_ok,
                })

            per_sample.append({
                "i": i,
                "pred_text": pred_text,                         # P0-2
                "pred_answer": pred_ans,
                "gt_answer": gt_ans,
                "answer_correct": ans_correct,
                "pred_grounding": pred_entries,                 # P0-2 (parsed)
                "gt_grounding": gt_entries,
                "matches": sample_grounding,
                "n_pred_boxes": n_valid_pred_boxes,
                "n_spurious": n_spur,
            })

            # T1 — completeness / per-view accumulation.
            # Run for every grounded sample regardless of `args.report_completeness`
            # so the per-sample dump can carry these later; aggregation into report
            # is the only thing gated by the flag.
            if has_gt_grounding:
                for g in gt_entries:
                    v = g.get("image_idx")
                    if isinstance(v, int) and 1 <= v <= 6:
                        per_view_gt_count[v - 1] += 1
                n_matched_thr = 0
                for mr in sample_grounding:
                    if not mr.get("iou_ok"):
                        continue
                    n_matched_thr += 1
                    gv = mr["gt"].get("image_idx")
                    if isinstance(gv, int) and 1 <= gv <= 6:
                        per_view_matched_count[gv - 1] += 1
                ref_completeness_sample_sum += n_matched_thr / max(len(gt_entries), 1)
                n_missing_boxes_sum += len(gt_entries) - n_matched_thr

                # view_confusion uses a LABEL-ONLY pairing so view errors on
                # emitted boxes are visible. The primary matcher above keys on
                # (view, label) and would force a diagonal — that's why we run
                # a second pass here.
                for gi, pi in _match_by_label_only(pred_entries, gt_entries):
                    gv = gt_entries[gi].get("image_idx")
                    pv = pred_entries[pi].get("image_idx")
                    if (isinstance(gv, int) and 1 <= gv <= 6
                            and isinstance(pv, int) and 1 <= pv <= 6):
                        view_confusion[gv - 1][pv - 1] += 1

            # Sanity print first N grounded samples per category.
            # Mirrors the visualizer's info density (grounding + reasoning
            # + answer) so log scanning has the same affordances.
            if (
                has_gt_grounding
                and args.sanity_print_n
                and sanity_printed < args.sanity_print_n
            ):
                gt_reasoning,   gt_answer_full   = _sanity_extract_reasoning_answer(gt_text)
                pred_reasoning, pred_answer_full = _sanity_extract_reasoning_answer(pred_text)
                mark = "OK" if ans_correct else "MISS"
                print(
                    f"  --- [SANITY {cat} #{sanity_printed}] val_idx={i}  "
                    f"answer:{mark}  "
                    f"gt:{(gt_ans or '')!r}  pred:{(pred_ans or '')!r}  "
                    f"GT_boxes:{len(gt_entries)}  PRED_boxes:{n_valid_pred_boxes}  "
                    f"spurious:{n_spur}"
                )
                # GT
                print(f"    GT grounding:   {_fmt_grounding_tuples(gt_entries)}")
                print(f"    GT answer:      {gt_answer_full or gt_ans or ''}")
                if gt_reasoning:
                    print("    GT reasoning:")
                    print(_indent(gt_reasoning, "      | "))
                # PRED
                print(f"    PRED grounding: {_fmt_grounding_tuples(pred_entries)}")
                print(f"    PRED answer:    {pred_answer_full or pred_ans or ''}")
                if pred_reasoning:
                    print("    PRED reasoning:")
                    print(_indent(pred_reasoning, "      | "))
                else:
                    print("    PRED reasoning: <none parsed>")
                # Per-box match breakdown (concise — visualizer has the full table)
                if sample_grounding:
                    print("    matches (gt_view·label  ->  pred_view·label  IoU  view_ok):")
                    for mr in sample_grounding:
                        gt_g  = mr["gt"]
                        prd   = mr.get("pred") or {}
                        iou_s = f"{mr['iou']:.3f}" if mr.get("iou") is not None else "-"
                        view_s = "OK" if mr.get("view_ok") else "X"
                        iou_t  = "OK" if mr.get("iou_ok")  else "X"
                        print(
                            f"      gt[{gt_g.get('image_idx')}·{gt_g.get('label','')}]"
                            f"  ->  pred[{prd.get('image_idx', '-')}·{prd.get('label','')}]"
                            f"  IoU={iou_s} ({iou_t})  view:{view_s}"
                        )
                sanity_printed += 1

            if (i + 1) % 25 == 0 or i == n_total - 1:
                running = n_ans_correct / max(i + 1, 1)
                print(f"  [{i+1}/{n_total}] running answer_acc={running:.3f}")

        answer_acc = n_ans_correct / max(n_total, 1)
        view_acc = gt_box_view_correct / max(gt_box_total, 1) if gt_box_total else None
        ground_acc = gt_box_iou_correct / max(gt_box_total, 1) if gt_box_total else None
        # P1-3: grounded samples whose pred grounding parses to >=1 valid box.
        # None when n_grounded == 0 (denominator undefined).
        if n_grounded > 0:
            fmt_valid = n_grounded_with_valid_pred / n_grounded
        else:
            fmt_valid = None
        parse_rate = n_parsed_answer / max(n_total, 1)

        report["metrics"][cat] = {
            "n": n_total,
            "n_grounded": n_grounded,
            "n_gt_boxes": gt_box_total,
            "answer_acc": answer_acc,
            "view_acc": view_acc,
            f"grounding_acc@{iou_threshold:.1f}": ground_acc,
            "grounding_format_valid": fmt_valid,
            "parse_rate": parse_rate,
            "spurious_pred_boxes": spurious_pred_boxes,
        }
        if getattr(args, "report_completeness", True):
            # referring_completeness is undefined on samples with no GT grounding
            # (0/0), so its denominator is n_grounded. n_missing and n_spurious
            # are symmetric — empty-GT samples contribute 0 missing and
            # potentially-nonzero spurious — so their denominator is n_total.
            referring_completeness = (
                ref_completeness_sample_sum / n_grounded if n_grounded > 0 else None
            )
            n_missing_boxes_mean  = n_missing_boxes_sum / max(n_total, 1)
            n_spurious_boxes_mean = spurious_pred_boxes / max(n_total, 1)
            per_view_recall = [
                (per_view_matched_count[v] / per_view_gt_count[v]) if per_view_gt_count[v] else None
                for v in range(6)
            ]
            report["metrics"][cat].update({
                "referring_completeness": referring_completeness,
                "n_missing_boxes":  n_missing_boxes_mean,
                "n_spurious_boxes": n_spurious_boxes_mean,
                "per_view_recall":  per_view_recall,
                "per_view_gt_count": per_view_gt_count,
                "view_confusion":   view_confusion,
                # Raw counts so the 8-GPU merger can sum + re-derive ratios
                # across workers. Not for human consumption — the *_mean /
                # per_view_recall keys above are. Schema-stable: append-only.
                "_raw_per_view_matched_count":   per_view_matched_count,
                "_raw_n_missing_boxes_sum":      n_missing_boxes_sum,
                "_raw_ref_completeness_sum":     ref_completeness_sample_sum,
            })
        cat_predictions[cat] = per_sample
        print(
            f"[{cat}] done: answer_acc={answer_acc:.3f}  "
            f"view_acc={view_acc if view_acc is not None else 'NA'}  "
            f"grounding_acc@{iou_threshold:.1f}={ground_acc if ground_acc is not None else 'NA'}  "
            f"format_valid={fmt_valid if fmt_valid is not None else 'NA (n_grounded=0)'}"
        )

    # Macro = mean over categories that have answer_acc.
    macro = (
        sum(m["answer_acc"] for m in report["metrics"].values())
        / len(report["metrics"]) if report["metrics"] else 0.0
    )
    report["macro_answer_acc"] = macro

    # Writes land in the EFFECTIVE LoRA path (the checkpoint subdir when
    # --lora_checkpoint_subdir is set; the stage dir otherwise). With
    # --report_suffix, multiple workers can write side-by-side files.
    out_path = os.path.join(effective_lora_path, f"eval_report{report_suffix}.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    if args.save_predictions:
        preds_path = os.path.join(effective_lora_path, f"eval_predictions{report_suffix}.json")
        with open(preds_path, "w") as f:
            json.dump(cat_predictions, f, indent=2)
        print(f"Per-sample predictions saved to: {preds_path}")

    print(f"\nv2 eval report saved to: {out_path}")
    if is_partial_worker:
        print(f"  (partial worker: stride={sample_stride} offset={sample_offset}; "
              f"merge via merge_partial_eval_reports.py before reading aggregate metrics)")
    print(f"Macro answer_acc (over THIS worker's slice): {macro:.3f}")


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
    parser.add_argument('--eval_v2', action='store_true',
                        help="Use v2 eval (md Section 7): answer_acc, view_acc, "
                             "grounding_acc@<iou_threshold>, grounding_format_valid. "
                             "Requires --per_category_eval_dir.")
    parser.add_argument('--iou_threshold', type=float, default=0.8,
                        help="IoU threshold for v2 grounding accuracy (default 0.8).")
    parser.add_argument('--sanity_print_n', type=int, default=2,
                        help="In v2 eval, print raw generation + parsed grounding for "
                             "the first N grounded samples per category (default 2; "
                             "set 0 to disable).")
    parser.add_argument('--lora_checkpoint_subdir', type=str, default=None,
                        help="Optional: load an intermediate checkpoint inside --lora_path "
                             "(e.g. 'checkpoint-1000') instead of the final adapter at the "
                             "stage dir's top level. Useful for measuring per-stage progress.")
    parser.add_argument('--sample_stride', type=int, default=1,
                        help="Sample-level data parallelism: a worker processes only samples "
                             "whose original index modulo `sample_stride` equals "
                             "`sample_offset`. Default 1 = process all.")
    parser.add_argument('--sample_offset', type=int, default=0,
                        help="Companion to --sample_stride; must be in [0, sample_stride).")
    parser.add_argument('--report_suffix', type=str, default="",
                        help="Suffix appended to eval_report.json / eval_predictions.json "
                             "filenames (e.g. '_worker3'). Used by the 8-GPU single-stage "
                             "wrapper so per-worker outputs don't clobber each other.")
    parser.add_argument('--report_completeness',
                        default=True,
                        action=argparse.BooleanOptionalAction,
                        help="T1: emit referring_completeness, n_missing/spurious_boxes, "
                             "per_view_recall, view_confusion in the v2 report. "
                             "Use --no-report_completeness to suppress.")

    args = parser.parse_args()

    # Per-category eval branch — skips the legacy NuScenesDataLoader path.
    if args.per_category_eval_dir:
        if args.eval_v2:
            run_per_category_eval_v2(args)
        else:
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
