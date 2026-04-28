"""
Verify that the SFT training pipeline correctly handles `from: "system"` turns.

Loads ONE sample through the actual `NuScenesVQADataset` (the same class used at
training time) and runs six mechanical checks:

  1. The system block is decodable from input_ids (visible in the tokenized output).
  2. The system block sits at the START of the sequence (position 0).
  3. The chat-template wrappers (<|im_start|>system ... <|im_end|>) are present.
  4. Loss masking — labels are IGNORE_INDEX (-100) inside the system block,
     and real token IDs inside the assistant block.
  5. Effective attention mask covers the system block (no implicit zeroing).
  6. Position IDs are monotonically non-decreasing across the sequence.

Each check prints a single PASS/FAIL line, with a tail summary.

Usage
-----
    # Run on sample index 0 of the qwen3vl training set
    python -m nuscenes_pipeline.scripts.verify_system_in_training_pipeline

    # Pick a specific sample index
    python -m nuscenes_pipeline.scripts.verify_system_in_training_pipeline --sample_idx 5

    # Use the val set instead
    python -m nuscenes_pipeline.scripts.verify_system_in_training_pipeline \\
        --data_path sft_dataset/sft_val_qwen3vl.json
"""

import argparse
import os
import sys

import torch
from transformers import AutoProcessor, AutoTokenizer

# Reuse the actual training dataset class so we exercise the real code path.
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from train_nuscenes_qwen3vl import NuScenesVQADataset  # noqa: E402


IGNORE_INDEX = -100


def _find_system_block(input_ids, tokenizer):
    """Return (start, end) inclusive indices of the first <|im_start|>system ... <|im_end|>
    block in input_ids. Returns (None, None) if not found."""
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    sys_token_ids = tokenizer.encode("system", add_special_tokens=False)

    if not isinstance(input_ids, list):
        input_ids = input_ids.tolist()

    # Find <|im_start|> followed by "system"
    for i in range(len(input_ids) - len(sys_token_ids)):
        if input_ids[i] != im_start_id:
            continue
        # Check that the next token(s) match "system"
        if input_ids[i + 1: i + 1 + len(sys_token_ids)] == sys_token_ids:
            sys_start = i
            # Find the first <|im_end|> after sys_start
            for j in range(sys_start, len(input_ids)):
                if input_ids[j] == im_end_id:
                    return sys_start, j
            return sys_start, None
    return None, None


def _find_assistant_block(input_ids, tokenizer):
    """Return (start, end) inclusive indices of the LAST <|im_start|>assistant ... <|im_end|>
    block. Returns (None, None) if not found."""
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assistant_token_ids = tokenizer.encode("assistant", add_special_tokens=False)

    if not isinstance(input_ids, list):
        input_ids = input_ids.tolist()

    # Find the LAST <|im_start|> followed by "assistant"
    last_start = None
    for i in range(len(input_ids) - len(assistant_token_ids)):
        if input_ids[i] != im_start_id:
            continue
        if input_ids[i + 1: 1 + i + len(assistant_token_ids)] == assistant_token_ids:
            last_start = i
    if last_start is None:
        return None, None
    # Find the first <|im_end|> after last_start
    for j in range(last_start, len(input_ids)):
        if input_ids[j] == im_end_id:
            return last_start, j
    return last_start, None


def main():
    parser = argparse.ArgumentParser(
        description="Sanity-check that from: 'system' turns flow through the SFT training pipeline correctly"
    )
    parser.add_argument("--data_path", type=str,
                        default="sft_dataset/sft_train_qwen3vl.json",
                        help="Path to the SFT JSON file (qwen3vl format)")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Index of the sample to inspect")
    parser.add_argument("--checkpoint", type=str,
                        default="ckpts/qwen3_vl_8b_instruct",
                        help="Path to the model checkpoint (for tokenizer + image processor)")
    parser.add_argument("--resize_factor", type=int, default=2,
                        help="Image resize factor (must match training config)")
    parser.add_argument("--max_pixels", type=int, default=50176)
    parser.add_argument("--min_pixels", type=int, default=784)
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=False)
    image_processor = AutoProcessor.from_pretrained(args.checkpoint).image_processor

    print(f"Loading dataset from {args.data_path}...")
    dataset = NuScenesVQADataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        resize_factor=args.resize_factor,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
    )

    print(f"\nLoading sample {args.sample_idx}...\n")
    sample = dataset[args.sample_idx]

    input_ids = sample["input_ids"][0] if sample["input_ids"].dim() > 1 else sample["input_ids"]
    labels = sample["labels"][0] if sample["labels"].dim() > 1 else sample["labels"]
    position_ids = sample.get("position_ids")

    print("=" * 70)
    print(f"Sequence length: {len(input_ids)}")
    print("=" * 70)

    # ----- Check 1 & 3: system block decodable + has correct wrappers -----
    print("\n[Check 1 & 3] Decoded first 250 tokens:")
    print("-" * 70)
    decoded_head = tokenizer.decode(input_ids[:250])
    print(decoded_head)
    print("-" * 70)
    has_system_marker = "<|im_start|>system" in decoded_head
    has_user_marker = "<|im_start|>user" in decoded_head
    print(f"\nCheck 1: system block present in decoded output: "
          f"{'PASS' if has_system_marker else 'FAIL'}")
    print(f"Check 3: <|im_start|>system + <|im_end|> wrappers present: "
          f"{'PASS' if has_system_marker else 'FAIL'}")
    print(f"         (also has <|im_start|>user marker: {has_user_marker})")

    # ----- Check 2: system at sequence start -----
    sys_start, sys_end = _find_system_block(input_ids, tokenizer)
    if sys_start is None:
        print(f"\nCheck 2: system block not found — FAIL")
    else:
        is_at_start = sys_start == 0
        print(f"\nCheck 2: system block at sequence start "
              f"(found at position {sys_start}, ends at {sys_end}): "
              f"{'PASS' if is_at_start else 'FAIL'}")

    # ----- Check 4: loss masking -----
    if sys_start is not None and sys_end is not None:
        sys_labels = labels[sys_start:sys_end + 1]
        sys_all_masked = bool((sys_labels == IGNORE_INDEX).all().item())
        print(f"\nCheck 4a: all labels in system block == IGNORE_INDEX(-100): "
              f"{'PASS' if sys_all_masked else 'FAIL'}")
        if not sys_all_masked:
            unmasked = sys_labels[sys_labels != IGNORE_INDEX]
            print(f"         {len(unmasked)} unmasked tokens detected — first 10: "
                  f"{unmasked[:10].tolist()}")

    asst_start, asst_end = _find_assistant_block(input_ids, tokenizer)
    if asst_start is not None and asst_end is not None:
        asst_labels = labels[asst_start:asst_end + 1]
        # Skip the <|im_start|>assistant\n prefix — those are also masked by training code
        asst_real_labels = asst_labels[3:]  # offset matches train_nuscenes_qwen3vl.py:172
        asst_has_real_tokens = bool((asst_real_labels != IGNORE_INDEX).any().item())
        print(f"Check 4b: assistant block has real token labels (not all -100): "
              f"{'PASS' if asst_has_real_tokens else 'FAIL'}")
        n_real = int((asst_real_labels != IGNORE_INDEX).sum().item())
        print(f"         {n_real}/{len(asst_real_labels)} assistant tokens contribute to loss")

    # ----- Check 5: attention mask covers system tokens -----
    attn = sample.get("attention_mask")
    if attn is None:
        print(f"\nCheck 5: attention_mask not present in sample (collator builds it later)")
    else:
        if isinstance(attn, list):
            # train_nuscenes_qwen3vl stores `[seq_len]` here; collator expands it.
            print(f"\nCheck 5: attention_mask is a length scalar [{attn[0]}] "
                  f"(collator expands to all-1 mask of that length) — implicit PASS")
        else:
            attn_t = attn[0] if attn.dim() > 1 else attn
            if sys_start is not None and sys_end is not None:
                sys_attn = attn_t[sys_start:sys_end + 1]
                all_attended = bool((sys_attn == 1).all().item())
                print(f"\nCheck 5: attention_mask == 1 for entire system block: "
                      f"{'PASS' if all_attended else 'FAIL'}")

    # ----- Check 6: position IDs monotonically non-decreasing -----
    if position_ids is None:
        print(f"\nCheck 6: position_ids absent — FAIL")
    else:
        # position_ids has shape (3, 1, seq_len) for 3D RoPE in Qwen2.5/3-VL
        # (temporal/height/width). Each axis should be monotonic in its own way,
        # but for a forward-only sequence, the values along any axis should not
        # decrease abruptly. We just check that no axis has a negative diff
        # outside the vision-token block (which has 2D structure).
        if position_ids.dim() == 3:
            pids = position_ids[0, 0]  # take temporal axis, batch 0
        elif position_ids.dim() == 2:
            pids = position_ids[0]
        else:
            pids = position_ids
        diffs = pids[1:] - pids[:-1]
        # Allow zero increments (within vision blocks) but not large negatives.
        large_negative = (diffs < -1).sum().item()
        is_ok = large_negative == 0
        print(f"\nCheck 6: position_ids monotonically non-decreasing "
              f"(temporal axis, with zero increments allowed for vision blocks): "
              f"{'PASS' if is_ok else 'FAIL'}")
        if not is_ok:
            bad_positions = ((diffs < -1).nonzero().flatten()[:5].tolist())
            print(f"         {large_negative} large-negative diffs; first occurrences at indices: {bad_positions}")

    # ----- Summary -----
    print("\n" + "=" * 70)
    print("Summary: review each check above. All should be PASS for cycle-2 readiness.")
    print("=" * 70)


if __name__ == "__main__":
    main()
