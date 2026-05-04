"""
Fine-tune Qwen3-VL on nuScenes driving VQA dataset.

Supports two modes:
  1. LoRA fine-tuning  (--mode lora)
  2. Full fine-tuning  (--mode full)

Handles multi-image (6 camera views) samples with custom system prompts
containing driving context (command, velocity, detected 3D objects).

Usage:
    # LoRA fine-tuning (recommended for 8B model on limited GPU)
    torchrun --nproc_per_node=1 train_nuscenes_qwen3vl.py \
        --mode lora --bf16 --output_dir ./output_lora

    # Full fine-tuning (needs more GPU memory / DeepSpeed ZeRO-3)
    torchrun --nproc_per_node=1 train_nuscenes_qwen3vl.py \
        --mode full --deepspeed ./ds_zero3.json --bf16 --output_dir ./output_full
"""

import os
import sys
import copy
import json
import math
import time
import random
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
from collections.abc import Sequence as SequenceABC

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import transformers
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    Trainer,
    TrainingArguments as HfTrainingArguments,
)

# Add qwen-vl-finetune to path for rope utilities
FINETUNE_ROOT = os.path.join(os.path.dirname(__file__), "qwen-vl-finetune")
sys.path.insert(0, os.path.abspath(FINETUNE_ROOT))
from qwenvl.data.rope2d import get_rope_index_25
from qwenvl.train.trainer import replace_qwen2_vl_attention_class

IGNORE_INDEX = -100

local_rank = None


def rank0_print(*args):
    if local_rank == 0 or local_rank is None:
        print(*args)


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="Qwen/Qwen3-VL-8B")
    mode: str = field(default="lora", metadata={"help": "Training mode: 'lora' or 'full'"})
    # Full fine-tuning options
    tune_mm_vision: bool = field(default=False)
    tune_mm_mlp: bool = field(default=True)
    tune_mm_llm: bool = field(default=True)
    # LoRA options
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated list of LoRA target modules"},
    )


@dataclass
class DataArguments:
    train_data_path: str = field(
        default="sft_dataset/sft_train_qwen3vl.json",
        metadata={"help": "Path to the training data JSON file"},
    )
    val_data_path: str = field(
        default="sft_dataset/sft_val_qwen3vl.json",
        metadata={"help": "Path to the validation data JSON file (optional)"},
    )
    data_flatten: bool = field(default=False)
    # Default max_pixels = 1600 * 900 ~= 1,440,208 (full nuScenes resolution
    # rounded to a 28x28 patch grid). The image processor's smart_resize
    # handles all downsampling inside this cap.
    max_pixels: int = field(default=1_440_208)
    min_pixels: int = field(default=28 * 28 * 16)


@dataclass
class TrainingArguments(HfTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=8192)
    mm_projector_lr: Optional[float] = field(default=None)
    vision_tower_lr: Optional[float] = field(default=None)


# ---------------------------------------------------------------------------
# Tokenization with custom system prompt support
# ---------------------------------------------------------------------------
def preprocess_with_system_prompt(
    sources: List[List[Dict]],
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw_image: List = None,
) -> Dict:
    """
    Tokenize conversations that may contain a custom 'system' turn.

    Expected source format:
        [
            {"from": "system", "value": "..."},
            {"from": "human", "value": "... <image> ... <image> ..."},
            {"from": "gpt", "value": "..."},
        ]
    """
    roles = {"human": "user", "gpt": "assistant", "system": "system"}

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    )
    tokenizer.chat_template = chat_template

    visual_replicate_index_image = 0
    input_ids, targets = [], []

    for source in sources:
        input_id, target = [], []

        for conv in source:
            role_key = conv.get("from", conv.get("role", ""))
            content = conv.get("value", conv.get("content", ""))
            role = roles.get(role_key, role_key)

            # Replace <image> placeholders with vision tokens
            if role == "user" and "<image>" in content and grid_thw_image:
                parts = content.split("<image>")
                new_parts = []
                for i in range(len(parts) - 1):
                    new_parts.append(parts[i])
                    replacement = (
                        "<|vision_start|>"
                        + "<|image_pad|>" * grid_thw_image[visual_replicate_index_image]
                        + "<|vision_end|>"
                    )
                    new_parts.append(replacement)
                    visual_replicate_index_image += 1
                new_parts.append(parts[-1])
                content = "".join(new_parts)

            conv_msg = [{"role": role, "content": content}]
            encode_id = tokenizer.apply_chat_template(conv_msg)
            input_id += encode_id

            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target_mask = encode_id.copy()
                target_mask[:3] = [IGNORE_INDEX] * 3  # Mask <|im_start|>assistant\n
                target += target_mask

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        input_ids.append(input_id)
        targets.append(target)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(input_ids=input_ids, labels=targets)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class NuScenesVQADataset(Dataset):
    """Dataset for nuScenes driving VQA SFT with 6-view images."""

    REAR_CAMERAS = {'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'}
    # Image indices that correspond to rear cameras (0-indexed in the 6-image list)
    # Order: FL(0), F(1), FR(2), BL(3), B(4), BR(5)
    REAR_IMAGE_INDICES = {3, 4, 5}

    def __init__(self, data_path: str, tokenizer, image_processor,
                 max_pixels: int = 1_440_208,
                 min_pixels: int = 28 * 28 * 16):
        super().__init__()

        rank0_print(f"Loading SFT data from: {data_path}")
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        rank0_print(f"Loaded {len(self.data)} training samples")

        self.tokenizer = tokenizer
        self.image_processor = copy.deepcopy(image_processor)
        self.get_rope_index = get_rope_index_25

        # Set pixel limits — the processor's smart_resize handles all
        # downsampling within this cap. No PIL pre-resize is needed.
        self.image_processor.max_pixels = max_pixels
        self.image_processor.min_pixels = min_pixels
        self.image_processor.size["longest_edge"] = max_pixels
        self.image_processor.size["shortest_edge"] = min_pixels

    def __len__(self):
        return len(self.data)

    def process_image(self, image_path: str, flip_horizontal: bool = False):
        """Load, optionally flip, and process a single image. smart_resize
        in the image processor handles all spatial downsampling."""
        img = Image.open(image_path).convert("RGB")

        # Flip rear cameras for egocentric view
        if flip_horizontal:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        processor = copy.deepcopy(self.image_processor)
        visual_processed = processor.preprocess(img, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, list):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        for attempt in range(3):
            try:
                return self._get_item(idx)
            except Exception as e:
                print(f"[Attempt {attempt}] Failed sample {idx}: {e}")
                time.sleep(0.5)
        # Final attempt
        return self._get_item(idx)

    def _get_item(self, idx) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        image_paths = sample["image"]  # list of 6 paths
        conversations = sample["conversations"]

        # Process 6 images
        images = []
        grid_thws = []
        for img_idx, img_path in enumerate(image_paths):
            flip = img_idx in self.REAR_IMAGE_INDICES
            img_tensor, grid_thw = self.process_image(img_path, flip_horizontal=flip)
            images.append(img_tensor)
            grid_thws.append(grid_thw)

        # Compute merged grid_thw for tokenization
        merge_size = self.image_processor.merge_size
        grid_thw_merged = [
            thw.prod() // (merge_size ** 2) for thw in grid_thws
        ]

        # Tokenize conversations
        data_dict = preprocess_with_system_prompt(
            [conversations],
            self.tokenizer,
            grid_thw_image=grid_thw_merged,
        )

        # Compute position IDs (3D RoPE)
        position_ids, _ = self.get_rope_index(
            merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.stack(grid_thws, dim=0),
            video_grid_thw=None,
            second_per_grid_ts=None,
        )

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [data_dict["input_ids"][0].size(0)]
        data_dict["pixel_values"] = torch.cat(images, dim=0)
        data_dict["image_grid_thw"] = torch.cat(
            [thw.unsqueeze(0) for thw in grid_thws], dim=0
        )

        return data_dict


# ---------------------------------------------------------------------------
# Data Collator
# ---------------------------------------------------------------------------
def pad_and_cat(tensor_list):
    max_length = max(t.shape[2] for t in tensor_list)
    padded = []
    for t in tensor_list:
        pad_len = max_length - t.shape[2]
        padded.append(torch.nn.functional.pad(t, (0, pad_len), "constant", 1))
    return torch.cat(padded, dim=1)


@dataclass
class NuScenesDataCollator:
    """Collate multi-image SFT samples with variable-length padding."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [inst["input_ids"].squeeze(0) for inst in instances]
        labels = [inst["labels"].squeeze(0) for inst in instances]
        position_ids = [inst["position_ids"] for inst in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)

        # Truncate to model_max_length
        max_len = self.tokenizer.model_max_length
        input_ids = input_ids[:, :max_len]
        labels = labels[:, :max_len]
        position_ids = position_ids[:, :, :max_len]

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            position_ids=position_ids,
        )

        # Aggregate pixel values
        images = [inst["pixel_values"] for inst in instances if "pixel_values" in inst]
        if images:
            batch["pixel_values"] = torch.cat(images, dim=0)
            grid_thws = [inst["image_grid_thw"] for inst in instances if "image_grid_thw" in inst]
            batch["image_grid_thw"] = torch.cat(grid_thws, dim=0)
        else:
            batch["pixel_values"] = None
            batch["image_grid_thw"] = None

        batch["pixel_values_videos"] = None
        batch["video_grid_thw"] = None

        return batch


@dataclass
class NuScenesFlattenedDataCollator(NuScenesDataCollator):
    """Collate into packed sequences (for data_flatten mode)."""

    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        import itertools

        input_ids = [inst["input_ids"] for inst in instances]
        labels = [inst["labels"] for inst in instances]
        position_ids = [inst["position_ids"] for inst in instances]
        attention_mask = list(
            itertools.chain(*(inst["attention_mask"] for inst in instances))
        )

        seq_lens = torch.tensor([0] + attention_mask, dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)

        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)
        position_ids = torch.cat(position_ids, dim=2)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
        )

        images = [inst["pixel_values"] for inst in instances if "pixel_values" in inst]
        if images:
            batch["pixel_values"] = torch.cat(images, dim=0)
            grid_thws = [inst["image_grid_thw"] for inst in instances if "image_grid_thw" in inst]
            batch["image_grid_thw"] = torch.cat(grid_thws, dim=0)
        else:
            batch["pixel_values"] = None
            batch["image_grid_thw"] = None

        batch["pixel_values_videos"] = None
        batch["video_grid_thw"] = None

        return batch


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------
def setup_full_finetune(model, model_args):
    """Configure model for full fine-tuning."""
    # Vision encoder
    for p in model.visual.named_parameters():
        p[1].requires_grad = model_args.tune_mm_vision
    # MLP merger
    for p in model.visual.merger.named_parameters():
        p[1].requires_grad = model_args.tune_mm_mlp
    # LLM
    for p in model.model.named_parameters():
        p[1].requires_grad = model_args.tune_mm_llm
    model.lm_head.requires_grad = model_args.tune_mm_llm
    return model


def setup_lora(model, model_args):
    """Configure model for LoRA fine-tuning."""
    from peft import LoraConfig, get_peft_model

    target_modules = [m.strip() for m in model_args.lora_target_modules.split(",")]

    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    rank0_print("=" * 60)
    rank0_print(f"Training mode: {model_args.mode}")
    rank0_print(f"Model: {model_args.model_name_or_path}")
    rank0_print(f"max_pixels: {data_args.max_pixels} | min_pixels: {data_args.min_pixels}")
    rank0_print(f"model_max_length: {training_args.model_max_length}")
    rank0_print("=" * 60)

    # Load model
    from transformers import Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16 if training_args.bf16 else None,
    )
    model.config.use_cache = False

    # Data flatten requires attention class replacement
    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()

    # Gradient checkpointing
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # Setup training mode
    if model_args.mode == "lora":
        model = setup_lora(model, model_args)
    elif model_args.mode == "full":
        model = setup_full_finetune(model, model_args)
    else:
        raise ValueError(f"Unknown training mode: {model_args.mode}. Use 'lora' or 'full'.")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # Image processor
    image_processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path
    ).image_processor

    # Datasets
    train_dataset = NuScenesVQADataset(
        data_path=data_args.train_data_path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_pixels=data_args.max_pixels,
        min_pixels=data_args.min_pixels,
    )

    eval_dataset = None
    if data_args.val_data_path and os.path.exists(data_args.val_data_path):
        eval_dataset = NuScenesVQADataset(
            data_path=data_args.val_data_path,
            tokenizer=tokenizer,
            image_processor=image_processor,
            max_pixels=data_args.max_pixels,
            min_pixels=data_args.min_pixels,
        )
        rank0_print(f"Loaded {len(eval_dataset)} validation samples")

    # Data collator
    if data_args.data_flatten:
        data_collator = NuScenesFlattenedDataCollator(tokenizer=tokenizer)
    else:
        data_collator = NuScenesDataCollator(tokenizer=tokenizer)

    # Trainer
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        rank0_print("Checkpoint found, resuming training...")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save
    trainer.save_state()
    image_processor.save_pretrained(training_args.output_dir)
    model.config.use_cache = True

    if model_args.mode == "lora":
        # Save LoRA adapter only
        model.save_pretrained(training_args.output_dir)
        rank0_print(f"LoRA adapter saved to {training_args.output_dir}")
    else:
        # Save full model
        if trainer.deepspeed:
            torch.cuda.synchronize()
            trainer.save_model(training_args.output_dir)
        else:
            state_dict = trainer.model.state_dict()
            if trainer.args.should_save:
                cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
                del state_dict
                trainer._save(training_args.output_dir, state_dict=cpu_state_dict)
        rank0_print(f"Full model saved to {training_args.output_dir}")


if __name__ == "__main__":
    train()
