"""
Qwen3-VL nuScenes SFT — v2 (composite loss + curriculum).

Differences vs train_nuscenes_qwen3vl.py:

- New `preprocess_with_system_prompt_v2` emits per-token role masks
  (answer/gate/coord/image_idx) and the per-sample grounding metadata
  (gt_boxes, box_image_idx, box_label, has_grounding) alongside `labels`.
- `NuScenesVQADatasetV2` parses each sample's `gpt` JSON, validates the
  grounding entries, derives ground-truth boxes in 0–1000 normalized coords,
  and threads everything into the model inputs.
- `NuScenesDataCollatorV2` pads the new mask tensors with `IGNORE_INDEX` /
  `False` to match `labels`, and keeps `gt_boxes` / `box_image_idx` /
  `box_label` as ragged Python lists (1 entry per sample). HF Trainer's
  `_prepare_inputs` ignores non-tensor list entries, so we manually pop them
  inside `compute_loss`.
- `CompositeWSDTrainer` overrides `compute_loss` (in addition to the existing
  WSD `create_scheduler`). Loss = base CE + w_ans·L_ans + w_gate·L_gate +
  λ_view·L_view + λ_iou·L_iou + λ_klal·L_klal. KLAL is wired but executes
  only when `lam_klal > 0`; this PR leaves it at 0 so FlashAttention stays
  on. The decision of `output_attentions` is taken **once** at construction
  time from `lam_klal > 0`, so the forward graph is identical across DDP
  ranks (md Section 8).
- Auxiliary terms are individually toggled by their λ; setting λ=0 disables
  the term cleanly (no extra compute beyond a 0-tensor add).

Usage:

    torchrun --nproc_per_node=N train_nuscenes_qwen3vl_v2.py \\
        --mode lora --bf16 \\
        --train_data_path sft_dataset/sft_train_qwen3vl_OBS.json \\
        --eval_dataset_paths_json <stage_dir>/eval_dataset_paths.json \\
        --w_ans 2.0 --w_gate 1.5 --lam_view 0.5 --lam_iou 0.5 --lam_klal 0.0 \\
        --use_wsd_scheduler True --wsd_warmup_ratio 0.1 --wsd_decay_ratio 0.2 \\
        --output_dir <stage_dir>

Driven by `qwen-vl-finetune/scripts/run_curriculum_v2.sh` +
`qwen-vl-finetune/configs/curriculum_v2.yaml`.
"""

from __future__ import annotations

import copy
import json
import os
import pathlib
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Trainer,
    TrainingArguments as HfTrainingArguments,
)

FINETUNE_ROOT = os.path.join(os.path.dirname(__file__), "qwen-vl-finetune")
sys.path.insert(0, os.path.abspath(FINETUNE_ROOT))

from qwenvl.data.rope2d import get_rope_index_25
from qwenvl.train.composite_loss import (
    CompositeLossConfig,
    base_ce,
    iou_aware_ce_tier2,
    klal_loss_stub,
    masked_token_ce,
    view_classification_loss,
)
from qwenvl.train.token_role_masks import (
    IGNORE_INDEX,
    VIEW_DIGIT_IDS,
    build_assistant_masks,
    verify_landmarks,
)
from qwenvl.train.trainer import replace_qwen2_vl_attention_class
from qwenvl.train.wsd_scheduler import get_wsd_schedule


local_rank: Optional[int] = None


def rank0_print(*args):
    if local_rank == 0 or local_rank is None:
        print(*args)


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="Qwen/Qwen3-VL-8B")
    mode: str = field(default="lora")
    tune_mm_vision: bool = field(default=False)
    tune_mm_mlp: bool = field(default=True)
    tune_mm_llm: bool = field(default=True)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    )
    lora_pretrained: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    train_data_path: str = field(default="sft_dataset/sft_train_qwen3vl.json")
    val_data_path: str = field(default="sft_dataset/sft_val_qwen3vl.json")
    eval_dataset_paths_json: Optional[str] = field(default=None)
    data_flatten: bool = field(default=False)
    max_pixels: int = field(default=1_440_208)
    min_pixels: int = field(default=28 * 28 * 16)
    # Drop samples whose tokenized assistant turn exceeds this many tokens,
    # to avoid the tail-truncation gotcha (md Section 8 item 6). 0 disables.
    max_assistant_tokens: int = field(default=6000)


@dataclass
class TrainingArguments(HfTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=16384)
    mm_projector_lr: Optional[float] = field(default=None)
    vision_tower_lr: Optional[float] = field(default=None)
    use_wsd_scheduler: bool = field(default=False)
    wsd_warmup_ratio: float = field(default=0.10)
    wsd_decay_ratio: float = field(default=0.20)
    # HF Trainer's RemoveColumnsCollator strips any dataset key that's not a
    # parameter of model.forward(). v2 emits token-role masks + ragged
    # grounding metadata that the model forward doesn't know about — keep
    # them so the v2 collator + compute_loss can see them.
    remove_unused_columns: bool = field(default=False)

    # Composite-loss knobs (md Section 10).
    w_ans:   float = field(default=2.0)
    w_gate:  float = field(default=1.5)
    lam_view: float = field(default=0.5)
    lam_iou:  float = field(default=0.5)
    lam_klal: float = field(default=0.0)
    klal_layers: str = field(default="-1",
                             metadata={"help": "Comma-separated layer indices for KLAL"})
    iou_alpha: float = field(default=1.0)
    # JSON-encoded view class-weight vector (6 floats) or "auto" for inverse-
    # frequency from the training-set histogram (computed at startup).
    view_class_weight: str = field(default="")


# ---------------------------------------------------------------------------
# Tokenization with system prompt + token-role mask emission
# ---------------------------------------------------------------------------
def preprocess_with_system_prompt_v2(
    sources: List[List[Dict]],
    tokenizer,
    grid_thw_image: Optional[List[int]] = None,
):
    """Same conversation tokenization as v1, but also builds and concatenates
    token-role masks. System/human/assistant-header positions get all-zero
    masks (and IGNORE_INDEX targets); the assistant-body span carries the
    real per-token masks from `build_assistant_masks`.
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
    out_input_ids: List[List[int]] = []
    out_labels: List[List[int]] = []
    out_answer_mask: List[List[bool]] = []
    out_gate_mask: List[List[bool]] = []
    out_coord_mask: List[List[bool]] = []
    out_image_idx_mask: List[List[bool]] = []
    out_image_idx_target: List[List[int]] = []
    out_coord_box_id: List[List[int]] = []

    for source in sources:
        input_id: List[int] = []
        target: List[int] = []
        answer_m: List[bool] = []
        gate_m: List[bool] = []
        coord_m: List[bool] = []
        image_idx_m: List[bool] = []
        image_idx_t: List[int] = []
        coord_bid: List[int] = []

        for conv in source:
            role_key = conv.get("from", conv.get("role", ""))
            content = conv.get("value", conv.get("content", ""))
            role = roles.get(role_key, role_key)

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
            if not isinstance(encode_id, list):
                encode_id = encode_id["input_ids"]
            input_id += encode_id

            n = len(encode_id)
            if role in ("user", "system"):
                target += [IGNORE_INDEX] * n
                answer_m       += [False] * n
                gate_m         += [False] * n
                coord_m        += [False] * n
                image_idx_m    += [False] * n
                image_idx_t    += [IGNORE_INDEX] * n
                coord_bid      += [IGNORE_INDEX] * n
            else:  # assistant
                # First 3 tokens (<|im_start|>, assistant, \n) are header,
                # masked from loss + no role assignment.
                tgt = list(encode_id)
                tgt[:3] = [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]
                target += tgt

                masks = build_assistant_masks(encode_id, tokenizer)
                answer_m       += masks.answer_token_mask
                gate_m         += masks.gate_token_mask
                coord_m        += masks.coord_token_mask
                image_idx_m    += masks.image_idx_mask
                image_idx_t    += masks.image_idx_target
                coord_bid      += masks.coord_box_id

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"

        out_input_ids.append(input_id)
        out_labels.append(target)
        out_answer_mask.append(answer_m)
        out_gate_mask.append(gate_m)
        out_coord_mask.append(coord_m)
        out_image_idx_mask.append(image_idx_m)
        out_image_idx_target.append(image_idx_t)
        out_coord_box_id.append(coord_bid)

    return {
        "input_ids":         torch.tensor(out_input_ids, dtype=torch.long),
        "labels":            torch.tensor(out_labels, dtype=torch.long),
        "answer_token_mask": torch.tensor(out_answer_mask, dtype=torch.bool),
        "gate_token_mask":   torch.tensor(out_gate_mask, dtype=torch.bool),
        "coord_token_mask":  torch.tensor(out_coord_mask, dtype=torch.bool),
        "image_idx_mask":    torch.tensor(out_image_idx_mask, dtype=torch.bool),
        "image_idx_target":  torch.tensor(out_image_idx_target, dtype=torch.long),
        "coord_box_id":      torch.tensor(out_coord_box_id, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Grounding parser — derives GT boxes from the JSON `ref` field
# ---------------------------------------------------------------------------
_BOX_RE = re.compile(
    r"<\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>"
)
_LABEL_RE = re.compile(
    r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>"
)


def parse_grounding_entries(grounding_list: Sequence[Dict]) -> Tuple[
    List[Tuple[int, int, int, int]], List[int], List[str]
]:
    """Return (boxes 0..1000 xyxy, image_idx 1..6, label strings) ordered as
    in the JSON. Entries that fail to parse are dropped to keep the per-box
    arrays in lock-step with the in-text boxes the tokenizer sees."""
    boxes, image_idx, labels = [], [], []
    for g in grounding_list:
        ref = g.get("ref", "")
        m = _BOX_RE.search(ref)
        if not m:
            continue
        x1, y1, x2, y2 = (int(v) for v in m.groups())
        if not (x1 < x2 and y1 < y2 and all(0 <= v <= 1000 for v in (x1, y1, x2, y2))):
            continue
        idx = g.get("image_idx")
        if not isinstance(idx, int) or not (1 <= idx <= 6):
            continue
        lab_m = _LABEL_RE.search(ref)
        label = lab_m.group(1).strip() if lab_m else ""
        boxes.append((x1, y1, x2, y2))
        image_idx.append(idx)
        labels.append(label)
    return boxes, image_idx, labels


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class NuScenesVQADatasetV2(Dataset):
    REAR_IMAGE_INDICES = {3, 4, 5}

    def __init__(
        self,
        data_path: str,
        tokenizer,
        image_processor,
        max_pixels: int = 1_440_208,
        min_pixels: int = 28 * 28 * 16,
        max_assistant_tokens: int = 6000,
    ):
        rank0_print(f"Loading SFT data from: {data_path}")
        with open(data_path) as f:
            self.data = json.load(f)
        rank0_print(f"  loaded {len(self.data)} samples")

        if max_assistant_tokens > 0:
            before = len(self.data)
            self.data = self._prefilter_long(self.data, tokenizer, max_assistant_tokens)
            dropped = before - len(self.data)
            if dropped:
                rank0_print(
                    f"  dropped {dropped} samples with assistant token count > "
                    f"{max_assistant_tokens} (truncation safety; md §8 item 6)"
                )

        self.tokenizer = tokenizer
        self.image_processor = copy.deepcopy(image_processor)
        self.get_rope_index = get_rope_index_25
        self.image_processor.max_pixels = max_pixels
        self.image_processor.min_pixels = min_pixels
        self.image_processor.size["longest_edge"] = max_pixels
        self.image_processor.size["shortest_edge"] = min_pixels

    @staticmethod
    def _prefilter_long(data, tokenizer, threshold):
        out = []
        for s in data:
            gpt = s["conversations"][2]["value"]
            n = len(tokenizer.encode(gpt, add_special_tokens=False))
            if n <= threshold:
                out.append(s)
        return out

    def __len__(self):
        return len(self.data)

    def process_image(self, image_path: str, flip_horizontal: bool = False):
        img = Image.open(image_path).convert("RGB")
        if flip_horizontal:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        processor = copy.deepcopy(self.image_processor)
        visual = processor.preprocess(img, return_tensors="pt")
        image_tensor = visual["pixel_values"]
        if isinstance(image_tensor, list):
            image_tensor = image_tensor[0]
        grid_thw = visual["image_grid_thw"][0]
        return image_tensor, grid_thw

    def __getitem__(self, idx):
        for attempt in range(3):
            try:
                return self._get_item(idx)
            except Exception as e:
                print(f"[Attempt {attempt}] Failed sample {idx}: {e}")
                time.sleep(0.5)
        return self._get_item(idx)

    def _get_item(self, idx):
        sample = self.data[idx]
        image_paths = sample["image"]
        conversations = sample["conversations"]

        # Vision
        images, grid_thws = [], []
        for img_idx, p in enumerate(image_paths):
            flip = img_idx in self.REAR_IMAGE_INDICES
            it, gt = self.process_image(p, flip_horizontal=flip)
            images.append(it)
            grid_thws.append(gt)
        merge_size = self.image_processor.merge_size
        grid_thw_merged = [t.prod() // (merge_size ** 2) for t in grid_thws]

        # Tokenization + masks
        data_dict = preprocess_with_system_prompt_v2(
            [conversations], self.tokenizer, grid_thw_image=grid_thw_merged
        )

        # Grounding metadata
        gpt = conversations[2]["value"]
        try:
            parsed = json.loads(gpt)
            grounding_list = parsed.get("grounding", []) or []
        except json.JSONDecodeError:
            grounding_list = []
        gt_boxes, box_image_idx, box_label = parse_grounding_entries(grounding_list)
        has_grounding = len(gt_boxes) > 0

        # P1-4: assert mask construction agrees with parsed grounding. If
        # `has_grounding` is True, both coord_token_mask and image_idx_mask
        # must fire. A mismatch indicates either:
        #   (a) token-id landmark drift (regenerate IDs in token_role_masks.py),
        #   (b) a category emitting box markup we don't recognize.
        if has_grounding:
            cm = data_dict["coord_token_mask"][0]
            im = data_dict["image_idx_mask"][0]
            if not (cm.any().item() and im.any().item()):
                # Diagnostic dump to help triage future regressions.
                raise RuntimeError(
                    f"Token-role mask construction failed for grounded sample "
                    f"with {len(gt_boxes)} box(es).\n"
                    f"  coord_token_mask.any()={cm.any().item()} "
                    f"image_idx_mask.any()={im.any().item()}\n"
                    f"  first GT box={gt_boxes[0]} image_idx={box_image_idx[0]} "
                    f"label={box_label[0]!r}\n"
                    f"  sample image[0]={image_paths[0]}\n"
                    f"  Inspect with qwenvl.train.token_role_masks.build_assistant_masks "
                    f"on this sample's assistant turn."
                )

        # RoPE
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
            [t.unsqueeze(0) for t in grid_thws], dim=0
        )

        # Per-sample grounding payload (kept as Python objects; collator
        # bundles them into a list-of-len-B and pop()s in compute_loss).
        data_dict["_grounding"] = {
            "gt_boxes": gt_boxes,
            "box_image_idx": box_image_idx,
            "box_label": box_label,
            "has_grounding": has_grounding,
        }
        return data_dict


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------
def _pad_2d(seqs: List[torch.Tensor], pad_value, dtype) -> torch.Tensor:
    """Right-pad a list of 1D tensors to (B, T_max), with `pad_value`."""
    L = max(s.size(0) for s in seqs)
    out = torch.full((len(seqs), L), pad_value, dtype=dtype)
    for i, s in enumerate(seqs):
        out[i, : s.size(0)] = s
    return out


def _pad_position_ids(tensor_list: List[torch.Tensor]) -> torch.Tensor:
    """RoPE position ids are shape (3, 1, T). Pad along T to T_max, pad value 1."""
    Lmax = max(t.shape[2] for t in tensor_list)
    padded = []
    for t in tensor_list:
        pad_len = Lmax - t.shape[2]
        padded.append(F.pad(t, (0, pad_len), mode="constant", value=1))
    return torch.cat(padded, dim=1)


@dataclass
class NuScenesDataCollatorV2:
    tokenizer: Any

    def __call__(self, instances: List[Dict]) -> Dict[str, Any]:
        # ---- 1D pad: input_ids / labels / all masks
        ids   = [inst["input_ids"].squeeze(0)         for inst in instances]
        labs  = [inst["labels"].squeeze(0)            for inst in instances]
        am    = [inst["answer_token_mask"].squeeze(0) for inst in instances]
        gm    = [inst["gate_token_mask"].squeeze(0)   for inst in instances]
        cm    = [inst["coord_token_mask"].squeeze(0)  for inst in instances]
        im    = [inst["image_idx_mask"].squeeze(0)    for inst in instances]
        it    = [inst["image_idx_target"].squeeze(0)  for inst in instances]
        cbid  = [inst["coord_box_id"].squeeze(0)      for inst in instances]
        pos   = [inst["position_ids"]                 for inst in instances]

        pad_id = self.tokenizer.pad_token_id
        input_ids = _pad_2d(ids, pad_id, torch.long)
        labels    = _pad_2d(labs, IGNORE_INDEX, torch.long)
        answer_token_mask = _pad_2d(am, False, torch.bool)
        gate_token_mask   = _pad_2d(gm, False, torch.bool)
        coord_token_mask  = _pad_2d(cm, False, torch.bool)
        image_idx_mask    = _pad_2d(im, False, torch.bool)
        image_idx_target  = _pad_2d(it, IGNORE_INDEX, torch.long)
        coord_box_id      = _pad_2d(cbid, IGNORE_INDEX, torch.long)
        position_ids      = _pad_position_ids(pos)

        # Truncate to model_max_length.
        L = self.tokenizer.model_max_length
        input_ids = input_ids[:, :L]
        labels    = labels[:, :L]
        answer_token_mask = answer_token_mask[:, :L]
        gate_token_mask   = gate_token_mask[:, :L]
        coord_token_mask  = coord_token_mask[:, :L]
        image_idx_mask    = image_idx_mask[:, :L]
        image_idx_target  = image_idx_target[:, :L]
        coord_box_id      = coord_box_id[:, :L]
        position_ids      = position_ids[:, :, :L]

        batch: Dict[str, Any] = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(pad_id),
            position_ids=position_ids,
            answer_token_mask=answer_token_mask,
            gate_token_mask=gate_token_mask,
            coord_token_mask=coord_token_mask,
            image_idx_mask=image_idx_mask,
            image_idx_target=image_idx_target,
            coord_box_id=coord_box_id,
        )

        # Vision
        images = [inst["pixel_values"] for inst in instances if "pixel_values" in inst]
        if images:
            batch["pixel_values"] = torch.cat(images, dim=0)
            grid_thws = [inst["image_grid_thw"] for inst in instances]
            batch["image_grid_thw"] = torch.cat(grid_thws, dim=0)
        else:
            batch["pixel_values"] = None
            batch["image_grid_thw"] = None
        batch["pixel_values_videos"] = None
        batch["video_grid_thw"] = None

        # Per-sample ragged grounding metadata (lists, NOT tensors — Trainer
        # passes them through unchanged; compute_loss pops them out).
        gt_boxes      = [inst["_grounding"]["gt_boxes"]      for inst in instances]
        box_image_idx = [inst["_grounding"]["box_image_idx"] for inst in instances]
        box_label     = [inst["_grounding"]["box_label"]     for inst in instances]
        has_grounding = torch.tensor(
            [inst["_grounding"]["has_grounding"] for inst in instances],
            dtype=torch.bool,
        )
        batch["_gt_boxes"] = gt_boxes
        batch["_box_image_idx"] = box_image_idx
        batch["_box_label"] = box_label
        batch["_has_grounding"] = has_grounding

        return batch


# ---------------------------------------------------------------------------
# Model setup (LoRA — identical to v1)
# ---------------------------------------------------------------------------
def setup_full_finetune(model, model_args):
    for p in model.visual.named_parameters():
        p[1].requires_grad = model_args.tune_mm_vision
    for p in model.visual.merger.named_parameters():
        p[1].requires_grad = model_args.tune_mm_mlp
    for p in model.model.named_parameters():
        p[1].requires_grad = model_args.tune_mm_llm
    model.lm_head.requires_grad = model_args.tune_mm_llm
    return model


def setup_lora(model, model_args):
    for p in model.parameters():
        p.requires_grad = False
    if model_args.lora_pretrained:
        from peft import PeftModel
        rank0_print(f"Warm-starting LoRA from: {model_args.lora_pretrained}")
        model = PeftModel.from_pretrained(model, model_args.lora_pretrained, is_trainable=True)
        model.print_trainable_parameters()
        return model
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
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# CompositeWSDTrainer
# ---------------------------------------------------------------------------
class CompositeWSDTrainer(Trainer):
    """HF Trainer override that:
      1. Replaces compute_loss with the composite loss (md Section 1).
      2. Swaps in the WSD LR schedule when args.use_wsd_scheduler is set
         (mirrors v1 WSDTrainer behavior).
    """

    def __init__(self, *args, loss_config: CompositeLossConfig, tokenizer_for_loss=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_config = loss_config
        # The composite loss decodes argmax token IDs back to coord text — we
        # need a tokenizer reference outside `self.processing_class` because
        # the latter may be wrapped.
        self.tokenizer_for_loss = tokenizer_for_loss or self.processing_class
        self._view_token_ids: Optional[torch.Tensor] = None
        self._view_class_weight: Optional[torch.Tensor] = None
        # Decided ONCE at construction (DDP-safe per md Section 8).
        self._needs_output_attentions = loss_config.needs_output_attentions

        # Track current LR by group for logging alongside loss components.
        self._last_loss_breakdown: Dict[str, float] = {}
        self._last_loss_was_training: bool = True

        rank0_print(
            f"[CompositeWSDTrainer] loss config: w_ans={loss_config.w_ans} "
            f"w_gate={loss_config.w_gate} lam_view={loss_config.lam_view} "
            f"lam_iou={loss_config.lam_iou} lam_klal={loss_config.lam_klal} "
            f"output_attentions={self._needs_output_attentions}"
        )

    # ---- LR schedule (reuse WSD from v1)
    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if not getattr(self.args, "use_wsd_scheduler", False):
            return super().create_scheduler(num_training_steps, optimizer)
        if self.lr_scheduler is not None:
            return self.lr_scheduler
        opt = optimizer if optimizer is not None else self.optimizer
        self.lr_scheduler = get_wsd_schedule(
            opt,
            num_training_steps=num_training_steps,
            warmup_ratio=self.args.wsd_warmup_ratio,
            decay_ratio=self.args.wsd_decay_ratio,
        )
        rank0_print(
            f"[CompositeWSDTrainer] WSD scheduler: total_steps={num_training_steps} "
            f"warmup_ratio={self.args.wsd_warmup_ratio} decay_ratio={self.args.wsd_decay_ratio}"
        )
        return self.lr_scheduler

    def _ensure_view_buffers(self, device, dtype):
        if self._view_token_ids is None:
            self._view_token_ids = torch.tensor(VIEW_DIGIT_IDS, dtype=torch.long, device=device)
        if (
            self.loss_config.view_class_weight is not None
            and self._view_class_weight is None
        ):
            # Aux terms run in fp32 (md §8). `dtype` arg is unused — kept on
            # the signature so future buffers (if any) can opt into the model
            # dtype individually.
            del dtype
            self._view_class_weight = torch.tensor(
                self.loss_config.view_class_weight,
                dtype=torch.float32,
                device=device,
            )

    # ---- Composite loss
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Pop the aux fields BEFORE handing to the model.
        answer_mask   = inputs.pop("answer_token_mask")
        gate_mask     = inputs.pop("gate_token_mask")
        coord_mask    = inputs.pop("coord_token_mask")
        image_idx_m   = inputs.pop("image_idx_mask")
        image_idx_t   = inputs.pop("image_idx_target")
        coord_box_id  = inputs.pop("coord_box_id")
        gt_boxes      = inputs.pop("_gt_boxes")
        box_image_idx = inputs.pop("_box_image_idx")
        _box_label    = inputs.pop("_box_label")          # eval-only; not in loss
        has_grounding = inputs.pop("_has_grounding")

        labels = inputs["labels"]

        # Forward — output_attentions decided once at construction. We do NOT
        # pass `labels` so the model returns logits and we apply base CE
        # manually (preserves num_items_in_batch normalization).
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(
            **model_inputs,
            output_attentions=self._needs_output_attentions,
        )
        logits = outputs.logits
        device = logits.device

        # Base CE — normalized by num_items_in_batch (matches HF's
        # ForCausalLMLoss semantics so grad-accum equals a larger batch).
        loss = base_ce(logits, labels, num_items_in_batch=num_items_in_batch)
        # Per-component breakdown logs BOTH the raw L_* (for trend monitoring)
        # AND the actual contribution to total (lambda * L_*, suffix `_w`).
        # By design, loss_base_ce + loss_answer_w + loss_gate_w + loss_view_w
        # + loss_iou_w + loss_klal_w == loss (within float precision).
        breakdown = {"loss_base_ce": float(loss.detach())}
        # Track grounding presence in this microbatch so the user can spot
        # "loss_view=0 because no grounded samples" vs "loss_view=0 because of
        # a bug." Fraction of samples with any grounded coord token.
        grounded_in_batch = float(coord_mask.any(dim=1).float().mean().detach())
        breakdown["grounded_frac"] = grounded_in_batch

        # Answer-field weighting (all samples)
        if self.loss_config.w_ans > 0:
            l_ans = masked_token_ce(logits, labels, answer_mask)
            contrib = self.loss_config.w_ans * l_ans
            loss = loss + contrib
            breakdown["loss_answer"] = float(l_ans.detach())
            breakdown["loss_answer_w"] = float(contrib.detach())

        # Presence-gate (all samples)
        if self.loss_config.w_gate > 0:
            l_gate = masked_token_ce(logits, labels, gate_mask)
            contrib = self.loss_config.w_gate * l_gate
            loss = loss + contrib
            breakdown["loss_gate"] = float(l_gate.detach())
            breakdown["loss_gate_w"] = float(contrib.detach())

        # View classification (grounded samples; mask is empty for non-grounded)
        if self.loss_config.lam_view > 0:
            self._ensure_view_buffers(device, logits.dtype)
            l_view = view_classification_loss(
                logits, image_idx_m, image_idx_t,
                view_token_ids=self._view_token_ids,
                class_weight=self._view_class_weight,
            )
            contrib = self.loss_config.lam_view * l_view
            loss = loss + contrib
            breakdown["loss_view"] = float(l_view.detach())
            breakdown["loss_view_w"] = float(contrib.detach())

        # IoU-aware CE (Tier 2)
        if self.loss_config.lam_iou > 0:
            l_iou = iou_aware_ce_tier2(
                logits, labels,
                coord_token_mask=coord_mask,
                coord_box_id=coord_box_id,
                gt_boxes=gt_boxes,
                box_image_idx=box_image_idx,
                tokenizer=self.tokenizer_for_loss,
                iou_alpha=self.loss_config.iou_alpha,
            )
            contrib = self.loss_config.lam_iou * l_iou
            loss = loss + contrib
            breakdown["loss_iou"] = float(l_iou.detach())
            breakdown["loss_iou_w"] = float(contrib.detach())

        # KLAL — scaffolded; no execution unless lam_klal>0 AND attentions returned
        if self.loss_config.lam_klal > 0:
            l_klal = klal_loss_stub(
                getattr(outputs, "attentions", None),
                self.loss_config.klal_layers,
                gt_boxes, box_image_idx, has_grounding,
            ).to(loss.device)
            contrib = self.loss_config.lam_klal * l_klal
            loss = loss + contrib
            breakdown["loss_klal"] = float(l_klal.detach())
            breakdown["loss_klal_w"] = float(contrib.detach())

        # Mark whether this call was from train or eval so the log() override
        # can scope which logs the breakdown bleeds into. HF Trainer toggles
        # `self.model.training` correctly under prediction_step.
        self._last_loss_breakdown = breakdown
        self._last_loss_was_training = model.training
        return (loss, outputs) if return_outputs else loss

    # Surface per-component loss numbers in HF's `log()` output. Called by
    # Trainer at every `logging_steps`; the dict we add is merged into the
    # log entry (and forwarded to all `report_to` backends).
    #
    # We only merge the breakdown into TRAINING-step logs. During eval, HF
    # calls log() once per dict-eval category with keys like `eval_<CAT>_loss`;
    # the breakdown's per-component keys (`loss_view`, ...) would attach to
    # whichever category log fires last and look misleadingly per-category.
    # Skip on eval logs so the only place breakdown appears is training steps.
    def log(self, logs, *args, **kwargs):
        is_eval_log = any(k.startswith("eval_") for k in logs)
        if self._last_loss_breakdown and not is_eval_log:
            for k, v in self._last_loss_breakdown.items():
                logs[k] = v
        return super().log(logs, *args, **kwargs)


# ---------------------------------------------------------------------------
# Helpers — class-weight inverse-frequency, layer parsing
# ---------------------------------------------------------------------------
def _compute_view_inverse_frequency(data_path: str) -> List[float]:
    """Walk an SFT JSON and compute inverse-frequency weights over image_idx
    1..6. Returns a length-6 list usable as the `weight` arg of
    F.cross_entropy. Falls back to uniform if grounding is sparse."""
    counts = [0] * 6
    with open(data_path) as f:
        data = json.load(f)
    for s in data:
        try:
            parsed = json.loads(s["conversations"][2]["value"])
        except (KeyError, IndexError, json.JSONDecodeError):
            continue
        for g in parsed.get("grounding", []) or []:
            i = g.get("image_idx")
            if isinstance(i, int) and 1 <= i <= 6:
                counts[i - 1] += 1
    total = sum(counts)
    if total == 0:
        return [1.0] * 6
    # Smoothed inverse-frequency, normalized to mean 1.
    inv = [(total / (6 * (c + 1))) for c in counts]
    mean_inv = sum(inv) / 6
    return [w / mean_inv for w in inv]


def _parse_klal_layers(s: str) -> Tuple[int, ...]:
    s = (s or "").strip()
    if not s:
        return (-1,)
    return tuple(int(x) for x in s.split(",") if x.strip())


def _resolve_view_class_weight(spec: str, train_path: str) -> Optional[List[float]]:
    spec = (spec or "").strip()
    if not spec:
        return None
    if spec.lower() == "auto":
        rank0_print(f"  computing view class weights (inverse-frequency) from {train_path} ...")
        w = _compute_view_inverse_frequency(train_path)
        rank0_print(f"  view class weights = {[round(x, 3) for x in w]}")
        return w
    try:
        w = json.loads(spec)
    except json.JSONDecodeError:
        raise ValueError(f"--view_class_weight must be 'auto' or JSON list, got: {spec!r}")
    if not (isinstance(w, list) and len(w) == 6):
        raise ValueError(f"--view_class_weight JSON must be length 6, got: {w}")
    return [float(x) for x in w]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def train():
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    rank0_print("=" * 60)
    rank0_print(f"Training mode: {model_args.mode}")
    rank0_print(f"Model: {model_args.model_name_or_path}")
    rank0_print(f"max_pixels: {data_args.max_pixels} | min_pixels: {data_args.min_pixels}")
    rank0_print(f"model_max_length: {training_args.model_max_length}")
    rank0_print("=" * 60)

    # Model
    from transformers import Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16 if training_args.bf16 else None,
    )
    model.config.use_cache = False

    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if model_args.mode == "lora":
        model = setup_lora(model, model_args)
    elif model_args.mode == "full":
        model = setup_full_finetune(model, model_args)
    else:
        raise ValueError(f"Unknown training mode: {model_args.mode}")

    # Tokenizer + processor
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # Validate tokenizer landmarks before training — fail fast on vocab drift.
    landmarks = verify_landmarks(tokenizer)
    bad = [k for k, (exp, act) in landmarks.items() if exp != act]
    if bad:
        raise RuntimeError(
            f"Token-role mask landmarks failed verification for keys {bad}. "
            f"Tokenizer vocab may have drifted from Qwen3-VL 8B Instruct. "
            f"Inspect qwen-vl-finetune/qwenvl/train/token_role_masks.py to "
            f"regenerate landmark IDs.\nFull report: {landmarks}"
        )

    image_processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path
    ).image_processor

    # Loss config
    view_w = _resolve_view_class_weight(
        training_args.view_class_weight, data_args.train_data_path
    )
    loss_config = CompositeLossConfig(
        w_ans=training_args.w_ans,
        w_gate=training_args.w_gate,
        lam_view=training_args.lam_view,
        lam_iou=training_args.lam_iou,
        lam_klal=training_args.lam_klal,
        klal_layers=_parse_klal_layers(training_args.klal_layers),
        iou_alpha=training_args.iou_alpha,
        view_class_weight=view_w,
    )

    # Datasets
    train_dataset = NuScenesVQADatasetV2(
        data_path=data_args.train_data_path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_pixels=data_args.max_pixels,
        min_pixels=data_args.min_pixels,
        max_assistant_tokens=data_args.max_assistant_tokens,
    )

    eval_dataset = None
    if data_args.eval_dataset_paths_json and os.path.exists(data_args.eval_dataset_paths_json):
        with open(data_args.eval_dataset_paths_json) as f:
            eval_paths = json.load(f)
        eval_dataset = {}
        for cat_name, cat_path in eval_paths.items():
            if not os.path.exists(cat_path):
                rank0_print(f"[warn] eval dataset for '{cat_name}' not found: {cat_path}")
                continue
            eval_dataset[cat_name] = NuScenesVQADatasetV2(
                data_path=cat_path,
                tokenizer=tokenizer,
                image_processor=image_processor,
                max_pixels=data_args.max_pixels,
                min_pixels=data_args.min_pixels,
                max_assistant_tokens=data_args.max_assistant_tokens,
            )
        rank0_print(f"Loaded dict-form eval_dataset: {sorted(eval_dataset)}")
    elif data_args.val_data_path and os.path.exists(data_args.val_data_path):
        eval_dataset = NuScenesVQADatasetV2(
            data_path=data_args.val_data_path,
            tokenizer=tokenizer,
            image_processor=image_processor,
            max_pixels=data_args.max_pixels,
            min_pixels=data_args.min_pixels,
            max_assistant_tokens=data_args.max_assistant_tokens,
        )

    data_collator = NuScenesDataCollatorV2(tokenizer=tokenizer)

    trainer = CompositeWSDTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        loss_config=loss_config,
        tokenizer_for_loss=tokenizer,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        rank0_print("Checkpoint found, resuming training...")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    if trainer.is_world_process_zero():
        trainer.model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
