"""
Curriculum config loader.

Reads a YAML file (see qwen-vl-finetune/configs/curriculum_v1.yaml) and merges
`defaults` into every stage entry, producing a list of fully-resolved Stage
dataclasses for the orchestrator to consume.

This file is intended to be invoked both from Python imports and from the bash
orchestrator via `python -m qwenvl.curriculum.config --emit_shell <path>`,
which prints `KEY=VALUE` lines that the shell script `eval`s.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Stage:
    name: str                              # category code, e.g. "OBS"
    epochs: int = 1
    # If > 0, caps optimizer steps for this stage and overrides `epochs`.
    # Useful for smoke runs.
    max_steps: int = -1
    peak_lr: float = 2.0e-4
    wsd: List[float] = field(default_factory=lambda: [0.10, 0.70, 0.20])
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 2
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    model_max_length: int = 16384
    max_pixels: int = 1_440_208
    min_pixels: int = 784
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
    bf16: bool = True
    report_to: str = "tensorboard"
    full_val_for_stage_end_eval: bool = False
    # v2 composite-loss knobs (Stage may override per-category).
    w_ans: float = 0.0
    w_gate: float = 0.0
    lam_view: float = 0.0
    lam_iou: float = 0.0
    lam_klal: float = 0.0
    klal_layers: str = "-1"
    iou_alpha: float = 1.0
    view_class_weight: str = ""    # "" | "auto" | JSON list of 6 floats
    max_assistant_tokens: int = 6000


@dataclass
class Curriculum:
    output_root: str
    base_model: str
    train_data_dir: str
    eval_subset_dir: str
    eval_categories: List[str]
    stages: List[Stage]


def _merge(defaults: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(defaults)
    merged.update({k: v for k, v in override.items() if v is not None})
    return merged


def load_curriculum(path: str) -> Curriculum:
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "pyyaml is required to parse curriculum configs. Install via `pip install pyyaml`."
        ) from e

    with open(path) as f:
        raw = yaml.safe_load(f)

    defaults = raw.get("defaults", {}) or {}
    # Top-level `loss:` block flattens into defaults (per md Section 10
    # config schema). Per-stage overrides may also live under a `loss:` key.
    loss_defaults = raw.get("loss", {}) or {}
    defaults = _merge(defaults, loss_defaults)

    stages_raw = raw["stages"]
    stage_fields = {f.name for f in Stage.__dataclass_fields__.values()}

    stages: List[Stage] = []
    for st in stages_raw:
        # Allow a `loss:` sub-block at the stage level too.
        st_loss = st.pop("loss", {}) if isinstance(st, dict) else {}
        merged = _merge(defaults, st)
        merged = _merge(merged, st_loss)
        # Filter to fields Stage accepts; reject unknown keys to fail loudly.
        unknown = set(merged) - stage_fields
        if unknown:
            raise ValueError(f"Unknown stage keys: {unknown}")
        stages.append(Stage(**merged))

    return Curriculum(
        output_root=raw["output_root"],
        base_model=raw["base_model"],
        train_data_dir=raw["train_data_dir"],
        eval_subset_dir=raw["eval_subset_dir"],
        eval_categories=list(raw["eval_categories"]),
        stages=stages,
    )


def _stage_dir(curriculum: Curriculum, idx: int) -> str:
    name = curriculum.stages[idx].name
    return os.path.join(curriculum.output_root, f"stage_{idx:02d}_{name}")


def emit_shell(curriculum: Curriculum, stage_idx: int) -> str:
    """
    Emit KEY=VALUE bash assignments for the requested stage. The orchestrator
    eval()s this string so subsequent torchrun invocations pick up the values.
    """
    stage = curriculum.stages[stage_idx]
    out_dir = _stage_dir(curriculum, stage_idx)
    lora_pretrained = ""
    if stage_idx > 0:
        lora_pretrained = _stage_dir(curriculum, stage_idx - 1)

    eval_paths = {
        cat: os.path.abspath(
            os.path.join(curriculum.eval_subset_dir, f"sft_val_qwen3vl_{cat}_subset.json")
        )
        for cat in curriculum.eval_categories
    }
    eval_json_path = os.path.join(out_dir, "eval_dataset_paths.json")

    train_path = os.path.abspath(
        os.path.join(curriculum.train_data_dir, f"sft_train_qwen3vl_{stage.name}.json")
    )

    wsd_warmup, _wsd_stable, wsd_decay = stage.wsd

    lines = [
        f"STAGE_IDX={stage_idx}",
        f"STAGE_NAME={stage.name}",
        f"STAGE_OUTPUT_DIR={out_dir}",
        f"STAGE_TRAIN_DATA={train_path}",
        f"STAGE_EVAL_JSON={eval_json_path}",
        f"STAGE_LORA_PRETRAINED={lora_pretrained}",
        f"STAGE_BASE_MODEL={curriculum.base_model}",
        f"STAGE_EPOCHS={stage.epochs}",
        f"STAGE_MAX_STEPS={stage.max_steps}",
        f"STAGE_PEAK_LR={stage.peak_lr}",
        f"STAGE_WSD_WARMUP_RATIO={wsd_warmup}",
        f"STAGE_WSD_DECAY_RATIO={wsd_decay}",
        f"STAGE_PER_DEVICE_TRAIN_BATCH_SIZE={stage.per_device_train_batch_size}",
        f"STAGE_PER_DEVICE_EVAL_BATCH_SIZE={stage.per_device_eval_batch_size}",
        f"STAGE_GRADIENT_ACCUMULATION_STEPS={stage.gradient_accumulation_steps}",
        f"STAGE_EVAL_STEPS={stage.eval_steps}",
        f"STAGE_SAVE_STEPS={stage.save_steps}",
        f"STAGE_SAVE_TOTAL_LIMIT={stage.save_total_limit}",
        f"STAGE_WEIGHT_DECAY={stage.weight_decay}",
        f"STAGE_MAX_GRAD_NORM={stage.max_grad_norm}",
        f"STAGE_MODEL_MAX_LENGTH={stage.model_max_length}",
        f"STAGE_MAX_PIXELS={stage.max_pixels}",
        f"STAGE_MIN_PIXELS={stage.min_pixels}",
        f"STAGE_LORA_R={stage.lora_r}",
        f"STAGE_LORA_ALPHA={stage.lora_alpha}",
        f"STAGE_LORA_DROPOUT={stage.lora_dropout}",
        f"STAGE_LORA_TARGET_MODULES={stage.lora_target_modules}",
        f"STAGE_GRADIENT_CHECKPOINTING={'true' if stage.gradient_checkpointing else 'false'}",
        f"STAGE_DATALOADER_NUM_WORKERS={stage.dataloader_num_workers}",
        f"STAGE_BF16={'true' if stage.bf16 else 'false'}",
        f"STAGE_REPORT_TO={stage.report_to}",
        f"STAGE_FULL_VAL_FOR_END_EVAL={'true' if stage.full_val_for_stage_end_eval else 'false'}",
        # v2 composite-loss knobs. Empty `view_class_weight` is emitted as
        # the empty string (= disabled); shell quotes it as needed.
        f"STAGE_W_ANS={stage.w_ans}",
        f"STAGE_W_GATE={stage.w_gate}",
        f"STAGE_LAM_VIEW={stage.lam_view}",
        f"STAGE_LAM_IOU={stage.lam_iou}",
        f"STAGE_LAM_KLAL={stage.lam_klal}",
        f"STAGE_KLAL_LAYERS={stage.klal_layers}",
        f"STAGE_IOU_ALPHA={stage.iou_alpha}",
        f"STAGE_VIEW_CLASS_WEIGHT={stage.view_class_weight}",
        f"STAGE_MAX_ASSISTANT_TOKENS={stage.max_assistant_tokens}",
    ]

    os.makedirs(out_dir, exist_ok=True)
    with open(eval_json_path, "w") as f:
        json.dump(eval_paths, f, indent=2)

    return "\n".join(lines)


def emit_summary(curriculum: Curriculum) -> str:
    return "\n".join(
        f"  stage {i:02d}  {st.name:>3s}  epochs={st.epochs}  peak_lr={st.peak_lr}  wsd={st.wsd}"
        for i, st in enumerate(curriculum.stages)
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--emit_shell", type=int, default=None,
                   help="Stage index to emit bash KEY=VALUE assignments for.")
    p.add_argument("--summary", action="store_true")
    p.add_argument("--num_stages", action="store_true",
                   help="Print number of stages and exit.")
    args = p.parse_args()

    curr = load_curriculum(args.config)

    if args.num_stages:
        print(len(curr.stages))
        return
    if args.summary:
        print(emit_summary(curr))
        return
    if args.emit_shell is not None:
        if not 0 <= args.emit_shell < len(curr.stages):
            print(f"stage_idx {args.emit_shell} out of range [0, {len(curr.stages)})",
                  file=sys.stderr)
            sys.exit(2)
        print(emit_shell(curr, args.emit_shell))
        return

    # Default: dump full resolved curriculum as JSON for inspection.
    print(json.dumps(asdict(curr), indent=2))


if __name__ == "__main__":
    main()
