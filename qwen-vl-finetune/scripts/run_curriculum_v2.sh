#!/bin/bash
# Curriculum-learning driver for Qwen3-VL 8B LoRA SFT — v2 (composite loss).
#
# Same structure as run_curriculum.sh but invokes train_nuscenes_qwen3vl_v2.py
# and threads the new loss knobs (w_ans/w_gate/lam_view/lam_iou/lam_klal/...).
# Each stage:
#   1. Resolves per-stage paths + loss hyperparams via the curriculum loader.
#   2. Builds dict-form eval_dataset JSON pointing at the per-category val
#      subsets, so per-category eval losses are logged every eval step.
#   3. Launches torchrun training with CompositeWSDTrainer.
#   4. Runs the v2 per-category generation eval (answer_acc, view_acc,
#      grounding_acc@0.8, grounding_format_valid).
#
# Usage:
#   bash qwen-vl-finetune/scripts/run_curriculum_v2.sh \
#       qwen-vl-finetune/configs/curriculum_v2.yaml
#
# Environment overrides:
#   NPROC_PER_NODE, NNODES, MASTER_ADDR, MASTER_PORT  (torchrun)
#   START_STAGE  (default 0)    — resume from a later stage
#   END_STAGE    (default last) — stop after this stage (inclusive)
#   SKIP_INTRAINING_EVAL=1      — disable HF Trainer's per-step dict-eval
#                                  (large speedup; eval LR-curves disappear)
#   SKIP_STAGE_END_EVAL=1       — skip generation eval after each stage's
#                                  training (run eval_curriculum_v2.sh later)
#   STAGE_END_EVAL_LIMIT=N      — cap samples per category for stage-end eval
#   EVAL_IOU_THRESHOLD          — IoU threshold for grounding_acc (default 0.8)
#
# Train-only mode (no evaluation during the run):
#   SKIP_INTRAINING_EVAL=1 SKIP_STAGE_END_EVAL=1 \
#       NPROC_PER_NODE=8 \
#       bash qwen-vl-finetune/scripts/run_curriculum_v2.sh CONFIG.yaml
#
# Run evaluations separately after training:
#   bash qwen-vl-finetune/scripts/eval_curriculum_v2.sh CONFIG.yaml

set -e

CONFIG=${1:?"usage: $0 <curriculum_config.yaml>"}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: config not found: $CONFIG" >&2
    exit 1
fi

export PYTHONPATH="$PROJECT_ROOT/qwen-vl-finetune:${PYTHONPATH:-}"

export NCCL_TIMEOUT=${NCCL_TIMEOUT:-3600}
export TORCH_NCCL_TIMEOUT_SEC=${TORCH_NCCL_TIMEOUT_SEC:-3600}
export TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-2000}
export TORCH_NCCL_DUMP_ON_TIMEOUT=${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
NNODES=${WORLD_SIZE:-1}

DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG-$PROJECT_ROOT/qwen-vl-finetune/scripts/zero2.json}"

NUM_STAGES=$(python -m qwenvl.curriculum.config "$CONFIG" --num_stages)
START_STAGE=${START_STAGE:-0}
END_STAGE=${END_STAGE:-$((NUM_STAGES - 1))}
EVAL_IOU_THRESHOLD=${EVAL_IOU_THRESHOLD:-0.8}

echo "============================================"
echo "  Curriculum v2: $CONFIG"
echo "  Stages: $NUM_STAGES total | running [$START_STAGE..$END_STAGE]"
echo "  GPUs:   $NPROC_PER_NODE"
echo "============================================"
python -m qwenvl.curriculum.config "$CONFIG" --summary
echo "============================================"

for STAGE_IDX in $(seq "$START_STAGE" "$END_STAGE"); do
    STAGE_ENV=$(python -m qwenvl.curriculum.config "$CONFIG" --emit_shell "$STAGE_IDX")
    eval "$STAGE_ENV"

    echo
    echo "============================================"
    echo "  STAGE $STAGE_IDX ($STAGE_NAME)"
    echo "  output:   $STAGE_OUTPUT_DIR"
    echo "  train:    $STAGE_TRAIN_DATA"
    echo "  warm:     ${STAGE_LORA_PRETRAINED:-<fresh>}"
    echo "  epochs:   $STAGE_EPOCHS  peak_lr: $STAGE_PEAK_LR"
    echo "  WSD:      warmup=$STAGE_WSD_WARMUP_RATIO decay=$STAGE_WSD_DECAY_RATIO"
    echo "  loss:     w_ans=$STAGE_W_ANS w_gate=$STAGE_W_GATE"
    echo "            lam_view=$STAGE_LAM_VIEW lam_iou=$STAGE_LAM_IOU lam_klal=$STAGE_LAM_KLAL"
    echo "============================================"

    if [ ! -f "$STAGE_TRAIN_DATA" ]; then
        echo "ERROR: train file missing: $STAGE_TRAIN_DATA" >&2
        exit 1
    fi
    if [ ! -f "$STAGE_EVAL_JSON" ]; then
        echo "ERROR: eval_dataset_paths.json was not written: $STAGE_EVAL_JSON" >&2
        exit 1
    fi

    mkdir -p "$STAGE_OUTPUT_DIR"

    WARM_START_FLAG=()
    if [ -n "$STAGE_LORA_PRETRAINED" ]; then
        if [ ! -d "$STAGE_LORA_PRETRAINED" ]; then
            echo "ERROR: warm-start dir missing: $STAGE_LORA_PRETRAINED" >&2
            exit 1
        fi
        WARM_START_FLAG=(--lora_pretrained "$STAGE_LORA_PRETRAINED")
    fi

    BF16_FLAG=()
    if [ "$STAGE_BF16" = "true" ]; then BF16_FLAG=(--bf16); fi

    GC_FLAG=(--gradient_checkpointing "$STAGE_GRADIENT_CHECKPOINTING")

    RUN_NAME="curriculum-v2-stage${STAGE_IDX}-${STAGE_NAME}"

    DEEPSPEED_FLAG=()
    if [ -n "$DEEPSPEED_CONFIG" ]; then
        if [ ! -f "$DEEPSPEED_CONFIG" ]; then
            echo "ERROR: DeepSpeed config not found: $DEEPSPEED_CONFIG" >&2
            exit 1
        fi
        DEEPSPEED_FLAG=(--deepspeed "$DEEPSPEED_CONFIG")
    fi

    # view_class_weight may be empty, "auto", or a JSON string — quote it.
    VIEW_CW_FLAG=()
    if [ -n "$STAGE_VIEW_CLASS_WEIGHT" ]; then
        VIEW_CW_FLAG=(--view_class_weight "$STAGE_VIEW_CLASS_WEIGHT")
    fi

    # In-training per-category eval — skip when SKIP_INTRAINING_EVAL=1.
    # This avoids the dict-form eval loop that runs every $STAGE_EVAL_STEPS
    # and dominates wall-clock when intra-stage feedback isn't needed.
    EVAL_FLAGS=()
    if [ "${SKIP_INTRAINING_EVAL:-0}" = "1" ]; then
        EVAL_FLAGS=(--eval_strategy "no")
        echo "[stage $STAGE_IDX] SKIP_INTRAINING_EVAL=1 — disabling in-training eval."
    else
        EVAL_FLAGS=(
            --eval_dataset_paths_json "$STAGE_EVAL_JSON"
            --eval_strategy "steps"
            --eval_steps "$STAGE_EVAL_STEPS"
        )
    fi

    torchrun \
        --nproc_per_node="$NPROC_PER_NODE" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        --nnodes="$NNODES" \
        train_nuscenes_qwen3vl_v2.py \
        "${DEEPSPEED_FLAG[@]}" \
        --mode lora \
        --model_name_or_path "$STAGE_BASE_MODEL" \
        --train_data_path "$STAGE_TRAIN_DATA" \
        "${EVAL_FLAGS[@]}" \
        "${WARM_START_FLAG[@]}" \
        --lora_r "$STAGE_LORA_R" \
        --lora_alpha "$STAGE_LORA_ALPHA" \
        --lora_dropout "$STAGE_LORA_DROPOUT" \
        --lora_target_modules "$STAGE_LORA_TARGET_MODULES" \
        "${BF16_FLAG[@]}" \
        --output_dir "$STAGE_OUTPUT_DIR" \
        --num_train_epochs "$STAGE_EPOCHS" \
        --max_steps "$STAGE_MAX_STEPS" \
        --per_device_train_batch_size "$STAGE_PER_DEVICE_TRAIN_BATCH_SIZE" \
        --per_device_eval_batch_size "$STAGE_PER_DEVICE_EVAL_BATCH_SIZE" \
        --gradient_accumulation_steps "$STAGE_GRADIENT_ACCUMULATION_STEPS" \
        --max_pixels "$STAGE_MAX_PIXELS" \
        --min_pixels "$STAGE_MIN_PIXELS" \
        --max_assistant_tokens "$STAGE_MAX_ASSISTANT_TOKENS" \
        --save_strategy "steps" \
        --save_steps "$STAGE_SAVE_STEPS" \
        --save_total_limit "$STAGE_SAVE_TOTAL_LIMIT" \
        --learning_rate "$STAGE_PEAK_LR" \
        --weight_decay "$STAGE_WEIGHT_DECAY" \
        --max_grad_norm "$STAGE_MAX_GRAD_NORM" \
        --use_wsd_scheduler True \
        --wsd_warmup_ratio "$STAGE_WSD_WARMUP_RATIO" \
        --wsd_decay_ratio "$STAGE_WSD_DECAY_RATIO" \
        --lr_scheduler_type "constant" \
        --logging_steps 1 \
        --model_max_length "$STAGE_MODEL_MAX_LENGTH" \
        "${GC_FLAG[@]}" \
        --dataloader_num_workers "$STAGE_DATALOADER_NUM_WORKERS" \
        --run_name "$RUN_NAME" \
        --report_to "$STAGE_REPORT_TO" \
        --w_ans "$STAGE_W_ANS" \
        --w_gate "$STAGE_W_GATE" \
        --lam_view "$STAGE_LAM_VIEW" \
        --lam_iou "$STAGE_LAM_IOU" \
        --lam_klal "$STAGE_LAM_KLAL" \
        --klal_layers "$STAGE_KLAL_LAYERS" \
        --iou_alpha "$STAGE_IOU_ALPHA" \
        "${VIEW_CW_FLAG[@]}"

    echo "[stage $STAGE_IDX] training done."

    # ----------------------------------------------------------------------
    # Per-category generation eval (v2 metrics).
    # ----------------------------------------------------------------------
    if [ "${SKIP_STAGE_END_EVAL:-0}" = "1" ]; then
        echo "[stage $STAGE_IDX] SKIP_STAGE_END_EVAL=1 — skipping generation eval."
        continue
    fi

    if [ "$STAGE_FULL_VAL_FOR_END_EVAL" = "true" ]; then
        EVAL_DIR="$PROJECT_ROOT/sft_dataset"
    else
        EVAL_DIR=$(python - <<PY
import json, os
with open("$STAGE_EVAL_JSON") as f:
    m = json.load(f)
print(os.path.dirname(next(iter(m.values()))))
PY
)
    fi

    EXTRA=()
    if [ -n "${STAGE_END_EVAL_LIMIT:-}" ]; then
        EXTRA+=(--n_per_cat_limit "$STAGE_END_EVAL_LIMIT")
    fi

    python -m nuscenes_pipeline.modules.sft_model_tester \
        --base_model "$STAGE_BASE_MODEL" \
        --lora_path "$STAGE_OUTPUT_DIR" \
        --per_category_eval_dir "$EVAL_DIR" \
        --eval_v2 \
        --iou_threshold "$EVAL_IOU_THRESHOLD" \
        --save_predictions \
        --resize_factor 1 \
        --max_new_tokens 1024 \
        "${EXTRA[@]}"

    echo "[stage $STAGE_IDX] v2 eval done."
done

echo
echo "============================================"
echo "  Curriculum v2 complete. Aggregating reports."
echo "============================================"
OUTPUT_ROOT=$(python -m qwenvl.curriculum.config "$CONFIG" | python -c "
import json, sys
d = json.load(sys.stdin)
print(d['output_root'])
")
python hf_dataset_train/aggregate_curriculum_reports.py --output_root "$OUTPUT_ROOT" || true
echo "Done."
