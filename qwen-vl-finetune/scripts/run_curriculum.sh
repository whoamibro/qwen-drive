#!/bin/bash
# Curriculum-learning driver for Qwen3-VL 8B LoRA SFT.
#
# Walks the stages listed in a curriculum YAML config. For each stage:
#   1. Resolves per-stage paths/hyperparameters via the curriculum loader
#      (python -m qwenvl.curriculum.config --emit_shell <i>).
#   2. Builds the dict-form eval_dataset JSON pointing at every per-category
#      val subset, so per-category eval losses are logged at every eval step.
#   3. Launches torchrun training with WSD scheduler + warm-start from the
#      previous stage's LoRA adapter (empty for stage 0).
#   4. Runs the per-category generation eval (writes eval_report.json into
#      the stage's output dir).
#
# Usage:
#   bash qwen-vl-finetune/scripts/run_curriculum.sh \
#       qwen-vl-finetune/configs/curriculum_v1.yaml
#
# Environment overrides:
#   NPROC_PER_NODE, NNODES, MASTER_ADDR, MASTER_PORT  (torchrun)
#   START_STAGE  (default 0)    — resume the curriculum from a later stage
#   END_STAGE    (default last) — stop after this stage (inclusive)
#   SKIP_STAGE_END_EVAL=1       — train only, skip generation eval per stage
#   STAGE_END_EVAL_LIMIT=N      — cap samples per category for stage-end eval

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

# NCCL timeout (eval/save sweeps can take a while across 10 categories)
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-3600}
export TORCH_NCCL_TIMEOUT_SEC=${TORCH_NCCL_TIMEOUT_SEC:-3600}
export TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-2000}
export TORCH_NCCL_DUMP_ON_TIMEOUT=${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
NNODES=${WORLD_SIZE:-1}

# DeepSpeed is optional. Override via env:
#   DEEPSPEED_CONFIG=""                  -> skip --deepspeed entirely
#   DEEPSPEED_CONFIG=path/to/zeroN.json  -> use that config
# Default = repo's zero2.json (matches run_nuscenes_lora.sh).
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG-$PROJECT_ROOT/qwen-vl-finetune/scripts/zero2.json}"

NUM_STAGES=$(python -m qwenvl.curriculum.config "$CONFIG" --num_stages)
START_STAGE=${START_STAGE:-0}
END_STAGE=${END_STAGE:-$((NUM_STAGES - 1))}

echo "============================================"
echo "  Curriculum: $CONFIG"
echo "  Stages: $NUM_STAGES total | running [$START_STAGE..$END_STAGE]"
echo "  GPUs:   $NPROC_PER_NODE"
echo "============================================"
python -m qwenvl.curriculum.config "$CONFIG" --summary
echo "============================================"

for STAGE_IDX in $(seq "$START_STAGE" "$END_STAGE"); do
    # Emit per-stage shell assignments (also writes eval_dataset_paths.json).
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

    # Optional warm-start flag — only pass when non-empty so stage 0 trains fresh.
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

    RUN_NAME="curriculum-stage${STAGE_IDX}-${STAGE_NAME}"

    DEEPSPEED_FLAG=()
    if [ -n "$DEEPSPEED_CONFIG" ]; then
        if [ ! -f "$DEEPSPEED_CONFIG" ]; then
            echo "ERROR: DeepSpeed config not found: $DEEPSPEED_CONFIG" >&2
            echo "Set DEEPSPEED_CONFIG=\"\" to disable DeepSpeed." >&2
            exit 1
        fi
        DEEPSPEED_FLAG=(--deepspeed "$DEEPSPEED_CONFIG")
    fi

    torchrun \
        --nproc_per_node="$NPROC_PER_NODE" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        --nnodes="$NNODES" \
        train_nuscenes_qwen3vl.py \
        "${DEEPSPEED_FLAG[@]}" \
        --mode lora \
        --model_name_or_path "$STAGE_BASE_MODEL" \
        --train_data_path "$STAGE_TRAIN_DATA" \
        --eval_dataset_paths_json "$STAGE_EVAL_JSON" \
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
        --eval_strategy "steps" \
        --eval_steps "$STAGE_EVAL_STEPS" \
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
        --report_to "$STAGE_REPORT_TO"

    echo "[stage $STAGE_IDX] training done."

    # ------------------------------------------------------------------
    # Per-category generation eval for this stage's final adapter.
    # ------------------------------------------------------------------
    if [ "${SKIP_STAGE_END_EVAL:-0}" = "1" ]; then
        echo "[stage $STAGE_IDX] SKIP_STAGE_END_EVAL=1 — skipping generation eval."
        continue
    fi

    if [ "$STAGE_FULL_VAL_FOR_END_EVAL" = "true" ]; then
        EVAL_DIR="$(dirname "$STAGE_EVAL_JSON")/../.."   # not used; see below
        EVAL_DIR="$PROJECT_ROOT/sft_dataset"
    else
        # Subset dir = the directory containing the per-cat subset JSONs.
        # We derive it from the first entry in eval_dataset_paths.json.
        EVAL_DIR=$(python - <<PY
import json, os
with open("$STAGE_EVAL_JSON") as f:
    m = json.load(f)
print(os.path.dirname(next(iter(m.values()))))
PY
)
    fi

    bash nuscenes_pipeline/scripts/run_stage_end_eval.sh \
        "$STAGE_OUTPUT_DIR" \
        "$EVAL_DIR" \
        "$STAGE_BASE_MODEL" \
        "${STAGE_END_EVAL_LIMIT:-}"

    echo "[stage $STAGE_IDX] eval done."
done

echo
echo "============================================"
echo "  Curriculum complete. Aggregating reports."
echo "============================================"
OUTPUT_ROOT=$(python -m qwenvl.curriculum.config "$CONFIG" | python -c "
import json, sys
d = json.load(sys.stdin)
print(d['output_root'])
")
python hf_dataset_train/aggregate_curriculum_reports.py --output_root "$OUTPUT_ROOT" || true
echo "Done."
