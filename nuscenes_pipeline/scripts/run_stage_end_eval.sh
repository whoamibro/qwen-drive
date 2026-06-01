#!/bin/bash
# Stage-end per-category generation eval for curriculum training.
#
# Loads the LoRA adapter at STAGE_OUTPUT_DIR (merged onto BASE_MODEL), runs
# inference on every per-category subset JSON inside EVAL_DIR, and writes
# eval_report.json into STAGE_OUTPUT_DIR.
#
# Usage:
#   bash nuscenes_pipeline/scripts/run_stage_end_eval.sh \
#       <STAGE_OUTPUT_DIR> <EVAL_DIR> [BASE_MODEL] [N_PER_CAT_LIMIT]

set -e

STAGE_OUTPUT_DIR=${1:?"usage: $0 <STAGE_OUTPUT_DIR> <EVAL_DIR> [BASE_MODEL] [N_PER_CAT_LIMIT]"}
EVAL_DIR=${2:?"usage: $0 <STAGE_OUTPUT_DIR> <EVAL_DIR> [BASE_MODEL] [N_PER_CAT_LIMIT]"}
BASE_MODEL=${3:-"ckpts/qwen3_vl_8b_instruct"}
N_PER_CAT_LIMIT=${4:-}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

if [ ! -d "$STAGE_OUTPUT_DIR" ]; then
    echo "ERROR: STAGE_OUTPUT_DIR does not exist: $STAGE_OUTPUT_DIR" >&2
    exit 1
fi
if [ ! -d "$EVAL_DIR" ]; then
    echo "ERROR: EVAL_DIR does not exist: $EVAL_DIR" >&2
    exit 1
fi

echo "============================================"
echo "  Stage-end per-category eval"
echo "============================================"
echo "  LoRA path:   $STAGE_OUTPUT_DIR"
echo "  Eval dir:    $EVAL_DIR"
echo "  Base model:  $BASE_MODEL"
echo "  Cap per cat: ${N_PER_CAT_LIMIT:-<no cap>}"
echo "============================================"

EXTRA=""
if [ -n "$N_PER_CAT_LIMIT" ]; then
    EXTRA="--n_per_cat_limit $N_PER_CAT_LIMIT"
fi

python -m nuscenes_pipeline.modules.sft_model_tester \
    --base_model "$BASE_MODEL" \
    --lora_path "$STAGE_OUTPUT_DIR" \
    --per_category_eval_dir "$EVAL_DIR" \
    --save_predictions \
    --resize_factor 1 \
    --max_new_tokens 1024 \
    $EXTRA

echo "Done. Report: $STAGE_OUTPUT_DIR/eval_report.json"
