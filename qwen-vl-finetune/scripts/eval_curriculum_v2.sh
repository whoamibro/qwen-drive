#!/bin/bash
# Standalone v2 per-category evaluation across all stages of a curriculum run.
#
# Discovers stage_XX_<NAME>/ dirs under <output_root> (read from the YAML),
# runs sft_model_tester --eval_v2 on each LoRA checkpoint, writes
# eval_report.json + eval_predictions.json into each stage dir, then runs
# aggregate_curriculum_reports.py to produce curriculum_report.{csv,md}.
#
# Designed to be run AFTER a train-only curriculum
# (SKIP_INTRAINING_EVAL=1 SKIP_STAGE_END_EVAL=1 bash run_curriculum_v2.sh ...).
#
# Usage:
#   bash qwen-vl-finetune/scripts/eval_curriculum_v2.sh CONFIG.yaml [EVAL_DIR]
#
# Arguments:
#   CONFIG    Path to the curriculum YAML used for training.
#   EVAL_DIR  (optional) Override the eval-subset dir. Default: the YAML's
#             `eval_subset_dir` field.
#
# Environment overrides:
#   START_STAGE        (default 0)            — evaluate from this stage
#   END_STAGE          (default last)         — stop after this stage
#   EVAL_IOU_THRESHOLD (default 0.8)          — grounding IoU threshold
#   N_PER_CAT_LIMIT    (default unset)        — cap samples/category (debug)
#   SANITY_PRINT_N     (default 2)            — raw-output samples to print
#   FORCE_REEVAL=1                            — re-eval stages that already
#                                               have eval_report.json
#   STAGES_FILTER="OBS,CHR"                   — only eval matching stage names
#   CUDA_VISIBLE_DEVICES                      — pin to specific GPU (eval is
#                                               single-GPU; generation in HF
#                                               doesn't shard across DDP)

set -e

CONFIG=${1:?"usage: $0 <curriculum_config.yaml> [eval_dir]"}
EVAL_DIR_OVERRIDE=${2:-}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: config not found: $CONFIG" >&2
    exit 1
fi

export PYTHONPATH="$PROJECT_ROOT/qwen-vl-finetune:${PYTHONPATH:-}"

# Pull output_root, base_model, eval_subset_dir from the YAML.
read -r OUTPUT_ROOT BASE_MODEL CFG_EVAL_DIR <<<"$(python -m qwenvl.curriculum.config "$CONFIG" | python -c "
import json, sys
d = json.load(sys.stdin)
print(d['output_root'], d['base_model'], d['eval_subset_dir'])
")"

EVAL_DIR="${EVAL_DIR_OVERRIDE:-$CFG_EVAL_DIR}"

if [ ! -d "$OUTPUT_ROOT" ]; then
    echo "ERROR: output_root does not exist: $OUTPUT_ROOT" >&2
    echo "  Has the curriculum been trained yet?" >&2
    exit 1
fi
if [ ! -d "$EVAL_DIR" ]; then
    echo "ERROR: eval_dir does not exist: $EVAL_DIR" >&2
    echo "  Run: python -m nuscenes_pipeline.postprocessing.build_eval_subset --n_per_cat 200" >&2
    exit 1
fi

NUM_STAGES=$(python -m qwenvl.curriculum.config "$CONFIG" --num_stages)
START_STAGE=${START_STAGE:-0}
END_STAGE=${END_STAGE:-$((NUM_STAGES - 1))}
EVAL_IOU_THRESHOLD=${EVAL_IOU_THRESHOLD:-0.8}
SANITY_PRINT_N=${SANITY_PRINT_N:-2}

echo "============================================"
echo "  Curriculum v2 standalone eval"
echo "  config:       $CONFIG"
echo "  output_root:  $OUTPUT_ROOT"
echo "  eval_dir:     $EVAL_DIR"
echo "  base_model:   $BASE_MODEL"
echo "  stages:       [$START_STAGE..$END_STAGE] of $NUM_STAGES"
echo "  iou_thresh:   $EVAL_IOU_THRESHOLD"
echo "============================================"

# Build the optional stage-name filter set.
STAGE_FILTER_OK=""  # empty = all pass
if [ -n "${STAGES_FILTER:-}" ]; then
    STAGE_FILTER_OK="$STAGES_FILTER"
fi

EXTRA_FLAGS=()
if [ -n "${N_PER_CAT_LIMIT:-}" ]; then
    EXTRA_FLAGS+=(--n_per_cat_limit "$N_PER_CAT_LIMIT")
fi

# Loop stages and eval each LoRA checkpoint.
EVALUATED=0
SKIPPED=0
for STAGE_IDX in $(seq "$START_STAGE" "$END_STAGE"); do
    # Look up the stage's directory name & path from the loader.
    STAGE_NAME=$(python -m qwenvl.curriculum.config "$CONFIG" --emit_shell "$STAGE_IDX" \
        | grep '^STAGE_NAME=' | head -n1 | cut -d= -f2)
    STAGE_DIR=$(python -m qwenvl.curriculum.config "$CONFIG" --emit_shell "$STAGE_IDX" \
        | grep '^STAGE_OUTPUT_DIR=' | head -n1 | cut -d= -f2)

    # Apply stage-name filter if set.
    if [ -n "$STAGE_FILTER_OK" ] && [[ ! ",$STAGE_FILTER_OK," == *",$STAGE_NAME,"* ]]; then
        echo "[stage $STAGE_IDX $STAGE_NAME] not in STAGES_FILTER — skipping"
        SKIPPED=$((SKIPPED+1))
        continue
    fi

    if [ ! -d "$STAGE_DIR" ]; then
        echo "[stage $STAGE_IDX $STAGE_NAME] no checkpoint dir at $STAGE_DIR — skipping"
        SKIPPED=$((SKIPPED+1))
        continue
    fi

    # Skip if eval_report.json already exists (unless FORCE_REEVAL=1).
    if [ -f "$STAGE_DIR/eval_report.json" ] && [ "${FORCE_REEVAL:-0}" != "1" ]; then
        echo "[stage $STAGE_IDX $STAGE_NAME] eval_report.json exists — skipping (FORCE_REEVAL=1 to overwrite)"
        SKIPPED=$((SKIPPED+1))
        continue
    fi

    echo
    echo "============================================"
    echo "  EVAL stage $STAGE_IDX ($STAGE_NAME)"
    echo "  lora_path: $STAGE_DIR"
    echo "============================================"

    python -m nuscenes_pipeline.modules.sft_model_tester \
        --base_model "$BASE_MODEL" \
        --lora_path "$STAGE_DIR" \
        --per_category_eval_dir "$EVAL_DIR" \
        --eval_v2 \
        --iou_threshold "$EVAL_IOU_THRESHOLD" \
        --sanity_print_n "$SANITY_PRINT_N" \
        --save_predictions \
        --resize_factor 1 \
        --max_new_tokens 1024 \
        "${EXTRA_FLAGS[@]}"

    EVALUATED=$((EVALUATED+1))
    echo "[stage $STAGE_IDX $STAGE_NAME] eval done -> $STAGE_DIR/eval_report.json"
done

echo
echo "============================================"
echo "  Evaluated: $EVALUATED  Skipped: $SKIPPED"
echo "  Aggregating eval_report.json across stages."
echo "============================================"
python hf_dataset_train/aggregate_curriculum_reports.py --output_root "$OUTPUT_ROOT" || true
echo "Done."
