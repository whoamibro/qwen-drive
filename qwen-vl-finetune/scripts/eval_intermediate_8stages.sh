#!/bin/bash
# One-off launcher: evaluate stages 1..8 of curriculum_v2_0609 against specific
# intermediate checkpoint subdirs, fanned out across 8 GPUs (one stage per GPU,
# all in parallel).
#
# Mapping (stage -> GPU -> checkpoint subdir):
#   stage_01_IDN  -> GPU 0 -> checkpoint-164
#   stage_02_AAS  -> GPU 1 -> checkpoint-252
#   stage_03_SRO  -> GPU 2 -> checkpoint-562
#   stage_04_TSS  -> GPU 3 -> checkpoint-151
#   stage_05_RML  -> GPU 4 -> checkpoint-113
#   stage_06_DRA  -> GPU 5 -> checkpoint-983
#   stage_07_RWP  -> GPU 6 -> checkpoint-266
#   stage_08_ESC  -> GPU 7 -> checkpoint-313
#
# Usage:
#   bash qwen-vl-finetune/scripts/eval_intermediate_8stages.sh
#
# Environment overrides:
#   OUTPUT_ROOT    (default: output/curriculum_v2_0609)
#   EVAL_DIR       (default: sft_dataset/eval_subset_200)
#   BASE_MODEL     (default: ckpts/qwen3_vl_8b_instruct)
#   FORCE_REEVAL=1 — re-run even if eval_report.json already exists per stage

set -e
set -o pipefail  # propagate non-zero status through `python | sed | tee`

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT/qwen-vl-finetune:${PYTHONPATH:-}"

OUTPUT_ROOT=${OUTPUT_ROOT:-output/curriculum_v2_0609}
EVAL_DIR=${EVAL_DIR:-sft_dataset/eval_subset_200}
BASE_MODEL=${BASE_MODEL:-ckpts/qwen3_vl_8b_instruct}
LOG_DIR="$OUTPUT_ROOT/eval_logs_intermediate"
mkdir -p "$LOG_DIR"

# Mapping: bash arrays (parallel arrays for portability across bash versions).
STAGES=(stage_01_IDN stage_02_AAS stage_03_SRO stage_04_TSS \
        stage_05_RML stage_06_DRA stage_07_RWP stage_08_ESC)
CKPTS=(checkpoint-164 checkpoint-252 checkpoint-562 checkpoint-151 \
       checkpoint-113 checkpoint-983 checkpoint-266 checkpoint-313)
GPUS=(0 1 2 3 4 5 6 7)

echo "============================================"
echo "  Intermediate-checkpoint eval (stages 1-8)"
echo "  output_root: $OUTPUT_ROOT"
echo "  eval_dir:    $EVAL_DIR"
echo "  base_model:  $BASE_MODEL"
echo "  log_dir:     $LOG_DIR"
echo "============================================"
for i in "${!STAGES[@]}"; do
    printf "  GPU %s  %s  ->  %s\n" "${GPUS[$i]}" "${STAGES[$i]}" "${CKPTS[$i]}"
done
echo "============================================"
echo "Streaming all 8 worker outputs to THIS terminal. Each line is prefixed"
echo "  with the stage tag, e.g. [04_TSS] ... so you can tell who said what."
echo "  Full per-stage logs are also written to $LOG_DIR/."
echo "============================================"

# Fan out — all 8 in parallel.
PIDS=()
for i in "${!STAGES[@]}"; do
    STAGE="${STAGES[$i]}"
    CKPT="${CKPTS[$i]}"
    GPU="${GPUS[$i]}"
    STAGE_DIR="$OUTPUT_ROOT/$STAGE"
    LOG_FILE="$LOG_DIR/eval_${STAGE}_${CKPT}_gpu${GPU}.log"

    if [ ! -d "$STAGE_DIR" ]; then
        echo "[skip] $STAGE: stage dir missing"
        continue
    fi
    if [ ! -d "$STAGE_DIR/$CKPT" ]; then
        echo "[skip] $STAGE: $CKPT subdir missing under $STAGE_DIR"
        continue
    fi

    # eval_report.json gets written into STAGE_DIR/$CKPT (the resolved
    # effective_lora_path). Skip if already done unless FORCE_REEVAL=1.
    REPORT="$STAGE_DIR/$CKPT/eval_report.json"
    if [ -f "$REPORT" ] && [ "${FORCE_REEVAL:-0}" != "1" ]; then
        echo "[skip] $STAGE/$CKPT: eval_report.json already exists ($REPORT) — FORCE_REEVAL=1 to overwrite"
        continue
    fi

    # Short stage label used as the per-line prefix; e.g. "01_IDN".
    PREFIX=$(echo "$STAGE" | sed -E 's/^stage_//')
    echo "[launch] GPU $GPU  $STAGE/$CKPT  (prefix [$PREFIX])  -> $LOG_FILE"
    {
        CUDA_VISIBLE_DEVICES="$GPU" python -u -m nuscenes_pipeline.modules.sft_model_tester \
            --base_model "$BASE_MODEL" \
            --lora_path "$STAGE_DIR" \
            --lora_checkpoint_subdir "$CKPT" \
            --per_category_eval_dir "$EVAL_DIR" \
            --eval_v2 \
            --iou_threshold 0.8 \
            --sanity_print_n 2 \
            --save_predictions \
            --resize_factor 1 \
            --max_new_tokens 1024 \
            2>&1
    } | sed -u "s/^/[$PREFIX] /" | tee "$LOG_FILE" &
    PIDS+=($!)
done

# Wait for all background workers.
echo
echo "Waiting for ${#PIDS[@]} eval(s) to complete ..."
FAIL=0
for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        echo "  pid $PID exited non-zero — check logs in $LOG_DIR"
        FAIL=$((FAIL + 1))
    fi
done

if [ "$FAIL" -gt 0 ]; then
    echo
    echo "WARNING: $FAIL eval(s) failed. Inspect $LOG_DIR/eval_*.log before trusting aggregate."
fi

echo
echo "============================================"
echo "  All evals done. Aggregating per-checkpoint reports."
echo "  Note: eval_report.json files are inside each stage_*/checkpoint-*/ subdir."
echo "  The aggregator currently scans stage_* (top-level) only; per-checkpoint"
echo "  reports must be aggregated separately or inspected individually."
echo "============================================"
for i in "${!STAGES[@]}"; do
    REPORT="$OUTPUT_ROOT/${STAGES[$i]}/${CKPTS[$i]}/eval_report.json"
    if [ -f "$REPORT" ]; then
        echo "  ${STAGES[$i]}/${CKPTS[$i]}:"
        python -c "
import json
r = json.load(open('$REPORT'))
m = r.get('metrics', {})
macro = r.get('macro_answer_acc', '?')
print(f'    macro_answer_acc = {macro}')
for cat in sorted(m):
    e = m[cat]
    aa = e.get('answer_acc')
    ga = e.get('grounding_acc@0.8')
    print(f'    {cat:>3s}  answer={aa if aa is not None else \"-\":.3f}  ground@0.8={ga if ga is not None else \"-\":.3f}  n_grounded={e.get(\"n_grounded\")}')
" 2>/dev/null || echo "    (failed to parse $REPORT)"
    else
        echo "  ${STAGES[$i]}/${CKPTS[$i]}: <no eval_report.json>"
    fi
done
echo "Done."
