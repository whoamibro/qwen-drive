#!/bin/bash
# Parallel v2 curriculum evaluation — fans stage-level eval out across N GPUs.
#
# Each background process pins to a different GPU via CUDA_VISIBLE_DEVICES
# and evaluates a single stage via eval_curriculum_v2.sh + STAGES_FILTER.
# Stages are processed in batches of NPROC; the script waits for one batch
# to finish before starting the next. After all stages complete, the
# aggregator runs once.
#
# Why stage-level parallelism (not data-parallel within one eval)?
#   - `sft_model_tester` is plain python (no torchrun); per-sample generation
#     is sequential. Putting a different LoRA stage on each GPU is the
#     cheapest path to ~Nx speedup with zero code change to the eval loop.
#   - Each invocation only loads the base model on its assigned GPU, so 8
#     parallel processes don't fight for memory (each takes ~16 GB bf16).
#   - eval_curriculum_v2.sh is idempotent — a final no-filter pass at the
#     end runs the aggregator over the now-complete set.
#
# Usage:
#   bash qwen-vl-finetune/scripts/eval_curriculum_v2_parallel.sh CONFIG.yaml
#
# Environment overrides:
#   NPROC              (default: number of GPUs from nvidia-smi, fallback 8)
#                      — number of stages to evaluate in parallel
#   GPU_IDS            (default: 0,1,...,NPROC-1)
#                      — comma-separated GPU IDs to use, e.g. "2,3,5,7"
#   START_STAGE        (default 0)
#   END_STAGE          (default last stage)
#   STAGES_FILTER      (e.g. "OBS,CHR") — restrict to specific stage names
#   FORCE_REEVAL=1     — re-evaluate stages whose eval_report.json exists
#   EVAL_IOU_THRESHOLD (default 0.8)
#   N_PER_CAT_LIMIT    (default unset) — debug; cap samples/category
#   SANITY_PRINT_N     (default 0 in parallel mode — keeps logs clean)
#   LOG_DIR            (default $OUTPUT_ROOT/eval_logs)
#                      — per-stage log files written here

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

# ----------------------------------------------------------------------
# Resolve curriculum metadata and GPU set.
# ----------------------------------------------------------------------
OUTPUT_ROOT=$(python -m qwenvl.curriculum.config "$CONFIG" | python -c "
import json, sys
print(json.load(sys.stdin)['output_root'])
")

NUM_STAGES=$(python -m qwenvl.curriculum.config "$CONFIG" --num_stages)
START_STAGE=${START_STAGE:-0}
END_STAGE=${END_STAGE:-$((NUM_STAGES - 1))}

# Detect GPU count from nvidia-smi, fall back to 8.
DETECTED_NPROC=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 0)
if [ "$DETECTED_NPROC" -le 0 ]; then DETECTED_NPROC=8; fi
NPROC=${NPROC:-$DETECTED_NPROC}

# Build the GPU ID array.
if [ -n "${GPU_IDS:-}" ]; then
    IFS=',' read -ra GPU_ARR <<<"$GPU_IDS"
    if [ "${#GPU_ARR[@]}" != "$NPROC" ]; then
        echo "ERROR: GPU_IDS has ${#GPU_ARR[@]} entries but NPROC=$NPROC" >&2
        exit 1
    fi
else
    GPU_ARR=()
    for ((i=0; i<NPROC; i++)); do GPU_ARR+=("$i"); done
fi

LOG_DIR=${LOG_DIR:-$OUTPUT_ROOT/eval_logs}
mkdir -p "$LOG_DIR"

# Resolve all stage names up front (one Python call per index is fine; 10x).
declare -a STAGE_NAMES
for STAGE_IDX in $(seq 0 $((NUM_STAGES - 1))); do
    NAME=$(python -m qwenvl.curriculum.config "$CONFIG" --emit_shell "$STAGE_IDX" \
        | grep '^STAGE_NAME=' | head -n1 | cut -d= -f2)
    STAGE_NAMES[$STAGE_IDX]="$NAME"
done

# Apply user STAGES_FILTER (intersection with [START_STAGE..END_STAGE]).
declare -a STAGES_TO_RUN
for STAGE_IDX in $(seq "$START_STAGE" "$END_STAGE"); do
    NAME="${STAGE_NAMES[$STAGE_IDX]}"
    if [ -n "${STAGES_FILTER:-}" ] && [[ ! ",$STAGES_FILTER," == *",$NAME,"* ]]; then
        continue
    fi
    STAGES_TO_RUN+=("$STAGE_IDX:$NAME")
done

TOTAL=${#STAGES_TO_RUN[@]}

echo "============================================"
echo "  Parallel v2 curriculum eval"
echo "  config:      $CONFIG"
echo "  output_root: $OUTPUT_ROOT"
echo "  GPUs:        ${GPU_ARR[*]}  (NPROC=$NPROC)"
echo "  stages:      $TOTAL to run -> ${STAGES_TO_RUN[*]}"
echo "  log_dir:     $LOG_DIR"
echo "  force_re:    ${FORCE_REEVAL:-0}"
echo "============================================"

if [ "$TOTAL" -eq 0 ]; then
    echo "Nothing to evaluate — running aggregator only."
else
    # ----------------------------------------------------------------------
    # Launch in batches of NPROC. Each batch fills the GPU pool; we wait for
    # the batch to finish before starting the next so GPU allocation stays
    # simple (no need for `wait -n` slot recycling).
    # ----------------------------------------------------------------------
    BATCH=0
    POS=0
    EXIT_CODE=0
    while [ "$POS" -lt "$TOTAL" ]; do
        BATCH=$((BATCH + 1))
        echo
        echo "--- batch $BATCH (stages $POS..$((POS + NPROC - 1)) of $TOTAL) ---"
        declare -a PIDS=()
        for ((SLOT=0; SLOT<NPROC && POS<TOTAL; SLOT++, POS++)); do
            ENTRY="${STAGES_TO_RUN[$POS]}"
            STAGE_IDX="${ENTRY%%:*}"
            NAME="${ENTRY##*:}"
            GPU="${GPU_ARR[$SLOT]}"
            LOG_FILE="$LOG_DIR/eval_stage${STAGE_IDX}_${NAME}_gpu${GPU}.log"

            echo "  slot $SLOT  GPU $GPU  stage $STAGE_IDX ($NAME)  -> $LOG_FILE"

            # Pass-through env vars + isolate to one GPU + filter to this stage.
            CUDA_VISIBLE_DEVICES="$GPU" \
            STAGES_FILTER="$NAME" \
            FORCE_REEVAL="${FORCE_REEVAL:-0}" \
            EVAL_IOU_THRESHOLD="${EVAL_IOU_THRESHOLD:-0.8}" \
            N_PER_CAT_LIMIT="${N_PER_CAT_LIMIT:-}" \
            SANITY_PRINT_N="${SANITY_PRINT_N:-0}" \
                bash qwen-vl-finetune/scripts/eval_curriculum_v2.sh "$CONFIG" \
                > "$LOG_FILE" 2>&1 &
            PIDS+=($!)
        done

        # Wait for all background jobs in this batch.
        for PID in "${PIDS[@]}"; do
            if ! wait "$PID"; then
                echo "WARNING: pid $PID exited with non-zero status; check logs in $LOG_DIR"
                EXIT_CODE=1
            fi
        done
        echo "  batch $BATCH complete."
    done

    if [ "$EXIT_CODE" -ne 0 ]; then
        echo
        echo "WARNING: at least one stage's eval reported a non-zero exit code."
        echo "         Inspect per-stage logs in $LOG_DIR before trusting the aggregator output."
    fi
fi

# ----------------------------------------------------------------------
# Final aggregator call (single-process, runs over all completed stages).
# ----------------------------------------------------------------------
echo
echo "============================================"
echo "  Aggregating eval_report.json across stages."
echo "============================================"
python hf_dataset_train/aggregate_curriculum_reports.py --output_root "$OUTPUT_ROOT" || true
echo "Done."
