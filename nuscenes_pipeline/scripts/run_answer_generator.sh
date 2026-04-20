#!/bin/bash

################################################################################
# Answer Generator - Stage 3
#
# Takes Stage 2 (question selector) outputs, generates positive QA pairs, and
# produces contrastive QA pairs with altered placeholders.
# Uses vLLM-served model via OpenAI-compatible API with multiprocessing.
#
# Prerequisites:
#   vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct --tensor-parallel-size 8
#   Stage 2 (question_selector) must be complete.
#
# Output directories:
#   qa_outputs_stage2/             - QA pair result files
#   answer_generator_logs/         - Log files
################################################################################

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR/.."

# Default parameters
START_IDX=${1:-0}
END_IDX=${2:-6018}
CATEGORY=${3:-"all"}
NUM_WORKERS=${4:-8}
MODE=${5:-"range"}         # "range" (default) or "from_stage1"
ANSWER_MODE=${6:-"a_r"}    # "a_r" (default, answer-first) or "r_a" (reasoning-first)

# Output directories
STAGE1_DIR="qa_outputs"
OUTPUT_DIR="qa_results"
LOG_DIR="answer_generator_logs"
RISK_RESULTS_DIR="risk_assessment_results"
TRAFFIC_RESULTS_DIR="traffic_analysis_results"

echo "================================================================================"
echo "Answer Generator - Stage 3"
echo "================================================================================"
echo "Category: $CATEGORY"
echo "Num Workers: $NUM_WORKERS"
echo "Mode: $MODE"
echo "Answer Mode: $ANSWER_MODE"
if [ "$MODE" = "from_stage1" ]; then
    echo "  -> Auto-discovering samples from Stage 1 outputs in $STAGE1_DIR"
else
    echo "Start Index: $START_IDX"
    echo "End Index: $END_IDX"
    echo "Total samples: $((END_IDX - START_IDX + 1))"
fi
echo ""
echo "Input:  $STAGE1_DIR"
echo "Output: $OUTPUT_DIR"
echo "Logs:   $LOG_DIR"
echo "================================================================================"
echo ""

# Build mode-specific arguments
if [ "$MODE" = "from_stage1" ]; then
    SAMPLE_ARGS="--from_stage1"
else
    SAMPLE_ARGS="--start_idx $START_IDX --end_idx $END_IDX"
fi

# Run the module
python3 -m nuscenes_pipeline.modules.answer_generator \
    $SAMPLE_ARGS \
    --category "$CATEGORY" \
    --num_workers $NUM_WORKERS \
    --max_new_tokens 16384 \
    --max_pairs 3 \
    --filter_distance 50 \
    --rear_filter 20 \
    --resize_factor 2 \
    --stage1_dir "$STAGE1_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --log_dir "$LOG_DIR" \
    --risk_results_dir "$RISK_RESULTS_DIR" \
    --traffic_results_dir "$TRAFFIC_RESULTS_DIR" \
    --answer_mode "$ANSWER_MODE"

echo ""
echo "================================================================================"
echo "Answer generation complete!"
echo "  Results:  $OUTPUT_DIR/"
echo "  Logs:     $LOG_DIR/"
echo "================================================================================"
