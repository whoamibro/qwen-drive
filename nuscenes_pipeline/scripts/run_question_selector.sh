#!/bin/bash

################################################################################
# Question Selector - Stage 2
#
# Selects applicable question templates from the question bank for each sample.
# Consumes Stage 1A (risk) and Stage 1B (traffic) results as prior context.
# Uses vLLM-served model via OpenAI-compatible API with multiprocessing.
#
# Prerequisites:
#   vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct --tensor-parallel-size 8
#   Stage 1A (risk_assessment) and Stage 1B (traffic_analysis) must be complete.
#
# Output directories:
#   qa_outputs/<category>/         - JSON result files per category
#   question_selector_logs/        - Log files
################################################################################

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR/.."

# Default parameters
START_IDX=${1:-0}
END_IDX=${2:-6018}
CATEGORY=${3:-"all"}
NUM_WORKERS=${4:-8}

# Output directories
OUTPUT_DIR="qa_outputs/$CATEGORY"
LOG_DIR="question_selector_logs"
RISK_RESULTS_DIR="risk_assessment_results"
TRAFFIC_RESULTS_DIR="traffic_analysis_results"

echo "================================================================================"
echo "Question Selector - Stage 2"
echo "================================================================================"
echo "Start Index: $START_IDX"
echo "End Index: $END_IDX"
echo "Category: $CATEGORY"
echo "Num Workers: $NUM_WORKERS"
echo "Total samples: $((END_IDX - START_IDX + 1))"
echo ""
echo "Output directories:"
echo "  Results:  $OUTPUT_DIR"
echo "  Logs:     $LOG_DIR"
echo "================================================================================"
echo ""

# Run the module
python3 -m nuscenes_pipeline.modules.question_selector \
    --start_idx $START_IDX \
    --end_idx $END_IDX \
    --category "$CATEGORY" \
    --num_workers $NUM_WORKERS \
    --filter_distance 50 \
    --rear_filter 20 \
    --batch_size 5 \
    --resize_factor 2 \
    --max_new_tokens 1024 \
    --output_dir "$OUTPUT_DIR" \
    --log_dir "$LOG_DIR" \
    --risk_results_dir "$RISK_RESULTS_DIR" \
    --traffic_results_dir "$TRAFFIC_RESULTS_DIR"

echo ""
echo "================================================================================"
echo "Question selection complete!"
echo "  Results:  $OUTPUT_DIR/"
echo "  Logs:     $LOG_DIR/"
echo "================================================================================"
