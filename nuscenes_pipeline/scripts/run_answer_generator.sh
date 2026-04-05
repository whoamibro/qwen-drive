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

# Output directories
STAGE1_DIR="qa_outputs_stage1"
OUTPUT_DIR="qa_outputs_stage2"
LOG_DIR="answer_generator_logs"

echo "================================================================================"
echo "Answer Generator - Stage 3"
echo "================================================================================"
echo "Start Index: $START_IDX"
echo "End Index: $END_IDX"
echo "Category: $CATEGORY"
echo "Num Workers: $NUM_WORKERS"
echo "Total samples: $((END_IDX - START_IDX + 1))"
echo ""
echo "Input:  $STAGE1_DIR"
echo "Output: $OUTPUT_DIR"
echo "Logs:   $LOG_DIR"
echo "================================================================================"
echo ""

# Run the module
python3 -m nuscenes_pipeline.modules.answer_generator \
    --start_idx $START_IDX \
    --end_idx $END_IDX \
    --category "$CATEGORY" \
    --num_workers $NUM_WORKERS \
    --max_new_tokens 16384 \
    --max_pairs 3 \
    --filter_distance 50 \
    --rear_filter 20 \
    --resize_factor 2 \
    --stage1_dir "$STAGE1_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --log_dir "$LOG_DIR"

echo ""
echo "================================================================================"
echo "Answer generation complete!"
echo "  Results:  $OUTPUT_DIR/"
echo "  Logs:     $LOG_DIR/"
echo "================================================================================"
