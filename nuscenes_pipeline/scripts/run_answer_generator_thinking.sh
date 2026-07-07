#!/bin/bash

################################################################################
# Answer Generator (Thinking variant) - Stage 3
#
# Same as run_answer_generator.sh, but targets the Qwen3-VL Thinking model
# instead of the Instruct model. The Thinking model emits a <think>...</think>
# block before the final JSON, so the vLLM server must be launched with the
# matching reasoning parser, and the per-call output budget must be larger.
#
# Outputs land in dedicated directories (qa_results_thinking/,
# answer_generator_thinking_logs/) so this run does not clobber the Instruct
# baseline outputs.
#
# Prerequisites:
#   vllm serve Qwen/Qwen3-VL-235B-A22B-Thinking \
#       --tensor-parallel-size 8 \
#       --reasoning-parser qwen3 \
#       --max-model-len 65536
#   Stage 2 (question_selector) must be complete.
#
# Output directories:
#   qa_results_thinking/             - QA pair result files
#   answer_generator_thinking_logs/  - Log files
################################################################################

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR/.."

# Default parameters (mirror run_answer_generator.sh)
START_IDX=${1:-0}
END_IDX=${2:-6018}
CATEGORY=${3:-"all"}
NUM_WORKERS=${4:-8}
MODE=${5:-"range"}         # "range" (default) or "from_stage1"
ANSWER_MODE=${6:-"a_r"}    # "a_r" (default, answer-first) or "r_a" (reasoning-first)
RESUME=${7:-""}            # "resume" to skip samples with existing results

# Thinking-variant overrides
MODEL_NAME="Qwen/Qwen3-VL-235B-A22B-Thinking"
MAX_NEW_TOKENS=32768

# Output directories (separate namespace from the Instruct run)
STAGE1_DIR="qa_outputs"
OUTPUT_DIR="qa_results_thinking"
LOG_DIR="answer_generator_thinking_logs"
RISK_RESULTS_DIR="risk_assessment_results"
TRAFFIC_RESULTS_DIR="traffic_signal_analysis_results"
SIGN_RESULTS_DIR="traffic_sign_results"
DISAGREEMENT_DIR="prior_disagreements_thinking"

echo "================================================================================"
echo "Answer Generator (Thinking) - Stage 3"
echo "================================================================================"
echo "Model:        $MODEL_NAME"
echo "Max tokens:   $MAX_NEW_TOKENS"
echo "Category:     $CATEGORY"
echo "Num Workers:  $NUM_WORKERS"
echo "Mode:         $MODE"
echo "Answer Mode:  $ANSWER_MODE"
if [ "$MODE" = "from_stage1" ]; then
    echo "  -> Auto-discovering samples from Stage 1 outputs in $STAGE1_DIR"
else
    echo "Start Index:  $START_IDX"
    echo "End Index:    $END_IDX"
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
if [ "$RESUME" = "resume" ]; then
    SAMPLE_ARGS="$SAMPLE_ARGS --skip_existing"
    echo "Resume mode: samples with existing results in $OUTPUT_DIR will be skipped"
fi

# Run the module with Thinking-variant overrides
python3 -m nuscenes_pipeline.modules.answer_generator \
    $SAMPLE_ARGS \
    --model_name "$MODEL_NAME" \
    --category "$CATEGORY" \
    --num_workers $NUM_WORKERS \
    --max_new_tokens $MAX_NEW_TOKENS \
    --max_pairs 3 \
    --filter_distance 50 \
    --rear_filter 20 \
    --resize_factor 1 \
    --stage1_dir "$STAGE1_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --log_dir "$LOG_DIR" \
    --risk_results_dir "$RISK_RESULTS_DIR" \
    --traffic_results_dir "$TRAFFIC_RESULTS_DIR" \
    --sign_results_dir "$SIGN_RESULTS_DIR" \
    --answer_mode "$ANSWER_MODE" \
    --disagreement_dir "$DISAGREEMENT_DIR"

echo ""
echo "================================================================================"
echo "Answer generation (Thinking) complete!"
echo "  Results:  $OUTPUT_DIR/"
echo "  Logs:     $LOG_DIR/"
echo "================================================================================"
