#!/bin/bash

################################################################################
# Answer Generator (RL/GRPO variant) - Stage 3
#
# Generates RL-training QA data with the fixed output contract:
#   - tier-banded "think" array (student <think> learning target)
#   - key order think -> reasoning -> answer
#   - contrast_status self-report (achieved / same_answer / skipped)
# Spec: thinking_answer_generator_4_rl_impl.md
#
# Prerequisites:
#   vllm serve Qwen/Qwen3-VL-235B-A22B-Thinking \
#       --tensor-parallel-size 8 \
#       --max-model-len 65536
#   (NO --reasoning-parser: the inline trace is kept for auditability and
#    stripped by the module's parser.)
#   Stage 2 (question_selector) must be complete.
#
# Output directories (isolated from the SFT pipeline):
#   qa_results_rl_thinking/       - QA pair result files + rl_run_report.json
#   answer_generator_rl_logs/     - Log files
#   prior_disagreements_rl/       - Prior-disagreement traces
#
# Smoke test (T10) — run before any full run:
#   bash nuscenes_pipeline/scripts/run_answer_generator_rl.sh smoke "100,228,304,371,759"
################################################################################

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR/.."

# Smoke-test mode: run_answer_generator_rl.sh smoke "<comma-separated indices>"
if [ "$1" = "smoke" ]; then
    INDICES=${2:?"smoke mode needs a comma-separated sample index list"}
    echo "================================================================================"
    echo "Answer Generator (RL) - SMOKE TEST"
    echo "  Samples: $INDICES | 2 templates per category"
    echo "================================================================================"
    python3 -m nuscenes_pipeline.modules.answer_generator_rl \
        --sample_indices "$INDICES" \
        --max_templates_per_category 2 \
        --num_workers 4 \
        --max_new_tokens 32768 \
        --max_pairs 3 \
        --filter_distance 50 \
        --rear_filter 20 \
        --resize_factor 1
    exit $?
fi

# Default parameters (mirror run_answer_generator_thinking.sh, minus ANSWER_MODE)
START_IDX=${1:-0}
END_IDX=${2:-6018}
CATEGORY=${3:-"all"}
NUM_WORKERS=${4:-8}
MODE=${5:-"range"}         # "range" (default) or "from_stage1"
RESUME=${6:-""}            # "resume" to skip samples with existing results

# Output directories
STAGE1_DIR="qa_outputs"
OUTPUT_DIR="qa_results_rl_thinking"
LOG_DIR="answer_generator_rl_logs"
RISK_RESULTS_DIR="risk_assessment_results"
TRAFFIC_RESULTS_DIR="traffic_signal_analysis_results"
SIGN_RESULTS_DIR="traffic_sign_results"
DISAGREEMENT_DIR="prior_disagreements_rl"

echo "================================================================================"
echo "Answer Generator (RL/GRPO) - Stage 3"
echo "================================================================================"
echo "Model:        Qwen/Qwen3-VL-235B-A22B-Thinking (module default)"
echo "Contract:     think -> reasoning -> answer + contrast_status"
echo "Category:     $CATEGORY"
echo "Num Workers:  $NUM_WORKERS"
echo "Mode:         $MODE"
if [ "$MODE" = "from_stage1" ]; then
    echo "  -> Auto-discovering samples from Stage 2 outputs in $STAGE1_DIR"
else
    echo "Start Index:  $START_IDX"
    echo "End Index:    $END_IDX"
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

# Run the module
python3 -m nuscenes_pipeline.modules.answer_generator_rl \
    $SAMPLE_ARGS \
    --category "$CATEGORY" \
    --num_workers $NUM_WORKERS \
    --max_new_tokens 32768 \
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
    --disagreement_dir "$DISAGREEMENT_DIR"

echo ""
echo "================================================================================"
echo "RL answer generation complete!"
echo "  Results:  $OUTPUT_DIR/  (incl. rl_run_report.json)"
echo "  Logs:     $LOG_DIR/"
echo "================================================================================"
