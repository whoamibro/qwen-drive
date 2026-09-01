#!/bin/bash

################################################################################
# Traffic Signal Analysis v2 - Stage 1B (upgraded)
#
# Same prompts as Stage 1B v1, plus the detected traffic-signal 2D boxes and
# their VLM-identified status (from the pkl tl_* arrays written by
# add_traffic_lights_to_infos.py + apply_traffic_signal_status.py) injected
# into the user prompt as a risk_assessment-style object list.
#
# Prerequisites:
#   - pkl contains tl_* arrays (run_traffic_signal_status.sh done for the split)
#   - vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct --tensor-parallel-size 8
#
# Output directories:
#   traffic_signal_analysis_v2_results/  - JSON results (suffix: _traffic_signal.json,
#                                          same suffix as v1 so downstream stages can
#                                          consume it via --traffic_results_dir)
#
# Usage:
#   bash nuscenes_pipeline/scripts/run_traffic_signal_analysis_v2.sh [START_IDX] [END_IDX] [NUM_WORKERS]
################################################################################

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR/.."

# Default parameters
START_IDX=${1:-0}
END_IDX=${2:-6018}
NUM_WORKERS=${3:-8}

RESULTS_DIR="traffic_signal_analysis_v2_results"

echo "================================================================================"
echo "Traffic Signal Analysis v2 - Stage 1B (detected boxes + status in prompt)"
echo "================================================================================"
echo "Start Index: $START_IDX"
echo "End Index: $END_IDX"
echo "Num Workers: $NUM_WORKERS"
echo "Total samples: $((END_IDX - START_IDX + 1))"
echo ""
echo "Output directories:"
echo "  Results:  $RESULTS_DIR"
echo "================================================================================"
echo ""

python3 -m nuscenes_pipeline.modules.traffic_signal_analysis_v2 \
    --start_idx $START_IDX \
    --end_idx $END_IDX \
    --num_workers $NUM_WORKERS \
    --resize_factor 1 \
    --max_new_tokens 4096 \
    --to_global \
    --results_dir "$RESULTS_DIR"

echo ""
echo "================================================================================"
echo "Traffic signal analysis v2 complete!"
echo "  Results:  $RESULTS_DIR/"
echo "================================================================================"
