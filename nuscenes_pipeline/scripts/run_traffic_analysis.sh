#!/bin/bash

################################################################################
# Traffic Analysis - Stage 1B
#
# Analyzes traffic signal states per nuScenes sample.
# Uses vLLM-served model via OpenAI-compatible API with multiprocessing.
#
# Prerequisites:
#   vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct --tensor-parallel-size 8
#
# Output directories:
#   traffic_analysis_results/      - JSON result files
################################################################################

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR/.."

# Default parameters
START_IDX=${1:-0}
END_IDX=${2:-6018}
NUM_WORKERS=${3:-8}

# Output directories
RESULTS_DIR="traffic_analysis_results"

# Question text
QUESTION="Identify all traffic signals visible in the images and determine their current state (red, yellow, or green). For each signal, specify which camera it appears in, its orientation relative to ego vehicle's travel direction, and whether it governs ego vehicle's lane."

echo "================================================================================"
echo "Traffic Analysis - Stage 1B"
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

# Run the module
python3 -m nuscenes_pipeline.modules.traffic_analysis \
    --start_idx $START_IDX \
    --end_idx $END_IDX \
    --num_workers $NUM_WORKERS \
    --question "$QUESTION" \
    --resize_factor 1 \
    --max_new_tokens 4096 \
    --to_global \
    --results_dir "$RESULTS_DIR"

echo ""
echo "================================================================================"
echo "Traffic analysis complete!"
echo "  Results:  $RESULTS_DIR/"
echo "================================================================================"
