#!/bin/bash

################################################################################
# Traffic Sign Extraction - Stage 1C
#
# Extracts and classifies traffic signs visible in 6-view camera images.
# Runs parallel to Stage 1B (signal analysis). Uses vLLM-served model via
# OpenAI-compatible API with multiprocessing.
#
# Prerequisites:
#   vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct --tensor-parallel-size 8
#
# Output directories:
#   traffic_sign_results/              - JSON result files (suffix: _sign.json)
################################################################################

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR/.."

# Default parameters
START_IDX=${1:-0}
END_IDX=${2:-6018}
NUM_WORKERS=${3:-8}

# Output directories
RESULTS_DIR="traffic_sign_results"

# Question text (ego-relevance-focused)
QUESTION="Extract all traffic signs visible across the 6 camera views. For each sign, report its category (regulatory/warning/guide), text or symbol, orientation, mount location, and whether it applies to the ego-vehicle per the Sign Ego-Relevance Test."

echo "================================================================================"
echo "Traffic Sign Extraction - Stage 1C"
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
python3 -m nuscenes_pipeline.modules.traffic_sign_extraction \
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
echo "Traffic sign extraction complete!"
echo "  Results:  $RESULTS_DIR/"
echo "================================================================================"
