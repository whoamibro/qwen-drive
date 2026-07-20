#!/bin/bash

################################################################################
# Traffic Light & Pole 3D Detection - Stage 1D
#
# Detects traffic signal light housings (3D bounding boxes) and their supporting
# poles/mast arms (3D line segments) per nuScenes sample. One vLLM API call per
# camera view; 2D detections are lifted to 3D (ego FLU frame) using camera
# intrinsics + sensor2ego extrinsics. Runs parallel to Stages 1A/1B/1C.
#
# Prerequisites (no special flags needed — the model's default pixel budget of
# 16.78M px passes the x2.5-upscaled 4000x2250 input through unchanged; detected
# coordinates are 0-1000 normalized, so alignment never depends on resizing):
#   vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct --tensor-parallel-size 8
#
# Output directories:
#   traffic_light_pole_3d_results/   - one JSON per sample: {idx:04d}_{token}.json
#   traffic_light_pole_3d_vis/       - (with "visualize") annotated 6-view + BEV
#                                      composite per sample: {idx:04d}_{token}.jpg
#
# Usage:
#   bash nuscenes_pipeline/scripts/run_traffic_light_pole_detection.sh [START_IDX] [END_IDX] [NUM_WORKERS] [front_only] [visualize]
#   ("front_only" and "visualize" are optional and order-independent)
################################################################################

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR/.."

# Default parameters
START_IDX=${1:-0}
END_IDX=${2:-6018}
NUM_WORKERS=${3:-8}
FRONT_ONLY_ARG=""
VISUALIZE_ARG=""
for opt in "${@:4}"; do
    case "$opt" in
        front_only) FRONT_ONLY_ARG="--front_only" ;;
        visualize)  VISUALIZE_ARG="--visualize" ;;
    esac
done

# Output directory
RESULTS_DIR="traffic_light_pole_3d_results"

#MODEL_NAME="Qwen/Qwen3-VL-235B-A22B-Thinking"
MODEL_NAME="Qwen/Qwen3-VL-235B-A22B-Instruct"

echo "================================================================================"
echo "Traffic Light & Pole 3D Detection - Stage 1D"
echo "================================================================================"
echo "Model: $MODEL_NAME"
echo "Start Index: $START_IDX"
echo "End Index: $END_IDX"
echo "Num Workers: $NUM_WORKERS"
echo "Total samples: $((END_IDX - START_IDX + 1))"
echo "Front only: $([ -n "$FRONT_ONLY_ARG" ] && echo yes || echo no)"
echo "Visualize: $([ -n "$VISUALIZE_ARG" ] && echo yes || echo no)"
echo ""
echo "Output directory:"
echo "  Results:  $RESULTS_DIR"
echo "================================================================================"
echo ""

# Run the module

# --upscale_factor 2.5 \
python3 -m nuscenes_pipeline.modules.traffic_light_pole_detection \
    --model_name "$MODEL_NAME" \
    --start_idx $START_IDX \
    --end_idx $END_IDX \
    --num_workers $NUM_WORKERS \
    --upscale_size 4000 4000 \
    --max_new_tokens 8192 \
    --results_dir "$RESULTS_DIR" \
    $FRONT_ONLY_ARG \
    $VISUALIZE_ARG

echo ""
echo "================================================================================"
echo "Traffic light & pole 3D detection complete!"
echo "  Results:  $RESULTS_DIR/"
echo "================================================================================"
