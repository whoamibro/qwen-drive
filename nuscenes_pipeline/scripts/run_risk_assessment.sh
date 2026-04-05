#!/bin/bash

################################################################################
# Risk Assessment - Stage 1A
#
# Analyzes driving risks, hazards, and TTC per nuScenes sample.
# Uses vLLM-served model via OpenAI-compatible API with multiprocessing.
#
# Prerequisites:
#   vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct --tensor-parallel-size 8
#
# Output directories:
#   risk_assessment_results/       - JSON result files
#   risk_assessment_logs/          - Log files
################################################################################

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR/.."

# Default parameters
START_IDX=${1:-0}
END_IDX=${2:-6018}
NUM_WORKERS=${3:-8}

# Output directories
RESULTS_DIR="risk_assessment_results"
LOG_DIR="risk_assessment_logs"

# Question text
QUESTION="Analyze risks and hazards following these categories: Risk Categories: 1. Static Hazards: Parked vehicles, road infrastructure, visibility obstructions 2. Dynamic Risks: Potential pedestrian/vehicle emergence zones, blind spots 3. Environmental Factors: Weather, lighting, road geometry 4. Situational Awareness: Areas requiring increased vigilance Output Format: 1. Immediate Risks (requires immediate attention) 2. Potential Risks (monitor closely) 3. Recommended Actions (specific driving advice) 4. Overall Risk Level(Low/Moderate/High with brief justification) Instructions: - Analyze the recommended actions in the overall context of ego-centric surrounding scenes. - Double-check all directions of images before finalizing. - Prioritize by severity and likelihood. - Focus on actionable insight. - For the assessment of the Overall Risk Level, compute the collision risk using the provided collision-risk formula by substituting the 3D information and velocity of all objects and the ego vehicle, and present the evaluated result accordingly."

echo "================================================================================"
echo "Risk Assessment - Stage 1A"
echo "================================================================================"
echo "Start Index: $START_IDX"
echo "End Index: $END_IDX"
echo "Num Workers: $NUM_WORKERS"
echo "Total samples: $((END_IDX - START_IDX + 1))"
echo ""
echo "Output directories:"
echo "  Results:  $RESULTS_DIR"
echo "  Logs:     $LOG_DIR"
echo "================================================================================"
echo ""

# Run the module
python3 -m nuscenes_pipeline.modules.risk_assessment \
    --start_idx $START_IDX \
    --end_idx $END_IDX \
    --num_workers $NUM_WORKERS \
    --question "$QUESTION" \
    --resize_factor 2 \
    --max_new_tokens 4096 \
    --to_global \
    --proj2img \
    --3dod \
    --filter_length 50 \
    --rear_filter 20 \
    --results_dir "$RESULTS_DIR" \
    --log_dir "$LOG_DIR"

echo ""
echo "================================================================================"
echo "Risk assessment complete!"
echo "  Results:  $RESULTS_DIR/"
echo "  Logs:     $LOG_DIR/"
echo "================================================================================"
