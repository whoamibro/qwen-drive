#!/bin/bash

################################################################################
# Verifier Visualizer — Inspect Stage 1 seed data outputs
#
# Launches a Flask web dashboard that renders, per sample:
#   - 6-view panoramic with OBJ bboxes from the risk_assessment prompt
#   - BEV overlay (ego + objects + velocity vectors)
#   - Risk / Signal / Sign analysis results in parallel columns
#
# Prerequisites:
#   Stages 1A / 1B / 1C must have output files in:
#     risk_assessment_results/
#     traffic_signal_analysis_results/
#     traffic_sign_results/
#
# Usage:
#   bash nuscenes_pipeline/scripts/run_verifier_visualizer.sh [PORT]
#
#   Default port: 6061. Open http://<server-ip>:6061 in browser.
################################################################################

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR/.."

# Default parameters
PORT=${1:-6061}

RISK_DIR="risk_assessment_results"
SIGNAL_DIR="traffic_signal_analysis_results"
SIGN_DIR="traffic_sign_results"
PKL_PATH=${NUSCENES_PKL_PATH:-"./data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl"}

echo "================================================================================"
echo "Verifier Visualizer"
echo "================================================================================"
echo "  Risk dir:    $RISK_DIR"
echo "  Signal dir:  $SIGNAL_DIR"
echo "  Sign dir:    $SIGN_DIR"
echo "  PKL path:    $PKL_PATH"
echo "  Port:        $PORT"
echo "  URL:         http://0.0.0.0:$PORT"
echo "================================================================================"
echo ""

python3 -m nuscenes_pipeline.visualization.verifier_visualizer \
    --risk_dir "$RISK_DIR" \
    --signal_dir "$SIGNAL_DIR" \
    --sign_dir "$SIGN_DIR" \
    --pkl_path "$PKL_PATH" \
    --port "$PORT" \
    --host 0.0.0.0
