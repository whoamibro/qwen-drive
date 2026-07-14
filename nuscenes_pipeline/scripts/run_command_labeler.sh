#!/bin/bash

################################################################################
# Driving Command Labeler — hand-label GT driving commands in the browser
#
# Launches a Flask web tool that renders, per sample, the 3 forward camera
# views and 7 command-label buttons (0-6). Labels are autosaved as one JSON
# per scene into driving_command_labels/.
#
# Usage:
#   bash nuscenes_pipeline/scripts/run_command_labeler.sh [PORT] [SPLIT]
#
#   PORT   default: 6062. Open http://<server-ip>:6062 in browser.
#   SPLIT  train or val (default: train). Selects the pkl file and keeps
#          labels in a split-specific output dir (driving_command_labels_<SPLIT>).
################################################################################

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR/.."

# Default parameters
PORT=${1:-6062}
SPLIT=${2:-train}

PKL_PATH="./data/nuscenes/nuscenes2d_ego_temporal_infos_${SPLIT}.pkl"
OUTPUT_DIR="driving_command_labels_${SPLIT}"

echo "================================================================================"
echo "Driving Command Labeler"
echo "================================================================================"
echo "  Split:       $SPLIT"
echo "  PKL path:    $PKL_PATH"
echo "  Output dir:  $OUTPUT_DIR"
echo "  Port:        $PORT"
echo "  URL:         http://0.0.0.0:$PORT"
echo "================================================================================"
echo ""

python3 -m nuscenes_pipeline.visualization.driving_command_labeler \
    --pkl_path "$PKL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --port "$PORT" \
    --host 0.0.0.0
