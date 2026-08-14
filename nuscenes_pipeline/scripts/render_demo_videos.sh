#!/bin/bash
################################################################################
# Render demo videos for every scene in a batch directory.
#
# Loops over <BATCH_DIR>/*/predictions.json and invokes
# nuscenes_pipeline.visualization.demo_scene_video for each scene, skipping
# scenes whose videos/ dir already contains MP4s (safe to re-run).
#
# Usage:
#   bash nuscenes_pipeline/scripts/render_demo_videos.sh [BATCH_DIR]
#
#   BATCH_DIR  batch directory containing <scene16>/predictions.json subdirs,
#              relative to the project root or absolute
#              (default: demo_test_results/c10_ckpt100_modeB)
#
# Environment overrides:
#   JOBS           (default: 8) scenes rendered concurrently; each job is one
#                  python+ffmpeg process, its output going to <scene>/videos/render.log
#   FRAMERATE      (default: 2) frames per second
#   RESIZE_FACTOR  (default: 2) downscale factor for the composite frames
#   GATHER=1       afterwards gather the MP4s via gather_demo_videos.sh
#   DEMO_VID_DIR   gather target when GATHER=1 (default: <scripts dir>/demo_vid)
#
# ffmpeg: if not already on PATH, the cosmos-transfer1 env's binary is used.
################################################################################

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

BATCH_DIR="${1:-demo_test_results/c10_ckpt100_modeB}"
JOBS="${JOBS:-8}"
FRAMERATE="${FRAMERATE:-2}"
RESIZE_FACTOR="${RESIZE_FACTOR:-2}"

[ -d "$BATCH_DIR" ] || { echo "ERROR: BATCH_DIR=$BATCH_DIR not found" >&2; exit 1; }

# Ensure an ffmpeg binary is available (demo_scene_video shells out to it).
if ! command -v ffmpeg >/dev/null 2>&1; then
    if [ -x /opt/conda/envs/cosmos-transfer1/bin/ffmpeg ]; then
        export PATH="/opt/conda/envs/cosmos-transfer1/bin:$PATH"
        echo "[render] using ffmpeg from cosmos-transfer1 env"
    else
        echo "ERROR: no ffmpeg on PATH and cosmos-transfer1 fallback not found" >&2
        exit 1
    fi
fi

shopt -s nullglob
PREDS=("$BATCH_DIR"/*/predictions.json)
TOTAL=${#PREDS[@]}
[ "$TOTAL" -gt 0 ] || { echo "ERROR: no */predictions.json under $BATCH_DIR" >&2; exit 1; }

echo "============================================================"
echo "Rendering $TOTAL scenes from $BATCH_DIR"
echo "  jobs=$JOBS  framerate=$FRAMERATE  resize_factor=$RESIZE_FACTOR"
echo "============================================================"

# Scene-level parallelism: scenes are fully independent (separate videos/
# dirs, separate _temp_frames), so run up to $JOBS renders concurrently.
# Each job's output goes to <scene>/videos/render.log to keep the console
# readable; failures are collected via an append-only temp file.
FAIL_LOG="$(mktemp)"
n=0
for p in "${PREDS[@]}"; do
    n=$((n+1))
    SCENE_DIR="$(dirname "$p")"
    if compgen -G "$SCENE_DIR/videos/*.mp4" > /dev/null; then
        echo "[$n/$TOTAL] $SCENE_DIR — videos exist, skipping"
        continue
    fi
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do
        wait -n || true
    done
    echo "[$n/$TOTAL] rendering $SCENE_DIR"
    (
        mkdir -p "$SCENE_DIR/videos"
        if python3 -m nuscenes_pipeline.visualization.demo_scene_video \
            --predictions "$p" --framerate "$FRAMERATE" --resize_factor "$RESIZE_FACTOR" \
            > "$SCENE_DIR/videos/render.log" 2>&1; then
            echo "        done $SCENE_DIR"
        else
            echo "$SCENE_DIR" >> "$FAIL_LOG"
            echo "RENDER_FAILED: $SCENE_DIR (see videos/render.log)" >&2
        fi
    ) &
done
wait
FAILED=()
while IFS= read -r line; do [ -n "$line" ] && FAILED+=("$line"); done < "$FAIL_LOG"
rm -f "$FAIL_LOG"

if [ "${GATHER:-0}" = "1" ]; then
    echo ""
    DEMO_VID_DIR="${DEMO_VID_DIR:-$SCRIPT_DIR/demo_vid}"
    bash "$SCRIPT_DIR/gather_demo_videos.sh" "$BATCH_DIR" "$DEMO_VID_DIR"
fi

echo ""
echo "============================================================"
echo "Render complete: $((TOTAL - ${#FAILED[@]}))/$TOTAL scenes OK"
if [ "${#FAILED[@]}" -gt 0 ]; then
    printf 'Failed scenes:\n'; printf '  %s\n' "${FAILED[@]}"
    exit 1
fi
