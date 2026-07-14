#!/bin/bash
# Fan out demo_tester across all local GPUs via frame-level data parallelism.
# Each worker owns a strided slice of the scene's frames (frame_pos % NPROC == i)
# and writes its own predictions_worker<i>.json. When all workers finish, the
# per-worker JSONs are merged into a canonical predictions.json.
#
# Usage:
#   bash nuscenes_pipeline/scripts/run_demo_tester_8gpu.sh \
#       <scene_token> <mode_arg>
#
#   <mode_arg> is either:
#     - a path to demo_questions.json (Mode A), or
#     - a plain-string question wrapped in single-quotes (Mode B).
#
# Examples:
#   # Mode A: all 40 demo questions across all frames of the scene
#   bash nuscenes_pipeline/scripts/run_demo_tester_8gpu.sh \
#       ff6af17f52c34e9c data/demo_questions.json
#
#   # Mode B: single custom question across all frames
#   bash nuscenes_pipeline/scripts/run_demo_tester_8gpu.sh \
#       b526c20f7eed49f0 'Is the ego-vehicle safe to change lanes right?'
#
# Environment overrides:
#   BASE_MODEL     (default: ckpts/qwen3_vl_8b_instruct)
#   LORA_PATH      (REQUIRED)
#   PKL_PATH       (default: data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl)
#   OUTPUT_DIR     (default: demo_test_results/<scene_token[:16]>)
#   NPROC          (default: 8) — number of GPU workers
#   GPU_IDS        (default: 0,1,...,NPROC-1)
#   RESIZE_FACTOR  (default: 1)
#   MAX_NEW_TOKENS (default: 1024)
#   MAX_FRAMES     (default: unset)
#   RENDER_VIDEO=1 — after merge, invoke demo_scene_video for per-category MP4s

set -e
set -o pipefail

SCENE_TOKEN="${1:?usage: $0 <scene_token> <demo_questions.json | \"question text\">}"
MODE_ARG="${2:?usage: $0 <scene_token> <demo_questions.json | \"question text\">}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
cd "$PROJECT_ROOT"

BASE_MODEL="${BASE_MODEL:-ckpts/qwen3_vl_8b_instruct}"
if [ -z "${LORA_PATH:-}" ]; then
    echo "ERROR: LORA_PATH is required (env var)." >&2
    exit 1
fi
PKL_PATH="${PKL_PATH:-data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl}"
RESIZE_FACTOR="${RESIZE_FACTOR:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
SCENE_SHORT="${SCENE_TOKEN:0:16}"
OUTPUT_DIR="${OUTPUT_DIR:-demo_test_results/${SCENE_SHORT}}"

# Decide Mode A vs B from the shape of MODE_ARG.
if [ -f "$MODE_ARG" ]; then
    MODE_FLAG=(--demo_questions_path "$MODE_ARG")
    MODE_DESC="A (question list from $MODE_ARG)"
else
    MODE_FLAG=(--q "$MODE_ARG")
    MODE_DESC="B (single custom question)"
fi

# GPU pool
DETECTED_NPROC=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 0)
[ "$DETECTED_NPROC" -le 0 ] && DETECTED_NPROC=8
NPROC="${NPROC:-$DETECTED_NPROC}"
if [ -n "${GPU_IDS:-}" ]; then
    IFS=',' read -ra GPU_ARR <<<"$GPU_IDS"
    if [ "${#GPU_ARR[@]}" != "$NPROC" ]; then
        echo "ERROR: GPU_IDS has ${#GPU_ARR[@]} entries but NPROC=$NPROC" >&2
        exit 1
    fi
else
    GPU_ARR=()
    for ((i=0; i<NPROC; i++)); do GPU_ARR+=($i); done
    GPU_IDS="$(IFS=,; echo "${GPU_ARR[*]}")"
fi

MAX_FRAMES_FLAG=()
if [ -n "${MAX_FRAMES:-}" ]; then
    MAX_FRAMES_FLAG=(--max_frames "$MAX_FRAMES")
fi

mkdir -p "$OUTPUT_DIR"
LOG_DIR="$OUTPUT_DIR/_worker_logs"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "demo_tester 8-GPU fan-out"
echo "  scene_token   : $SCENE_TOKEN  (short=$SCENE_SHORT)"
echo "  mode          : $MODE_DESC"
echo "  base_model    : $BASE_MODEL"
echo "  lora_path     : $LORA_PATH"
echo "  pkl_path      : $PKL_PATH"
echo "  output_dir    : $OUTPUT_DIR"
echo "  NPROC / GPUs  : $NPROC / [${GPU_IDS}]"
echo "  resize_factor : $RESIZE_FACTOR"
echo "  max_new_tokens: $MAX_NEW_TOKENS"
[ -n "${MAX_FRAMES:-}" ] && echo "  max_frames    : $MAX_FRAMES"
echo "============================================================"

# Launch workers in parallel.
PIDS=()
for ((i=0; i<NPROC; i++)); do
    GPU_ID="${GPU_ARR[$i]}"
    LOG="$LOG_DIR/worker${i}.log"
    (
        CUDA_VISIBLE_DEVICES="$GPU_ID" \
        python -m nuscenes_pipeline.modules.demo_tester \
            --scene_token "$SCENE_TOKEN" \
            "${MODE_FLAG[@]}" \
            --base_model "$BASE_MODEL" \
            --lora_path "$LORA_PATH" \
            --pkl_path "$PKL_PATH" \
            --output_dir "$OUTPUT_DIR" \
            --resize_factor "$RESIZE_FACTOR" \
            --max_new_tokens "$MAX_NEW_TOKENS" \
            --sample_stride "$NPROC" \
            --sample_offset "$i" \
            --print_every 5 \
            "${MAX_FRAMES_FLAG[@]}" \
            >"$LOG" 2>&1
    ) &
    PIDS+=($!)
    echo "  launched worker $i on GPU $GPU_ID (pid=${PIDS[-1]}, log=$LOG)"
done

FAIL=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo "ERROR: worker $i (pid=${PIDS[$i]}) exited non-zero. Log: $LOG_DIR/worker${i}.log" >&2
        FAIL=1
    fi
done
if [ "$FAIL" -ne 0 ]; then
    echo "ABORTING before merge — one or more workers failed." >&2
    exit 2
fi

echo
echo "============================================================"
echo "Merging $NPROC worker JSONs -> $OUTPUT_DIR/predictions.json"
echo "============================================================"
python -m nuscenes_pipeline.modules.merge_demo_predictions \
    --worker_dir "$OUTPUT_DIR" \
    --output "$OUTPUT_DIR/predictions.json"

if [ "${RENDER_VIDEO:-0}" = "1" ]; then
    echo
    echo "============================================================"
    echo "Rendering per-category MP4s"
    echo "============================================================"
    python -m nuscenes_pipeline.visualization.demo_scene_video \
        --predictions "$OUTPUT_DIR/predictions.json" \
        --output_dir "$OUTPUT_DIR/videos" \
        --framerate 2 \
        --resize_factor 2
fi

echo
echo "Done. Predictions: $OUTPUT_DIR/predictions.json"
[ "${RENDER_VIDEO:-0}" = "1" ] && echo "Videos:       $OUTPUT_DIR/videos/"
