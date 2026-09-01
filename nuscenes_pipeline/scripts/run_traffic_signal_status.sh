#!/bin/bash

################################################################################
# Traffic Signal Status — crop -> VLM classify -> write back into the infos pkl
#
# Steps (all resumable / idempotent):
#   1. crop_traffic_signals        tl_bboxes2d (signals only, never poles) -> cropped_ts_p/{split}/
#   2. traffic_signal_status_classifier   one vLLM request per crop -> traffic_signal_status_results/{split}_status.jsonl
#   3. apply_traffic_signal_status        -> tl_signal_type2d / tl_light_observable2d / tl_light_color2d /
#                                            tl_lit_shape2d / tl_status_conf2d in the pkl
#
# Prerequisite: a vLLM OpenAI-compatible server, e.g.
#   vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct --tensor-parallel-size 8              (port 8000)
#   CUDA_VISIBLE_DEVICES=4 vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8010 \
#       --gpu-memory-utilization 0.4 --max-model-len 4096 --limit-mm-per-prompt '{"image":1}'
#
# Usage:
#   bash nuscenes_pipeline/scripts/run_traffic_signal_status.sh SPLIT [MODEL_NAME] [API_BASE] [WORKERS] [skip_crop] [skip_apply]
#   e.g. bash nuscenes_pipeline/scripts/run_traffic_signal_status.sh val Qwen/Qwen3-VL-8B-Instruct http://localhost:8010/v1 48
#
# Env: NUSCENES_PKL_PATH overrides the pkl (default data/nuscenes/nuscenes2d_ego_temporal_infos_${SPLIT}.pkl)
#      TS_PAD (default 8), TS_MIN_SIDE (default 160) are forwarded to the classifier.
################################################################################

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR/.."

SPLIT=${1:?usage: run_traffic_signal_status.sh SPLIT [MODEL_NAME] [API_BASE] [WORKERS] [skip_crop] [skip_apply]}
MODEL_NAME=${2:-"Qwen/Qwen3-VL-235B-A22B-Instruct"}
API_BASE=${3:-"http://localhost:8000/v1"}
WORKERS=${4:-32}
SKIP_CROP=""; SKIP_APPLY=""
for opt in "${@:5}"; do
    case "$opt" in
        skip_crop)  SKIP_CROP=1 ;;
        skip_apply) SKIP_APPLY=1 ;;
    esac
done

PKL_PATH=${NUSCENES_PKL_PATH:-"data/nuscenes/nuscenes2d_ego_temporal_infos_${SPLIT}.pkl"}
CROPS_DIR="./cropped_ts_p"
MANIFEST="${CROPS_DIR}/${SPLIT}_manifest.jsonl"
STATUS_DIR="traffic_signal_status_results"
STATUS_FILE="${STATUS_DIR}/${SPLIT}_status.jsonl"
mkdir -p "$STATUS_DIR" logs

echo "=== Traffic signal status: split=$SPLIT pkl=$PKL_PATH model=$MODEL_NAME api=$API_BASE ==="

if [ -z "$SKIP_CROP" ]; then
    echo "--- Step 1: crop traffic signals ---"
    python -m nuscenes_pipeline.postprocessing.crop_traffic_signals \
        --pkl_path "$PKL_PATH" --split "$SPLIT" --output_dir "$CROPS_DIR" || exit 1
fi

echo "--- Step 2: classify crops with $MODEL_NAME ---"
python -m nuscenes_pipeline.modules.traffic_signal_status_classifier \
    --manifest "$MANIFEST" --crops_root "$CROPS_DIR" --output "$STATUS_FILE" \
    --model_name "$MODEL_NAME" --api_base "$API_BASE" --workers "$WORKERS" \
    --pad "${TS_PAD:-8}" --min_side "${TS_MIN_SIDE:-160}" \
    2>&1 | tee "logs/traffic_signal_status_${SPLIT}_$(date +%Y%m%d_%H%M%S).log"
[ "${PIPESTATUS[0]}" -eq 0 ] || exit 1

if [ -z "$SKIP_APPLY" ]; then
    echo "--- Step 3: write status into pkl ---"
    python -m nuscenes_pipeline.postprocessing.apply_traffic_signal_status \
        --pkl_path "$PKL_PATH" --status "$STATUS_FILE" || exit 1
fi
echo "=== Done ==="
