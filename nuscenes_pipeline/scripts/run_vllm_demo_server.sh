#!/bin/bash
################################################################################
# Launch a vLLM OpenAI-compatible server for the demo tester.
#
# Serves the base Qwen3-VL-8B as `qwen3vl-8b`, and — when LORA_PATH is set —
# additionally registers the adapter under LORA_NAME, so one server offers
# BOTH modes; clients pick per request via the `model` field:
#   model="qwen3vl-8b"   -> base only
#   model="$LORA_NAME"   -> base + LoRA
#
# Usage:
#   bash nuscenes_pipeline/scripts/run_vllm_demo_server.sh [DP_SIZE]
#
#   DP_SIZE  data-parallel replicas, one GPU each (default: 1)
#
# Environment overrides:
#   BASE_MODEL    (default: ckpts/qwen3_vl_8b_instruct)
#   LORA_PATH     adapter dir to register (optional; omit for base-only server)
#   LORA_NAME     served adapter name (default: dir basename, e.g. ckpt_100)
#   PORT          (default: 8000)
#   GPU_MEM_UTIL  fraction of GPU memory vLLM may claim (default: 0.9)
#
# Demo runs against it with:
#   VLLM_URL=http://localhost:$PORT/v1 MODEL_NAME=<name> \
#   bash nuscenes_pipeline/scripts/run_demo_batch.sh 30 42
################################################################################

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

DP_SIZE="${1:-1}"
BASE_MODEL="${BASE_MODEL:-ckpts/qwen3_vl_8b_instruct}"
PORT="${PORT:-8000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"

LORA_ARGS=()
if [ -n "${LORA_PATH:-}" ] && [ "${LORA_PATH,,}" != "none" ]; then
    LORA_NAME="${LORA_NAME:-$(basename "$LORA_PATH")}"
    LORA_ARGS=(--enable-lora --max-lora-rank 64
               --lora-modules "${LORA_NAME}=${LORA_PATH}")
    echo "[vllm_demo_server] serving base 'qwen3vl-8b' + LoRA '${LORA_NAME}' (${LORA_PATH})"
else
    echo "[vllm_demo_server] serving base 'qwen3vl-8b' only (no LORA_PATH given)"
fi

exec vllm serve "$BASE_MODEL" \
    --served-model-name qwen3vl-8b \
    --data-parallel-size "$DP_SIZE" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --limit-mm-per-prompt '{"image": 6}' \
    --port "$PORT" \
    "${LORA_ARGS[@]}"
