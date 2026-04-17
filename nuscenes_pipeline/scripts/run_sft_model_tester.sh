#!/bin/bash

################################################################################
# SFT Model Tester
#
# Runs inference on a LoRA-fine-tuned Qwen3-VL-8B model using the same prompt
# structure as training (no-objlist variant). Saves results in the same format
# as sft_train_no_objlist.json for easy comparison / visualization.
#
# Usage:
#   bash nuscenes_pipeline/scripts/run_sft_model_tester.sh              # default val GT test
#   bash nuscenes_pipeline/scripts/run_sft_model_tester.sh val 0 20     # val indices 0..20
#   bash nuscenes_pipeline/scripts/run_sft_model_tester.sh range 0 10   # sample indices 0..10
################################################################################

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR/.."

MODE=${1:-"val"}     # "val" or "range"
START=${2:-0}
END=${3:-9}

BASE_MODEL=${BASE_MODEL:-"ckpts/qwen3_vl_8b_instruct"}
LORA_PATH=${LORA_PATH:-"output_nuscenes_lora_no_objlist_cleansing_qads/checkpoint-5500"}
PKL_PATH=${PKL_PATH:-"/home/yongjinjeon/datasets/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl"}
VAL_PATH=${VAL_PATH:-"/home/yongjinjeon/workspace/qa_dataset/qwen3vl_8b_sft_dataset/sft_val_no_objlist.json"}
OUTPUT_DIR=${OUTPUT_DIR:-"/home/yongjinjeon/workspace/qa_dataset/qwen3vl_8b_sft_dataset"}

echo "============================================"
echo "  SFT Model Tester (Qwen3-VL-8B LoRA)"
echo "============================================"
echo "  Base model: $BASE_MODEL"
echo "  LoRA path:  $LORA_PATH"
echo "  Mode:       $MODE"
echo "  Range:      $START .. $END"
echo "  Output:     $OUTPUT_DIR"
echo "============================================"

if [ "$MODE" = "val" ]; then
    VAL_INDICES=$(seq -s ' ' $START $END)
    python -m nuscenes_pipeline.modules.sft_model_tester \
        --base_model "$BASE_MODEL" \
        --lora_path "$LORA_PATH" \
        --pkl_path "$PKL_PATH" \
        --val_path "$VAL_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --from_val \
        --val_indices $VAL_INDICES
else
    python -m nuscenes_pipeline.modules.sft_model_tester \
        --base_model "$BASE_MODEL" \
        --lora_path "$LORA_PATH" \
        --pkl_path "$PKL_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --start_idx $START --end_idx $END
fi
