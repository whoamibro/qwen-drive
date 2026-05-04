#!/bin/bash
#
# Full SFT post-processing pipeline (5 stages).
#
# Reads:  qa_results/sample_*_qa_results.json   (Stage 3 output, OBJ IDs)
# Writes: sft_dataset/sft_{train,val}_qwen3vl.json
#
# Bbox coordinate flow:
#   Step 1 projects 3D->2D bboxes in 1600x900 native pixel space
#          (--resize_factor 1) and mirrors x-coords for rear cameras.
#   Step 5 normalizes pixel coords to the [0, 1000] grid that Qwen3-VL
#          grounding tokens were pretrained on, using each image's
#          actual PIL-read dimensions.
#
# Overrides via env vars:
#   PKL_PATH        path to the nuScenes infos pkl
#   DATA_ROOT       data root (where samples/ lives)
#   QA_INPUT_DIR    Stage 3 output dir (default: qa_results)
#   SFT_DIR         intermediate + final output dir (default: sft_dataset)
#
# Usage:
#   bash nuscenes_pipeline/scripts/run_postprocessing.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR/.."

PKL_PATH="${PKL_PATH:-./data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl}"
DATA_ROOT="${DATA_ROOT:-./data/nuscenes}"
QA_INPUT_DIR="${QA_INPUT_DIR:-qa_results}"
SFT_DIR="${SFT_DIR:-sft_dataset}"

echo "================================================================================"
echo "  Post-processing pipeline"
echo "================================================================================"
echo "  pkl:         $PKL_PATH"
echo "  data_root:   $DATA_ROOT"
echo "  qa input:    $QA_INPUT_DIR"
echo "  sft output:  $SFT_DIR"
echo "================================================================================"

echo ""
echo "--- Step 1/5: transform_obj_to_bbox (3D->2D, rear cameras x-mirrored) ---"
python -m nuscenes_pipeline.postprocessing.transform_obj_to_bbox \
    --input_dir "$QA_INPUT_DIR" \
    --output_dir "$SFT_DIR" \
    --data_root "$DATA_ROOT" \
    --pkl_path "$PKL_PATH" \
    --resize_factor 1

echo ""
echo "--- Step 2/5: prepare_sft_dataset (build train/val SFT JSONs) ---"
python -m nuscenes_pipeline.postprocessing.prepare_sft_dataset \
    --qa_dir "$SFT_DIR" \
    --output_dir "$SFT_DIR" \
    --data_root "$DATA_ROOT" \
    --pkl_path "$PKL_PATH"

echo ""
echo "--- Step 3/5: cleanse_obj_references (gpt turns only) ---"
python -m nuscenes_pipeline.postprocessing.cleanse_obj_references \
    --input "$SFT_DIR/sft_train_no_objlist.json"
python -m nuscenes_pipeline.postprocessing.cleanse_obj_references \
    --input "$SFT_DIR/sft_val_no_objlist.json"

echo ""
echo "--- Step 4/5: fix_motion_states (against GT velocity) ---"
python -m nuscenes_pipeline.postprocessing.fix_motion_states \
    --data_dir "$SFT_DIR" \
    --data_root "$DATA_ROOT" \
    --pkl_path "$PKL_PATH"

echo ""
echo "--- Step 5/5: convert_to_qwen3vl_format (system + native grounding tokens, 0-1000) ---"
python -m nuscenes_pipeline.postprocessing.convert_to_qwen3vl_format \
    --input  "$SFT_DIR/sft_train_no_objlist.json" \
    --output "$SFT_DIR/sft_train_qwen3vl.json" \
    --force
python -m nuscenes_pipeline.postprocessing.convert_to_qwen3vl_format \
    --input  "$SFT_DIR/sft_val_no_objlist.json" \
    --output "$SFT_DIR/sft_val_qwen3vl.json" \
    --force

echo ""
echo "================================================================================"
echo "  Done. Final SFT files:"
echo "    $SFT_DIR/sft_train_qwen3vl.json"
echo "    $SFT_DIR/sft_val_qwen3vl.json"
echo "================================================================================"
