#!/bin/bash
#
# Full SFT post-processing pipeline (5 stages).
#
# Reads:  qa_results/sample_*_qa_results.json   (Stage 3 output, OBJ IDs)
#
# Two output modes selected via MODE env var:
#
#   MODE=full         (default)
#     Step 2 emits one mixed train + val pair:
#       sft_dataset/sft_{train,val}_qwen3vl.json
#
#   MODE=curriculum
#     Step 2 buckets samples by question-bank category and emits 20 files
#     (10 cats x train+val), suffixed _<ABBR> where ABBR ∈
#       OBS, IDN, AAS, SRO, TSS, RML, DRA, RWP, ESC, CHR
#     e.g. sft_dataset/sft_train_qwen3vl_OBS.json,
#          sft_dataset/sft_val_qwen3vl_OBS.json, ...
#     Steps 3 and 5 loop the 20 files; Step 4 makes one pass with shared
#     pkl + image-index + velocity cache so its startup cost is paid once.
#
# Bbox coordinate flow (same in both modes):
#   Step 1 projects 3D->2D bboxes in 1600x900 native pixel space
#          (--resize_factor 1) and mirrors x-coords for rear cameras.
#   Step 5 normalizes pixel coords to the [0, 1000] grid that Qwen3-VL
#          grounding tokens were pretrained on, using each image's
#          actual PIL-read dimensions.
#
# Overrides via env vars:
#   MODE            full (default) | curriculum
#   PKL_PATH        path to the nuScenes infos pkl
#   DATA_ROOT       data root (where samples/ lives)
#   QA_INPUT_DIR    Stage 3 output dir (default: qa_results)
#   SFT_DIR         intermediate + final output dir (default: sft_dataset)
#
# Usage:
#   bash nuscenes_pipeline/scripts/run_postprocessing.sh                  # full
#   MODE=curriculum bash nuscenes_pipeline/scripts/run_postprocessing.sh  # 20 splits

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR/.."

MODE="${MODE:-full}"
PKL_PATH="${PKL_PATH:-./data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl}"
DATA_ROOT="${DATA_ROOT:-./data/nuscenes}"
QA_INPUT_DIR="${QA_INPUT_DIR:-qa_results}"
SFT_DIR="${SFT_DIR:-sft_dataset}"

if [[ "$MODE" != "full" && "$MODE" != "curriculum" ]]; then
    echo "ERROR: MODE must be 'full' or 'curriculum' (got '$MODE')" >&2
    exit 1
fi

CURRICULUM_ABBRS=(OBS IDN AAS SRO TSS RML DRA RWP ESC CHR)

echo "================================================================================"
echo "  Post-processing pipeline"
echo "================================================================================"
echo "  mode:        $MODE"
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
echo "--- Step 2/5: prepare_sft_dataset (mode=$MODE) ---"
if [[ "$MODE" == "curriculum" ]]; then
    PREPARE_MODE_FLAG="--curriculum"
else
    PREPARE_MODE_FLAG="--full"
fi
python -m nuscenes_pipeline.postprocessing.prepare_sft_dataset \
    --qa_dir "$SFT_DIR" \
    --output_dir "$SFT_DIR" \
    --data_root "$DATA_ROOT" \
    --pkl_path "$PKL_PATH" \
    "$PREPARE_MODE_FLAG"

# Resolve the list of intermediate files Step 2 produced.
if [[ "$MODE" == "full" ]]; then
    INTERMEDIATE_TRAIN=("$SFT_DIR/sft_train_no_objlist.json")
    INTERMEDIATE_VAL=(  "$SFT_DIR/sft_val_no_objlist.json")
else
    INTERMEDIATE_TRAIN=()
    INTERMEDIATE_VAL=()
    for abbr in "${CURRICULUM_ABBRS[@]}"; do
        tpath="$SFT_DIR/sft_train_no_objlist_${abbr}.json"
        vpath="$SFT_DIR/sft_val_no_objlist_${abbr}.json"
        [[ -f "$tpath" ]] && INTERMEDIATE_TRAIN+=("$tpath")
        [[ -f "$vpath" ]] && INTERMEDIATE_VAL+=("$vpath")
    done
fi

echo ""
echo "--- Step 3/5: cleanse_obj_references (gpt turns only) ---"
for fpath in "${INTERMEDIATE_TRAIN[@]}" "${INTERMEDIATE_VAL[@]}"; do
    python -m nuscenes_pipeline.postprocessing.cleanse_obj_references \
        --input "$fpath"
done

echo ""
echo "--- Step 4/5: fix_motion_states (against GT velocity) ---"
if [[ "$MODE" == "curriculum" ]]; then
    # Single invocation amortizes pkl load + image index + velocity cache
    # across all 20 curriculum files.
    python -m nuscenes_pipeline.postprocessing.fix_motion_states \
        --data_dir "$SFT_DIR" \
        --data_root "$DATA_ROOT" \
        --pkl_path "$PKL_PATH" \
        --curriculum
else
    python -m nuscenes_pipeline.postprocessing.fix_motion_states \
        --data_dir "$SFT_DIR" \
        --data_root "$DATA_ROOT" \
        --pkl_path "$PKL_PATH"
fi

echo ""
echo "--- Step 5/5: convert_to_qwen3vl_format (system + native grounding tokens, 0-1000) ---"
for fpath in "${INTERMEDIATE_TRAIN[@]}"; do
    out="${fpath/sft_train_no_objlist/sft_train_qwen3vl}"
    python -m nuscenes_pipeline.postprocessing.convert_to_qwen3vl_format \
        --input  "$fpath" --output "$out" --force
done
for fpath in "${INTERMEDIATE_VAL[@]}"; do
    out="${fpath/sft_val_no_objlist/sft_val_qwen3vl}"
    python -m nuscenes_pipeline.postprocessing.convert_to_qwen3vl_format \
        --input  "$fpath" --output "$out" --force
done

echo ""
echo "================================================================================"
echo "  Done. Final SFT files:"
if [[ "$MODE" == "full" ]]; then
    echo "    $SFT_DIR/sft_train_qwen3vl.json"
    echo "    $SFT_DIR/sft_val_qwen3vl.json"
else
    for abbr in "${CURRICULUM_ABBRS[@]}"; do
        echo "    $SFT_DIR/sft_train_qwen3vl_${abbr}.json"
    done
    for abbr in "${CURRICULUM_ABBRS[@]}"; do
        echo "    $SFT_DIR/sft_val_qwen3vl_${abbr}.json"
    done
fi
echo "================================================================================"
