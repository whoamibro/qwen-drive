#!/bin/bash
# Full fine-tuning for Qwen3-VL-8B on nuScenes driving VQA dataset
#
# Requires DeepSpeed ZeRO-3 for 8B model.
#
# Usage:
#   cd /path/to/qwen-drive
#   NPROC_PER_NODE=4 bash qwen-vl-finetune/scripts/run_nuscenes_full.sh  # multi-GPU recommended
#   NPROC_PER_NODE=1 bash qwen-vl-finetune/scripts/run_nuscenes_full.sh  # single GPU (needs ~80GB VRAM)

set -e

# Project root = Qwen3-VL directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
NNODES=${WORLD_SIZE:-1}

# DeepSpeed config (ZeRO-3 required for full fine-tuning 8B)
DEEPSPEED_CONFIG="$PROJECT_ROOT/qwen-vl-finetune/scripts/zero3.json"

# Model
MODEL_PATH="ckpts/qwen3_vl_8b_instruct"

# Data paths
TRAIN_DATA="$PROJECT_ROOT/sft_dataset/sft_train_no_objlist.json"
VAL_DATA="$PROJECT_ROOT/sft_dataset/sft_val_no_objlist.json"

# Output
OUTPUT_DIR="$PROJECT_ROOT/output_nuscenes_full_no_objlist"
RUN_NAME="nuscenes-qwen3vl-8b-full-no-objlist"

# Training hyperparameters
LR=2e-5
BATCH_SIZE=1
GRAD_ACCUM=16
NUM_EPOCHS=2
MAX_LENGTH=8192
RESIZE_FACTOR=2

echo "============================================"
echo "  Full Fine-tuning: Qwen3-VL-8B"
echo "============================================"
echo "  Project root:  $PROJECT_ROOT"
echo "  Model:         $MODEL_PATH"
echo "  Train data:    $TRAIN_DATA"
echo "  Output:        $OUTPUT_DIR"
echo "  Learning rate: $LR"
echo "  Batch size:    $BATCH_SIZE x $GRAD_ACCUM (accum)"
echo "  Resize factor: $RESIZE_FACTOR"
echo "  GPUs:          $NPROC_PER_NODE"
echo "============================================"

torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --nnodes=$NNODES \
    train_nuscenes_qwen3vl.py \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --mode full \
    --model_name_or_path "$MODEL_PATH" \
    --train_data_path "$TRAIN_DATA" \
    --val_data_path "$VAL_DATA" \
    --resize_factor $RESIZE_FACTOR \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --data_flatten True \
    --bf16 \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy "steps" \
    --eval_steps 500 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate $LR \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length $MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name "$RUN_NAME" \
    --report_to tensorboard

echo "Training complete. Model saved to: $OUTPUT_DIR"
