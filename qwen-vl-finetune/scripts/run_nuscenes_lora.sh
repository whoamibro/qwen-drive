#!/bin/bash
# LoRA fine-tuning for Qwen3-VL-8B on nuScenes driving VQA dataset
#
# Usage:
#   cd /path/to/qwen-drive
#   NPROC_PER_NODE=1 bash qwen-vl-finetune/scripts/run_nuscenes_lora.sh
#   NPROC_PER_NODE=4 bash qwen-vl-finetune/scripts/run_nuscenes_lora.sh  # multi-GPU

set -e

# Project root = Qwen3-VL directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# NCCL timeout + telemetry (prevents 10-min default timeout when eval/save runs long)
export NCCL_TIMEOUT=3600
export TORCH_NCCL_TIMEOUT_SEC=3600
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
NNODES=${WORLD_SIZE:-1}

# DeepSpeed config (ZeRO-2 is sufficient for LoRA)
DEEPSPEED_CONFIG="$PROJECT_ROOT/qwen-vl-finetune/scripts/zero2.json"

# Model
MODEL_PATH="ckpts/qwen3_vl_8b_instruct"

# Data paths
TRAIN_DATA="$PROJECT_ROOT/sft_dataset/sft_train_qwen3vl.json"
# Small held-out subset (300 samples) for in-training eval.
# Full val set (sft_val_no_objlist.json, 35K samples) is reserved for final evaluation.
VAL_DATA="$PROJECT_ROOT/sft_dataset/sft_val_qwen3vl.json"

# Output
OUTPUT_DIR="$PROJECT_ROOT/output_nuscenes_lora_5p_test_qwen3vl_v9"
RUN_NAME="nuscenes-qwen3vl-8b-lora-5p-test-qwen3vl"

# LoRA hyperparameters
LORA_R=64
LORA_ALPHA=128
LORA_DROPOUT=0.05

# Training hyperparameters
LR=2e-4
BATCH_SIZE=1
GRAD_ACCUM=4
NUM_EPOCHS=3
MAX_LENGTH=16384
# max_pixels = 1,440,208 ~= 1600 * 900 -> ViT receives near-native nuScenes
# resolution (rounded to a 28x28 patch grid). The image processor's
# smart_resize handles all downsampling inside this cap; no PIL pre-resize
# is performed in train_nuscenes_qwen3vl.py.
MAX_PIXELS=1440208
MIN_PIXELS=784

echo "============================================"
echo "  LoRA Fine-tuning: Qwen3-VL-8B"
echo "============================================"
echo "  Project root:  $PROJECT_ROOT"
echo "  Model:         $MODEL_PATH"
echo "  Train data:    $TRAIN_DATA"
echo "  Output:        $OUTPUT_DIR"
echo "  LoRA rank:     $LORA_R"
echo "  Learning rate: $LR"
echo "  Batch size:    $BATCH_SIZE x $GRAD_ACCUM (accum)"
echo "  max_pixels:    $MAX_PIXELS"
echo "  max_length:    $MAX_LENGTH"
echo "  GPUs:          $NPROC_PER_NODE"
echo "============================================"

torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --nnodes=$NNODES \
    train_nuscenes_qwen3vl.py \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --mode lora \
    --model_name_or_path "$MODEL_PATH" \
    --train_data_path "$TRAIN_DATA" \
    --val_data_path "$VAL_DATA" \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --bf16 \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --max_pixels $MAX_PIXELS \
    --min_pixels $MIN_PIXELS \
    --eval_strategy "steps" \
    --eval_steps 1000 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 3 \
    --learning_rate $LR \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length $MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name "$RUN_NAME" \
    --report_to tensorboard

echo "Training complete. LoRA adapter saved to: $OUTPUT_DIR"
