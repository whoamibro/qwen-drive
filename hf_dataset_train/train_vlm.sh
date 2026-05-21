# Force offline mode: don't reach out to HF Hub — use local ckpt only
#export TRANSFORMERS_OFFLINE=1
#export HF_HUB_OFFLINE=1

# Use the qwen3vl-b200 conda env (torch 2.11.0 + CUDA 13 + cuDNN 9.19 — required for Blackwell).
# PYTHONNOUSERSITE prevents ~/.local/lib site packages (older trl/datasets/peft) from
# shadowing this env's installs.
export PYTHONNOUSERSITE=1
ENV_BIN=/home/stradvision_bo/.conda/envs/qwen3vl-b200/bin

# Move to project root so ./datas_v9 and ./output/... resolve correctly
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

run_name="qwen3_vl_8b_lora_ourqa_0428_batch16_v9"

$ENV_BIN/accelerate launch --num_processes 8 ./hf_dataset_train/train_vlm.py \
    --model_name_or_path ./ckpts/qwen3_vl_8b_instruct \
    --dataset_name ./datas_v9 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --logging_steps 10 \
    --optim paged_adamw_8bit \
    --lr_scheduler_type 'cosine' \
    --learning_rate 2e-5 \
    --output_dir ./output/$run_name \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 20 \
    --dtype bfloat16 \
    --bf16 true \
    --dataloader_num_workers 4 \
    --dataloader_persistent_workers true \
    --dataloader_pin_memory true \
    --report_to tensorboard \
    --logging_dir ./output/$run_name/tb_logs \
    --run_name $run_name \
    --attn_implementation flash_attention_2 \
    --use_peft \
    --lora_target_modules all-linear \
    --ddp_find_unused_parameters false \
    --lora_r 64 \
    --lora_alpha 128 \
    --gradient_checkpointing true \
    --max_length 16384 \
    --log_level error \
    --log_level_replica error \
