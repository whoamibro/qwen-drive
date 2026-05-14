export WANDB_PROJECT="Qwen_VL_Local"
export WANDB_RUN_NAME="qwen3_vl_8b_lora_ourqa_v8"

run_name="qwen3_vl_8b_lora_ourqa_0428_batch16_v8"

accelerate launch --num_processes 8 ./train_vlm.py \
    --model_name_or_path Qwen/Qwen3-VL-8B-Instruct\
    --dataset_name ./datas_v8 \
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
    --report_to wandb \
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
