#!/bin/bash

# 평가할 모델 경로 (학습 시 설정한 output_dir)
MODEL_PATH="../output/qwen3_vl_8b_lora_ourqa_0428_batch16_v8"

# 결과 저장 폴더 이름 (Default Resolution 사용)
SAVE_NAME="temp"

# Qwen/Qwen2.5-VL-7B-Instruct
# Qwen3-VL-8B-Instruct

CUDA_VISIBLE_DEVICES=0 python ./qwen_vllm_eval.py \
    --model_path $MODEL_PATH \
    --save_name $SAVE_NAME \
    --tensor_parallel_size 1 \
    

# CUDA_VISIBLE_DEVICES=0 python ./qwen_vllm_eval.py \
#     --model_path $MODEL_PATH \
#     --save_name $SAVE_NAME \
#     --tensor_parallel_size 1 \
#     --use_base_model
#     --base_model_name Qwen/Qwen2.5-VL-7B-Instruct

echo "Evaluation for $SAVE_NAME finished."
