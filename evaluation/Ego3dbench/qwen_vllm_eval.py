import torch
from datasets import load_dataset
import json
import os
import sys
from transformers import AutoProcessor
# from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import argparse
from vllm import LLM, SamplingParams
try:
    from vllm.lora.request import LoRARequest
except ImportError:
    from vllm import LoRARequest
from vllm.inputs import TokensPrompt
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.eval import *
from utils.common import *

import re

    
### fix the seed:
torch.manual_seed(42)

def extract_answer(response: str) -> str:
    match = re.search(r'"answer"\s*:\s*"?([^",}\n]+)"?', response)
    return match.group(1).strip() if match else ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained LoRA model or base model")
    parser.add_argument("--save_name", type=str, required=True, help="Name for the log directory")
    parser.add_argument("--image_root", type=str, default='../nuScenes/samples')
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--use_base_model", action="store_true", help="Evaluate the base model even if LoRA adapter exists")
    parser.add_argument("--base_model_name", type=str)
    args = parser.parse_args()

    # Use default processor settings (will use longest_edge: 16777216 from preprocessor_config.json)
    mm_processor_kwargs = None

    if args.use_base_model:
        # base_model_path = "Qwen/Qwen3-VL-8B-Instruct"
        base_model_path = base_model_name
        print(f">>> Using base model: {base_model_path}")
        lora_request = None
        enable_lora = False
        model_to_load = base_model_path
    else:
        ## check if it is a LoRA adapter
        adapter_config_path = os.path.join(args.model_path, "adapter_config.json")
        lora_request = None
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            base_model = adapter_config.get("base_model_name_or_path")
            print(f">>> LoRA adapter detected. Path: {args.model_path}")
            print(f">>> Base model: {base_model}")
            model_to_load = base_model
            lora_request = LoRARequest("adapter", 1, args.model_path)
            enable_lora = True
        else:
            model_to_load = args.model_path
            enable_lora = False
        
        print(f">>> LoRA enabled: {enable_lora}")
        print(f">>> Resolution Config: {mm_processor_kwargs}")

    ## load model via vLLM
    llm = LLM(
        model=model_to_load,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        enable_lora=enable_lora,
        max_lora_rank=128, # Training used alpha=128
        limit_mm_per_prompt={"image": 6}
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=2048,
    )

    # Use base model for processor template
    processor_qwen = AutoProcessor.from_pretrained(model_to_load, use_fast=True, trust_remote_code=True)

    ## load dataset
    dataset = load_dataset("vbdai/Ego3D-Bench")['test']
    dataset = dataset.filter(lambda x: x["source"].lower() == "nuscenes")
    print(f"Filtered to {len(dataset)} nuScenes samples.")

    ## output file setup
    processsed = {}
    save_path = {}
    idx = {}
    if not os.path.exists(f"logs/{args.save_name}"):
        os.makedirs(f"logs/{args.save_name}", exist_ok=True)
    
    for category in set(dataset['category']):
        save_path[category] = f"logs/{args.save_name}/{category}.jsonl"
        processsed[category] = 0
        idx[category] = 0
        if os.path.exists(save_path[category]):
            with open(save_path[category]) as f:
                processsed[category] = sum(1 for _ in f)
            print(f"Resuming from {processsed[category]} in {category}.")

    for sample in tqdm(dataset):
        idx[sample['category']] += 1
        if idx[sample['category']] <= processsed[sample['category']]:
            continue

        image_path = sample['images']
        question = sample['question']
        options = sample['options']

        # Fix camera naming differences if necessary
        question = question.replace("Back Right", "Back_Right").replace("Back Left", "Back_Left")
        question = question.replace("Back_Right", "Back Left").replace("Back_Left", "Back Right")

        if options:
            for option in options:
                question += '\n' + option
                
        if sample['category'] in ['Ego_Centric_Absolute_Distance', 'Object_Centric_Absolute_Distance']:
            answer_format = "a number only (no units, no extra text)"
        else:
            answer_format = "only the letter of the choice (e.g., A, B, C, D)"

        question += (
            "\n\nOutput your response strictly as a JSON object with the following format:\n"
            '{"reasoning": "<reasoning process>", '
            f'"answer": "<{answer_format}>"}}\n'
            "Do not include any text outside the JSON object.")

        # ego3dbench's original format
        # if sample['category'] in ['Ego_Centric_Absolute_Distance','Object_Centric_Absolute_Distance']:
        #     question += "\nOutput the thinking process in <think> </think> and final answer (number only) in <answer> </answer> tags."
        # else:
        #     question += "\nOutput the thinking process in <think> </think> and final answer (only the letter of the choice) in <answer> </answer> tags."

        image_order = ['Front_Left', 'Front', 'Front_Right', 'Back_Left', 'Back', 'Back_Right'] 
        image_path_sorted = [os.path.join(args.image_root, f"CAM_{img.upper()}", image_path[img]) for img in image_order]

        messages = convert_to_qwen_input(question, image_path_sorted)

        text_prompt = processor_qwen.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        pil_images = []
        for img_p in image_path_sorted:
            if os.path.exists(img_p):
                pil_images.append(Image.open(img_p).convert("RGB"))

        vllm_input = {
            "prompt": text_prompt,
            "multi_modal_data": {"image": pil_images},
        }

        outputs = llm.generate([vllm_input], sampling_params=sampling_params, lora_request=lora_request)
        response = outputs[0].outputs[0].text

        # response_processed = response.split('<answer>')[-1].split('</answer>')[0]
        # response_processed = response_processed.replace('\n', '').strip()

        response_processed = extract_answer(response)

        with open(save_path[sample['category']], "a") as file_out:
            file_out.write(json.dumps({
                'Question': question,
                'GT': sample['answer'],
                'Pred': response,
                'Processed_Pred': response_processed,
                'Category': sample['category'],
                'Images': image_path_sorted
            }) + "\n")

    print(f"\nEvaluation finished. Logs saved in logs/{args.save_name}")

    ### Evaluate the stored log:
    eval_order = [
        'Ego_Centric_Absolute_Distance',
        'Ego_Centric_Absolute_Distance_MultiChoice',
        'Ego_Centric_Relative_Distance',
        'Ego_Centric_Motion_Reasoning',
        'Object_Centric_Absolute_Distance',
        'Object_Centric_Absolute_Distance_MultiChoice',
        'Object_Centric_Relative_Distance',
        'Object_Centric_Motion_Reasoning',
        'Localization',
        'Travel_Time',
    ]
    eval_results = {}
    for category in eval_order:
        if category not in save_path:
            save_path[category] = f"logs/{args.save_name}/{category}.jsonl"
        if not os.path.exists(save_path[category]):
            continue

        if category in ['Ego_Centric_Absolute_Distance', 'Object_Centric_Absolute_Distance']:
            multi_choice = False
        else:
            multi_choice = True
        result = eval_logs(save_path[category], multi_choice=multi_choice)
        if result:
            eval_results[category] = result

    ### Save eval results to JSON:
    eval_save_path = f"logs/{args.save_name}/eval_results.json"
    with open(eval_save_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nEval results saved to {eval_save_path}")


