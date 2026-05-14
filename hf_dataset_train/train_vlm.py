import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoModelForImageTextToText, set_seed, AutoProcessor
from trl import ModelConfig, ScriptArguments, SFTConfig, SFTTrainer, TrlParser, get_peft_config

if __name__ == "__main__":
    set_seed(41)
    
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
        trust_remote_code=True
    )

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    )

        
    print(f"Loading model: {model_args.model_name_or_path}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_args.model_name_or_path, **model_kwargs
    )
    
    dataset = load_from_disk(script_args.dataset_name)
    train_dataset = dataset.shuffle(seed=41)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=get_peft_config(model_args),
        processing_class=processor,
    )
    
    trainer.train()
    trainer.save_model(training_args.output_dir)