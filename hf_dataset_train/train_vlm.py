# Workaround: transformer_engine_cu12-2.4.0+3cd6870c has an empty/malformed RECORD file
# in its .dist-info, which crashes importlib.metadata.packages_distributions() with
# "make_file() missing 1 required positional argument: 'name'". trl/import_utils.py
# calls packages_distributions() at import time. Patch Distribution.files in both the
# stdlib module and the importlib_metadata backport so iteration skips the bad dist.
import importlib.metadata as _stdlib_im
_modules_to_patch = [_stdlib_im]
try:
    import importlib_metadata as _backport_im
    _modules_to_patch.append(_backport_im)
except ImportError:
    pass

def _make_safe_files(orig):
    is_prop = isinstance(orig, property)
    def _safe_files(self):
        try:
            return orig.fget(self) if is_prop else orig(self)
        except TypeError:
            return None
    return property(_safe_files) if is_prop else _safe_files

for _mod in _modules_to_patch:
    _mod.Distribution.files = _make_safe_files(_mod.Distribution.files)

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