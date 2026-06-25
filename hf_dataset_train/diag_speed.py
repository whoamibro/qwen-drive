"""
Diagnostic: identify the source of 5x slowdown in train_vlm.sh on B200.

Tests two hypotheses:
  H1: flash_attention_2 silently falls back on B200 (Blackwell, sm_100).
  H2: Vision processor / data collator is the per-batch bottleneck.

Runs on a single GPU. Doesn't touch the dataset on disk — pulls one sample.
"""

# Reuse the importlib.metadata workaround so this script can also import trl-related deps if needed.
import importlib.metadata as _stdlib_im
_mods = [_stdlib_im]
try:
    import importlib_metadata as _b
    _mods.append(_b)
except ImportError:
    pass

def _safe(orig):
    is_prop = isinstance(orig, property)
    def f(self):
        try:
            return orig.fget(self) if is_prop else orig(self)
        except TypeError:
            return None
    return property(f) if is_prop else f

for _m in _mods:
    _m.Distribution.files = _safe(_m.Distribution.files)

import time
import torch
from datasets import load_from_disk
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_PATH = "./ckpts/qwen3_vl_8b_instruct"
DATA_PATH = "./datas_v9"

device = torch.device("cuda:0")

def banner(s):
    print("\n" + "=" * 70 + f"\n{s}\n" + "=" * 70)


def dump_attn_impl(model, n=4):
    """Inspect the actual attention class on the first n decoder layers."""
    found = []
    for name, m in model.named_modules():
        cls = m.__class__.__name__
        if "Attention" in cls and "Module" not in cls:
            found.append((name, cls))
            if len(found) >= n:
                break
    return found


def time_forward(model, processor, sample, label):
    """Time one forward pass on a single sample."""
    images = sample["images"]
    user_text = []
    for c in sample["prompt"][1]["content"]:
        if c["type"] == "text":
            user_text.append(c["text"])
    system_text = sample["prompt"][0]["content"][0]["text"]

    msgs = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": [
            *[{"type": "image", "image": img} for img in images],
            {"type": "text", "text": " ".join(user_text)},
        ]},
    ]

    # Time processor
    t0 = time.perf_counter()
    inputs = processor.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    )
    t_proc = time.perf_counter() - t0
    seq_len = inputs["input_ids"].shape[1]

    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    # Time forward (warmup + measure)
    with torch.no_grad():
        _ = model(**inputs)  # warmup
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        _ = model(**inputs)
        torch.cuda.synchronize()
        t_fwd = time.perf_counter() - t1

    print(f"[{label}]  seq_len={seq_len}  processor={t_proc*1000:.0f}ms  forward={t_fwd*1000:.0f}ms")
    return t_proc, t_fwd, seq_len


def run_with_attn(attn_impl):
    banner(f"Loading model with attn_implementation={attn_impl!r}")
    t0 = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    ).to(device).eval()
    print(f"loaded in {time.perf_counter()-t0:.1f}s")

    attns = dump_attn_impl(model)
    print(f"attention classes found:")
    for n, c in attns:
        print(f"  {c}    ({n})")

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    banner("Loading 1 sample from datas_v9")
    ds = load_from_disk(DATA_PATH)
    sample = ds[0]
    print(f"images per sample: {len(sample['images'])}")

    banner("Timing 3 forward passes")
    for i in range(3):
        time_forward(model, processor, sample, f"{attn_impl} run{i+1}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print("torch:", torch.__version__)
    print("cuda:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0), "cc:", torch.cuda.get_device_capability(0))
    try:
        import flash_attn
        print("flash_attn:", flash_attn.__version__)
    except ImportError:
        print("flash_attn: NOT INSTALLED")

    # Test H1: does flash_attention_2 actually engage on B200?
    run_with_attn("flash_attention_2")
    # Compare against sdpa as a fast baseline (should always work)
    run_with_attn("sdpa")
    # Eager is the worst case
    run_with_attn("eager")
