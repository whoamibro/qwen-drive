"""Profile a single forward pass — find where 142s goes."""
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
dev = torch.device("cuda:0")

print("Loading model with sdpa (matches train_vlm.sh path)...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, attn_implementation="sdpa",
).to(dev).eval()
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

ds = load_from_disk(DATA_PATH)
sample = ds[0]
system_text = sample["prompt"][0]["content"][0]["text"]
user_text = " ".join(c["text"] for c in sample["prompt"][1]["content"] if c.get("type") == "text")
msgs = [
    {"role": "system", "content": [{"type": "text", "text": system_text}]},
    {"role": "user", "content": [*[{"type": "image", "image": im} for im in sample["images"]],
                                   {"type": "text", "text": user_text}]},
]
inputs = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                        return_dict=True, return_tensors="pt")
inputs = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in inputs.items()}
print("input_ids shape:", inputs["input_ids"].shape)
print("pixel_values shape:", inputs.get("pixel_values", inputs.get("pixel_values_videos"))
      .shape if "pixel_values" in inputs or "pixel_values_videos" in inputs else "N/A")
print("Vision-input keys:", [k for k in inputs.keys() if "pixel" in k or "grid" in k])

print("\n=== Inspect a few model parameters' devices/dtypes ===")
for name, p in list(model.named_parameters())[:5]:
    print(f"  {name}: device={p.device} dtype={p.dtype}")
print(f"... total params: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")
print(f"params on cpu: {sum(p.numel() for p in model.parameters() if p.device.type=='cpu')}")
print(f"params on gpu: {sum(p.numel() for p in model.parameters() if p.device.type=='cuda')}")

# Warm up
print("\n=== Warm up forward (timed) ===")
with torch.no_grad():
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = model(**inputs)
    torch.cuda.synchronize()
    print(f"first forward (warmup): {time.perf_counter()-t0:.2f}s")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = model(**inputs)
    torch.cuda.synchronize()
    print(f"second forward:         {time.perf_counter()-t0:.2f}s")

# Now profile a forward
print("\n=== torch.profiler over 1 forward ===")
from torch.profiler import profile, ProfilerActivity, record_function

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
    with record_function("FULL_FORWARD"):
        with torch.no_grad():
            _ = model(**inputs)
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
print("\n--- by CPU self time ---")
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))
