"""
Split-the-forward diagnostic.
Times vision encoder and LLM body separately so we know which is the 1400x slowdown.
"""
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

import time, torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

MODEL_PATH = "./ckpts/qwen3_vl_8b_instruct"
dev = torch.device("cuda:0")

print("Loading model (sdpa, bf16)...")
t0 = time.perf_counter()
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, attn_implementation="sdpa",
).to(dev).eval()
print(f"loaded {sum(p.numel() for p in model.parameters())/1e9:.2f}B params in {time.perf_counter()-t0:.1f}s")

print("\nModel attribute layout:")
print(f"  has model.visual?  {hasattr(model, 'visual')}")
print(f"  has model.language_model?  {hasattr(model, 'language_model')}")
print(f"  model type: {type(model).__name__}")
print(f"  top-level submodules: {[n for n, _ in model.named_children()][:10]}")

# ---------- 1) TEXT-ONLY forward (no images) ----------
print("\n=== TEXT-ONLY forward (just input_ids, no images) ===")
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
text = "Describe the scene from the front camera of an autonomous vehicle. " * 50  # ~500 tokens
ids = tok(text, return_tensors="pt").input_ids.to(dev)
print(f"input_ids shape: {ids.shape}")

with torch.no_grad():
    # Warmup
    _ = model(input_ids=ids)
    torch.cuda.synchronize()
    times = []
    for i in range(3):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(input_ids=ids)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"  run {i+1}: {dt*1000:.1f} ms  ({ids.shape[1]/dt:.0f} tok/s)")

# ---------- 2) Bigger text-only forward (long sequence, matches our training seq_len) ----------
print("\n=== TEXT-ONLY long sequence (~9000 tokens, no images) ===")
long_ids = torch.randint(0, 100000, (1, 9000), device=dev)
with torch.no_grad():
    _ = model(input_ids=long_ids)
    torch.cuda.synchronize()
    for i in range(3):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(input_ids=long_ids)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"  run {i+1}: {dt*1000:.1f} ms")

# ---------- 3) Vision encoder only ----------
print("\n=== VISION ENCODER ONLY (no LLM) ===")
visual = model.visual if hasattr(model, "visual") else model.model.visual
print(f"  visual class: {type(visual).__name__}")

# Make a fake pixel input matching nuScenes 1600x900 -> smart_resize to multiple of 32
# Visual config: patch_size=16, temporal_patch_size=2, spatial_merge_size=2
# A typical image flattened: (num_patches, channels*temporal_patch*patch*patch) = (N, 3*2*16*16) = (N, 1536)
H, W = 896, 1568  # smart_resize for 900x1600 rounded to /32
n_h, n_w = H // 16, W // 16  # patch grid
n_patches_per_image = n_h * n_w
# Each "frame" in qwen3-vl ViT is processed as temporal_patch=2, so 1 image gives n_patches//2 tokens
flat_dim = 3 * 2 * 16 * 16  # 1536
num_images = 6
pixel_values = torch.randn(num_images * n_patches_per_image, flat_dim, device=dev, dtype=torch.bfloat16)
image_grid_thw = torch.tensor([[1, n_h, n_w]] * num_images, device=dev, dtype=torch.long)
print(f"  pixel_values: {pixel_values.shape}  total ViT tokens: {pixel_values.shape[0]}  per-image grid: {n_h}x{n_w}")

with torch.no_grad():
    try:
        # warmup
        _ = visual(pixel_values, grid_thw=image_grid_thw)
        torch.cuda.synchronize()
        for i in range(3):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = visual(pixel_values, grid_thw=image_grid_thw)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            print(f"  run {i+1}: {dt*1000:.1f} ms")
    except TypeError as e:
        print(f"  visual signature mismatch ({e}); trying alternative call...")
        try:
            _ = visual(pixel_values)
            print("  (worked with positional only)")
        except Exception as e2:
            print(f"  alt call failed: {e2}")

print("\n=== Summary ===")
print("Compare:")
print("  Raw matmul 9135x3584@3584x18944 bf16 = ~0.76 ms")
print("  An 8B-model forward on 9000 tokens SHOULD be ~100-300 ms on B200.")
print("  Anything above ~1 sec means model code is dominating, not compute.")
