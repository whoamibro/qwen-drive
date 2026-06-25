"""Benchmark raw matmul on B200 to see if compute is broken."""
import torch, time

print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0), "cc:", torch.cuda.get_device_capability(0))
dev = torch.device("cuda:0")

def bench(M, N, K, dtype, n_iter=20):
    a = torch.randn(M, K, device=dev, dtype=dtype)
    b = torch.randn(K, N, device=dev, dtype=dtype)
    # Warmup
    for _ in range(3):
        _ = a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        c = a @ b
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n_iter
    flops = 2 * M * N * K
    tflops = flops / dt / 1e12
    print(f"  {dtype} {M}x{K} @ {K}x{N}: {dt*1000:.2f}ms  {tflops:.1f} TFLOPS")
    return dt

print("\n--- bf16 matmul (B200 peak ~2.2 PFLOPS bf16) ---")
for M, N, K in [(4096, 4096, 4096), (8192, 8192, 8192), (4096, 14336, 4096)]:
    bench(M, N, K, torch.bfloat16)

print("\n--- fp16 ---")
bench(8192, 8192, 8192, torch.float16)

print("\n--- fp32 ---")
bench(4096, 4096, 4096, torch.float32)

print("\n--- realistic LLM matmul: hidden 3584 x intermediate 18944 (Qwen3-8B sizes) ---")
# Qwen3-VL-8B: hidden_size=3584, intermediate_size=18944, num_heads=28, head_dim=128
bench(9135, 18944, 3584, torch.bfloat16)  # MLP up_proj-like shape with seq_len=9135
bench(9135, 3584, 18944, torch.bfloat16)  # MLP down_proj-like shape

print("\n--- memory bandwidth test (copy 1 GB) ---")
x = torch.randn(256 * 1024 * 1024, device=dev, dtype=torch.bfloat16)  # 512 MB
y = torch.empty_like(x)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(10):
    y.copy_(x)
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / 10
print(f"  copy 512MB bf16: {dt*1000:.2f}ms  {(x.numel()*2)/dt/1e9:.1f} GB/s")
