from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="Qwen/Qwen3-VL-235B-A22B-Thinking",  # Hugging Face repo ID
    local_dir="./ckpts/qwen3_vl_235b_thinking",
    local_dir_use_symlinks=False
)

print("Saving Finished.")
