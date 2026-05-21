from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="Qwen/Qwen3-VL-8B-Instruct",  # Hugging Face repo ID
    local_dir="./ckpts/qwen3_vl_8b_instruct",
    local_dir_use_symlinks=False
)

print("Saving Finished.")
