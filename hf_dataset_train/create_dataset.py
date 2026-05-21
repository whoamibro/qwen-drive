import os
import json
import re
from tqdm import tqdm
from datasets import Dataset, Features, Sequence, Image as DatasetsImage

def get_json_dataset(json_path, image_root_map):
    print(f"Loading dataset from {json_path}...")
    with open(json_path, 'r') as f:
        data_list = json.load(f)

    all_images = []
    all_prompts = []
    all_completions = []

    for entry in tqdm(data_list, desc="Processing JSON Dataset"):
        try:
            # 1. check the path of the image and processing
            image_paths = []
            for p in entry.get('image', []):
                if p.startswith("./data/nuscenes/"):
                    rel_path = p.replace("./data/nuscenes/", "", 1)
                    full_path = os.path.join(image_root_map, rel_path)
                else:
                    full_path = p
                image_paths.append(full_path)

            if not all(os.path.exists(p) for p in image_paths):
                print(f"Missing images for entry {entry.get('id', 'unknown')}")
                continue  # pass if there's no file exist.

            # 2. Parsing the conversation data
            conversations = entry.get('conversations', [])
            system_msg = next((msg['value'] for msg in conversations if msg['from'] == 'system'), "")
            user_msg = next((msg['value'] for msg in conversations if msg['from'] == 'human'), None)
            assistant_msg = next((msg['value'] for msg in conversations if msg['from'] == 'gpt'), None)

            if not user_msg or not assistant_msg:
                continue

            # 3. Create the Qwen/LLaVA styled content structure
            content = []
            parts = re.split(r"<image>", user_msg)
            for i, part in enumerate(parts):
                part = part.strip()
                if part:
                    content.append({"type": "text", "text": part})
                if i < len(image_paths):
                    content.append({"type": "image", "text": None})

            prompt = [
                    {"role": "system", "content": [{"type": "text", "text": system_msg}]},
                    {"role": "user", "content": content}
                ]
            completion = [
                    {"role": "assistant", "content": [{"type": "text", "text": assistant_msg}]}
                ]

            all_images.append(image_paths)
            all_prompts.append(prompt)
            all_completions.append(completion)

        except Exception as e:
            print(f"Error processing entry: {e}")
            continue

    # Create Dataset
    ds = Dataset.from_dict({
        "images": all_images,
        "prompt": all_prompts,
        "completion": all_completions,
    })

    # Type Casting the Image column to the Real Image type (including the image binaries)
    ds = ds.cast_column("images", Sequence(DatasetsImage()))

    return ds

if __name__ == "__main__":
    # Path Setting
    JSON_PATH = "./sft_dataset/sft_train_qwen3vl.json"
    IMAGE_ROOT_MAP = "./data/nuscenes"
    SAVE_PATH = "./datas_v9"
    
    # Create the Dataset
    dataset = get_json_dataset(JSON_PATH, IMAGE_ROOT_MAP)

    # Create the Directory if not exist
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # Save into the Disk
    print(f"Saving dataset to {SAVE_PATH}...")
    dataset.save_to_disk(SAVE_PATH, num_proc=32)
    print("DONE!")

