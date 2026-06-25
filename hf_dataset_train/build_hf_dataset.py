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
            # 1. 이미지 경로 확인 및 처리
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
                continue  # 파일이 없으면 건너뜁니다.

            # 2. 대화 데이터 파싱
            conversations = entry.get('conversations', [])
            system_msg = next((msg['value'] for msg in conversations if msg['from'] == 'system'), "")
            user_msg = next((msg['value'] for msg in conversations if msg['from'] == 'human'), None)
            assistant_msg = next((msg['value'] for msg in conversations if msg['from'] == 'gpt'), None)

            if not user_msg or not assistant_msg:
                continue

            # 3. Qwen/LLaVA 스타일의 content 구조 생성
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

    # 데이터셋 생성
    ds = Dataset.from_dict({
        "images": all_images,
        "prompt": all_prompts,
        "completion": all_completions,
    })

    # 이미지 컬럼을 실제 Image 타입으로 캐스팅 (이미지 바이너리가 데이터셋에 포함됨)
    ds = ds.cast_column("images", Sequence(DatasetsImage()))

    return ds


if __name__ == "__main__":
    # 설정 경로 (run from project root)
    JSON_PATH = "./sft_dataset/sft_train_qwen3vl.json"
    IMAGE_ROOT_MAP = "./data/nuscenes"
    SAVE_PATH = "./datas_v9"  # 저장 경로

    # 데이터셋 생성
    dataset = get_json_dataset(JSON_PATH, IMAGE_ROOT_MAP)

    # 디렉토리가 없으면 생성
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # 디스크에 저장
    print(f"Saving dataset to {SAVE_PATH}...")
    dataset.save_to_disk(SAVE_PATH, num_proc=32)
    print("Done!")
