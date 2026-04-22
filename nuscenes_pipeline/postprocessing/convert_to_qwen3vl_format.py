"""
Convert the SFT dataset from the legacy Qwen-VL inline-bbox format to the
unified Qwen3-VL JSON format.

Input format (legacy, inline bbox in prose)
-------------------------------------------
    "conversations": [
      {"from": "system", "value": "You are an autonomous driving..."},
      {"from": "human",  "value": "<image>...<image>\n...TASK: ..."},
      {"from": "gpt",    "value": "No. - pedestrian (Image 1 (Front-left) bbox[982,426,1128,684]) is present, ... - Answer: No"}
    ]

Output format (unified JSON in gpt.value)
-----------------------------------------
    "conversations": [
      {"from": "human", "value": "<system prompt merged>\n...<image>...TASK: ..."},
      {"from": "gpt",   "value": "{\n  \"reasoning\": \"- ...\",\n  \"grounding\": [{\"image_idx\": 1, \"camera\": \"Front-left\", \"bbox_2d\": [982, 426, 1128, 684], \"label\": \"pedestrian\"}],\n  \"answer\": \"No\"\n}"}
    ]

The gpt.value is a JSON string that parses to:
    {
      "reasoning": "<bullet-point reasoning, bboxes removed>",
      "grounding": [{"image_idx": N, "camera": "...", "bbox_2d": [x1,y1,x2,y2], "label": "..."}],
      "answer": "<short answer>"
    }

Usage
-----
    # Dry-run first to inspect parsing stats + a sample transformation
    python -m nuscenes_pipeline.postprocessing.convert_to_qwen3vl_format \\
        --input sft_dataset/sft_train_no_objlist.json \\
        --output sft_dataset/sft_train_qwen3vl.json \\
        --dry_run

    # Actual run
    python -m nuscenes_pipeline.postprocessing.convert_to_qwen3vl_format \\
        --input sft_dataset/sft_train_no_objlist.json \\
        --output sft_dataset/sft_train_qwen3vl.json
"""

import os
import re
import json
import argparse


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Matches:  label (Image N (CamName) bbox[x1,y1,x2,y2])
# - label is a word possibly containing underscores (e.g., traffic_cone, construction_vehicle)
# - CamName is the human-readable camera name (Front, Front-left, Rear-right, etc.)
# Captures: label, image_num, camera_name, x1, y1, x2, y2
_BBOX_LABELED_RE = re.compile(
    r'(\b\w[\w_]*)\s*\(Image\s+(\d+)\s*\(([^)]+)\)\s*bbox\[(\d+),(\d+),(\d+),(\d+)\]\)'
)

# Matches a bare bbox parenthetical without a category prefix (less common):
#    (Image 3 (Front-right) bbox[...])
# Used for clean-up only — we've already extracted all bboxes via the labeled regex.
_BBOX_BARE_RE = re.compile(
    r' ?\(Image\s+\d+\s*\([^)]+\)\s*bbox\[\d+,\d+,\d+,\d+\]\)'
)

# Final answer bullets that duplicate the short answer (to be stripped from reasoning)
_ANSWER_BULLET_RE = re.compile(
    r'(?mi)^\s*-\s*(?:Answer|Conclusion)\s*:\s*.+?$'
)

# Heuristic boundary between the short answer clause and the bullets
# Matches the first occurrence of " - " that starts a bullet list
# Works on single-line strings like "No. - pedestrian ..." as well as multi-line
_FIRST_BULLET_RE = re.compile(r'\s*\n?\s*-\s+', re.MULTILINE)


# ---------------------------------------------------------------------------
# Bbox extraction + text cleanup
# ---------------------------------------------------------------------------

def extract_grounding(text: str) -> tuple[list[dict], str]:
    """Extract bbox-referenced objects from text into a grounding list and return
    the text with bbox parentheticals stripped.

    Returns: (grounding_list, cleaned_text)
    """
    grounding = []
    seen = set()

    # Collect all labeled bboxes first (preserves label)
    for m in _BBOX_LABELED_RE.finditer(text):
        label = m.group(1)
        img_idx = int(m.group(2))
        camera = m.group(3).strip()
        x1, y1, x2, y2 = int(m.group(4)), int(m.group(5)), int(m.group(6)), int(m.group(7))
        key = (img_idx, x1, y1, x2, y2, label)
        if key in seen:
            continue
        seen.add(key)
        grounding.append({
            "image_idx": img_idx,
            "camera": camera,
            "bbox_2d": [x1, y1, x2, y2],
            "label": label,
        })

    # Replace "label (Image N (CamName) bbox[...])" with just "label"
    cleaned = _BBOX_LABELED_RE.sub(r'\1', text)
    # Any remaining bare bbox parentheticals → remove entirely
    cleaned = _BBOX_BARE_RE.sub('', cleaned)
    return grounding, cleaned


# ---------------------------------------------------------------------------
# Answer + reasoning separation
# ---------------------------------------------------------------------------

def extract_answer_and_reasoning(text: str) -> tuple[str, str, str]:
    """Parse gpt text into (answer, reasoning, heuristic_used).

    Primary heuristic: the first clause before the first '- ' bullet marker is
    the short answer; everything after (the bullets) is the reasoning.

    Fallback: if no bullet delimiter is found, treat the first sentence as the
    answer and the rest as reasoning.
    """
    text = text.strip()
    if not text:
        return "", "", "empty"

    # Heuristic 1: split on first '- ' bullet
    m = _FIRST_BULLET_RE.search(text)
    if m:
        answer = text[:m.start()].strip().rstrip('.').strip()
        reasoning = text[m.start():].strip()
        # Drop trailing "- Answer: X" / "- Conclusion: X" bullets (duplicate info)
        reasoning = _ANSWER_BULLET_RE.sub('', reasoning).strip()
        # Collapse double-blank runs introduced by the strip above
        reasoning = re.sub(r'\n{3,}', '\n\n', reasoning)
        if answer and len(answer) <= 400:
            return answer, reasoning, "first_clause_before_bullets"

    # Heuristic 2: first sentence as the answer
    parts = text.split('. ', 1)
    if len(parts) == 2 and len(parts[0]) <= 200:
        answer = parts[0].strip().rstrip('.').strip()
        reasoning = parts[1].strip()
        return answer, reasoning, "first_sentence"

    # Fallback: whole text is the answer (no reasoning extractable)
    return text, "", "whole_text_fallback"


# ---------------------------------------------------------------------------
# Per-sample conversion
# ---------------------------------------------------------------------------

def convert_gpt_value(value: str) -> tuple[str, dict]:
    """Convert one gpt turn's value string.

    Returns: (new_value_json_string, per_sample_stats)
    """
    stats = {"grounding_count": 0, "heuristic_used": None}

    grounding, cleaned = extract_grounding(value)
    stats["grounding_count"] = len(grounding)

    answer, reasoning, heuristic = extract_answer_and_reasoning(cleaned)
    stats["heuristic_used"] = heuristic

    unified = {
        "reasoning": reasoning,
        "grounding": grounding,
        "answer": answer,
    }
    new_value = json.dumps(unified, indent=2, ensure_ascii=False)
    return new_value, stats


def convert_sample(sample: dict) -> tuple[dict, dict]:
    """Convert a single SFT sample to Qwen3-VL unified-JSON format.

    Returns (new_sample, stats).
    """
    convs = sample.get("conversations", [])
    new_convs = []
    system_content = None
    per_sample_stats = {
        "system_merged": False,
        "gpt_turns": 0,
        "grounding_count": 0,
        "heuristic_used": None,
    }

    for turn in convs:
        role = turn.get("from")
        value = turn.get("value", "")

        if role == "system":
            # Stash and defer to the next human turn
            system_content = value
            per_sample_stats["system_merged"] = True
            continue

        if role == "human":
            if system_content:
                value = system_content + "\n\n" + value
                system_content = None
            new_convs.append({"from": "human", "value": value})
            continue

        if role == "gpt":
            new_value, turn_stats = convert_gpt_value(value)
            new_convs.append({"from": "gpt", "value": new_value})
            per_sample_stats["gpt_turns"] += 1
            per_sample_stats["grounding_count"] += turn_stats["grounding_count"]
            per_sample_stats["heuristic_used"] = turn_stats["heuristic_used"]
            continue

        # Unknown role — pass through unchanged
        new_convs.append(turn)

    if system_content:
        # System turn without a following human turn — edge case; prepend as a human preamble
        new_convs.insert(0, {"from": "human", "value": system_content})

    new_sample = dict(sample)
    new_sample["conversations"] = new_convs
    return new_sample, per_sample_stats


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert SFT dataset to Qwen3-VL unified JSON format (reasoning + grounding + answer)"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSON (e.g., sft_dataset/sft_train_no_objlist.json)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON (e.g., sft_dataset/sft_train_qwen3vl.json)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Report stats without writing output file")
    parser.add_argument("--force", action="store_true",
                        help="Skip idempotency check and overwrite existing output")
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    marker_path = os.path.join(out_dir, ".format_qwen3vl")
    if os.path.exists(marker_path) and os.path.exists(args.output) and not args.force and not args.dry_run:
        print(f"ERROR: Output directory already has a Qwen3-VL-format marker at {marker_path}")
        print(f"  A previous run already produced {args.output}. Pass --force to overwrite.")
        return

    print(f"Loading {args.input}...")
    with open(args.input, "r") as f:
        data = json.load(f)
    print(f"  {len(data):,} samples")

    stats = {
        "total": 0,
        "system_merged": 0,
        "with_grounding": 0,
        "without_grounding": 0,
        "total_bboxes": 0,
        "heuristic": {"first_clause_before_bullets": 0, "first_sentence": 0, "whole_text_fallback": 0, "empty": 0},
    }

    new_data = []
    for sample in data:
        new_sample, s = convert_sample(sample)
        new_data.append(new_sample)
        stats["total"] += 1
        if s["system_merged"]:
            stats["system_merged"] += 1
        if s["grounding_count"] > 0:
            stats["with_grounding"] += 1
        else:
            stats["without_grounding"] += 1
        stats["total_bboxes"] += s["grounding_count"]
        h = s["heuristic_used"] or "empty"
        stats["heuristic"][h] = stats["heuristic"].get(h, 0) + 1

    print("\n=== CONVERSION STATS ===")
    print(f"  Total samples:        {stats['total']:,}")
    print(f"  System turns merged:  {stats['system_merged']:,}")
    print(f"  With grounding:       {stats['with_grounding']:,} ({stats['with_grounding']*100/stats['total']:.1f}%)")
    print(f"  Without grounding:    {stats['without_grounding']:,} ({stats['without_grounding']*100/stats['total']:.1f}%)")
    print(f"  Total bboxes:         {stats['total_bboxes']:,}")
    print("  Answer-extraction heuristic:")
    for heuristic, count in stats["heuristic"].items():
        print(f"    {heuristic:35s}  {count:,}")

    if args.dry_run:
        print("\n=== SAMPLE OUTPUT (first sample, gpt turn) ===")
        for turn in new_data[0]["conversations"]:
            if turn.get("from") == "gpt":
                print(turn["value"][:2000])
                break
        print("\n[DRY RUN] No file written.")
        return

    print(f"\nWriting {args.output}...")
    with open(args.output, "w") as f:
        json.dump(new_data, f, ensure_ascii=False)
    with open(marker_path, "w") as f:
        f.write(f"Qwen3-VL unified-JSON format written by convert_to_qwen3vl_format.py\n")
    print(f"  Saved: {args.output}")
    print(f"  Marker: {marker_path}")


if __name__ == "__main__":
    main()
