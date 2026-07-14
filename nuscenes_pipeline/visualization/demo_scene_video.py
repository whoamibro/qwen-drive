"""
Demo Scene Video — assemble per-category MP4s from `demo_tester.py` outputs.

Input: the `predictions.json` written by
`nuscenes_pipeline.modules.demo_tester`. That JSON carries, per frame:
    - 6 camera image paths
    - N per-question results (question text, pred_answer, pred_grounding, ...)

Output granularity (per user spec): **one MP4 per category** (10 total for the
canonical 40-question bank). Each MP4 cycles through the category's questions
sequentially, playing each question across every frame of the scene:

    Q1 across frames 1..N  ->  Q2 across frames 1..N  ->  Q3 ...  ->  Q4 ...

Layout per rendered frame:
    +--------+--------+--------+   Top row : Front-left  | Front   | Front-right
    | tile0  | tile1  | tile2  |   Row 1   : Back-left * | Back  * | Back-right *
    +--------+--------+--------+   (* = horizontally flipped, same as training)
    | tile3  | tile4  | tile5  |
    +--------+--------+--------+
    +------------------------------------------------------+
    | header: category | question_id | frame_pos/N (t=...) |
    | question text                                        |
    | PRED answer                                          |
    | pred reasoning (first ~2 lines)                      |
    +------------------------------------------------------+

Bboxes: PRED grounding is drawn as pink outlined boxes on the tile matching
`image_idx` (1..6). Coordinates in predictions are in Qwen-VL's [0, 1000]
normalized space (after any rear-view flip already applied at inference time),
so they map straight onto the panoramic tile positions — no unflip needed.

Usage:
    python -m nuscenes_pipeline.visualization.demo_scene_video \\
        --predictions demo_test_results/ff6af17f52c34e9c/predictions.json \\
        --output_dir  demo_test_results/ff6af17f52c34e9c/videos \\
        --framerate 2

Category → 3-letter code mapping matches
`nuscenes_pipeline.postprocessing.split_sft_by_category.CATEGORY_TO_ABBR`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import textwrap
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GRID_COLS = 3
GRID_ROWS = 2
CAMERA_ORDER = [
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",  "CAM_BACK",  "CAM_BACK_RIGHT",
]
REAR_TILE_INDICES = {3, 4, 5}  # image indices in CAMERA_ORDER that get H-flipped

# Category name -> 3-letter abbrev (source of truth in split_sft_by_category.py).
CATEGORY_TO_ABBR = {
    "Observation":                          "OBS",
    "Identification":                       "IDN",
    "Attributes_and_States":                "AAS",
    "Attributes and States":                "AAS",
    "Spatial_Relationships_and_Occlusion":  "SRO",
    "Spatial Relationships and Occlusion":  "SRO",
    "Traffic_Signs_and_Signals":            "TSS",
    "Traffic Signs and Signals":            "TSS",
    "Road_Markings_and_Lane_Configuration": "RML",
    "Road Markings and Lane Configuration": "RML",
    "Dynamic_Agents_and_Risk_Assessment":   "DRA",
    "Dynamic Agents and Risk Assessment":   "DRA",
    "Right_of_Way_and_Planning":            "RWP",
    "Right of Way and Planning":            "RWP",
    "Environmental_and_Sensor_Conditions":  "ESC",
    "Environmental and Sensor Conditions":  "ESC",
    "Causal_and_Hypothetical_Reasoning":    "CHR",
    "Causal and Hypothetical Reasoning":    "CHR",
    "User":                                 "USR",  # Mode B (single custom question)
}


# ---------------------------------------------------------------------------
# Font resolution
# ---------------------------------------------------------------------------
def _resolve_font(size: int) -> ImageFont.FreeTypeFont:
    """Try common truetype fonts; fall back to PIL's bitmap default."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Panoramic build (self-contained — mirrors pan_generator, no NuScenesDataLoader dep)
# ---------------------------------------------------------------------------
def build_panoramic_from_paths(image_paths: List[str],
                                resize_factor: int = 2) -> Image.Image:
    """Load 6 camera images in CAMERA_ORDER, downscale, flip rears, tile 2x3."""
    tiles: List[Image.Image] = []
    for view_idx, img_path in enumerate(image_paths):
        img = Image.open(img_path).convert("RGB")
        if resize_factor > 1:
            w, h = img.size
            img = img.resize((w // resize_factor, h // resize_factor), Image.LANCZOS)
        if view_idx in REAR_TILE_INDICES:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        tiles.append(img)

    tile_w, tile_h = tiles[0].size
    canvas = Image.new("RGB", (tile_w * GRID_COLS, tile_h * GRID_ROWS))
    for idx, tile in enumerate(tiles):
        if tile.size != (tile_w, tile_h):
            tile = tile.resize((tile_w, tile_h), Image.LANCZOS)
        row, col = divmod(idx, GRID_COLS)
        canvas.paste(tile, (col * tile_w, row * tile_h))
    return canvas, tile_w, tile_h


# ---------------------------------------------------------------------------
# Bbox overlay
# ---------------------------------------------------------------------------
_QWEN_BOX_RE = re.compile(r"\((\d+),(\d+)\),\((\d+),(\d+)\)")


def _extract_bbox_from_ref(ref) -> Optional[Tuple[int, int, int, int]]:
    """Predictions from `_parse_grounding_from_token_ids` carry either:
        {"image_idx": N, "label": "...", "box": [x1, y1, x2, y2]}
    (the token-id parser's normalized shape) or a raw ref string. Handle both.
    """
    if isinstance(ref, dict):
        b = ref.get("box")
        if b and len(b) == 4:
            return int(b[0]), int(b[1]), int(b[2]), int(b[3])
    if isinstance(ref, str):
        m = _QWEN_BOX_RE.search(ref)
        if m:
            return tuple(int(x) for x in m.groups())
    return None


def draw_pred_boxes(canvas: Image.Image, pred_grounding: List[dict],
                    tile_w: int, tile_h: int) -> None:
    """Draw pink outlined bboxes on the tile matching each entry's image_idx.
    Coordinates are Qwen [0, 1000]; scale to tile size."""
    draw = ImageDraw.Draw(canvas)
    label_font = _resolve_font(max(12, tile_h // 30))

    for entry in pred_grounding:
        img_idx = entry.get("image_idx")
        if not isinstance(img_idx, int) or not (1 <= img_idx <= 6):
            continue
        box = entry.get("box")
        if box is None:
            box = _extract_bbox_from_ref(entry.get("ref"))
        if not box:
            continue
        x1n, y1n, x2n, y2n = box
        # Convert normalized [0, 1000] to tile pixel.
        # In the token-id parser output, box is ALREADY in [0, 1000] space.
        x_off = ((img_idx - 1) % GRID_COLS) * tile_w
        y_off = ((img_idx - 1) // GRID_COLS) * tile_h
        x1 = x_off + int(x1n * tile_w / 1000)
        y1 = y_off + int(y1n * tile_h / 1000)
        x2 = x_off + int(x2n * tile_w / 1000)
        y2 = y_off + int(y2n * tile_h / 1000)
        draw.rectangle([x1, y1, x2, y2], outline=(255, 105, 180), width=3)
        label = str(entry.get("label", ""))
        if label:
            draw.text((x1 + 3, max(y1 - int(label_font.size * 1.1), y_off + 2)),
                      label, fill=(255, 255, 255), font=label_font,
                      stroke_width=1, stroke_fill=(0, 0, 0))


# ---------------------------------------------------------------------------
# Frame composition: pan + text pane
# ---------------------------------------------------------------------------
def _parse_reasoning(pred_text: str) -> str:
    """Best-effort pull of the `reasoning` string out of a unified-JSON pred."""
    try:
        obj = json.loads(pred_text)
        return str(obj.get("reasoning", "")).strip()
    except (json.JSONDecodeError, TypeError):
        return ""


def compose_frame(pan: Image.Image, tile_w: int, tile_h: int,
                  header_text: str, question_text: str,
                  answer_text: str, reasoning_snippet: str,
                  text_pane_height: int) -> Image.Image:
    """Stack the panoramic grid on top of a text pane. Even dims for H.264."""
    pan_w, pan_h = pan.size
    total_h = pan_h + text_pane_height
    if total_h % 2:
        total_h += 1
    if pan_w % 2:
        pan_w += 1
        # left-align; the extra pixel column is black.
        pad = Image.new("RGB", (pan_w, pan_h), (0, 0, 0))
        pad.paste(pan, (0, 0))
        pan = pad
    canvas = Image.new("RGB", (pan_w, total_h), (18, 18, 22))
    canvas.paste(pan, (0, 0))

    draw = ImageDraw.Draw(canvas)
    header_font = _resolve_font(max(18, tile_h // 22))
    body_font   = _resolve_font(max(16, tile_h // 26))
    reason_font = _resolve_font(max(14, tile_h // 30))

    x_pad = 16
    y = pan_h + 10
    # Header: category / question_id / frame progress
    draw.text((x_pad, y), header_text, fill=(120, 200, 255), font=header_font)
    y += header_font.size + 10

    # Question (wrapped)
    wrap_w = max(30, pan_w // (body_font.size // 2 or 8))
    for line in textwrap.wrap(question_text, width=wrap_w)[:2]:
        draw.text((x_pad, y), line, fill=(240, 240, 240), font=body_font)
        y += body_font.size + 4

    # Predicted answer
    y += 4
    draw.text((x_pad, y), f"PRED: {answer_text}", fill=(255, 190, 100), font=body_font,
              stroke_width=1, stroke_fill=(0, 0, 0))
    y += body_font.size + 6

    # Reasoning snippet
    if reasoning_snippet:
        for line in textwrap.wrap(reasoning_snippet, width=wrap_w)[:3]:
            draw.text((x_pad, y), line, fill=(180, 180, 200), font=reason_font)
            y += reason_font.size + 2

    return canvas


# ---------------------------------------------------------------------------
# Per-category rendering
# ---------------------------------------------------------------------------
def render_category(cat_label: str, cat_abbr: str,
                    cat_questions: List[dict],
                    frames: List[dict],
                    output_mp4: str, temp_dir: str,
                    resize_factor: int, framerate: int) -> None:
    """Render one MP4 for the category. Sequence:
        for question in cat_questions:
            for frame in frames: render one PNG
    Then ffmpeg the PNGs into an MP4 at `output_mp4`.
    """
    os.makedirs(temp_dir, exist_ok=True)

    # Pre-index question results per frame for O(1) lookup: {(frame_pos, qid): entry}
    result_lookup: Dict[Tuple[int, str], dict] = {}
    for f in frames:
        for r in f["results"]:
            result_lookup[(f["frame_pos"], r["question_id"])] = r

    frame_paths: List[str] = []
    counter = 0
    for q_meta in cat_questions:
        for f in frames:
            key = (f["frame_pos"], q_meta["question_id"])
            r = result_lookup.get(key)
            if r is None:
                # Shouldn't happen if predictions.json is complete for this cat.
                continue

            pan, tile_w, tile_h = build_panoramic_from_paths(
                f["image_paths"], resize_factor=resize_factor
            )
            draw_pred_boxes(pan, r.get("pred_grounding") or [], tile_w, tile_h)

            n_frames = len(frames)
            ts_s = ""
            ts = f.get("timestamp")
            if isinstance(ts, (int, float)):
                # nuScenes timestamps are microseconds since epoch; render as t = X.Ys within scene.
                ts_s = f"  t={float(ts)/1e6:.2f}s"

            header = (f"[{cat_abbr}] {r['question_id']}   "
                      f"frame {f['frame_pos']+1}/{n_frames}"
                      f"{ts_s}")

            answer_text = (r.get("pred_answer") or "(unparsed)").strip()
            if q_meta.get("answer_type") == "mcq" and q_meta.get("options"):
                # Attach option text next to the letter for readability
                try:
                    letter_i = ord(answer_text.strip("() ").upper()[0]) - ord("A")
                    if 0 <= letter_i < len(q_meta["options"]):
                        answer_text = f"{answer_text}  = \"{q_meta['options'][letter_i]}\""
                except (IndexError, TypeError):
                    pass

            reasoning = _parse_reasoning(r.get("pred_text", ""))
            text_pane_h = max(140, tile_h // 3)
            composed = compose_frame(pan, tile_w, tile_h, header,
                                     q_meta["question"], answer_text,
                                     reasoning, text_pane_h)

            frame_path = os.path.join(temp_dir, f"frame_{counter:06d}.png")
            composed.save(frame_path)
            frame_paths.append(frame_path)
            counter += 1

    if not frame_paths:
        print(f"  [{cat_abbr}] no frames to render, skipping.")
        return

    # ffmpeg pass
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(framerate),
        "-i", os.path.join(temp_dir, "frame_%06d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        output_mp4,
    ], check=True)

    for fp in frame_paths:
        try:
            os.remove(fp)
        except OSError:
            pass
    print(f"  [{cat_abbr}] {output_mp4}  ({counter} frames)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        prog="python -m nuscenes_pipeline.visualization.demo_scene_video",
        description=(
            "Assemble per-category MP4s from a demo_tester predictions.json. "
            "One MP4 per category; each cycles through that category's "
            "questions x all frames of the scene."
        ),
    )
    p.add_argument("--predictions", required=True, type=str,
                   help="Path to demo_tester's predictions.json.")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output dir for the MP4s. Default: sibling 'videos/' next "
                        "to predictions.json.")
    p.add_argument("--framerate", type=int, default=2,
                   help="Output MP4 framerate (default 2 = nuScenes annotation rate).")
    p.add_argument("--resize_factor", type=int, default=2,
                   help="Per-tile downscale factor for the panoramic (default 2 -> "
                        "800x450 tiles). Larger = higher res, slower rendering.")
    p.add_argument("--only_category", type=str, default=None,
                   help="If set, render only this category abbrev (e.g. OBS). "
                        "Handy for iterating on layout / colors.")
    p.add_argument("--keep_temp", action="store_true",
                   help="Don't delete the per-frame PNG scratch dir when done.")
    args = p.parse_args()

    with open(args.predictions) as f:
        pred = json.load(f)

    frames = pred["frames"]
    scene_tok = pred["scene_token"]
    scene_short = scene_tok[:16]

    # Determine question metadata per (question_id) from the first frame's
    # results — every frame carries the same question set in the same order.
    if not frames or not frames[0].get("results"):
        print("ERROR: predictions.json has no frames or no per-frame results.")
        return 2
    first_results = frames[0]["results"]

    # Group questions by category (curriculum-friendly ordering already baked
    # into demo_tester's traversal).
    cats_ordered: "OrderedDict[str, List[dict]]" = OrderedDict()
    for r in first_results:
        cat = r.get("category", "Unknown")
        cats_ordered.setdefault(cat, []).append({
            "question_id":  r["question_id"],
            "question":     r["question"],
            "answer_type":  r.get("answer_type", ""),
            "options":      r.get("options"),
        })

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.predictions)), "videos"
    )
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = os.path.join(output_dir, "_temp_frames")

    n_cats_planned = 0
    for cat_label in cats_ordered:
        abbr = CATEGORY_TO_ABBR.get(cat_label, cat_label[:3].upper())
        if args.only_category and abbr != args.only_category.upper():
            continue
        n_cats_planned += 1
    print(f"[demo_scene_video] scene={scene_tok}  frames={len(frames)}  "
          f"cats_to_render={n_cats_planned}  -> {output_dir}/")

    n_done = 0
    for cat_label, cat_questions in cats_ordered.items():
        abbr = CATEGORY_TO_ABBR.get(cat_label, cat_label[:3].upper())
        if args.only_category and abbr != args.only_category.upper():
            continue
        output_mp4 = os.path.join(output_dir, f"scene_{scene_short}_{abbr}.mp4")
        render_category(cat_label, abbr, cat_questions, frames,
                        output_mp4, temp_dir, args.resize_factor, args.framerate)
        n_done += 1

    if not args.keep_temp and os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"[demo_scene_video] done. {n_done} MP4s written to {output_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
