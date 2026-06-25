"""
Eval Sample Visualizer — qualitative GT-vs-PRED comparison for v2 eval.

Mirrors the UI/UX of qa_visualizer.py (dark theme, 6-view image grid, right
side text pane) but specialised for the curriculum-v2 evaluation outputs:

  - **Data source**: `eval_predictions.json` written by
    `nuscenes_pipeline/modules/sft_model_tester.py:run_per_category_eval_v2`,
    plus the per-category subset JSONs at `--val_subset_dir` for image paths
    and human/system text.

  - **Sample selection**: matches the on-disk sanity-print output — the first
    N grounded samples per category (default N=2 → 20 samples total). Same
    deterministic ordering as `--sanity_print_n` in the eval CLI, so what
    you see here is exactly what showed up in the training-time logs.

  - **Overlay**: every image carries BOTH a GT box (green) AND a PRED box
    (pink) when applicable, grouped by `image_idx`. Coords are 0–1000
    normalized (Qwen3-VL native space); scaled to the displayed image's
    actual width at render time.

  - **Per-image matching**: the right pane lists every GT box, the predicted
    box it was matched to under `(image_idx, label)` keyed greedy-IoU
    matching, and the IoU score / view-correctness flag — same matching
    logic that produces `view_acc` and `grounding_acc@<iou>` in the report.

  - **Spurious predictions** (pred boxes not matched to any GT) are listed
    separately and drawn with a dashed pink outline.

Usage:
    # Single stage's predictions
    python -m nuscenes_pipeline.visualization.eval_sample_visualizer \\
        --eval_predictions output/curriculum_v2_0609/stage_00_OBS/eval_predictions.json \\
        --val_subset_dir   sft_dataset/eval_subset_200 \\
        --port 6061

    # All stages discovered under a curriculum output_root (stage selector
    # in the header lets you switch between them)
    python -m nuscenes_pipeline.visualization.eval_sample_visualizer \\
        --curriculum_root output/curriculum_v2_0609 \\
        --val_subset_dir  sft_dataset/eval_subset_200 \\
        --port 6061
"""

from __future__ import annotations

import argparse
import base64
import glob
import io
import json
import os
import re
from typing import Dict, List, Optional, Tuple

from flask import Flask, jsonify, render_template_string, request
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Constants — mirrored from qa_visualizer / sft_model_tester
# ---------------------------------------------------------------------------
CAMERA_ORDER = [
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
]
CAM_LABELS_DISPLAY = [
    "1: Front-left", "2: Front", "3: Front-right",
    "4: Rear-left (flip)", "5: Rear (flip)", "6: Rear-right (flip)",
]
REAR_INDICES = {3, 4, 5}
QWEN_BOX_NORM = 1000  # Coord space of <|box_start|>(x1,y1),(x2,y2)<|box_end|>

# Colors (match qa_visualizer palette).
GT_COLOR     = (46, 204, 113)   # green
PRED_COLOR   = (233, 30, 140)   # magenta / pink
PRED_SPUR    = (233, 145, 200)  # paler magenta for spurious preds (dashed)
GT_COLOR_HEX     = "#2ecc71"
PRED_COLOR_HEX   = "#e91e8c"
PRED_SPUR_HEX    = "#e991c8"


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def _load_font(size: int = 14):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
        )
    except Exception:
        return ImageFont.load_default()


def _scale_box_to_image(box, img_w, img_h):
    """Map a 0..QWEN_BOX_NORM xyxy box onto pixel coords of (img_w, img_h)."""
    x1, y1, x2, y2 = box
    sx = img_w / QWEN_BOX_NORM
    sy = img_h / QWEN_BOX_NORM
    return (int(round(x1 * sx)), int(round(y1 * sy)),
            int(round(x2 * sx)), int(round(y2 * sy)))


def _draw_box(draw, box_px, label, color, dashed=False, font=None):
    x1, y1, x2, y2 = box_px
    if dashed:
        # Quick-and-dirty dashed rect: draw 4 dashed lines.
        dash_len = 6
        for x in range(x1, x2, dash_len * 2):
            draw.line([(x, y1), (min(x + dash_len, x2), y1)], fill=color, width=2)
            draw.line([(x, y2), (min(x + dash_len, x2), y2)], fill=color, width=2)
        for y in range(y1, y2, dash_len * 2):
            draw.line([(x1, y), (x1, min(y + dash_len, y2))], fill=color, width=2)
            draw.line([(x2, y), (x2, min(y + dash_len, y2))], fill=color, width=2)
    else:
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

    if label and font:
        # Background pill behind the label so it's readable on bright images.
        tb = draw.textbbox((x1, y1 - 18), label, font=font)
        draw.rectangle([tb[0] - 2, tb[1] - 1, tb[2] + 2, tb[3] + 1], fill=color)
        draw.text((x1, y1 - 18), label, fill=(255, 255, 255), font=font)


def _image_to_b64(img: Image.Image, fmt="JPEG", quality=85) -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode()}"


def _placeholder(text: str, w=640, h=360) -> Image.Image:
    img = Image.new("RGB", (w, h), (40, 40, 50))
    d = ImageDraw.Draw(img)
    f = _load_font(13)
    d.text((10, h // 2 - 8), text[:120], fill=(220, 100, 100), font=f)
    return img


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
_REASONING_RE = re.compile(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', re.DOTALL)
_ANSWER_RE    = re.compile(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', re.DOTALL)


def _unescape_json_string(s: str) -> str:
    """Decode common JSON string escapes back to readable text."""
    try:
        # json.loads on a quoted string handles all standard escapes.
        return json.loads(f'"{s}"')
    except (json.JSONDecodeError, ValueError):
        return (s.replace("\\n", "\n").replace("\\t", "\t")
                  .replace('\\"', '"').replace("\\\\", "\\"))


def _extract_reasoning_and_answer(text: str) -> Tuple[str, str]:
    """Pull `reasoning` and `answer` strings out of a Qwen3-VL unified-JSON
    envelope. Robust to:
      - Clean JSON envelopes (common case).
      - Truncated/garbled JSON where the closing brace got dropped.
      - Markdown-fenced JSON (```json ... ```) — strip the fences first.

    Returns (reasoning, answer); either may be "" on parse failure.
    """
    if not isinstance(text, str) or not text:
        return "", ""

    candidate = text.strip()
    # Strip ```json ... ``` or ``` ... ``` fences if present.
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
        candidate = re.sub(r"\s*```\s*$", "", candidate)

    # Path A: clean JSON parse.
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return str(obj.get("reasoning", "") or ""), str(obj.get("answer", "") or "")
    except (json.JSONDecodeError, ValueError):
        pass

    # Path B: regex fallback. Reasoning often dominates and contains embedded
    # quotes/braces that confuse a partial-JSON repair, so a tolerant regex
    # is simpler and recovers most truncated outputs.
    reasoning = ""
    answer = ""
    m = _REASONING_RE.search(candidate)
    if m:
        reasoning = _unescape_json_string(m.group(1))
    m = _ANSWER_RE.search(candidate)
    if m:
        answer = _unescape_json_string(m.group(1))
    return reasoning, answer


def _stage_label_from_path(path: str) -> str:
    """Pretty stage label from a curriculum stage dir, e.g. 'stage_04_TSS'."""
    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
    return parent if parent.startswith("stage_") else parent


def _discover_predictions(args) -> List[Tuple[str, str]]:
    """Return [(stage_label, eval_predictions_path), ...]. Sorted by stage idx."""
    out = []
    if args.eval_predictions:
        for p in args.eval_predictions:
            if not os.path.exists(p):
                raise FileNotFoundError(f"--eval_predictions not found: {p}")
            out.append((_stage_label_from_path(p), os.path.abspath(p)))
    if args.curriculum_root:
        root = os.path.abspath(args.curriculum_root)
        if not os.path.isdir(root):
            raise FileNotFoundError(f"--curriculum_root not found: {root}")
        for sub in sorted(os.listdir(root)):
            stage_dir = os.path.join(root, sub)
            ep = os.path.join(stage_dir, "eval_predictions.json")
            if os.path.isdir(stage_dir) and os.path.exists(ep):
                out.append((sub, ep))
    if not out:
        raise FileNotFoundError(
            "No eval_predictions.json found. Pass --eval_predictions <path> "
            "or --curriculum_root <dir>."
        )
    return out


def _load_val_subset_index(val_subset_dir: str) -> Dict[str, List[dict]]:
    """Map category code -> list of val-subset samples (image paths + convs)."""
    out = {}
    if not os.path.isdir(val_subset_dir):
        raise FileNotFoundError(f"--val_subset_dir not found: {val_subset_dir}")
    pat = re.compile(r"sft_val_qwen3vl_([A-Z]+)(?:_subset)?\.json$")
    for path in sorted(glob.glob(os.path.join(val_subset_dir, "sft_val_qwen3vl_*.json"))):
        m = pat.search(os.path.basename(path))
        if not m:
            continue
        with open(path) as f:
            out[m.group(1)] = json.load(f)
    return out


def _select_sanity_samples(
    predictions: dict,
    n_per_cat: int,
) -> Dict[str, List[dict]]:
    """For each category, return the first n_per_cat samples that have GT
    grounding (matches the sanity-print selection rule in sft_model_tester)."""
    out = {}
    for cat in sorted(predictions.keys()):
        samples = predictions[cat]
        picked = []
        for s in samples:
            if s.get("error"):
                continue
            gt = s.get("gt_grounding") or []
            if not gt:
                continue
            picked.append(s)
            if len(picked) >= n_per_cat:
                break
        out[cat] = picked
    return out


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)

# Lazy-loaded singletons (set in main()).
_PRED_PATHS: List[Tuple[str, str]] = []        # [(stage_label, predictions_path)]
_PRED_CACHE: Dict[str, dict] = {}               # stage_label -> predictions dict
_SANITY_CACHE: Dict[Tuple[str, int], Dict[str, List[dict]]] = {}
_VAL_SUBSET: Dict[str, List[dict]] = {}         # cat -> list of val samples
_VAL_SUBSET_DIR: str = ""
_N_PER_CAT: int = 2


def _get_predictions(stage_label: str) -> dict:
    if stage_label not in _PRED_CACHE:
        path = dict(_PRED_PATHS)[stage_label]
        with open(path) as f:
            _PRED_CACHE[stage_label] = json.load(f)
    return _PRED_CACHE[stage_label]


def _get_sanity_samples(stage_label: str) -> Dict[str, List[dict]]:
    key = (stage_label, _N_PER_CAT)
    if key not in _SANITY_CACHE:
        _SANITY_CACHE[key] = _select_sanity_samples(_get_predictions(stage_label), _N_PER_CAT)
    return _SANITY_CACHE[key]


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/info")
def api_info():
    """Return: stages, per-stage category breakdown + sample counts."""
    out_stages = []
    for label, _ in _PRED_PATHS:
        cats = _get_sanity_samples(label)
        per_cat = {cat: len(items) for cat, items in cats.items() if items}
        out_stages.append({"stage": label, "per_cat": per_cat, "total": sum(per_cat.values())})
    return jsonify({
        "stages": out_stages,
        "n_per_cat": _N_PER_CAT,
        "categories": sorted({c for s in out_stages for c in s["per_cat"]}),
    })


def _render_sample(stage_label: str, cat: str, slot: int) -> dict:
    cats = _get_sanity_samples(stage_label)
    pool = cats.get(cat, [])
    if not pool or slot < 0 or slot >= len(pool):
        return {"error": f"no sample at stage={stage_label} cat={cat} slot={slot}"}

    pred_sample = pool[slot]
    sample_i = int(pred_sample["i"])    # index in val subset JSON

    val_sample = _VAL_SUBSET.get(cat, [None])[sample_i] if cat in _VAL_SUBSET else None
    if val_sample is None:
        return {"error": f"val sample idx {sample_i} for cat {cat} not in --val_subset_dir"}

    image_paths = val_sample["image"]
    convs = val_sample["conversations"]
    user_text = convs[1]["value"]
    # Split off the long camera-image prelude so the question is readable.
    if "** EGO VEHICLE's DRIVING STATUS **" in user_text:
        question_text = user_text.split("** EGO VEHICLE's DRIVING STATUS **", 1)[1]
        question_text = "** EGO VEHICLE's DRIVING STATUS **" + question_text
    else:
        question_text = user_text

    gt_grounding   = pred_sample.get("gt_grounding") or []
    pred_grounding = pred_sample.get("pred_grounding") or []
    matches        = pred_sample.get("matches") or []
    pred_text      = pred_sample.get("pred_text") or ""
    gt_text        = convs[2]["value"]

    # Group GT and predicted boxes by image_idx (1..6 → 0..5).
    gt_by_img:   Dict[int, List[dict]] = {}
    pred_by_img: Dict[int, List[dict]] = {}
    spurious_idx_set = set(range(len(pred_grounding)))
    for m in matches:
        if m.get("pred_idx") is not None:
            spurious_idx_set.discard(m["pred_idx"])

    for g in gt_grounding:
        idx = (g.get("image_idx") or 0) - 1
        if 0 <= idx < 6:
            gt_by_img.setdefault(idx, []).append(g)
    for pi, p in enumerate(pred_grounding):
        idx = (p.get("image_idx") or 0) - 1
        if 0 <= idx < 6:
            p2 = dict(p); p2["_idx"] = pi
            pred_by_img.setdefault(idx, []).append(p2)

    images_b64 = []
    for i, img_path in enumerate(image_paths):
        try:
            img = Image.open(img_path).convert("RGB")
            if i in REAR_INDICES:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            iw, ih = img.size
            draw = ImageDraw.Draw(img)
            font = _load_font(15)

            # GT boxes (green, solid).
            for g in gt_by_img.get(i, []):
                box = g.get("box")
                if not box:
                    continue
                label = f"GT:{g.get('label', '?')}"
                _draw_box(draw, _scale_box_to_image(box, iw, ih), label,
                          GT_COLOR, dashed=False, font=font)

            # Predicted boxes (pink for matched, dashed paler pink for spurious).
            for p in pred_by_img.get(i, []):
                box = p.get("box")
                if not box:
                    continue
                pi = p["_idx"]
                if pi in spurious_idx_set:
                    label = f"PRED!:{p.get('label', '?')}"
                    color = PRED_SPUR
                    dashed = True
                else:
                    label = f"PRED:{p.get('label', '?')}"
                    color = PRED_COLOR
                    dashed = False
                _draw_box(draw, _scale_box_to_image(box, iw, ih), label,
                          color, dashed=dashed, font=font)

            # Downsize for transport (keeps payload ~150KB / image).
            w, h = img.size
            img = img.resize((w // 2, h // 2), Image.LANCZOS)
            images_b64.append(_image_to_b64(img))
        except Exception as e:
            images_b64.append(_image_to_b64(_placeholder(f"{e}")))

    # Build per-box match table for the right panel.
    match_rows = []
    for m in matches:
        gi = m["gt"]
        pi_idx = m.get("pred_idx")
        pred_obj = pred_grounding[pi_idx] if pi_idx is not None else None
        match_rows.append({
            "gt_view":   gi.get("image_idx"),
            "gt_label":  gi.get("label"),
            "gt_box":    gi.get("box"),
            "pred_view": pred_obj.get("image_idx") if pred_obj else None,
            "pred_label": pred_obj.get("label") if pred_obj else None,
            "pred_box":  pred_obj.get("box") if pred_obj else None,
            "iou":       m.get("iou"),
            "view_ok":   bool(m.get("view_ok")),
            "iou_ok":    bool(m.get("iou_ok")),
        })

    spurious_preds = [
        {
            "view":  pred_grounding[i].get("image_idx"),
            "label": pred_grounding[i].get("label"),
            "box":   pred_grounding[i].get("box"),
        }
        for i in sorted(spurious_idx_set)
        if pred_grounding[i].get("box") is not None
    ]

    # Parse reasoning + full answer from both sides' JSON envelopes.
    gt_reasoning,   gt_answer_full   = _extract_reasoning_and_answer(gt_text)
    pred_reasoning, pred_answer_full = _extract_reasoning_and_answer(pred_text)

    return {
        "stage": stage_label,
        "cat": cat,
        "slot": slot,
        "val_idx": sample_i,
        "images": images_b64,
        "answer_correct":     bool(pred_sample.get("answer_correct")),
        "pred_answer":        pred_sample.get("pred_answer"),
        "gt_answer":          pred_sample.get("gt_answer"),
        "pred_answer_full":   pred_answer_full,
        "gt_answer_full":     gt_answer_full,
        "pred_reasoning":     pred_reasoning,
        "gt_reasoning":       gt_reasoning,
        "n_gt_boxes":         len(gt_grounding),
        "n_pred_boxes":       int(pred_sample.get("n_pred_boxes", 0)),
        "n_spurious":         int(pred_sample.get("n_spurious", 0)),
        "matches":            match_rows,
        "spurious_preds":     spurious_preds,
        "pred_text":          pred_text,
        "gt_text":            gt_text,
        "question_text":      question_text,
    }


@app.route("/api/sample")
def api_sample():
    stage = request.args.get("stage", "")
    cat = request.args.get("cat", "")
    slot = int(request.args.get("slot", 0))
    return jsonify(_render_sample(stage, cat, slot))


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html><head>
<title>Eval Sample Visualizer</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Segoe UI', Tahoma, sans-serif; background: #1a1a2e; color: #e0e0e0; }
.header {
    background: #16213e; padding: 10px 20px;
    display: flex; align-items: center; gap: 14px; flex-wrap: wrap;
    border-bottom: 2px solid #0f3460;
}
.header h1 { font-size: 17px; color: #53d8fb; }
.controls { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.controls select, .controls button {
    padding: 5px 10px; border-radius: 4px; border: 1px solid #0f3460;
    background: #1a1a2e; color: #e0e0e0; font-size: 13px;
}
.controls button { background: #0f3460; cursor: pointer; font-weight: bold; }
.controls button:hover { background: #53d8fb; color: #1a1a2e; }
.controls label { font-size: 12px; color: #aaa; }
.info-badge { background: #0f3460; padding: 3px 8px; border-radius: 10px; font-size: 11px; color: #53d8fb; }
.main { display: flex; height: calc(100vh - 56px); }
.left-panel { flex: 3; padding: 6px; display: flex; flex-direction: column; gap: 3px; }
.image-grid {
    display: grid; grid-template-columns: 1fr 1fr 1fr; grid-template-rows: 1fr 1fr;
    gap: 3px; flex: 1;
}
.image-cell { position: relative; background: #111; border-radius: 3px; overflow: hidden; }
.image-cell img { width: 100%; height: 100%; object-fit: contain; }
.image-cell .cam-label {
    position: absolute; top: 3px; left: 3px; background: rgba(0,0,0,0.75);
    color: #53d8fb; padding: 1px 6px; border-radius: 2px; font-size: 11px; font-weight: bold;
}

.right-panel { flex: 2; display: flex; flex-direction: column; padding: 6px; gap: 6px; overflow: hidden; }
.card {
    background: #16213e; border-radius: 5px; padding: 10px;
    border: 1px solid #0f3460;
    overflow-y: auto;
}
.card h3 { font-size: 13px; margin-bottom: 6px; display: flex; align-items: center; gap: 6px; }
.badge { padding: 1px 6px; border-radius: 3px; font-size: 10px; color: #fff; }
.badge-gt    { background: #2ecc71; }
.badge-pred  { background: #e91e8c; }
.badge-warn  { background: #e67e22; }
.badge-ok    { background: #3498db; }
.badge-bad   { background: #c0392b; }

.scroll { font-size: 12px; line-height: 1.55; white-space: pre-wrap; word-break: break-word; }
.match-tbl { width: 100%; border-collapse: collapse; font-size: 11px; }
.match-tbl th, .match-tbl td { padding: 4px 6px; border-bottom: 1px solid #243a5e; }
.match-tbl th { background: #0f3460; text-align: left; }
.match-tbl tr:hover { background: rgba(83,216,251,0.06); }
.box-coord { font-family: monospace; color: #88c5ff; font-size: 10px; }
.match-ok  { color: #2ecc71; }
.match-bad { color: #e67e22; }

.summary { display: flex; gap: 10px; flex-wrap: wrap; font-size: 11px; color: #ddd; padding-bottom: 6px; }
.summary span { background: #0f3460; padding: 2px 8px; border-radius: 8px; }

/* Side-by-side reasoning / answer comparison blocks. Left border tints by
   side so the eye picks GT vs PRED at a glance even when scrolling. */
.cmp { display: flex; flex-direction: column; gap: 6px; }
.cmp-block { padding: 8px 10px; border-radius: 4px; background: #1f2c4a; }
.cmp-block.gt   { border-left: 4px solid #2ecc71; }
.cmp-block.pred { border-left: 4px solid #e91e8c; }
.cmp-label { font-size: 10px; font-weight: bold; color: #aaa; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.5px; }
.cmp-label .badge { vertical-align: middle; margin-right: 4px; }
.cmp-text  { font-size: 12px; line-height: 1.55; white-space: pre-wrap; word-break: break-word; color: #ddd; }
.cmp-text.empty { color: #777; font-style: italic; }
.cmp-text.diff   { background: rgba(255, 200, 0, 0.05); }   /* subtle highlight when mismatched */
.answer-row { display: flex; gap: 8px; align-items: stretch; }
.answer-row > .cmp-block { flex: 1; }
.answer-row .cmp-text { font-size: 14px; font-weight: 500; }
.bullet-count { color: #888; font-size: 10px; margin-left: 6px; }
.collapsed { max-height: 80px; overflow: hidden; position: relative; }
.collapsed::after {
    content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 25px;
    background: linear-gradient(transparent, #1f2c4a);
}
.expand-btn {
    background: none; color: #53d8fb; border: none; cursor: pointer;
    font-size: 11px; padding: 2px 0; margin-top: 4px;
}
.expand-btn:hover { text-decoration: underline; }

.legend { display: flex; gap: 12px; font-size: 11px; }
.legend-item { display: flex; align-items: center; gap: 4px; }
.legend-box { width: 13px; height: 13px; border-radius: 2px; }
.legend-box.dashed { border: 2px dashed #e991c8; background: transparent; }

.cat-pill {
    background: #243a5e; color: #ccc; border-radius: 12px; padding: 2px 9px;
    font-size: 11px; margin-right: 4px; cursor: pointer; user-select: none;
    border: 1px solid transparent;
}
.cat-pill:hover { background: #305a8a; }
.cat-pill.active { border-color: #53d8fb; color: #53d8fb; background: #0f3460; font-weight: bold; }
</style>
</head>
<body>
<div class="header">
    <h1>Eval Sample Visualizer</h1>
    <div class="controls">
        <label>Stage:</label>
        <select id="stage-sel" onchange="onStageChange()"></select>
        <label>Slot:</label>
        <select id="slot-sel" onchange="loadSample()"></select>
        <div class="nav-buttons">
            <button onclick="navigate(-1)">&larr; prev</button>
            <button onclick="navigate(1)">next &rarr;</button>
        </div>
        <span id="info-badge" class="info-badge">--</span>
    </div>
    <div class="legend">
        <div class="legend-item"><div class="legend-box" style="background:#2ecc71"></div> GT</div>
        <div class="legend-item"><div class="legend-box" style="background:#e91e8c"></div> Pred (matched)</div>
        <div class="legend-item"><div class="legend-box dashed"></div> Pred (spurious)</div>
    </div>
</div>

<div class="header" style="border-top: 1px solid #0f3460; padding: 8px 20px;">
    <label>Category:</label>
    <div id="cat-pills"></div>
</div>

<div class="main">
    <div class="left-panel">
        <div class="image-grid" id="image-grid"></div>
    </div>
    <div class="right-panel">
        <div class="card" style="flex: 0 0 auto;">
            <div class="summary" id="summary"></div>
        </div>

        <div class="card" style="flex: 0 0 auto;">
            <h3>answer (final) <span class="bullet-count" id="answer-status"></span></h3>
            <div class="answer-row">
                <div class="cmp-block gt">
                    <div class="cmp-label"><span class="badge badge-gt">GT</span> answer</div>
                    <div class="cmp-text" id="gt-answer-full"></div>
                </div>
                <div class="cmp-block pred">
                    <div class="cmp-label"><span class="badge badge-pred">PRED</span> answer</div>
                    <div class="cmp-text" id="pred-answer-full"></div>
                </div>
            </div>
        </div>

        <div class="card" style="flex: 1.4;">
            <h3>reasoning <span class="bullet-count" id="reasoning-counts"></span></h3>
            <div class="cmp">
                <div class="cmp-block gt">
                    <div class="cmp-label"><span class="badge badge-gt">GT</span> reasoning</div>
                    <div class="cmp-text" id="gt-reasoning"></div>
                </div>
                <div class="cmp-block pred">
                    <div class="cmp-label"><span class="badge badge-pred">PRED</span> reasoning</div>
                    <div class="cmp-text" id="pred-reasoning"></div>
                </div>
            </div>
        </div>

        <div class="card" style="flex: 1;">
            <h3><span class="badge badge-gt">GT</span> ↔ <span class="badge badge-pred">PRED</span>
                per-box match (IoU / view)</h3>
            <table class="match-tbl" id="match-tbl">
                <thead><tr>
                    <th>#</th><th>GT</th><th>PRED</th>
                    <th>IoU</th><th>View</th>
                </tr></thead>
                <tbody id="match-tbody"></tbody>
            </table>
            <div id="spurious-block" style="margin-top: 10px; display: none;">
                <h3 style="font-size: 12px;"><span class="badge badge-warn">spurious</span> predicted boxes
                    with no matching GT</h3>
                <ul id="spurious-list" style="margin-left: 18px; font-size: 11px;"></ul>
            </div>
        </div>
        <details class="card" style="flex: 0 0 auto;">
            <summary style="cursor: pointer; font-size: 13px;">
                <span class="badge badge-pred">PRED</span> raw generation (full JSON envelope)
            </summary>
            <div class="scroll" id="pred-text" style="margin-top: 6px;"></div>
        </details>
        <details class="card" style="flex: 0 0 auto;">
            <summary style="cursor: pointer; font-size: 13px;">
                <span class="badge badge-gt">GT</span> raw assistant text (full JSON envelope)
            </summary>
            <div class="scroll" id="gt-text" style="margin-top: 6px;"></div>
        </details>
        <details class="card" style="flex: 0 0 auto;">
            <summary style="cursor: pointer; font-size: 13px;">Q context (user turn)</summary>
            <div class="scroll" id="question-text" style="margin-top: 6px;"></div>
        </details>
    </div>
</div>

<script>
let INFO = null;
let curStage = '';
let curCat = '';
let curSlot = 0;

async function bootstrap() {
    const r = await fetch('/api/info');
    INFO = await r.json();
    const stageSel = document.getElementById('stage-sel');
    stageSel.innerHTML = '';
    INFO.stages.forEach(s => {
        const opt = document.createElement('option');
        opt.value = s.stage;
        opt.textContent = `${s.stage} (${s.total} samples)`;
        stageSel.appendChild(opt);
    });
    curStage = INFO.stages[0].stage;
    renderCatPills();
    pickFirstAvailable();
    loadSample();
}

function renderCatPills() {
    const stage = INFO.stages.find(s => s.stage === curStage);
    const pills = document.getElementById('cat-pills');
    pills.innerHTML = '';
    INFO.categories.forEach(cat => {
        const n = (stage.per_cat[cat] || 0);
        const pill = document.createElement('span');
        pill.className = 'cat-pill' + (cat === curCat ? ' active' : '');
        pill.textContent = `${cat} (${n})`;
        pill.onclick = () => { if (n > 0) { curCat = cat; curSlot = 0; renderCatPills(); rebuildSlotSel(); loadSample(); } };
        pills.appendChild(pill);
    });
}

function pickFirstAvailable() {
    const stage = INFO.stages.find(s => s.stage === curStage);
    for (const cat of INFO.categories) {
        if ((stage.per_cat[cat] || 0) > 0) { curCat = cat; curSlot = 0; break; }
    }
    rebuildSlotSel();
}

function rebuildSlotSel() {
    const stage = INFO.stages.find(s => s.stage === curStage);
    const n = (stage.per_cat[curCat] || 0);
    const sel = document.getElementById('slot-sel');
    sel.innerHTML = '';
    for (let i = 0; i < n; i++) {
        const opt = document.createElement('option');
        opt.value = i; opt.textContent = `${i + 1} / ${n}`;
        sel.appendChild(opt);
    }
    sel.value = curSlot;
}

function onStageChange() {
    curStage = document.getElementById('stage-sel').value;
    pickFirstAvailable();
    renderCatPills();
    loadSample();
}

function navigate(d) {
    const stage = INFO.stages.find(s => s.stage === curStage);
    const cats = INFO.categories.filter(c => (stage.per_cat[c] || 0) > 0);
    const ci = cats.indexOf(curCat);
    let n = stage.per_cat[curCat] || 0;
    curSlot += d;
    if (curSlot < 0) {
        const newCi = (ci - 1 + cats.length) % cats.length;
        curCat = cats[newCi];
        curSlot = stage.per_cat[curCat] - 1;
    } else if (curSlot >= n) {
        const newCi = (ci + 1) % cats.length;
        curCat = cats[newCi];
        curSlot = 0;
    }
    renderCatPills();
    rebuildSlotSel();
    loadSample();
}

async function loadSample() {
    document.getElementById('slot-sel').value = curSlot;
    const grid = document.getElementById('image-grid');
    grid.innerHTML = '';
    const camLabels = [
        '1: Front-left', '2: Front', '3: Front-right',
        '4: Rear-left (flip)', '5: Rear (flip)', '6: Rear-right (flip)'
    ];
    for (let i = 0; i < 6; i++) {
        const cell = document.createElement('div');
        cell.className = 'image-cell';
        cell.innerHTML = `<div class="cam-label">${camLabels[i]}</div><img id="img-${i}">`;
        grid.appendChild(cell);
    }
    const r = await fetch(`/api/sample?stage=${encodeURIComponent(curStage)}&cat=${curCat}&slot=${curSlot}`);
    const d = await r.json();
    if (d.error) {
        document.getElementById('summary').textContent = d.error;
        return;
    }
    for (let i = 0; i < 6; i++) document.getElementById('img-' + i).src = d.images[i];

    const ans = d.answer_correct ? '<span class="badge badge-ok">answer ✓</span>' : '<span class="badge badge-bad">answer ✗</span>';
    document.getElementById('summary').innerHTML = `
        <span>stage: <b>${d.stage}</b></span>
        <span>cat: <b>${d.cat}</b></span>
        <span>val_idx: ${d.val_idx}</span>
        <span>${ans}</span>
        <span>pred: <code>${escapeHtml(String(d.pred_answer))}</code></span>
        <span>gt: <code>${escapeHtml(String(d.gt_answer))}</code></span>
        <span>GT boxes: ${d.n_gt_boxes}</span>
        <span>Pred boxes: ${d.n_pred_boxes}</span>
        <span>spurious: ${d.n_spurious}</span>
    `;
    document.getElementById('info-badge').textContent = `${d.cat} ${d.slot + 1}/${INFO.stages.find(s => s.stage === d.stage).per_cat[d.cat]}`;

    const tbody = document.getElementById('match-tbody');
    tbody.innerHTML = '';
    d.matches.forEach((m, i) => {
        const iouTxt = m.iou == null ? '-' : (m.iou).toFixed(3);
        const iouCls = m.iou_ok ? 'match-ok' : 'match-bad';
        const viewCls = m.view_ok ? 'match-ok' : 'match-bad';
        const viewTxt = m.pred_view == null ? '-' : (m.view_ok ? '✓' : '✗ (pred=' + m.pred_view + ')');
        const gtBox = m.gt_box ? `<span class="box-coord">[${m.gt_box.join(', ')}]</span>` : '-';
        const predBox = m.pred_box ? `<span class="box-coord">[${m.pred_box.join(', ')}]</span>` : '<span style="color:#888">no pred</span>';
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${i + 1}</td>
            <td>view ${m.gt_view} · ${escapeHtml(m.gt_label || '')}<br>${gtBox}</td>
            <td>${m.pred_label != null ? escapeHtml(m.pred_label) : '-'}<br>${predBox}</td>
            <td class="${iouCls}">${iouTxt}</td>
            <td class="${viewCls}">${viewTxt}</td>
        `;
        tbody.appendChild(tr);
    });

    if (d.spurious_preds && d.spurious_preds.length) {
        document.getElementById('spurious-block').style.display = '';
        const ul = document.getElementById('spurious-list');
        ul.innerHTML = '';
        d.spurious_preds.forEach(p => {
            const li = document.createElement('li');
            const box = p.box ? `[${p.box.join(', ')}]` : '?';
            li.innerHTML = `view ${p.view} · ${escapeHtml(p.label || '')} <span class="box-coord">${box}</span>`;
            ul.appendChild(li);
        });
    } else {
        document.getElementById('spurious-block').style.display = 'none';
    }

    // Reasoning & answer comparison panels.
    const gtR  = (d.gt_reasoning   || '').trim();
    const prR  = (d.pred_reasoning || '').trim();
    setCmpText('gt-reasoning',   gtR, '(no reasoning parsed from GT)');
    setCmpText('pred-reasoning', prR, '(no reasoning parsed from PRED)');

    const gtBullets   = countBullets(gtR);
    const predBullets = countBullets(prR);
    document.getElementById('reasoning-counts').textContent =
        `GT ${gtBullets} bullet(s) · PRED ${predBullets} bullet(s)`;

    const gtA  = (d.gt_answer_full   || d.gt_answer   || '').trim();
    const prA  = (d.pred_answer_full || d.pred_answer || '').trim();
    setCmpText('gt-answer-full',   gtA, '(no answer parsed)');
    setCmpText('pred-answer-full', prA, '(no answer parsed)');
    document.getElementById('answer-status').textContent =
        d.answer_correct ? '✓ exact match' : '✗ mismatch';
    document.getElementById('answer-status').style.color = d.answer_correct ? '#2ecc71' : '#e67e22';

    document.getElementById('pred-text').textContent = d.pred_text || '(empty)';
    document.getElementById('gt-text').textContent   = d.gt_text || '(empty)';
    document.getElementById('question-text').textContent = d.question_text || '(empty)';
}

function setCmpText(id, text, emptyMsg) {
    const el = document.getElementById(id);
    if (!text) {
        el.textContent = emptyMsg;
        el.classList.add('empty');
    } else {
        el.textContent = text;
        el.classList.remove('empty');
    }
}

function countBullets(text) {
    if (!text) return 0;
    // Bullets are emitted as "- " at line starts (project convention).
    const m = text.match(/^\s*[-*•]\s+/gm);
    return m ? m.length : (text.trim() ? 1 : 0);
}

function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

document.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    if (e.key === 'ArrowLeft')  navigate(-1);
    if (e.key === 'ArrowRight') navigate(1);
});

bootstrap();
</script>
</body></html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Eval Sample Visualizer (curriculum v2)")
    src = parser.add_mutually_exclusive_group(required=False)
    src.add_argument(
        "--eval_predictions", nargs="+", default=None,
        help="One or more eval_predictions.json paths (stage selector lists them).",
    )
    src.add_argument(
        "--curriculum_root", default=None,
        help="Curriculum output root; auto-discovers every stage_*/eval_predictions.json.",
    )
    parser.add_argument(
        "--val_subset_dir", required=True,
        help="Directory containing sft_val_qwen3vl_<CAT>(_subset)?.json files "
             "(needed for image paths + system/user text).",
    )
    parser.add_argument(
        "--samples_per_cat", type=int, default=2,
        help="First N grounded samples per category (matches eval --sanity_print_n; "
             "default 2 -> 20 samples per stage).",
    )
    parser.add_argument("--port", type=int, default=6061)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    if not args.eval_predictions and not args.curriculum_root:
        parser.error("Provide either --eval_predictions or --curriculum_root.")

    global _PRED_PATHS, _VAL_SUBSET, _VAL_SUBSET_DIR, _N_PER_CAT
    _PRED_PATHS = _discover_predictions(args)
    _VAL_SUBSET = _load_val_subset_index(args.val_subset_dir)
    _VAL_SUBSET_DIR = args.val_subset_dir
    _N_PER_CAT = args.samples_per_cat

    print("Eval Sample Visualizer")
    print(f"  Stages discovered ({len(_PRED_PATHS)}):")
    for label, path in _PRED_PATHS:
        print(f"    {label:30s}  <- {path}")
    print(f"  val subset dir:  {_VAL_SUBSET_DIR}")
    print(f"  cats from subset: {sorted(_VAL_SUBSET.keys())}")
    print(f"  samples / category: {_N_PER_CAT}")
    # Pre-warm one stage so the user sees if anything blows up before serving.
    if _PRED_PATHS:
        try:
            n = sum(len(v) for v in _get_sanity_samples(_PRED_PATHS[0][0]).values())
            print(f"  first stage has {n} sanity samples ready")
        except Exception as e:
            print(f"  [warn] failed to pre-warm first stage: {e}")
    print(f"  Open  http://<server-ip>:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
