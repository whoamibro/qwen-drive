"""
Traffic Signal Status Classifier — VLM pass over cropped traffic signals.

Reads the crop manifest written by
`nuscenes_pipeline.postprocessing.crop_traffic_signals` and asks a vLLM-served
Qwen3-VL model, one crop per request, to report:

    signal_type       vehicle | pedestrian | other | not_a_signal
    light_observable  true if at least one lamp state is clearly readable
    light_color       red | yellow | green | off | unknown
                      ("off" = housing visible, no lamp lit; "unknown" when not observable)
    lit_shape         circle | arrow_left | arrow_right | arrow_straight | pedestrian | other | unknown
    confidence        0.0 - 1.0

Results go to ONE jsonl per split (`--output`), one row per crop keyed by the
crop `file` (= pkl address, see crop_traffic_signals.py). The run is resumable:
rows already present in the output are skipped.

By default each crop is re-cut from the source camera image with `--pad 8`
context pixels and upsampled (LANCZOS) so its shorter side is at least
`--min_side 160`. On a 64-crop val probe with Qwen3-VL-8B this cut false
"not_a_signal" answers from 42/64 (raw 1:1 crops) to 9/64 with the colors
of the readable lamps unchanged. `--pad 0` sends the stored crop file as-is.

Prerequisite:
    vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct --tensor-parallel-size 8
    (or a smaller Qwen3-VL on one GPU; pass --model_name / --api_base)

Usage:
    python -m nuscenes_pipeline.modules.traffic_signal_status_classifier \\
        --manifest cropped_ts_p/val_manifest.jsonl \\
        --output traffic_signal_status_results/val_status.jsonl \\
        --model_name Qwen/Qwen3-VL-8B-Instruct --api_base http://localhost:8010/v1 --workers 32
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from PIL import Image

from nuscenes_pipeline.modules.traffic_light_pole_detection import (
    extract_json_object,
    image_to_base64_data_uri,
)

SIGNAL_TYPES = ("vehicle", "pedestrian", "other", "not_a_signal")
COLORS = ("red", "yellow", "green", "off", "unknown")
SHAPES = ("circle", "arrow_left", "arrow_right", "arrow_straight", "pedestrian", "other", "unknown")

SYSTEM_PROMPT = """You are an expert traffic-signal annotator for autonomous-driving data.
You are shown ONE small image crop taken from a car's camera. A detector proposed it as a traffic signal housing; the crop may be tiny, blurry, cut off, seen from the side or the back, or may not be a traffic signal at all.
Answer ONLY with a single JSON object, no prose."""

USER_PROMPT = """Classify this crop.

Return exactly:
{
  "signal_type": "vehicle" | "pedestrian" | "other" | "not_a_signal",
  "light_observable": true | false,
  "light_color": "red" | "yellow" | "green" | "off" | "unknown",
  "lit_shape": "circle" | "arrow_left" | "arrow_right" | "arrow_straight" | "pedestrian" | "other" | "unknown",
  "confidence": <0.0-1.0>
}

Rules:
- Most crops ARE traffic signals (a detector selected them). Small size, blur, or low resolution is NEVER a reason to answer "not_a_signal". A dark housing with round lamp positions, or any glowing red/amber/green lamp, is a signal.
- signal_type: "vehicle" = round or arrow lamps for road vehicles (stacked dark housing); "pedestrian" = a glowing standing/walking human figure or hand symbol; "other" = bicycle/tram/lane-control or other signal devices; "not_a_signal" ONLY when the content is clearly something else (street lamp, road sign, window, car tail light, reflection, building).
- light_observable = true if you can tell which lamp color is lit, even from a blurry glowing blob (a red/orange blob in the top lamp position = red; a green/cyan blob = green; an amber blob in the middle = yellow), OR if the front face is clearly visible and no lamp is lit. It is false when no lamp face is visible: side or back view, only the housing edge, too dark, or it is not a signal.
- light_color: color of the lit lamp. "off" = front face visible, nothing lit. "unknown" whenever light_observable is false. If two lamps are lit (e.g. red + green arrow), report the one governing straight-ahead vehicles for vehicle signals.
- lit_shape: shape of the lit lamp ("pedestrian" for a figure/hand lamp); "unknown" when light_observable is false.
- Do not infer a color from the housing position alone when nothing glows."""


def load_manifest(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["file"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def prepare_image(row, crops_root, pad, min_side):
    if pad > 0:
        img = Image.open(row["image_path"]).convert("RGB")
        w, h = img.size
        x1, y1, x2, y2 = row["bbox_xyxy"]
        box = (max(0, int(x1 - pad)), max(0, int(y1 - pad)),
               min(w, int(round(x2 + pad))), min(h, int(round(y2 + pad))))
        img = img.crop(box)
    else:
        img = Image.open(os.path.join(crops_root, row["file"])).convert("RGB")
    if min_side and min(img.size) < min_side:
        s = min_side / min(img.size)
        img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.LANCZOS)
    return img


def normalize(parsed):
    """Coerce the model JSON into the fixed label vocabulary."""
    if not isinstance(parsed, dict):
        return None
    st = str(parsed.get("signal_type", "")).strip().lower()
    st = st if st in SIGNAL_TYPES else "other" if st else None
    obs = parsed.get("light_observable")
    if isinstance(obs, str):
        obs = obs.strip().lower() in ("true", "yes", "1")
    obs = bool(obs) if obs is not None else None
    color = str(parsed.get("light_color", "")).strip().lower()
    color = color if color in COLORS else "unknown"
    shape = str(parsed.get("lit_shape", "")).strip().lower()
    shape = shape if shape in SHAPES else "unknown"
    try:
        conf = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    if st is None or obs is None:
        return None
    if not obs:
        color, shape = "unknown", "unknown"
    return {"signal_type": st, "light_observable": obs, "light_color": color,
            "lit_shape": shape, "confidence": max(0.0, min(1.0, conf))}


def classify_one(client, model_name, row, crops_root, pad, min_side, max_tokens, retries):
    img = prepare_image(row, crops_root, pad, min_side)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_to_base64_data_uri(img)}},
            {"type": "text", "text": USER_PROMPT},
        ]},
    ]
    last_err = None
    for attempt in range(retries + 1):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=model_name, messages=messages, max_tokens=max_tokens, temperature=0.0)
            text = resp.choices[0].message.content
            label = normalize(extract_json_object(text))
            if label is None and attempt < retries:
                last_err = ValueError(f"unparseable/out-of-enum response: {text[:120]!r}")
                continue  # re-ask the model
            return {
                "file": row["file"], "split": row["split"], "sample_idx": row["sample_idx"],
                "token": row["token"], "camera": row["camera"], "cam_idx": row["cam_idx"],
                "sig_idx": row["sig_idx"], "det_id": row.get("det_id"),
                "bbox_xyxy": row["bbox_xyxy"], "det_score": row.get("score"),
                "input_size": list(img.size), "pad_px": pad,
                "parse_ok": label is not None, "label": label, "raw_response": text,
                "inference_time_sec": round(time.time() - t0, 3),
            }
        except Exception as e:  # noqa: BLE001 — network/API errors, retry
            last_err = e
            time.sleep(min(2 ** attempt, 10))
    return {"file": row["file"], "split": row["split"], "sample_idx": row["sample_idx"],
            "token": row["token"], "camera": row["camera"], "cam_idx": row["cam_idx"],
            "sig_idx": row["sig_idx"], "det_id": row.get("det_id"), "bbox_xyxy": row["bbox_xyxy"],
            "det_score": row.get("score"), "input_size": None, "pad_px": pad,
            "parse_ok": False, "label": None, "raw_response": None,
            "error": repr(last_err), "inference_time_sec": None}


def main():
    p = argparse.ArgumentParser(description="Classify cropped traffic signals with a vLLM-served VLM")
    p.add_argument("--manifest", required=True, help="cropped_ts_p/{split}_manifest.jsonl")
    p.add_argument("--crops_root", default="./cropped_ts_p")
    p.add_argument("--output", required=True, help="Output jsonl (appended; resumable)")
    p.add_argument("--model_name", default="Qwen/Qwen3-VL-235B-A22B-Instruct")
    p.add_argument("--api_base", default="http://localhost:8000/v1")
    p.add_argument("--api_key", default="EMPTY")
    p.add_argument("--workers", type=int, default=32, help="Concurrent requests")
    p.add_argument("--pad", type=int, default=8,
                   help="Re-crop from source image with N context px (0 = use stored crop as-is). "
                        "8 px + --min_side 160 cut false 'not_a_signal' from 66%% to 14%% on a val probe")
    p.add_argument("--min_side", type=int, default=160, help="Upsample crops whose short side is below this")
    p.add_argument("--max_tokens", type=int, default=150)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--limit", type=int, default=None, help="Only the first N pending crops (debug)")
    p.add_argument("--sample_indices", type=int, nargs="+", default=None, help="Restrict to these pkl sample indices")
    p.add_argument("--min_det_score", type=float, default=None, help="Skip crops with detector score below this")
    args = p.parse_args()

    rows = load_manifest(args.manifest)
    if args.sample_indices:
        keep = set(args.sample_indices)
        rows = [r for r in rows if r["sample_idx"] in keep]
    if args.min_det_score is not None:
        rows = [r for r in rows if (r.get("score") or 0) >= args.min_det_score]
    done = load_done(args.output)
    pending = [r for r in rows if r["file"] not in done]
    if args.limit:
        pending = pending[:args.limit]
    print(f"manifest rows: {len(rows)}  already done: {len(done)}  pending: {len(pending)}")
    if not pending:
        return

    client = OpenAI(base_url=args.api_base, api_key=args.api_key, timeout=120)
    try:
        served = [m.id for m in client.models.list().data]
        if args.model_name not in served:
            print(f"WARNING: {args.model_name} not in served models {served}", file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"Cannot reach vLLM at {args.api_base}: {e}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    lock = threading.Lock()
    n_ok = n_fail = 0
    t_start = time.time()
    with open(args.output, "a") as out, ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(classify_one, client, args.model_name, r, args.crops_root,
                          args.pad, args.min_side, args.max_tokens, args.retries) for r in pending]
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            with lock:
                out.write(json.dumps(res) + "\n")
                if i % 500 == 0 or i == len(pending):
                    out.flush()
            if res["parse_ok"]:
                n_ok += 1
            else:
                n_fail += 1
            if i % 500 == 0 or i == len(pending):
                rate = i / (time.time() - t_start)
                eta = (len(pending) - i) / rate if rate else 0
                print(f"  {i}/{len(pending)}  ok={n_ok} fail={n_fail}  {rate:.1f} crops/s  ETA {eta / 60:.1f} min",
                      flush=True)
    print(f"Done: {n_ok} labeled, {n_fail} failed/unparsed -> {args.output}")


if __name__ == "__main__":
    main()
