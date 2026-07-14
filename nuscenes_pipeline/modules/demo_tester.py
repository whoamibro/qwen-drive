"""
Demo Tester — run the SFT-trained Qwen3-VL model on every frame of a nuScenes
scene, answering either the full 40-question demo bank or a single custom
question at each frame.

Two modes (mutually exclusive):

    Mode A — Question list from `data/demo_questions.json`.
        For each of the 40 canonical questions, run inference on every frame
        of the scene. Output groups predictions by frame; each frame carries
        its 40 answers.

    Mode B — Single custom question via `--q "..."`.
        Same per-frame loop, but with one user-supplied question. Output has
        one result per frame.

Scene selection:
    --scene_token accepts either the full nuScenes scene_token or the 16-char
    prefix (e.g. 'ff6af17f52c34e9c'). The pkl is scanned and every frame with
    a matching scene_token becomes a test frame, kept in original pkl order.

Reuses the exact prompt/inference pipeline that `sft_model_tester` uses so
predictions are byte-comparable to the training-time distribution:
    - `build_system_prompt_no_objects` / `build_user_prompt_no_objects`
    - `prepare_messages` (6-view interleave + rear-flip)
    - `run_inference_with_ids` + `_parse_grounding_from_token_ids`

Usage (from project root):

    # Mode A — all 40 demo questions, every frame
    python -m nuscenes_pipeline.modules.demo_tester \\
        --scene_token ff6af17f52c34e9c \\
        --demo_questions_path data/demo_questions.json \\
        --base_model ckpts/qwen3_vl_8b_instruct \\
        --lora_path output/curriculum_v2_f3_0618/F3__seed0/ckpt_100 \\
        --pkl_path data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl \\
        --output_dir demo_test_results/f3_ff6af17f

    # Mode B — single custom question, every frame
    python -m nuscenes_pipeline.modules.demo_tester \\
        --scene_token b526c20f7eed49f0 \\
        --q "Is the ego-vehicle safe to change lanes right?" \\
        --base_model ckpts/qwen3_vl_8b_instruct \\
        --lora_path output/curriculum_v2_f3_0618/F3__seed0/ckpt_100 \\
        --pkl_path data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional

from nuscenes_pipeline.core.nuscenes_data_loader import NuScenesDataLoader
from nuscenes_pipeline.modules.sft_model_tester import (
    load_model,
    prepare_messages,
    run_inference_with_ids,
    _parse_grounding_from_token_ids,
    _extract_answer,
    _normalize_answer,
)


# ---------------------------------------------------------------------------
# Scene lookup
# ---------------------------------------------------------------------------
def find_scene_frames(loader: NuScenesDataLoader, scene_token: str) -> List[int]:
    """Return frame indices (sorted, original pkl order) belonging to the
    scene. Accepts either the full token or a prefix — the pkl uses full
    32-char hex; the CLI often passes the 16-char prefix used in filenames.
    """
    st = scene_token.strip()
    if not st:
        raise ValueError("--scene_token is empty.")
    frames: List[int] = []
    for i, info in enumerate(loader.infos):
        info_tok = info.get("scene_token") or ""
        if info_tok == st or (len(st) < len(info_tok) and info_tok.startswith(st)):
            frames.append(i)
    return frames


# ---------------------------------------------------------------------------
# Question loading
# ---------------------------------------------------------------------------
def load_demo_questions(path: str) -> List[dict]:
    """Flatten the demo_questions.json category structure into an ordered
    list of question dicts, each carrying its category label.

    Order = curriculum ordering across categories, question order within
    each category as written in the source JSON.
    """
    with open(path) as f:
        bank = json.load(f)
    out: List[dict] = []
    for cat_block in bank.get("categories", []):
        cat = cat_block.get("category", "Unknown")
        for q in cat_block.get("questions", []):
            entry = {
                "question_id":  q["id"],
                "category":     cat,
                "question":     q["question"],
                "answer_type":  q.get("answer_type", ""),
            }
            if "options" in q:
                entry["options"] = q["options"]
            out.append(entry)
    return out


def format_question_with_options(q: dict) -> str:
    """Match sft_prompt_builder's user-turn structure for MCQ questions: the
    training-time human turn appends an 'Options:' block for mcq questions.
    For y_or_n / num_count / open_ended, the bare question is fine."""
    text = q["question"].strip()
    if q.get("answer_type") == "mcq" and q.get("options"):
        lines = [text, "", "Options:"]
        for i, opt in enumerate(q["options"]):
            letter = chr(ord("A") + i)
            lines.append(f"({letter}) {opt}")
        return "\n".join(lines)
    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        prog="python -m nuscenes_pipeline.modules.demo_tester",
        description=(
            "Run the SFT-trained Qwen3-VL model on every frame of a nuScenes "
            "scene, answering either all 40 demo questions (Mode A) or a "
            "single custom question (Mode B)."
        ),
    )
    p.add_argument("--scene_token", required=True, type=str,
                   help="Full or 16-char-prefix scene_token, e.g. ff6af17f52c34e9c.")

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo_questions_path", type=str,
                      help="Mode A: JSON path (e.g. data/demo_questions.json).")
    mode.add_argument("--q", type=str, dest="q",
                      help="Mode B: a single custom question to ask at every frame.")

    p.add_argument("--base_model", type=str, default="ckpts/qwen3_vl_8b_instruct")
    p.add_argument("--lora_path",  type=str, required=True)
    p.add_argument("--pkl_path",   type=str,
                   default="data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl")
    p.add_argument("--data_root",  type=str, default=None,
                   help="Passed to NuScenesDataLoader; None lets it auto-derive.")
    p.add_argument("--resize_factor", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--output_dir", type=str, default=None,
                   help="Default: demo_test_results/<scene_token[:16]>/")
    p.add_argument("--max_frames", type=int, default=None,
                   help="Cap on frames processed (smoke tests). None = all.")
    p.add_argument("--print_every", type=int, default=1,
                   help="Print progress every N (frame, question) inferences. "
                        "0 = silent (only frame-level progress prints).")
    # Multi-GPU fan-out: a worker owns frames whose position-within-scene
    # modulo `sample_stride` equals `sample_offset`. Default stride=1 =
    # single-worker (all frames). Each worker writes predictions_worker<N>.json
    # (unless --output_name is explicit). The shell wrapper runs 8 workers +
    # merges them via `merge_demo_predictions`.
    p.add_argument("--sample_stride", type=int, default=1,
                   help="Frame-level sharding: worker processes only frames with "
                        "frame_pos %% sample_stride == sample_offset. Default 1 = all.")
    p.add_argument("--sample_offset", type=int, default=0,
                   help="Companion to --sample_stride; must be in [0, stride).")
    p.add_argument("--output_name", type=str, default=None,
                   help="Explicit output filename. Default: "
                        "predictions.json (stride=1) or "
                        "predictions_worker<offset>.json (stride>1).")
    args = p.parse_args()

    # ---- Load data ----
    print(f"[demo_tester] loading pkl: {args.pkl_path}")
    loader = NuScenesDataLoader(args.pkl_path, data_root=args.data_root)
    frames = find_scene_frames(loader, args.scene_token)
    if not frames:
        print(f"ERROR: no frames found for scene_token={args.scene_token!r} "
              f"in {args.pkl_path}", file=sys.stderr)
        return 2
    if args.max_frames is not None:
        frames = frames[: args.max_frames]

    # Resolve the FULL scene_token from the first matched frame (so downstream
    # filenames use the canonical form regardless of what the user passed).
    full_scene_tok = loader.infos[frames[0]]["scene_token"]
    scene_short = full_scene_tok[:16]

    # Multi-GPU sharding: keep frames' original positions within the scene so
    # the merger can stitch by frame_pos across workers.
    stride = max(1, int(args.sample_stride))
    offset = int(args.sample_offset)
    if offset < 0 or offset >= stride:
        print(f"ERROR: --sample_offset must be in [0, {stride}); got {offset}",
              file=sys.stderr)
        return 2
    n_full = len(frames)
    shard = [(pos, frame_idx) for pos, frame_idx in enumerate(frames)
             if pos % stride == offset]
    shard_tag = (f"  (worker stride={stride} offset={offset}; "
                 f"{len(shard)}/{n_full} frames)"
                 if stride > 1 else "")
    print(f"[demo_tester] scene_token={full_scene_tok}  frames={n_full}"
          f"{shard_tag}")

    # ---- Load questions ----
    if args.demo_questions_path:
        questions = load_demo_questions(args.demo_questions_path)
        mode = "A"
        print(f"[demo_tester] mode A — {len(questions)} demo questions loaded "
              f"from {args.demo_questions_path}")
    else:
        questions = [{
            "question_id":  "USER-000",
            "category":     "User",
            "question":     args.q,
            "answer_type":  "",
        }]
        mode = "B"
        print(f"[demo_tester] mode B — 1 custom question: {args.q!r}")

    total_inferences = len(shard) * len(questions)
    print(f"[demo_tester] total inferences (this worker): "
          f"{len(shard)} frames x {len(questions)} questions = {total_inferences}")

    # ---- Output path ----
    output_dir = args.output_dir or os.path.join("demo_test_results", scene_short)
    os.makedirs(output_dir, exist_ok=True)
    if args.output_name:
        out_name = args.output_name
    elif stride > 1:
        out_name = f"predictions_worker{offset}.json"
    else:
        out_name = "predictions.json"
    out_path = os.path.join(output_dir, out_name)

    # ---- Load model ----
    print(f"[demo_tester] loading model...")
    model, processor = load_model(args.base_model, args.lora_path)
    tokenizer = processor.tokenizer

    # ---- Inference loop ----
    result = {
        "scene_token":  full_scene_tok,
        "mode":         mode,
        "base_model":   args.base_model,
        "lora_path":    os.path.abspath(args.lora_path),
        "pkl_path":     os.path.abspath(args.pkl_path),
        "resize_factor": args.resize_factor,
        "max_new_tokens": args.max_new_tokens,
        "n_frames":     n_full,          # total frames in the scene (not this shard)
        "n_questions":  len(questions),
        "frames":       [],
    }
    if stride > 1:
        result["sample_stride"] = stride
        result["sample_offset"] = offset
        result["n_frames_this_worker"] = len(shard)
    if args.demo_questions_path:
        result["demo_questions_path"] = os.path.abspath(args.demo_questions_path)

    t_start = time.time()
    done = 0
    for fi_pos, frame_idx in shard:
        sample = loader.get_sample(frame_idx)
        frame_entry = {
            "frame_pos":    fi_pos,           # position within the scene (0..N-1)
            "frame_idx":    frame_idx,        # absolute pkl index
            "sample_token": sample.token,
            "timestamp":    sample.timestamp,
            "image_paths":  [sample.cameras[cam].image_path
                             for cam in ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
                                         "CAM_BACK_LEFT",  "CAM_BACK",  "CAM_BACK_RIGHT"]],
            "results":      [],
        }
        for qi, q in enumerate(questions):
            user_q_text = format_question_with_options(q)
            messages, sys_prompt, user_prompt = prepare_messages(
                sample, loader, user_q_text, resize_factor=args.resize_factor,
            )
            pred_text, pred_token_ids = run_inference_with_ids(
                model, processor, messages, args.max_new_tokens,
            )
            pred_answer = _extract_answer(pred_text)
            pred_grounding = _parse_grounding_from_token_ids(pred_token_ids, tokenizer)

            entry = {
                "question_id":    q["question_id"],
                "category":       q["category"],
                "question":       q["question"],
                "answer_type":    q.get("answer_type", ""),
                "pred_text":      pred_text,
                "pred_answer":    pred_answer,
                "pred_answer_norm": _normalize_answer(pred_answer) if pred_answer else None,
                "pred_grounding": pred_grounding,
            }
            if q.get("options"):
                entry["options"] = q["options"]
            frame_entry["results"].append(entry)

            done += 1
            if args.print_every and (done % args.print_every == 0 or done == total_inferences):
                elapsed = time.time() - t_start
                rate = done / max(elapsed, 1e-6)
                eta = (total_inferences - done) / max(rate, 1e-6)
                print(f"  [{done}/{total_inferences}]  "
                      f"frame_pos={fi_pos+1}/{n_full}  q {qi+1}/{len(questions)} "
                      f"({q['question_id']})  "
                      f"pred={(pred_answer or '')!r}  "
                      f"rate={rate:.2f}/s  eta={eta/60:.1f}min")

        result["frames"].append(frame_entry)

        # Incremental save after each frame — a crash mid-loop still leaves
        # the completed frames on disk. Cheap: the dict is entirely serializable.
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

    elapsed = time.time() - t_start
    print(f"[demo_tester] done in {elapsed:.1f}s "
          f"({total_inferences/max(elapsed,1e-6):.2f} inferences/s)")
    print(f"[demo_tester] predictions saved to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
