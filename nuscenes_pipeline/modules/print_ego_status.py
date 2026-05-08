"""
Ego Vehicle Driving-Status Printer (no LLM)

Iterates nuScenes samples and prints the ego vehicle's driving status block:
driving command, pose (global ENU), velocity (ego-relative FLU and global ENU),
speed, and motion state. Reuses the same ego extraction logic that the VQA
pipeline (risk_assessment, question_selector, answer_generator) feeds into the
prompt, so the printout matches what the model sees at inference time.

Usage:
    # Print samples 0..9 to stdout
    python -m nuscenes_pipeline.modules.print_ego_status --start_idx 0 --end_idx 9

    # Print specific sample indices (range flags are ignored if --indices is given)
    python -m nuscenes_pipeline.modules.print_ego_status --indices 0,42,1694

    # Print and dump per-sample JSON
    python -m nuscenes_pipeline.modules.print_ego_status \\
        --start_idx 0 --end_idx 100 --output_dir ego_status_results
"""

import os
import json
import argparse
from typing import Dict, List, Optional

import numpy as np

from nuscenes_pipeline.core.nuscenes_data_loader import NuScenesDataLoader
from nuscenes_pipeline.core.qa_utils import SceneAnalyzer


def format_ego_block(sample_idx: int, ego_info: Dict, scene_meta: Optional[Dict] = None) -> str:
    """Render the ego_info dict as a printable block (matches the
    ** EGO VEHICLE's DRIVING STATUS ** style used in the VQA prompts)."""
    pos = ego_info["position"]
    vel = ego_info["velocity"]
    speed = ego_info["speed"]
    speed_kmh = ego_info["speed_kmh"]
    yaw = ego_info["yaw"]
    yaw_deg = ego_info["yaw_degrees"]

    lines = []
    if scene_meta:
        lines.append(
            f"[sample_idx={sample_idx}, "
            f"scene={scene_meta.get('scene_token', '')[:8]}, "
            f"token={scene_meta.get('token', '')[:8]}]"
        )
    else:
        lines.append(f"[sample_idx={sample_idx}]")

    lines.append("** EGO VEHICLE's DRIVING STATUS **")
    lines.append(f"  - Driving Command: {ego_info['driving_command']}")
    lines.append(
        f"  - Position (global ENU): [x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}] m"
    )
    lines.append(f"  - Yaw (global): {yaw:.4f} rad ({yaw_deg:.2f}°)")
    # ego_info['velocity'] is global ENU [vx, vy] (matches qa_utils._extract_ego_info)
    lines.append(
        f"  - Velocity (global ENU): [vx={vel[0]:.2f}, vy={vel[1]:.2f}] m/s"
    )
    lines.append(f"  - Speed: {speed:.2f} m/s ({speed_kmh:.2f} km/h)")
    lines.append(f"  - Motion State: {ego_info['motion_state']}")
    return "\n".join(lines)


def collect_ego_status(
    loader: NuScenesDataLoader,
    sample_idx: int,
) -> Dict:
    """Extract ego_info + minimal scene metadata for a single sample."""
    analyzer = SceneAnalyzer(loader)
    sample = loader.get_sample(sample_idx)
    ego_info = analyzer._extract_ego_info(sample)
    scene_meta = {
        "sample_idx": sample_idx,
        "token": sample.token,
        "scene_token": sample.scene_token,
        "location": sample.location,
        "timestamp": sample.timestamp,
    }
    return {"scene_meta": scene_meta, "ego_info": ego_info}


def main():
    parser = argparse.ArgumentParser(
        description="Print the ego vehicle's driving status for a range of nuScenes samples (no LLM)."
    )
    parser.add_argument(
        "--pkl_path",
        type=str,
        default=os.environ.get(
            "NUSCENES_PKL_PATH",
            "./data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl",
        ),
        help="Path to nuScenes pkl file. Defaults to "
             "$NUSCENES_PKL_PATH if set, otherwise "
             "./data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl.",
    )
    parser.add_argument("--start_idx", type=int, default=0, help="Starting sample index (inclusive). Ignored if --indices is given.")
    parser.add_argument("--end_idx", type=int, default=9, help="Ending sample index (inclusive). Ignored if --indices is given.")
    parser.add_argument(
        "--indices",
        type=str,
        default=None,
        help="Comma-separated list of explicit sample indices to print "
             "(e.g. '0,42,1694'). Overrides --start_idx/--end_idx.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="If set, also save per-sample ego status as JSON in this directory.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress stdout printing; only write JSON (requires --output_dir).",
    )
    args = parser.parse_args()

    if args.quiet and not args.output_dir:
        parser.error("--quiet requires --output_dir to actually persist output.")

    if args.indices is not None:
        try:
            indices = [int(tok.strip()) for tok in args.indices.split(",") if tok.strip()]
        except ValueError as e:
            parser.error(f"--indices must be a comma-separated list of integers ({e})")
        if not indices:
            parser.error("--indices is empty after parsing")
        selection_label = f"explicit indices ({len(indices)} sample{'s' if len(indices) != 1 else ''}): {indices}"
    else:
        if args.end_idx < args.start_idx:
            parser.error("--end_idx must be >= --start_idx")
        indices = list(range(args.start_idx, args.end_idx + 1))
        selection_label = f"range [{args.start_idx}, {args.end_idx}] ({len(indices)} samples)"

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("Ego Vehicle Driving-Status Printer")
    print("=" * 80)
    print(f"  pkl_path:   {args.pkl_path}")
    print(f"  selection:  {selection_label}")
    print(f"  output_dir: {args.output_dir or '(stdout only)'}")
    print("=" * 80)

    loader = NuScenesDataLoader(args.pkl_path)
    total = len(loader)
    print(f"Loaded {total} samples from pkl.\n")
    out_of_range = [i for i in indices if i < 0 or i >= total]
    if out_of_range:
        print(
            f"WARNING: {len(out_of_range)} indices out of range [0, {total - 1}]; they will be skipped."
        )

    success = 0
    failures: List[Dict] = []

    for idx in indices:
        if idx < 0 or idx >= total:
            continue
        try:
            payload = collect_ego_status(loader, idx)
        except Exception as e:
            failures.append({"sample_idx": idx, "error": str(e)})
            print(f"  [sample {idx}] FAILED: {e}\n")
            continue

        if not args.quiet:
            print(format_ego_block(idx, payload["ego_info"], payload["scene_meta"]))
            print()

        if args.output_dir:
            out_path = os.path.join(args.output_dir, f"sample_{idx:04d}_ego_status.json")
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False, default=_np_default)

        success += 1

    print("=" * 80)
    print(f"Done. {success} succeeded, {len(failures)} failed.")
    if failures:
        print("Failed sample indices:", [f["sample_idx"] for f in failures])
    print("=" * 80)


def _np_default(obj):
    """JSON encoder fallback for numpy scalars / arrays."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


if __name__ == "__main__":
    main()
