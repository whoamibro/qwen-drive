"""Standalone eval phase for an already-trained experiment.

This is the counterpart to `qwenvl.experiments.run_experiment --skip_eval`:
runs `nuscenes_pipeline.modules.sft_model_tester` against every saved adapter
in the experiment's output dir, then computes `summary.json`.

Idempotent — checkpoints whose `eval_report.json` already exists are skipped
unless `--force` is passed. So if the eval is interrupted mid-run, just
re-invoke and it picks up where it left off.

Usage:
    # Sequential experiment — evaluates every stage_XX_<NAME>/ adapter
    python -m qwenvl.experiments.eval_run \\
        --run_dir output/curriculum_v2_exp/B0__seed0

    # Mixed experiment — evaluates every ckpt_NNN/ checkpoint
    python -m qwenvl.experiments.eval_run \\
        --run_dir output/curriculum_v2_exp/F3__seed0

    # Force re-eval (e.g. after fixing a metric bug)
    python -m qwenvl.experiments.eval_run \\
        --run_dir output/curriculum_v2_exp/B0__seed0 --force

The experiment's `run_manifest.json` tells us which mode (sequential vs
mixed) and which checkpoints to look for, so the user doesn't have to
re-specify any of the training-time knobs here.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from typing import List, Optional, Tuple

# Make the qwenvl package importable when invoked from the project root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "qwen-vl-finetune"))

from qwenvl.experiments.config import ExpConfig   # noqa: E402
from qwenvl.experiments.run_experiment import (    # noqa: E402
    CURRICULUM_ORDER, build_summary, load_base_yaml,
)


_STAGE_DIR_RE = re.compile(r"^stage_(\d+)_([A-Za-z]+)$")
_CKPT_DIR_RE = re.compile(r"^ckpt_(\d{3})$")


def _restore_cfg_from_manifest(run_dir: str) -> ExpConfig:
    """Reconstruct the ExpConfig from the manifest written at train time.

    `run_manifest.json` is the single source of truth for what knobs the
    training run used. We re-hydrate an ExpConfig so build_summary has the
    same metadata it would during the training phase.
    """
    path = os.path.join(run_dir, "run_manifest.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"run_manifest.json not found at {path}. Was training started "
            f"via `qwenvl.experiments.run_experiment`?"
        )
    with open(path) as f:
        manifest = json.load(f)
    resolved = manifest["resolved_config"]
    gf = resolved.get("grounding_floor")
    er = resolved.get("replay")
    return ExpConfig(
        exp_id=resolved["exp_id"],
        mode=resolved["mode"],
        lr_schedule=resolved["lr_schedule"],
        grounding_floor=None if gf == "off" else gf,
        replay=None if er == "off" else er,
        sampler=resolved.get("sampler", "uniform"),
        output_root=resolved["output_root"],
        seed=int(resolved["seed"]),
        config=resolved["config"],
        max_steps=resolved.get("max_steps"),
        dry_run=False,
        skip_eval=False,
    )


def _discover_checkpoint_dirs(run_dir: str, mode: str) -> List[Tuple[str, str]]:
    """Return ordered [(label, dir_path), ...] for every adapter checkpoint
    in `run_dir`. Order matters for the forgetting metric — sequential is
    OBS→CHR, mixed is ascending step fraction."""
    out: List[Tuple[int, str, str]] = []
    for name in sorted(os.listdir(run_dir)):
        sub = os.path.join(run_dir, name)
        if not os.path.isdir(sub):
            continue
        if mode == "sequential":
            m = _STAGE_DIR_RE.match(name)
            if m and os.path.exists(os.path.join(sub, "adapter_model.safetensors")):
                out.append((int(m.group(1)), name, sub))
        elif mode == "mixed":
            m = _CKPT_DIR_RE.match(name)
            if m and os.path.exists(os.path.join(sub, "adapter_model.safetensors")):
                out.append((int(m.group(1)), name, sub))
    out.sort(key=lambda x: x[0])
    return [(label, path) for _, label, path in out]


def _run_one_eval(
    base_model: str,
    lora_path: str,
    eval_dir: str,
    extra_eval_args: Optional[List[str]] = None,
) -> int:
    cmd = [
        sys.executable, "-m", "nuscenes_pipeline.modules.sft_model_tester",
        "--base_model", base_model,
        "--lora_path", lora_path,
        "--per_category_eval_dir", eval_dir,
        "--eval_v2",
        "--iou_threshold", "0.8",
        "--sanity_print_n", "2",
        "--save_predictions",
        "--resize_factor", "1",
        "--max_new_tokens", "1024",
    ]
    if extra_eval_args:
        cmd += extra_eval_args
    print(f"[eval_run] {' '.join(shlex.quote(c) for c in cmd)}")
    full_env = dict(os.environ)
    full_env["PYTHONPATH"] = (
        os.path.join(_PROJECT_ROOT, "qwen-vl-finetune")
        + os.pathsep + full_env.get("PYTHONPATH", "")
    )
    return subprocess.call(cmd, env=full_env, cwd=_PROJECT_ROOT)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m qwenvl.experiments.eval_run",
        description=__doc__,
    )
    p.add_argument(
        "--run_dir", required=True,
        help="The experiment's output dir, e.g. output/curriculum_v2_exp/B0__seed0",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-evaluate checkpoints even if eval_report.json already exists.",
    )
    p.add_argument(
        "--only", default=None,
        help="Restrict eval to specific labels (comma-separated). For sequential "
             "use stage labels like 'stage_04_TSS,stage_06_DRA'; for mixed use "
             "ckpt labels like 'ckpt_050,ckpt_100'.",
    )
    p.add_argument(
        "--extra_eval_args", default="",
        help="Extra raw args appended to every sft_model_tester invocation "
             "(quoted CLI string). Use sparingly.",
    )
    args = p.parse_args(argv)

    if not os.path.isdir(args.run_dir):
        print(f"ERROR: run_dir does not exist: {args.run_dir}")
        return 2

    cfg = _restore_cfg_from_manifest(args.run_dir)
    base_yaml = load_base_yaml(cfg.config)
    base_model = base_yaml["base_model"]
    eval_dir = base_yaml["eval_subset_dir"]

    print("=" * 60)
    print("Standalone eval phase")
    print(f"  exp_id:      {cfg.exp_id}")
    print(f"  mode:        {cfg.mode}")
    print(f"  run_dir:     {args.run_dir}")
    print(f"  base_model:  {base_model}")
    print(f"  eval_dir:    {eval_dir}")
    print(f"  force:       {args.force}")
    if args.only:
        print(f"  only:        {args.only}")
    print("=" * 60)

    checkpoint_dirs = _discover_checkpoint_dirs(args.run_dir, cfg.mode)
    if not checkpoint_dirs:
        print(f"[eval_run] no adapter checkpoints found under {args.run_dir}. "
              f"Did training complete?")
        return 1

    only_set: Optional[set] = None
    if args.only:
        only_set = {x.strip() for x in args.only.split(",") if x.strip()}

    extra_eval_args = shlex.split(args.extra_eval_args) if args.extra_eval_args else []

    failed = 0
    skipped = 0
    evaluated = 0
    for label, ck_dir in checkpoint_dirs:
        if only_set is not None and label not in only_set:
            continue
        report_path = os.path.join(ck_dir, "eval_report.json")
        if os.path.exists(report_path) and not args.force:
            print(f"[eval_run] [skip] {label}: eval_report.json exists "
                  f"(--force to overwrite)")
            skipped += 1
            continue
        print(f"\n[eval_run] {label}  ->  {ck_dir}")
        rc = _run_one_eval(base_model, ck_dir, eval_dir, extra_eval_args)
        if rc != 0:
            print(f"[eval_run] {label} FAILED with rc={rc}")
            failed += 1
        else:
            evaluated += 1

    print()
    print("=" * 60)
    print(f"Eval pass complete. evaluated={evaluated} skipped={skipped} failed={failed}")
    print("=" * 60)

    # Build / refresh the summary.json over whatever eval_reports we now have.
    # Skip nothing here — build_summary tolerates checkpoints whose
    # eval_report.json is missing (each row becomes None).
    if cfg.mode == "sequential":
        per_stage_steps = None
        if cfg.config and os.path.exists(cfg.config):
            try:
                base_yaml = load_base_yaml(cfg.config)
                # Recompute per-stage step estimates only for the manifest
                # extra block; cheap and self-consistent.
                from qwenvl.experiments.run_experiment import _estimate_stage_steps
                defaults = base_yaml["defaults"]
                train_dir = base_yaml["train_data_dir"]
                per_stage_steps = {
                    cat: _estimate_stage_steps(
                        os.path.join(train_dir, f"sft_train_qwen3vl_{cat}.json"),
                        defaults,
                    )
                    for cat in CURRICULUM_ORDER
                }
            except Exception:
                pass
        extra = {"per_stage_steps": per_stage_steps} if per_stage_steps else {}
    else:
        extra = {"step_fractions": [round(0.1 * i, 2) for i in range(1, 11)]}

    summary = build_summary(cfg, checkpoint_dirs, extra=extra)
    summary_path = os.path.join(args.run_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval_run] summary -> {summary_path}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
