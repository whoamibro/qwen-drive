"""Single-experiment ablation orchestrator.

One invocation = one experiment. Parses CLI knobs, resolves them into an
ExpConfig, then orchestrates the training + eval by SUBPROCESSING:

  - `qwen-vl-finetune/qwenvl/experiments/train.py`   (training)
  - `nuscenes_pipeline.modules.sft_model_tester`     (per-checkpoint eval)

This script never imports or modifies anything in the curriculum-v2 command's
code path; all interaction is via subprocess + on-disk artifacts.

Output layout (under --output_root):

    <output_root>/<exp_id>__seed<seed>/
        run_manifest.json                          # resolved knobs
        stage_00_OBS/ ... stage_09_CHR/            # sequential
            adapter_*.safetensors
            eval_report.json                       # per stage, all 10 cats
        ckpt_010/ ... ckpt_100/                    # mixed step-fractions
            adapter_*.safetensors
            eval_report.json
        summary.json                               # §6 headline metrics

Usage:

    python -m qwenvl.experiments.run_experiment \\
        --exp_id F1 --mode sequential --lr_schedule per_stage_wsd \\
        --grounding_floor 0.25 --replay off \\
        --output_root output/curriculum_v2_exp --seed 0

    # Verify config without GPU:
    python -m qwenvl.experiments.run_experiment ... --dry_run

GPU is limited; one experiment at a time. Use `qwenvl.experiments.aggregate`
afterward to refresh the master comparison table.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

# So `python -m qwenvl.experiments.run_experiment` can find the package
# without --module-search-path tweaks.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "qwen-vl-finetune"))

from qwenvl.experiments.config import ExpConfig, parse_cli   # noqa: E402

# Canonical 10-category sequence (matches CURRICULUM_ORDER everywhere else).
CURRICULUM_ORDER: List[str] = [
    "OBS", "IDN", "AAS", "SRO", "TSS",
    "RML", "DRA", "RWP", "ESC", "CHR",
]


# ---------------------------------------------------------------------------
# YAML helper — only used to source the frozen-loss defaults + paths.
# ---------------------------------------------------------------------------
def load_base_yaml(path: str) -> dict:
    try:
        import yaml
    except ImportError as e:
        raise ImportError("pyyaml required for run_experiment.py") from e
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------
def _maybe_nproc() -> int:
    """How many GPUs to torchrun across. Defaults to nvidia-smi count or 1."""
    n = os.environ.get("NPROC_PER_NODE")
    if n:
        return int(n)
    try:
        out = subprocess.check_output(["nvidia-smi", "--list-gpus"], text=True)
        return max(1, len(out.strip().splitlines()))
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 1


def _run_torchrun(extra_args: List[str], env: Optional[Dict[str, str]] = None,
                  cwd: Optional[str] = None) -> int:
    """Launch torchrun on the experiment training script. Streams stdout/stderr
    to this process's stdout so the foreground experience matches the user's
    request."""
    nproc = _maybe_nproc()
    master_port = os.environ.get("MASTER_PORT") or str(20000 + (int(time.time()) % 10000))
    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc}",
        f"--master_port={master_port}",
        "-m", "qwenvl.experiments.train",
    ] + extra_args
    print(f"[run_experiment] launching: {' '.join(shlex.quote(c) for c in cmd)}")
    full_env = dict(os.environ)
    full_env["PYTHONPATH"] = (
        os.path.join(_PROJECT_ROOT, "qwen-vl-finetune")
        + os.pathsep + full_env.get("PYTHONPATH", "")
    )
    if env:
        full_env.update(env)
    return subprocess.call(cmd, env=full_env, cwd=cwd or _PROJECT_ROOT)


def _run_eval(base_model: str, lora_path: str, eval_dir: str) -> int:
    """Single-stage v2 eval (existing sft_model_tester, called as subprocess).

    We rely on the existing eval path being correct; we never touch its code.
    """
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
    print(f"[run_experiment] eval: {' '.join(shlex.quote(c) for c in cmd)}")
    full_env = dict(os.environ)
    full_env["PYTHONPATH"] = (
        os.path.join(_PROJECT_ROOT, "qwen-vl-finetune")
        + os.pathsep + full_env.get("PYTHONPATH", "")
    )
    return subprocess.call(cmd, env=full_env, cwd=_PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Summary computation (§6 headline metrics)
# ---------------------------------------------------------------------------
def _per_cat_value(report_metrics: dict, cat: str, metric_key: str):
    m = report_metrics.get(cat) or {}
    return m.get(metric_key)


def _macro(report_metrics: dict, metric_key: str) -> Optional[float]:
    """Average a metric over categories with non-null values."""
    vals = []
    for cat in CURRICULUM_ORDER:
        v = _per_cat_value(report_metrics, cat, metric_key)
        if v is not None:
            vals.append(v)
    return sum(vals) / len(vals) if vals else None


def build_summary(
    cfg: ExpConfig,
    checkpoint_dirs: List[Tuple[str, str]],
    extra: Optional[dict] = None,
) -> dict:
    """Compute §6 headline metrics across a sequence of checkpoint eval_reports.

    Args:
        cfg: ExpConfig (used for run-level metadata).
        checkpoint_dirs: ordered list of (label, dir_path) — each dir must
            contain eval_report.json. Order matters for forgetting; the
            LAST entry is treated as "final".
        extra: optional run-level metadata to embed (e.g. total_steps used).

    Returns dict (the summary.json contents).
    """
    primary_metrics = ["answer_acc", "view_acc", "grounding_acc@0.8",
                       "grounding_format_valid"]

    # Build the checkpoint x category matrix per metric.
    matrix: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {
        m: {} for m in primary_metrics
    }
    macro_history: Dict[str, List[Tuple[str, Optional[float]]]] = {
        m: [] for m in primary_metrics
    }
    for label, ck_dir in checkpoint_dirs:
        report_path = os.path.join(ck_dir, "eval_report.json")
        if not os.path.exists(report_path):
            print(f"[summary] warn: missing {report_path}")
            for m in primary_metrics:
                matrix[m][label] = {c: None for c in CURRICULUM_ORDER}
                macro_history[m].append((label, None))
            continue
        with open(report_path) as f:
            rep = json.load(f)
        rep_metrics = rep.get("metrics") or {}
        for m in primary_metrics:
            row = {c: _per_cat_value(rep_metrics, c, m) for c in CURRICULUM_ORDER}
            matrix[m][label] = row
            macro_history[m].append((label, _macro(rep_metrics, m)))

    # final_macro per metric (last checkpoint).
    final_macro = {m: macro_history[m][-1][1] if macro_history[m] else None
                   for m in primary_metrics}

    # macro_forgetting per metric = mean over categories of (best_earlier - final),
    # where "earlier" excludes the final checkpoint.
    macro_forgetting: Dict[str, Optional[float]] = {}
    for m in primary_metrics:
        per_cat_forget = []
        if len(checkpoint_dirs) < 2:
            macro_forgetting[m] = None
            continue
        labels = [lbl for lbl, _ in checkpoint_dirs]
        final_label = labels[-1]
        for cat in CURRICULUM_ORDER:
            final_val = matrix[m][final_label].get(cat)
            if final_val is None:
                continue
            earlier_vals = [matrix[m][lbl].get(cat) for lbl in labels[:-1]]
            earlier_vals = [v for v in earlier_vals if v is not None]
            if not earlier_vals:
                continue
            best_earlier = max(earlier_vals)
            per_cat_forget.append(best_earlier - final_val)
        macro_forgetting[m] = (sum(per_cat_forget) / len(per_cat_forget)
                               if per_cat_forget else None)

    # collapse_indicator = min over checkpoints of (macro format_valid)
    fv_macros = [v for _, v in macro_history["grounding_format_valid"] if v is not None]
    collapse_indicator = min(fv_macros) if fv_macros else None

    summary = {
        "schema_version": "1",
        "exp_id": cfg.exp_id,
        "seed": cfg.seed,
        "mode": cfg.mode,
        "lr_schedule": cfg.lr_schedule,
        "grounding_floor": cfg.grounding_floor,
        "replay": cfg.replay,
        "sampler": cfg.sampler,
        "n_checkpoints": len(checkpoint_dirs),
        "checkpoint_labels": [lbl for lbl, _ in checkpoint_dirs],
        # Headline metrics (§6)
        "final_macro": final_macro,
        "macro_forgetting": macro_forgetting,
        "collapse_indicator": collapse_indicator,
        "final_macro_answer_acc": final_macro.get("answer_acc"),
        # Full matrix for downstream analysis
        "matrix": matrix,
        # Macro trajectory per metric
        "macro_history": {m: dict(h) for m, h in macro_history.items()},
    }
    if extra:
        summary["extra"] = extra
    return summary


# ---------------------------------------------------------------------------
# Mode-specific orchestration
# ---------------------------------------------------------------------------
def _resolve_paths(base_yaml: dict, cfg: ExpConfig) -> dict:
    """Pull data-side paths out of the base YAML. Same fields as curriculum_v2.yaml."""
    train_dir = base_yaml["train_data_dir"]
    eval_subset_dir = base_yaml["eval_subset_dir"]
    base_model = base_yaml["base_model"]
    return {
        "base_model": base_model,
        "train_dir": train_dir,
        "eval_dir": eval_subset_dir,
        # Per-category JSON files
        "train_jsons": {
            cat: os.path.join(train_dir, f"sft_train_qwen3vl_{cat}.json")
            for cat in CURRICULUM_ORDER
        },
        "eval_paths_json_per_stage_factory": lambda stage_dir: os.path.join(
            stage_dir, "eval_dataset_paths.json"
        ),
    }


def _write_eval_dataset_paths_json(stage_dir: str, eval_subset_dir: str) -> str:
    """Mirror what qwenvl.curriculum.config does — produce a {cat: path} JSON
    so v2 train's --eval_dataset_paths_json flag works."""
    paths = {
        cat: os.path.abspath(os.path.join(
            eval_subset_dir, f"sft_val_qwen3vl_{cat}_subset.json"
        ))
        for cat in CURRICULUM_ORDER
    }
    os.makedirs(stage_dir, exist_ok=True)
    out_path = os.path.join(stage_dir, "eval_dataset_paths.json")
    with open(out_path, "w") as f:
        json.dump(paths, f, indent=2)
    return out_path


def _common_train_args(base_yaml: dict, defaults: dict) -> List[str]:
    """Shared train flags pulled from curriculum_v2.yaml defaults."""
    d = defaults
    return [
        "--mode", "lora",
        "--per_device_train_batch_size", str(d["per_device_train_batch_size"]),
        "--per_device_eval_batch_size", str(d["per_device_eval_batch_size"]),
        "--gradient_accumulation_steps", str(d["gradient_accumulation_steps"]),
        "--max_pixels", str(d["max_pixels"]),
        "--min_pixels", str(d["min_pixels"]),
        "--max_assistant_tokens", str(d.get("max_assistant_tokens", 6000)),
        "--save_strategy", "steps",
        "--save_steps", str(d["save_steps"]),
        "--save_total_limit", str(d["save_total_limit"]),
        "--learning_rate", str(d["peak_lr"]),
        "--weight_decay", str(d["weight_decay"]),
        "--max_grad_norm", str(d["max_grad_norm"]),
        "--use_wsd_scheduler", "True",
        "--wsd_warmup_ratio", str(d["wsd"][0]),
        "--wsd_decay_ratio", str(d["wsd"][2]),
        "--lr_scheduler_type", "constant",
        "--logging_steps", "1",
        "--model_max_length", str(d["model_max_length"]),
        "--gradient_checkpointing", "true" if d.get("gradient_checkpointing", True) else "false",
        "--dataloader_num_workers", str(d.get("dataloader_num_workers", 4)),
        "--bf16",
        "--report_to", d.get("report_to", "tensorboard"),
        "--num_train_epochs", str(d.get("epochs", 1)),
        # Frozen composite-loss coefficients (passed through verbatim)
        "--w_ans", "2.0", "--w_gate", "1.5",
        "--lam_view", "0.5", "--lam_iou", "0.5", "--lam_klal", "0.0",
        "--klal_layers", "-1", "--iou_alpha", "1.0",
        "--view_class_weight", "auto",
        # Eval-time intra-training eval disabled — orchestrator runs gen-eval
        # explicitly between stages instead.
        "--eval_strategy", "no",
    ]


def _train_args_for_lr_schedule(
    cfg: ExpConfig,
    total_steps: int,
    stage_idx: Optional[int] = None,
    stage_grounded_fraction: Optional[float] = None,
) -> List[str]:
    """Convert ExpConfig.lr_schedule + total_steps into the right train.py flags.

    Per spec §2.4:
    - global_wsd uses warmup_frac=0.03, decay_frac=0.10 of TOTAL steps (NOT the
      YAML's per-stage [0.10, 0.20] defaults). Overrides the upstream flags
      that v2 ordinarily reads from the YAML.
    - per_stage_wsd_relaxed needs stage_idx + stage_grounded_fraction for the
      stage-0-warmup and sparse-peak special cases.
    """
    args = ["--lr_schedule_kind", cfg.lr_schedule]
    if cfg.lr_schedule == "global_wsd":
        args += [
            "--global_total_steps", str(total_steps),
            "--wsd_warmup_ratio", "0.03",
            "--wsd_decay_ratio", "0.10",
        ]
    if cfg.lr_schedule == "per_stage_wsd_relaxed":
        if stage_idx is not None:
            args += ["--stage_idx", str(stage_idx)]
        if stage_grounded_fraction is not None:
            args += ["--stage_grounded_fraction", str(stage_grounded_fraction)]
    return args


def _train_args_for_gf_er(
    cfg: ExpConfig,
    prior_data_paths: List[str],
    global_grounded_paths: List[str],
) -> List[str]:
    """GF + ER args. Defaults preserve B0 behavior bit-for-bit.

    Per spec §2.2, GF's pool is GLOBAL: the orchestrator passes all 10
    category JSONs as `--global_grounded_data_paths` so the trainer can
    build the union-of-grounded pool used for the GF top-up draw.

    Per spec §2.3 (`buffer: prior_stages — sequential: categories 0..i-1;
    stage 0 = no replay`): at the first sequential stage there is no prior
    pool to draw from, so the replay args are skipped entirely regardless
    of what the user passed on the CLI. This is a no-op for that stage —
    the loop reactivates replay at stage 1 once prior_data_paths is
    non-empty.
    """
    args = []
    if cfg.grounding_floor is not None:
        args += ["--grounding_floor", str(cfg.grounding_floor)]
        if global_grounded_paths:
            args += ["--global_grounded_data_paths", ",".join(global_grounded_paths)]
    if cfg.replay is not None:
        if not prior_data_paths:
            # Spec §2.3 — first sequential stage gets no replay. We do not
            # pass --replay_fraction so the trainer treats this as the
            # no-ER fast-path on its end.
            print("[run_experiment] replay=on but no prior stages yet "
                  "(first sequential stage); skipping --replay_fraction "
                  "per spec §2.3 ('stage 0 = no replay'). Replay kicks in "
                  "from stage 1 onward.")
        else:
            args += ["--replay_fraction", str(cfg.replay)]
            args += ["--prior_data_paths", ",".join(prior_data_paths)]
    return args


def _estimate_stage_steps(stage_data_path: str, defaults: dict) -> int:
    """Estimate optimizer steps for a stage from samples + grad_accum + epochs."""
    with open(stage_data_path) as f:
        n = len(json.load(f))
    eff_batch = defaults["per_device_train_batch_size"] * defaults["gradient_accumulation_steps"]
    nproc = _maybe_nproc()
    eff_global = eff_batch * nproc
    epochs = defaults.get("epochs", 1)
    return max(1, (n * epochs + eff_global - 1) // eff_global)


def _stage_grounded_fraction(stage_data_path: str) -> float:
    """Compute the grounded sample fraction for a stage's training file.

    Used by per_stage_wsd_relaxed to decide the sparse-stage peak factor
    per spec §2.4. A stage is "sparse" if this fraction < 0.15."""
    with open(stage_data_path) as f:
        data = json.load(f)
    if not data:
        return 0.0
    n_g = 0
    for s in data:
        try:
            obj = json.loads(s["conversations"][2]["value"])
            if obj.get("grounding"):
                n_g += 1
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            pass
    return n_g / len(data)


def run_sequential(cfg: ExpConfig, base_yaml: dict) -> int:
    """Loop the 10 stages OBS->...->CHR. Warm-start each stage from the
    previous adapter. Apply GF/ER/lr knobs."""
    paths = _resolve_paths(base_yaml, cfg)
    defaults = base_yaml["defaults"]

    stage_steps: List[int] = []
    for cat in CURRICULUM_ORDER:
        stage_steps.append(_estimate_stage_steps(paths["train_jsons"][cat], defaults))
    total_steps = sum(stage_steps)
    if cfg.max_steps:
        total_steps = min(total_steps, cfg.max_steps)
        print(f"[run_experiment] --max_steps={cfg.max_steps} -> capping total {total_steps}")

    print(f"[run_experiment] estimated per-stage steps: {dict(zip(CURRICULUM_ORDER, stage_steps))}")
    print(f"[run_experiment] sequential total optimizer steps: {total_steps}")

    cfg.write_manifest(os.path.join(cfg.run_dir, "run_manifest.json"))

    checkpoint_dirs: List[Tuple[str, str]] = []
    prior_warm_start: Optional[str] = None
    prior_data_paths: List[str] = []
    for stage_idx, cat in enumerate(CURRICULUM_ORDER):
        stage_dir = os.path.join(cfg.run_dir, f"stage_{stage_idx:02d}_{cat}")
        os.makedirs(stage_dir, exist_ok=True)
        eval_json_path = _write_eval_dataset_paths_json(stage_dir, paths["eval_dir"])

        max_steps_this_stage = stage_steps[stage_idx]
        if cfg.max_steps:
            remaining = cfg.max_steps - sum(stage_steps[:stage_idx])
            if remaining <= 0:
                print(f"[run_experiment] --max_steps exhausted before stage {cat}; stopping early.")
                break
            max_steps_this_stage = min(max_steps_this_stage, remaining)

        train_args = (
            _common_train_args(base_yaml, defaults)
            + [
                "--model_name_or_path", paths["base_model"],
                "--train_data_path", paths["train_jsons"][cat],
                "--eval_dataset_paths_json", eval_json_path,
                "--output_dir", stage_dir,
                "--max_steps", str(max_steps_this_stage),
                "--seed", str(cfg.seed),
                "--lora_r", str(defaults["lora_r"]),
                "--lora_alpha", str(defaults["lora_alpha"]),
                "--lora_dropout", str(defaults["lora_dropout"]),
                "--lora_target_modules", defaults["lora_target_modules"],
                "--run_name", f"{cfg.exp_id}-seed{cfg.seed}-stage{stage_idx:02d}-{cat}",
            ]
            + _train_args_for_lr_schedule(
                cfg, total_steps,
                stage_idx=stage_idx,
                stage_grounded_fraction=_stage_grounded_fraction(paths["train_jsons"][cat]),
            )
            + _train_args_for_gf_er(
                cfg, prior_data_paths,
                global_grounded_paths=[paths["train_jsons"][c] for c in CURRICULUM_ORDER],
            )
        )
        if prior_warm_start:
            train_args += ["--lora_pretrained", prior_warm_start]

        rc = _run_torchrun(train_args)
        if rc != 0:
            print(f"[run_experiment] stage {cat} training FAILED with rc={rc}")
            return rc

        if not cfg.skip_eval:
            rc = _run_eval(paths["base_model"], stage_dir, paths["eval_dir"])
            if rc != 0:
                print(f"[run_experiment] stage {cat} eval FAILED with rc={rc}")
                # Continue — partial eval is still useful for summary

        checkpoint_dirs.append((f"stage_{stage_idx:02d}_{cat}", stage_dir))
        prior_warm_start = stage_dir
        prior_data_paths.append(paths["train_jsons"][cat])

    if cfg.skip_eval:
        print(f"[run_experiment] --skip_eval: training done. Run "
              f"`qwenvl.experiments.eval_run --run_dir {cfg.run_dir}` to "
              f"do the eval pass and produce summary.json.")
        return 0

    summary = build_summary(
        cfg, checkpoint_dirs,
        extra={"total_steps": total_steps,
               "per_stage_steps": dict(zip(CURRICULUM_ORDER, stage_steps))},
    )
    summary_path = os.path.join(cfg.run_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[run_experiment] sequential summary -> {summary_path}")
    return 0


def run_mixed(cfg: ExpConfig, base_yaml: dict) -> int:
    """One continuous run over all 10 categories with the uniform sampler.

    Same total optimizer steps as the sequential baseline. Checkpoints at
    10 evenly-spaced step fractions (10%..100%); each is evaluated against
    all 10 categories."""
    paths = _resolve_paths(base_yaml, cfg)
    defaults = base_yaml["defaults"]

    # Compute total steps as the sequential sum.
    stage_steps = [
        _estimate_stage_steps(paths["train_jsons"][cat], defaults)
        for cat in CURRICULUM_ORDER
    ]
    total_steps = sum(stage_steps)
    if cfg.max_steps:
        total_steps = min(total_steps, cfg.max_steps)
        print(f"[run_experiment] --max_steps={cfg.max_steps} -> capping total {total_steps}")
    print(f"[run_experiment] mixed total optimizer steps: {total_steps}")

    cfg.write_manifest(os.path.join(cfg.run_dir, "run_manifest.json"))

    fractions = [round(f, 2) for f in [0.1 * i for i in range(1, 11)]]
    fractions_str = ",".join(str(f) for f in fractions)
    mix_paths = [paths["train_jsons"][c] for c in CURRICULUM_ORDER]

    train_args = (
        _common_train_args(base_yaml, defaults)
        + [
            "--model_name_or_path", paths["base_model"],
            # train_data_path is unused when mode_mixed=True but required by
            # HF arg parser; point it at OBS to keep the parser happy.
            "--train_data_path", paths["train_jsons"]["OBS"],
            "--output_dir", cfg.run_dir,
            "--max_steps", str(total_steps),
            "--seed", str(cfg.seed),
            "--lora_r", str(defaults["lora_r"]),
            "--lora_alpha", str(defaults["lora_alpha"]),
            "--lora_dropout", str(defaults["lora_dropout"]),
            "--lora_target_modules", defaults["lora_target_modules"],
            "--run_name", f"{cfg.exp_id}-seed{cfg.seed}-mixed",
            "--mode_mixed", "True",
            "--mixed_category_data_paths", ",".join(mix_paths),
            "--checkpoint_step_fractions", fractions_str,
            "--lr_schedule_kind", cfg.lr_schedule,   # always global_wsd for mixed
            "--global_total_steps", str(total_steps),
        ]
        + _train_args_for_gf_er(
            cfg, [],
            global_grounded_paths=[paths["train_jsons"][c] for c in CURRICULUM_ORDER],
        )
    )

    rc = _run_torchrun(train_args)
    if rc != 0:
        print(f"[run_experiment] mixed training FAILED with rc={rc}")
        return rc

    # Eval each ckpt_NNN/.
    checkpoint_dirs: List[Tuple[str, str]] = []
    for f in fractions:
        label = f"ckpt_{int(round(f*100)):03d}"
        ck_dir = os.path.join(cfg.run_dir, label)
        if not os.path.isdir(ck_dir):
            print(f"[run_experiment] missing checkpoint dir {ck_dir}; skipping eval")
            continue
        if not cfg.skip_eval:
            rc = _run_eval(paths["base_model"], ck_dir, paths["eval_dir"])
            if rc != 0:
                print(f"[run_experiment] eval at {label} FAILED with rc={rc}")
        checkpoint_dirs.append((label, ck_dir))

    if cfg.skip_eval:
        print(f"[run_experiment] --skip_eval: training done. Run "
              f"`qwenvl.experiments.eval_run --run_dir {cfg.run_dir}` to "
              f"do the eval pass and produce summary.json.")
        return 0

    summary = build_summary(
        cfg, checkpoint_dirs,
        extra={"total_steps": total_steps, "step_fractions": fractions},
    )
    summary_path = os.path.join(cfg.run_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[run_experiment] mixed summary -> {summary_path}")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    cfg = parse_cli(argv)
    cfg.print_resolved()

    if cfg.dry_run:
        print("\n--dry_run: not training. Resolved config above; no files written.")
        return 0

    base_yaml = load_base_yaml(cfg.config)
    os.makedirs(cfg.run_dir, exist_ok=True)

    if cfg.mode == "sequential":
        return run_sequential(cfg, base_yaml)
    if cfg.mode == "mixed":
        return run_mixed(cfg, base_yaml)
    raise ValueError(f"Unknown mode: {cfg.mode}")


if __name__ == "__main__":
    sys.exit(main())
