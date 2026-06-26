"""Argument-driven experiment configuration — no preset table.

A single experiment is fully described by:

    mode             : 'sequential' | 'mixed'
    lr_schedule      : 'per_stage_wsd' | 'global_wsd' | 'per_stage_wsd_relaxed'
    grounding_floor  : float in (0, 1) | None   (None means disabled)
    replay           : float in (0, 1) | None
    sampler          : 'uniform' | 'difficulty_weighted'
                       (difficulty_weighted is deferred -> NotImplementedError)
    exp_id           : free-form label for output dir naming
    seed             : int
    output_root      : path
    config           : path to base curriculum_v2.yaml
    max_steps        : optional cap for smoke tests
    dry_run          : if True, print resolved config and exit

CLI args are the single source of truth. The base YAML supplies upstream
defaults (e.g. WSD ratios) that the knobs inherit when not overridden, but
the knob VALUES come from the CLI only — no preset lookup, no static
registry of canonical (B0/F1/...) flag combinations.

The 16 canonical flag combinations live as a reference TABLE in
`curriculum_v2_ablation_experiments.md` at the project root. To run B0 the
user types out B0's flag set; the canonical labels exist only so humans can
talk about them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Optional


VALID_MODES = ("sequential", "mixed")
VALID_LR_SCHEDULES = ("per_stage_wsd", "global_wsd", "per_stage_wsd_relaxed")
VALID_SAMPLERS = ("uniform", "difficulty_weighted")


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def _parse_off_or_float(raw: Optional[str], name: str) -> Optional[float]:
    """Accept the literal 'off' (case-insensitive) or a float in (0, 1).

    Returns None for 'off' so downstream code does a clean `if cfg.grounding_floor:`
    rather than special-casing a sentinel string.
    """
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s == "off":
        return None
    try:
        v = float(s)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"--{name} must be 'off' or a float in (0, 1); got {raw!r}"
        ) from e
    if not (0.0 < v < 1.0):
        raise argparse.ArgumentTypeError(
            f"--{name} float must be strictly in (0, 1); got {v}"
        )
    return v


def _grounding_floor_type(raw):
    return _parse_off_or_float(raw, "grounding_floor")


def _replay_type(raw):
    return _parse_off_or_float(raw, "replay")


# ---------------------------------------------------------------------------
# Resolved experiment config
# ---------------------------------------------------------------------------
@dataclass
class ExpConfig:
    """The fully-resolved knob set for one experiment run.

    Note: `grounding_floor` and `replay` are stored as Optional[float] —
    None means "disabled". Manifest serialization writes them back as 'off'
    so the on-disk JSON matches the CLI surface.
    """
    # Required CLI knobs
    exp_id: str
    mode: str
    lr_schedule: str
    grounding_floor: Optional[float]
    replay: Optional[float]
    sampler: str
    output_root: str
    seed: int
    config: str
    max_steps: Optional[int] = None
    dry_run: bool = False
    # When True, the orchestrator runs training only — no per-stage /
    # per-checkpoint sft_model_tester calls, and no summary.json is written
    # (since the summary needs eval data). Use `qwenvl.experiments.eval_run`
    # afterward to do the eval pass and produce summary.json.
    skip_eval: bool = False

    # T3 — view-stratified grounding-floor sampling. Only meaningful when
    # grounding_floor is enabled.
    view_stratified: bool = False
    view_stratified_min_bucket: int = 50
    view_stratified_view_cap: float = 2.0

    # Frozen loss config — surfaced in the manifest for reproducibility.
    # These values mirror curriculum_v2.yaml:loss and are NEVER varied.
    # Updating them in train_nuscenes_qwen3vl_v2.py without updating here
    # is a bug; the acceptance criterion "B0 reproduces baseline bit-for-bit"
    # depends on this constant.
    frozen_loss: dict = field(default_factory=lambda: {
        "w_ans": 2.0,
        "w_gate": 1.5,
        "lam_view": 0.5,
        "lam_iou": 0.5,
        "lam_klal": 0.0,
        "klal_layers": "-1",
        "iou_alpha": 1.0,
        "view_class_weight": "auto",
    })

    # ---- Validation
    def __post_init__(self):
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"mode must be one of {VALID_MODES}; got {self.mode!r}"
            )
        if self.lr_schedule not in VALID_LR_SCHEDULES:
            raise ValueError(
                f"lr_schedule must be one of {VALID_LR_SCHEDULES}; got {self.lr_schedule!r}"
            )
        if self.sampler not in VALID_SAMPLERS:
            raise ValueError(
                f"sampler must be one of {VALID_SAMPLERS}; got {self.sampler!r}"
            )

        # Mixed-mode coherence check — the spec has no per_stage variants in
        # mixed mode (one continuous run -> only global_wsd makes sense).
        if self.mode == "mixed" and self.lr_schedule != "global_wsd":
            raise ValueError(
                f"mode='mixed' requires lr_schedule='global_wsd' (mixed runs "
                f"are one continuous schedule, so per_stage_wsd / "
                f"per_stage_wsd_relaxed have no meaning here). "
                f"Got lr_schedule={self.lr_schedule!r}."
            )

        # Difficulty-weighted sampler is the Appendix-A deferred follow-up.
        if self.sampler == "difficulty_weighted":
            raise NotImplementedError(
                "sampler='difficulty_weighted' is the deferred follow-up "
                "(Appendix A of curriculum_v2_ablation_experiments.md). Its "
                "mixture weights are derived from the results of B0 and F3 "
                "runs that do not exist yet. Use --sampler uniform until "
                "those results are in."
            )

        if not self.exp_id:
            raise ValueError(
                "exp_id is required (free-form label used for output dir naming). "
                "Use 'B0', 'F1', etc. for canonical runs, or any other label "
                "for ad-hoc experiments."
            )
        if not self.output_root:
            raise ValueError("output_root is required.")
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0; got {self.seed}")

    @property
    def run_dir(self) -> str:
        """Per-run output directory: <output_root>/<exp_id>__seed<seed>/."""
        return os.path.join(self.output_root, f"{self.exp_id}__seed{self.seed}")

    # ---- Serialization
    def to_manifest_dict(self) -> dict:
        """Manifest representation: floats stay floats; None becomes 'off' so
        the JSON matches the CLI surface."""
        d = asdict(self)
        d["grounding_floor"] = "off" if self.grounding_floor is None else self.grounding_floor
        d["replay"] = "off" if self.replay is None else self.replay
        return d

    def write_manifest(self, path: str) -> None:
        """Write run_manifest.json so the run can be reproduced from this file
        alone. Adds a small metadata block alongside the resolved knobs."""
        manifest = {
            "schema_version": "1",
            "resolved_config": self.to_manifest_dict(),
            "cli_argv": list(sys.argv),
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)

    def print_resolved(self, stream=None) -> None:
        """Pretty-print the resolved config (used by --dry_run)."""
        stream = stream or sys.stdout
        d = self.to_manifest_dict()
        # Pop bulky dict for compact display, then re-emit at the bottom.
        frozen = d.pop("frozen_loss")
        print("=" * 60, file=stream)
        print("RESOLVED EXPERIMENT CONFIG", file=stream)
        print("=" * 60, file=stream)
        for k, v in d.items():
            print(f"  {k:>16} = {v!r}", file=stream)
        print(f"  {'frozen_loss':>16} = (held constant across all experiments)", file=stream)
        for k, v in frozen.items():
            print(f"    {k:>14} = {v!r}", file=stream)
        print(f"  {'run_dir':>16} = {self.run_dir}", file=stream)
        print("=" * 60, file=stream)


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m qwenvl.experiments.run_experiment",
        description=(
            "Single-experiment ablation launcher for curriculum-v2. Every "
            "knob is set via CLI args; no preset table. See "
            "curriculum_v2_ablation_experiments.md for the 16 canonical "
            "flag combinations."
        ),
    )
    p.add_argument(
        "--exp_id", required=True, type=str,
        help="Free-form label used for output dir naming "
             "(<output_root>/<exp_id>__seed<seed>/). Use 'B0', 'F1', etc. for "
             "canonical experiments, or any string for ad-hoc runs.",
    )
    p.add_argument(
        "--mode", required=True, choices=VALID_MODES,
        help="Training topology. 'sequential' = 10 stages OBS->...->CHR with "
             "warm-start. 'mixed' = one continuous run over all 10 categories.",
    )
    p.add_argument(
        "--lr_schedule", required=True, choices=VALID_LR_SCHEDULES,
        help="LR schedule variant. per_stage_wsd = baseline (one WSD cycle per "
             "stage, decays to 0 between stages). global_wsd = one continuous WSD "
             "across all stages. per_stage_wsd_relaxed = per-stage WSD with "
             "shorter warmup/decay (INFERENCE: [0.05, 0.85, 0.10]; see spec).",
    )
    p.add_argument(
        "--grounding_floor", default="off", type=_grounding_floor_type,
        metavar="off|FLOAT",
        help="GF: minimum fraction of grounding-bearing samples per training step. "
             "'off' = disabled (baseline). Float in (0,1) enables — e.g. 0.25.",
    )
    p.add_argument(
        "--replay", default="off", type=_replay_type,
        metavar="off|FLOAT",
        help="ER: fraction of samples drawn from PRIOR stages (sequential only). "
             "'off' = disabled (baseline). Float in (0,1) enables — e.g. 0.10.",
    )
    p.add_argument(
        "--sampler", default="uniform", choices=VALID_SAMPLERS,
        help="Mixed-mode sampler. uniform = default. difficulty_weighted = the "
             "Appendix-A deferred follow-up (raises NotImplementedError).",
    )
    p.add_argument(
        "--output_root", required=True, type=str,
        help="Results land in <output_root>/<exp_id>__seed<seed>/.",
    )
    p.add_argument("--seed", default=0, type=int)
    p.add_argument(
        "--config", default="qwen-vl-finetune/configs/curriculum_v2.yaml", type=str,
        help="Path to the base curriculum_v2.yaml (provides frozen loss + WSD "
             "defaults that the knobs build on).",
    )
    p.add_argument(
        "--max_steps", default=None, type=int,
        help="Optional cap on total optimizer steps (smoke tests).",
    )
    p.add_argument(
        "--dry_run", action="store_true",
        help="Print the fully-resolved config and exit without training.",
    )
    p.add_argument(
        "--skip_eval", action="store_true",
        help="Training only — skip per-stage / per-checkpoint eval and "
             "summary.json. Run `qwenvl.experiments.eval_run --run_dir <...>` "
             "afterward to do the eval pass separately.",
    )
    # T3 — view-stratified GF (load-bearing: caps prevent sparse-view overfit
    # from polluting the data-only-vs-T2 gating decision).
    p.add_argument(
        "--view_stratified", action="store_true",
        help="T3: stratify the GF top-up pool per view (image_idx 1..6) "
             "instead of per category only. No effect if --grounding_floor is off.",
    )
    p.add_argument(
        "--view_stratified_min_bucket", default=50, type=int,
        help="T3: drop (view, cat) buckets smaller than this so sparse buckets "
             "can't be sampled hundreds of times per epoch. Default 50.",
    )
    p.add_argument(
        "--view_stratified_view_cap", default=2.0, type=float,
        help="T3: cap any view's GF selection probability at cap/6. Default 2.0 "
             "(= 33%% per view ceiling).",
    )
    return p


def parse_cli(argv=None) -> ExpConfig:
    """Parse argv into a validated ExpConfig.

    Resolution order is implicit and trivial because there's no preset
    layer: argparse handles its own defaults; --grounding_floor / --replay
    default to 'off' (parsed to None). The base YAML (--config) supplies
    frozen loss defaults that propagate into train_nuscenes_qwen3vl_v2.py
    independently of the knobs here.
    """
    args = build_argparser().parse_args(argv)
    return ExpConfig(
        exp_id=args.exp_id,
        mode=args.mode,
        lr_schedule=args.lr_schedule,
        grounding_floor=args.grounding_floor,
        replay=args.replay,
        sampler=args.sampler,
        output_root=args.output_root,
        seed=args.seed,
        config=args.config,
        max_steps=args.max_steps,
        dry_run=args.dry_run,
        skip_eval=args.skip_eval,
        view_stratified=args.view_stratified,
        view_stratified_min_bucket=args.view_stratified_min_bucket,
        view_stratified_view_cap=args.view_stratified_view_cap,
    )
