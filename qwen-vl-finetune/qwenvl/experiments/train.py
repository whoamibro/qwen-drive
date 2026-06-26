"""Experiment training entry point.

This is a NEW training script for the ablation framework. It imports the
existing curriculum-v2 classes (NuScenesVQADatasetV2, NuScenesDataCollatorV2,
CompositeWSDTrainer, ...) from `train_nuscenes_qwen3vl_v2.py` as a library —
NO modification to that file or any other file in the curriculum-v2 command's
code path.

What this script adds on top of v2:

1. **New CLI flags** (all default to no-op so a vanilla call mirrors v2):
   - `--grounding_floor FLOAT|off`     (GF)
   - `--replay_fraction FLOAT|0`       (ER)
   - `--prior_data_paths PATH[,PATH]`  (the prior-stage train JSONs for ER)
   - `--lr_schedule_kind {per_stage_wsd, global_wsd, per_stage_wsd_relaxed}`
   - `--global_total_steps INT`        (only used by global_wsd)
   - `--mode_mixed BOOL`               (mixed-mode toggle)
   - `--mixed_category_data_paths PATH[,PATH]` (when mode_mixed=True)
   - `--checkpoint_step_fractions FLOAT[,FLOAT]` (mixed-mode checkpoints)

2. **A custom Trainer subclass** that picks the right LR schedule via the
   `--lr_schedule_kind` flag, and (for mixed mode) saves the LoRA adapter at
   user-specified step fractions via a callback.

3. **A custom dataset wrapper** that produces a torch ConcatDataset of
   (current + prior) and a §5.2-compliant Sampler. When both GF and ER are
   off, the wrapper returns the original v2 dataset unchanged, so B0
   reproduces the v2 baseline bit-for-bit.

Usage is mediated by `qwenvl.experiments.run_experiment` — humans normally
don't invoke this script directly. But it accepts standard torchrun launch
so debugging single-stage invocations is straightforward.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import transformers
from torch.utils.data import ConcatDataset
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Trainer,
    TrainingArguments as HfTrainingArguments,
    TrainerCallback,
)

# --- Import the existing v2 building blocks. We never modify them. -----
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.insert(0, _PROJECT_ROOT)

import train_nuscenes_qwen3vl_v2 as v2   # noqa: E402

# --- Experiment-side helpers (this package) ----------------------------
from qwenvl.experiments.lr_schedules import (   # noqa: E402
    get_global_wsd_schedule,
    get_per_stage_wsd_relaxed_schedule,
)
from qwenvl.experiments.samplers import build_sampler   # noqa: E402

# --- Pull through the v2 essentials so we don't re-import them downstream
IGNORE_INDEX = v2.IGNORE_INDEX
rank0_print = v2.rank0_print


# ===========================================================================
# Extended arg dataclasses
# ===========================================================================
@dataclass
class ExpModelArguments(v2.ModelArguments):
    pass


@dataclass
class ExpDataArguments(v2.DataArguments):
    grounding_floor: float = field(
        default=-1.0,
        metadata={"help": "GF target fraction in (0,1); -1 = disabled."},
    )
    replay_fraction: float = field(
        default=0.0,
        metadata={"help": "ER probability of drawing from the prior pool; 0 = disabled."},
    )
    prior_data_paths: str = field(
        default="",
        metadata={"help": "Comma-separated JSON paths for the prior-stage training "
                          "data (used when replay_fraction > 0)."},
    )
    global_grounded_data_paths: str = field(
        default="",
        metadata={"help": "Comma-separated JSON paths covering all 10 categories. "
                          "Used to build the GLOBAL_GROUNDED pool per spec §2.2 "
                          "(pool: global) when grounding_floor > 0."},
    )
    # T3 — view-stratified grounding floor.
    view_stratified: bool = field(
        default=False,
        metadata={"help": "T3: stratify the GF top-up pool per image_idx (1..6). "
                          "Reuses per-category-uniform two-stage sampler pattern, "
                          "keyed on view. Default off."},
    )
    view_stratified_min_bucket: int = field(
        default=50,
        metadata={"help": "T3: drop (view, cat) buckets smaller than this so a "
                          "handful of rear-view samples can't be sampled hundreds "
                          "of times per epoch."},
    )
    view_stratified_view_cap: float = field(
        default=2.0,
        metadata={"help": "T3: cap any view's GF selection probability at cap/6. "
                          "Excess re-routes to uncapped views; falls back to "
                          "uniform-among-populated when cap can't be respected."},
    )
    # Mixed-mode knobs
    mode_mixed: bool = field(default=False)
    mixed_category_data_paths: str = field(
        default="",
        metadata={"help": "Comma-separated JSON paths covering all 10 categories. "
                          "Required when mode_mixed=True."},
    )


@dataclass
class ExpTrainingArguments(v2.TrainingArguments):
    lr_schedule_kind: str = field(
        default="per_stage_wsd",
        metadata={"help": "per_stage_wsd | global_wsd | per_stage_wsd_relaxed"},
    )
    global_total_steps: int = field(
        default=-1,
        metadata={"help": "Total optimizer steps for global_wsd (sum of per-stage budgets)."},
    )
    # Spec §2.4 per_stage_wsd_relaxed inputs: stage_idx=0 keeps a 0.10 warmup;
    # stage_grounded_fraction < SPARSE_STAGE_THRESHOLD triggers the peak factor.
    stage_idx: int = field(
        default=0,
        metadata={"help": "Sequential stage index (0..9). Used by "
                          "per_stage_wsd_relaxed to decide stage-0 warmup."},
    )
    stage_grounded_fraction: float = field(
        default=-1.0,
        metadata={"help": "Grounded sample fraction of this stage's training data. "
                          "If < 0.15, per_stage_wsd_relaxed applies peak factor 0.3. "
                          "-1 (default) = unknown -> treated as not sparse."},
    )
    checkpoint_step_fractions: str = field(
        default="",
        metadata={"help": "Comma-separated step fractions (e.g. '0.1,0.2,...,1.0') "
                          "for mixed-mode adapter snapshots. Empty = no extra "
                          "snapshots beyond save_strategy='steps'."},
    )


# ===========================================================================
# Custom Trainer: picks LR schedule + mixed-mode adapter snapshots
# ===========================================================================
class StepFractionCheckpointCallback(TrainerCallback):
    """Saves the LoRA adapter at a specified set of step fractions of
    `state.max_steps`. Used for mixed-mode: ckpt_010/, ckpt_020/, ..., ckpt_100/.

    Saving is rank-0 only, mirrors trainer.save_pretrained behavior, and is
    independent of HF Trainer's regular `save_strategy='steps'` checkpointing.
    """

    def __init__(self, fractions: List[float], output_root: str, trainer_ref):
        super().__init__()
        self.fractions = sorted(set(fractions))
        self.output_root = output_root
        self._trainer_ref = trainer_ref
        self._fired = set()

    def on_step_end(self, args, state, control, **kwargs):
        if not self.fractions or state.max_steps <= 0:
            return
        cur = state.global_step
        for f in self.fractions:
            if f in self._fired:
                continue
            target = max(1, int(round(f * state.max_steps)))
            if cur >= target:
                trainer = self._trainer_ref()
                if trainer is None or not trainer.is_world_process_zero():
                    self._fired.add(f)
                    continue
                sub = os.path.join(self.output_root, f"ckpt_{int(round(f*100)):03d}")
                os.makedirs(sub, exist_ok=True)
                trainer.model.save_pretrained(sub)
                rank0_print(f"[StepFractionCallback] saved ckpt_{int(round(f*100)):03d} "
                            f"at step {cur}/{state.max_steps} -> {sub}")
                self._fired.add(f)


class ExperimentTrainer(v2.CompositeWSDTrainer):
    """v2 CompositeWSDTrainer + alternative LR schedules.

    When `lr_schedule_kind == 'per_stage_wsd'` this is byte-identical to
    v2 (the v2 path returns the standard WSD via `super().create_scheduler`).
    Otherwise we plug in the global or relaxed variant from
    `qwenvl.experiments.lr_schedules`.
    """

    def __init__(self, *args, lr_schedule_kind="per_stage_wsd",
                 global_total_steps=-1, **kw):
        super().__init__(*args, **kw)
        self.lr_schedule_kind = lr_schedule_kind
        self.global_total_steps = global_total_steps

    def create_scheduler(self, num_training_steps, optimizer=None):
        kind = self.lr_schedule_kind
        if kind == "per_stage_wsd":
            return super().create_scheduler(num_training_steps, optimizer)
        if self.lr_scheduler is not None:
            return self.lr_scheduler
        opt = optimizer if optimizer is not None else self.optimizer
        if kind == "global_wsd":
            total = self.global_total_steps if self.global_total_steps > 0 \
                    else num_training_steps
            # Spec §2.4 pins global_wsd to warmup_frac=0.03, decay_frac=0.10 of
            # TOTAL steps. get_global_wsd_schedule hard-codes these — we don't
            # forward the upstream YAML ratios (which are per-stage WSD's
            # 0.10/0.20 baseline) because they have different semantics.
            self.lr_scheduler = get_global_wsd_schedule(
                opt,
                total_training_steps=total,
            )
            from qwenvl.experiments.lr_schedules import (
                GLOBAL_WSD_WARMUP_FRAC, GLOBAL_WSD_DECAY_FRAC,
            )
            rank0_print(f"[ExperimentTrainer] global_wsd: total_steps={total} "
                        f"warmup_frac={GLOBAL_WSD_WARMUP_FRAC} "
                        f"decay_frac={GLOBAL_WSD_DECAY_FRAC}")
        elif kind == "per_stage_wsd_relaxed":
            stage_idx = int(getattr(self.args, "stage_idx", 0))
            gf = float(getattr(self.args, "stage_grounded_fraction", -1.0))
            gf_arg = gf if gf >= 0 else None
            self.lr_scheduler = get_per_stage_wsd_relaxed_schedule(
                opt,
                num_training_steps=num_training_steps,
                is_first_stage=(stage_idx == 0),
                stage_grounded_fraction=gf_arg,
            )
            from qwenvl.experiments.lr_schedules import (
                SPARSE_STAGE_THRESHOLD, SPARSE_STAGE_PEAK_FACTOR,
                PER_STAGE_RELAXED_STAGE0_WARMUP, PER_STAGE_RELAXED_RATIOS,
            )
            warmup = PER_STAGE_RELAXED_STAGE0_WARMUP if stage_idx == 0 else PER_STAGE_RELAXED_RATIOS[0]
            sparse = gf_arg is not None and gf_arg < SPARSE_STAGE_THRESHOLD
            rank0_print(f"[ExperimentTrainer] per_stage_wsd_relaxed: "
                        f"num_steps={num_training_steps} stage_idx={stage_idx} "
                        f"warmup={warmup} decay={PER_STAGE_RELAXED_RATIOS[2]} "
                        f"grounded_frac={gf_arg} sparse={sparse} "
                        f"peak_factor={SPARSE_STAGE_PEAK_FACTOR if sparse else 1.0}")
        else:
            raise ValueError(f"Unknown lr_schedule_kind={kind!r}")
        return self.lr_scheduler


# ===========================================================================
# Dataset wrapper — combines current + prior, builds sampler
# ===========================================================================
def _read_samples(path: str) -> List[dict]:
    with open(path) as f:
        return json.load(f)


def _filter_grounded(samples) -> List[dict]:
    """Keep only samples with non-empty grounding."""
    from qwenvl.experiments.samplers import compute_grounding_flags
    flags = compute_grounding_flags(samples)
    return [s for s, f in zip(samples, flags) if f]


_PATH_CAT_RE = __import__("re").compile(r"sft_train_qwen3vl_([A-Z]+)\.json$")


def _cat_from_path(path: str) -> str:
    """Recover the category code (e.g. 'OBS') from a train JSON path. Used
    so the sampler can do per-category uniform within each pool."""
    m = _PATH_CAT_RE.search(os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)


def _build_concat_dataset_and_sampler(
    train_paths_current: List[str],
    train_paths_prior: List[str],
    train_paths_global_grounded: List[str],
    tokenizer,
    image_processor,
    data_args: ExpDataArguments,
    training_args: ExpTrainingArguments,
    num_training_samples: int,
    seed: int,
):
    """Build the flat ConcatDataset + the per-category-uniform sampler.

    Per spec §2.1 / §2.2 / §2.3 / §5.2: each pool (CURRENT, PRIOR,
    GLOBAL_GROUNDED) is composed of per-category sub-ranges. The sampler
    does two-stage uniform within each pool (pick category uniformly, then
    a sample uniformly within that category).

    B0 fast-path (sampler=None) requires single-category CURRENT and no
    GF / no ER. Multi-category CURRENT (mixed mode) ALWAYS builds a
    sampler so per-category uniform applies — this is the §2.1 fix.
    """
    target_floor = data_args.grounding_floor if data_args.grounding_floor > 0 else None
    p_replay = float(data_args.replay_fraction) if data_args.replay_fraction else 0.0

    # B0 fast-path: single-category CURRENT, no aux pools.
    if target_floor is None and p_replay == 0 and len(train_paths_current) == 1:
        ds = v2.NuScenesVQADatasetV2(
            data_path=train_paths_current[0],
            tokenizer=tokenizer,
            image_processor=image_processor,
            max_pixels=data_args.max_pixels,
            min_pixels=data_args.min_pixels,
            max_assistant_tokens=data_args.max_assistant_tokens,
        )
        return ds, None

    # Per-pool, per-category sample lists (in the SAME order they'll be laid
    # out in the flat combined JSON).
    current_per_cat: List[Tuple[str, List[dict]]] = [
        (_cat_from_path(p), _read_samples(p)) for p in train_paths_current
    ]

    prior_per_cat: List[Tuple[str, List[dict]]] = []
    if p_replay > 0:
        for p in train_paths_prior:
            prior_per_cat.append((_cat_from_path(p), _read_samples(p)))

    global_grounded_per_cat: List[Tuple[str, List[dict]]] = []
    if target_floor is not None:
        for p in train_paths_global_grounded:
            global_grounded_per_cat.append((_cat_from_path(p), _filter_grounded(_read_samples(p))))

    # Materialize combined JSON (rank 0 writes; others wait).
    combined_path = _materialize_combined_json(
        current_per_cat, prior_per_cat, global_grounded_per_cat,
        training_args.output_dir,
    )
    ds = v2.NuScenesVQADatasetV2(
        data_path=combined_path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_pixels=data_args.max_pixels,
        min_pixels=data_args.min_pixels,
        max_assistant_tokens=data_args.max_assistant_tokens,
    )

    # T3 — when view-stratified GF is on, derive per-sample view sets from the
    # already-loaded global_grounded samples. Iteration order MUST match the
    # combined-JSON GG block (i.e. concat of global_grounded_per_cat[*].samples
    # in the given order) so absolute indices line up with what
    # _materialize_combined_json writes.
    gg_per_sample_views: Optional[List[List[int]]] = None
    if data_args.view_stratified and target_floor is not None and global_grounded_per_cat:
        gg_per_sample_views = []
        for _, samples in global_grounded_per_cat:
            for s in samples:
                vs: List[int] = []
                try:
                    obj = json.loads(s["conversations"][2]["value"])
                    for g in obj.get("grounding") or []:
                        v = g.get("image_idx")
                        if isinstance(v, int) and 1 <= v <= 6 and v not in vs:
                            vs.append(v)
                except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                    pass
                gg_per_sample_views.append(vs)

    sampler, plan = build_sampler(
        current_per_cat=[(c, len(s)) for c, s in current_per_cat],
        prior_per_cat=[(c, len(s)) for c, s in prior_per_cat],
        global_grounded_per_cat=[(c, len(s)) for c, s in global_grounded_per_cat],
        target_floor=target_floor,
        replay_fraction=p_replay,
        num_training_samples=num_training_samples,
        seed=seed,
        global_grounded_per_sample_views=gg_per_sample_views,
        view_stratified=data_args.view_stratified,
        view_stratified_min_bucket=data_args.view_stratified_min_bucket,
        view_stratified_view_cap=data_args.view_stratified_view_cap,
    )
    rank0_print(
        f"[experiments] PoolPlan: |CURRENT|={plan.n_current} "
        f"({len(plan.current_ranges)} cats), |PRIOR|={plan.n_prior} "
        f"({len(plan.prior_ranges)} cats), |GLOBAL_GROUNDED|={plan.n_global_grounded} "
        f"({len(plan.global_grounded_ranges)} cats), total={plan.total}"
    )

    # Write the sampler plan to disk so the user can verify expected per-cat
    # weights without launching GPUs. Useful for spec acceptance checks.
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0 and sampler is not None:
        expected = plan.expected_weights(p_replay, target_floor)
        plan_dump = {
            "current_pool": {r.name: {"n": r.n, "start": r.start}
                             for r in plan.current_ranges},
            "prior_pool":   {r.name: {"n": r.n, "start": r.start}
                             for r in plan.prior_ranges},
            "global_grounded_pool": {r.name: {"n": r.n, "start": r.start}
                                     for r in plan.global_grounded_ranges},
            "expected_per_category_weights": expected,
            "p_replay": p_replay,
            "target_floor": target_floor,
        }
        if plan.stratified_gf:
            plan_dump["global_grounded_view_stratified"] = {
                "min_bucket": data_args.view_stratified_min_bucket,
                "view_cap":   data_args.view_stratified_view_cap,
                "view_weights": {str(v): w for v, w in plan.global_grounded_view_weights.items()},
                "buckets": [{"view": b.view, "cat": b.cat, "n": b.n}
                            for b in plan.global_grounded_view_buckets],
                "dropped_below_min_bucket": [
                    {"view": v, "cat": c, "n": n}
                    for v, c, n in plan.global_grounded_dropped_buckets
                ],
            }
        plan_path = os.path.join(training_args.output_dir, "sampler_plan.json")
        with open(plan_path, "w") as f:
            json.dump(plan_dump, f, indent=2)
        rank0_print(f"[experiments] sampler plan -> {plan_path}")

    return ds, sampler


def _materialize_combined_json(current_per_cat, prior_per_cat,
                               global_grounded_per_cat, scratch_dir) -> str:
    """Write the PoolPlan-ordered combined JSON. RANK-AWARE.

    Layout (matches PoolPlan offsets):
        current_per_cat[0].samples + current_per_cat[1].samples + ...
        + prior_per_cat[0].samples + ...
        + global_grounded_per_cat[0].samples + ...

    Args:
        current_per_cat: list of (cat_name, [samples]) for the CURRENT pool
        prior_per_cat:   list of (cat_name, [samples]) for the PRIOR pool
        global_grounded_per_cat: list of (cat_name, [grounded_samples])

    Rank-0-writes + others-wait pattern (torch.distributed isn't initialized
    yet at this point — use torchrun's RANK env var)."""
    import time
    rank = int(os.environ.get("RANK", "0"))

    os.makedirs(scratch_dir, exist_ok=True)
    path = os.path.join(scratch_dir, "_combined_train_data.json")
    done_marker = path + ".done"

    if rank == 0:
        if os.path.exists(done_marker):
            os.remove(done_marker)

        combined = []
        for _, samples in current_per_cat:
            combined.extend(samples)
        for _, samples in prior_per_cat:
            combined.extend(samples)
        for _, samples in global_grounded_per_cat:
            combined.extend(samples)

        tmp_path = f"{path}.tmp.{os.getpid()}"
        with open(tmp_path, "w") as f:
            json.dump(combined, f)
        os.replace(tmp_path, path)
        with open(done_marker, "w") as f:
            f.write("")

        rank0_print(
            f"[experiments] materialized combined train JSON: "
            f"CURRENT={sum(len(s) for _, s in current_per_cat)} "
            f"({[(c, len(s)) for c, s in current_per_cat]})  "
            f"PRIOR={sum(len(s) for _, s in prior_per_cat)} "
            f"({[(c, len(s)) for c, s in prior_per_cat]})  "
            f"GG={sum(len(s) for _, s in global_grounded_per_cat)} "
            f"({[(c, len(s)) for c, s in global_grounded_per_cat]})  "
            f"-> {path}"
        )
    else:
        timeout = 600.0
        start = time.time()
        while not os.path.exists(done_marker):
            if time.time() - start > timeout:
                raise TimeoutError(
                    f"Rank {rank}: timed out after {timeout}s waiting for "
                    f"{done_marker}. Rank 0 may have crashed before "
                    f"materializing the combined dataset."
                )
            time.sleep(0.5)
        time.sleep(0.5)

    return path


# ===========================================================================
# Main
# ===========================================================================
def train_exp():
    parser = transformers.HfArgumentParser(
        (ExpModelArguments, ExpDataArguments, ExpTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    v2.local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    rank0_print("=" * 60)
    rank0_print("Experiment training run")
    rank0_print(f"  mode_mixed: {data_args.mode_mixed}")
    rank0_print(f"  lr_schedule_kind: {training_args.lr_schedule_kind}")
    rank0_print(f"  grounding_floor: {data_args.grounding_floor}")
    rank0_print(f"  replay_fraction: {data_args.replay_fraction}")
    rank0_print(f"  prior_data_paths: {data_args.prior_data_paths!r}")
    rank0_print(f"  checkpoint_step_fractions: {training_args.checkpoint_step_fractions!r}")
    rank0_print("=" * 60)

    # ---- Model + LoRA + tokenizer (identical to v2)
    from transformers import Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16 if training_args.bf16 else None,
    )
    model.config.use_cache = False
    if data_args.data_flatten:
        v2.replace_qwen2_vl_attention_class()
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def _hook(module, inp, out): out.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(_hook)

    if model_args.mode == "lora":
        model = v2.setup_lora(model, model_args)
    elif model_args.mode == "full":
        model = v2.setup_full_finetune(model, model_args)
    else:
        raise ValueError(f"Unknown training mode: {model_args.mode}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right", use_fast=False,
    )
    landmarks = v2.verify_landmarks(tokenizer)
    bad = [k for k, (exp, act) in landmarks.items() if exp != act]
    if bad:
        raise RuntimeError(f"Token landmarks failed for {bad}: {landmarks}")
    image_processor = AutoProcessor.from_pretrained(model_args.model_name_or_path).image_processor

    # ---- Loss config — same composite as v2 (NEVER varied per the spec)
    view_w = v2._resolve_view_class_weight(
        training_args.view_class_weight, data_args.train_data_path
    )
    loss_config = v2.CompositeLossConfig(
        w_ans=training_args.w_ans, w_gate=training_args.w_gate,
        lam_view=training_args.lam_view, lam_iou=training_args.lam_iou,
        lam_klal=training_args.lam_klal,
        klal_layers=v2._parse_klal_layers(training_args.klal_layers),
        iou_alpha=training_args.iou_alpha,
        view_class_weight=view_w,
    )

    # ---- Resolve train file list (current vs prior vs global_grounded)
    if data_args.mode_mixed:
        if not data_args.mixed_category_data_paths:
            raise ValueError("mode_mixed=True requires --mixed_category_data_paths")
        train_paths_current = [p for p in data_args.mixed_category_data_paths.split(",") if p]
        train_paths_prior: List[str] = []
        if data_args.replay_fraction and data_args.replay_fraction > 0:
            rank0_print("[warn] replay_fraction > 0 with mode_mixed=True is "
                        "unusual (mixed already sees all categories every step). "
                        "Treating prior_data_paths as additional pool for the ER draw.")
            train_paths_prior = [p for p in (data_args.prior_data_paths or "").split(",") if p]
    else:
        train_paths_current = [data_args.train_data_path]
        train_paths_prior = [p for p in (data_args.prior_data_paths or "").split(",") if p]

    # Global grounded pool (spec §2.2 `pool: global`) — same paths are
    # passed for sequential AND mixed, since GF semantics are the same.
    train_paths_global_grounded = [
        p for p in (data_args.global_grounded_data_paths or "").split(",") if p
    ]

    # ---- Estimate num_training_samples for the sampler. HF Trainer scales
    # internally by epochs/grad_accum so we just need a reasonable len; a
    # large multiple of the dataset size suffices.
    approx_total_samples = sum(
        len(_read_samples(p)) for p in train_paths_current
    )
    num_training_samples = max(approx_total_samples, 1) * int(
        max(1, training_args.num_train_epochs)
    )

    train_dataset, custom_sampler = _build_concat_dataset_and_sampler(
        train_paths_current=train_paths_current,
        train_paths_prior=train_paths_prior,
        train_paths_global_grounded=train_paths_global_grounded,
        tokenizer=tokenizer,
        image_processor=image_processor,
        data_args=data_args,
        training_args=training_args,
        num_training_samples=num_training_samples,
        seed=training_args.seed if hasattr(training_args, "seed") else 0,
    )

    # ---- Eval dataset (identical to v2 plumbing)
    eval_dataset = None
    if data_args.eval_dataset_paths_json and os.path.exists(data_args.eval_dataset_paths_json):
        with open(data_args.eval_dataset_paths_json) as f:
            eval_paths = json.load(f)
        eval_dataset = {}
        for cat_name, cat_path in eval_paths.items():
            if not os.path.exists(cat_path):
                rank0_print(f"[warn] eval dataset for {cat_name!r} not found: {cat_path}")
                continue
            eval_dataset[cat_name] = v2.NuScenesVQADatasetV2(
                data_path=cat_path,
                tokenizer=tokenizer, image_processor=image_processor,
                max_pixels=data_args.max_pixels, min_pixels=data_args.min_pixels,
                max_assistant_tokens=data_args.max_assistant_tokens,
            )

    data_collator = v2.NuScenesDataCollatorV2(tokenizer=tokenizer)

    # ---- Build the ExperimentTrainer; install the step-fraction callback
    # for mixed mode if requested.
    trainer = ExperimentTrainer(
        model=model, processing_class=tokenizer, args=training_args,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        data_collator=data_collator,
        loss_config=loss_config, tokenizer_for_loss=tokenizer,
        lr_schedule_kind=training_args.lr_schedule_kind,
        global_total_steps=training_args.global_total_steps,
    )

    if custom_sampler is not None:
        # HF Trainer normally builds its own DistributedSampler/SequentialSampler;
        # override via the `_get_train_sampler` hook so multi-GPU + grad-accum
        # plumbing still works downstream.
        #
        # Two transformers-version concerns:
        #   (a) Recent versions changed the signature from `_get_train_sampler(self)`
        #       to `_get_train_sampler(self, dataset=None)`. Accept both via *args.
        #   (b) DDP correctness: CompositeSampler is DDP-aware internally
        #       (seeds with `seed + rank`, length scales by world_size) so
        #       returning the same sampler instance for every rank is OK.
        trainer._custom_train_sampler = custom_sampler

        def _override_sampler(self, *args, **kwargs):
            return self._custom_train_sampler
        # Bind to the instance, not the class (so the v2 trainer used by the
        # legacy command remains untouched).
        import types
        trainer._get_train_sampler = types.MethodType(_override_sampler, trainer)

    if data_args.mode_mixed and training_args.checkpoint_step_fractions:
        fractions = [float(x) for x in training_args.checkpoint_step_fractions.split(",") if x]
        import weakref
        trainer.add_callback(StepFractionCheckpointCallback(
            fractions=fractions,
            output_root=training_args.output_dir,
            trainer_ref=weakref.ref(trainer),
        ))

    # ---- Train + save final (matches v2 exactly)
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        rank0_print("Checkpoint found, resuming training...")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    if trainer.is_world_process_zero():
        trainer.model.save_pretrained(training_args.output_dir)

    # ---- Dump per-rank sampler exposure stats so we can verify per-category
    # uniform was actually realized (user-added §9 acceptance check).
    if custom_sampler is not None:
        rank = int(os.environ.get("RANK", "0"))
        exp_dir = os.path.join(training_args.output_dir, "sampler_exposure")
        os.makedirs(exp_dir, exist_ok=True)
        rank_path = os.path.join(exp_dir, f"rank{rank}.json")
        with open(rank_path, "w") as f:
            json.dump(custom_sampler.exposure_summary(), f, indent=2)
        rank0_print(f"[experiments] sampler exposure (rank {rank}) -> {rank_path}")

        # Rank 0 aggregates across ranks (after a brief barrier).
        if rank == 0:
            import glob as _glob, time as _time
            # Wait a couple seconds for other ranks to finish writing.
            for _ in range(30):
                if len(_glob.glob(os.path.join(exp_dir, "rank*.json"))) >= int(os.environ.get("WORLD_SIZE", "1")):
                    break
                _time.sleep(0.5)

            agg_counts: Dict[Tuple[str, str], int] = {}
            for p in sorted(_glob.glob(os.path.join(exp_dir, "rank*.json"))):
                with open(p) as f:
                    d = json.load(f)
                for pool_name, pool in d["per_pool"].items():
                    for cat, info in pool["categories"].items():
                        key = (pool_name, cat)
                        agg_counts[key] = agg_counts.get(key, 0) + info["count"]

            total = sum(agg_counts.values())
            agg: Dict[str, Dict] = {}
            for (pool_name, cat), count in agg_counts.items():
                agg.setdefault(pool_name, {})[cat] = count
            # Expected weights come from the plan, which we re-derive from
            # sampler_plan.json so a stale exposure dump doesn't lie.
            try:
                with open(os.path.join(training_args.output_dir, "sampler_plan.json")) as f:
                    plan_dump = json.load(f)
                expected = plan_dump.get("expected_per_category_weights", {})
            except FileNotFoundError:
                expected = {}

            agg_summary = {
                "total_draws": total,
                "world_size": int(os.environ.get("WORLD_SIZE", "1")),
                "per_pool": {},
            }
            for pool_name in ("current", "prior", "global_grounded"):
                cats = agg.get(pool_name, {})
                pool_total = sum(cats.values())
                exp_map = expected.get(pool_name, {})
                agg_summary["per_pool"][pool_name] = {
                    "pool_total": pool_total,
                    "share_of_total": (pool_total / total) if total else 0.0,
                    "categories": {
                        cat: {
                            "count": count,
                            "share_of_pool": (count / pool_total) if pool_total else 0.0,
                            "share_of_total": (count / total) if total else 0.0,
                            "expected_share_of_total": exp_map.get(cat, 0.0),
                        }
                        for cat, count in cats.items()
                    },
                }
            agg_path = os.path.join(training_args.output_dir, "sampler_exposure.json")
            with open(agg_path, "w") as f:
                json.dump(agg_summary, f, indent=2)
            rank0_print(f"[experiments] sampler exposure (aggregated) -> {agg_path}")


if __name__ == "__main__":
    train_exp()
