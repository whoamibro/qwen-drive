"""LR schedule variants for the curriculum-v2 ablation framework.

Per spec §2.4:

1. **global_wsd** — one continuous WSD across the sum of per-stage step
   budgets. **warmup_frac=0.03** of TOTAL steps, **decay_frac=0.10** of
   TOTAL steps, stable in between. Peak LR = 2e-4 (unchanged).

2. **per_stage_wsd_relaxed** — per-stage WSD with relaxed ratios so the
   between-stage LR-to-zero transient is softened. Ratios per spec §2.4:
   - `warmup_ratio = 0.0`  (warm-started stages need no fresh warmup;
     stage 0 keeps 0.10 — handled via the `is_first_stage` flag below)
   - `stable_ratio = 0.80`
   - `decay_ratio  = 0.20`
   PLUS a `sparse_stage_peak_factor = 0.3` applied as a uniform LR
   multiplier when the stage's grounded fraction is below
   `sparse_threshold = 0.15`. This makes TSS (~3.5%) and RML (~5%) train
   at peak * 0.3 to prevent a high-LR cycle from overwriting the shared
   LoRA weights on a near-empty grounding signal.

Both helpers return torch.optim.lr_scheduler.LambdaLR instances so they
slot directly into HF Trainer's create_scheduler() override pattern used
by ExperimentTrainer.create_scheduler.
"""

from __future__ import annotations

from typing import Optional, Tuple

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


# Spec §2.4 — relaxed per-stage WSD ratios.
PER_STAGE_RELAXED_RATIOS: Tuple[float, float, float] = (0.0, 0.80, 0.20)

# Stage 0 (no prior LoRA warm-start) keeps a normal warmup so the model
# isn't slammed with peak LR on step 1.
PER_STAGE_RELAXED_STAGE0_WARMUP: float = 0.10

# Sparse-stage cap — grounded fraction below this threshold triggers the
# peak-factor LR reduction.
SPARSE_STAGE_THRESHOLD: float = 0.15
SPARSE_STAGE_PEAK_FACTOR: float = 0.3

# Spec §2.4 — global WSD ratios (over TOTAL steps, not per-stage).
GLOBAL_WSD_WARMUP_FRAC: float = 0.03
GLOBAL_WSD_DECAY_FRAC: float = 0.10


def _trapezoid_lambda(num_training_steps: int,
                      warmup_ratio: float,
                      decay_ratio: float,
                      peak_factor: float = 1.0):
    """Closed-form WSD multiplier.

    `peak_factor` is a uniform scalar applied to every step's LR multiplier
    (so even the "stable plateau" caps at `peak_factor` instead of 1.0).
    Used for the spec §2.4 sparse-stage peak reduction.
    """
    if num_training_steps <= 0:
        raise ValueError(f"num_training_steps must be > 0, got {num_training_steps}")
    if warmup_ratio < 0 or decay_ratio < 0 or warmup_ratio + decay_ratio > 1.0:
        raise ValueError(
            f"warmup_ratio + decay_ratio must lie in [0, 1]; "
            f"got warmup={warmup_ratio} decay={decay_ratio}"
        )
    warmup_steps = int(round(warmup_ratio * num_training_steps))
    decay_steps = int(round(decay_ratio * num_training_steps))
    decay_start = num_training_steps - decay_steps

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            base = float(current_step) / max(1, warmup_steps)
        elif current_step < decay_start:
            base = 1.0
        else:
            progress = (current_step - decay_start) / max(1, decay_steps)
            base = max(0.0, 1.0 - progress)
        return base * peak_factor

    return lr_lambda


def get_global_wsd_schedule(
    optimizer: Optimizer,
    total_training_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    """Spec §2.4 global WSD.

    Ignores any per-stage WSD ratios from the YAML — uses the spec's
    `warmup_frac=0.03` and `decay_frac=0.10` of TOTAL steps.
    """
    return LambdaLR(
        optimizer,
        _trapezoid_lambda(
            total_training_steps,
            warmup_ratio=GLOBAL_WSD_WARMUP_FRAC,
            decay_ratio=GLOBAL_WSD_DECAY_FRAC,
        ),
        last_epoch=last_epoch,
    )


def get_per_stage_wsd_relaxed_schedule(
    optimizer: Optimizer,
    num_training_steps: int,
    is_first_stage: bool = False,
    stage_grounded_fraction: Optional[float] = None,
    last_epoch: int = -1,
) -> LambdaLR:
    """Spec §2.4 relaxed per-stage WSD.

    Args:
        is_first_stage: stage 0 keeps the normal 0.10 warmup since the model
            has no prior warm-start. Subsequent stages warmup_ratio=0.0
            (warm-starting means we don't need to ramp from 0).
        stage_grounded_fraction: if < SPARSE_STAGE_THRESHOLD (0.15), apply
            SPARSE_STAGE_PEAK_FACTOR (0.3) as a uniform LR multiplier so a
            high-LR cycle on a near-empty grounding signal can't overwrite
            shared LoRA weights. `None` (unknown) is treated as "not sparse".
    """
    warmup = PER_STAGE_RELAXED_STAGE0_WARMUP if is_first_stage else PER_STAGE_RELAXED_RATIOS[0]
    decay = PER_STAGE_RELAXED_RATIOS[2]

    peak_factor = 1.0
    if (stage_grounded_fraction is not None
            and stage_grounded_fraction < SPARSE_STAGE_THRESHOLD):
        peak_factor = SPARSE_STAGE_PEAK_FACTOR

    return LambdaLR(
        optimizer,
        _trapezoid_lambda(
            num_training_steps,
            warmup_ratio=warmup,
            decay_ratio=decay,
            peak_factor=peak_factor,
        ),
        last_epoch=last_epoch,
    )


__all__ = [
    "PER_STAGE_RELAXED_RATIOS",
    "PER_STAGE_RELAXED_STAGE0_WARMUP",
    "SPARSE_STAGE_THRESHOLD",
    "SPARSE_STAGE_PEAK_FACTOR",
    "GLOBAL_WSD_WARMUP_FRAC",
    "GLOBAL_WSD_DECAY_FRAC",
    "get_global_wsd_schedule",
    "get_per_stage_wsd_relaxed_schedule",
]
