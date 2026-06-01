"""
Warmup -> Stable -> Decay (WSD / trapezoidal) learning-rate schedule.

For each curriculum stage we want an independent LR cycle:

    lr_mult
       1.0 |        ___________________
           |      /                     \\
           |    /                         \\
       0.0 |__/                             \\___
              |--warmup--|------stable------|--decay--|
              0                                       total_steps

Implemented as a LambdaLR multiplier so HF Trainer plumbing is untouched.
"""

from __future__ import annotations

from typing import Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_wsd_lr_lambda(
    num_training_steps: int,
    warmup_ratio: float,
    decay_ratio: float,
):
    """Return the LR-multiplier function for a single stage's WSD schedule."""
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
            return float(current_step) / max(1, warmup_steps)
        if current_step < decay_start:
            return 1.0
        # Decay phase: linear ramp from 1.0 -> 0.0 across decay_steps.
        progress = (current_step - decay_start) / max(1, decay_steps)
        return max(0.0, 1.0 - progress)

    return lr_lambda


def get_wsd_schedule(
    optimizer: Optimizer,
    num_training_steps: int,
    warmup_ratio: float = 0.10,
    decay_ratio: float = 0.20,
    last_epoch: int = -1,
) -> LambdaLR:
    """LambdaLR scheduler implementing a single-stage WSD cycle."""
    lr_lambda = get_wsd_lr_lambda(num_training_steps, warmup_ratio, decay_ratio)
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


__all__ = ["get_wsd_lr_lambda", "get_wsd_schedule"]
