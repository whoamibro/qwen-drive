"""Data samplers for the curriculum-v2 ablation framework.

Implements three knobs and their composition, per
`curriculum_v2_ablation_experiments.md` §2.1 / §2.2 / §2.3 / §5.

- **Mixed-mode uniform (§2.1)**: `mixed.sampler: uniform` ⇒ per-category
  weight `w_c = 0.10` (Appendix A v0). Each draw picks a category
  uniformly, then a sample uniformly within that category. NOT per-sample
  uniform over the concatenated dataset (that would degrade to
  size-proportional, e.g. OBS 23% / RML 2.6% on this dataset).

- **Grounding Floor (§2.2)**: `pool: global` — top-up draws come from
  grounded samples across ALL 10 categories, with **per-category uniform
  weighting** among the categories that have any grounded samples.

- **Experience Replay (§2.3)**: PRIOR pool is the union of stages 0..i-1
  categories, also drawn with **per-category uniform** so small prior cats
  (e.g. IDN 5226) aren't dominated by large ones (OBS 32551).

- **§5.2 composition**:
    1. Replay first: prob `p_replay` ⇒ draw from PRIOR pool.
    2. Else, prob `target_floor` (if GF enabled) ⇒ draw from GLOBAL_GROUNDED.
    3. Else ⇒ draw from CURRENT pool.
    Per spec §5.1, step count is fixed regardless of knob composition.

Every pool sample uses **two-stage uniform**: (1) pick category uniformly
from the pool's category list, (2) pick a sample uniformly within that
category's index range. Single-category pools (sequential CURRENT)
degenerate to pure within-pool uniform.

Per-(pool, category) draw counts are tracked during __iter__ and exposed
via `exposure_summary()` for the manifest dump.
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Grounding-flag precomputation
# ---------------------------------------------------------------------------
def compute_grounding_flags(samples) -> List[bool]:
    """True iff sample's GPT response carries non-empty grounding."""
    flags = []
    for s in samples:
        try:
            obj = json.loads(s["conversations"][2]["value"])
            flags.append(bool(obj.get("grounding")))
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            flags.append(False)
    return flags


# ---------------------------------------------------------------------------
# Pool plan with per-category sub-ranges
# ---------------------------------------------------------------------------
@dataclass
class CategoryRange:
    name: str
    start: int      # absolute index in the flattened combined dataset
    n: int          # number of samples in this category within this pool

    @property
    def end(self) -> int:
        return self.start + self.n


@dataclass
class PoolPlan:
    """Three pools (CURRENT, PRIOR, GLOBAL_GROUNDED) each composed of one or
    more per-category sub-ranges within a single flat ConcatDataset.

    Two-stage uniform sampling: pick a category uniformly from a pool's
    sub-range list, then a sample uniformly within that category. Empty
    sub-ranges (cats with 0 samples in this pool) are dropped at plan-build
    time so they're not selected.
    """
    current_ranges: List[CategoryRange] = field(default_factory=list)
    prior_ranges: List[CategoryRange] = field(default_factory=list)
    global_grounded_ranges: List[CategoryRange] = field(default_factory=list)

    @property
    def n_current(self) -> int:
        return sum(r.n for r in self.current_ranges)

    @property
    def n_prior(self) -> int:
        return sum(r.n for r in self.prior_ranges)

    @property
    def n_global_grounded(self) -> int:
        return sum(r.n for r in self.global_grounded_ranges)

    @property
    def total(self) -> int:
        return self.n_current + self.n_prior + self.n_global_grounded

    def expected_weights(self, p_replay: float, target_floor: Optional[float]) -> Dict[str, Dict[str, float]]:
        """Return a {pool: {cat: expected_weight, ...}} dict for the manifest.

        Per spec §2.1 + §5.2:
        - pool selection: P(prior) = p_replay; P(global_grounded) = (1-p_replay) * tf;
          P(current) = (1-p_replay) * (1-tf).  (tf is 0 if GF disabled or pool empty.)
        - within a pool: 1/k for each of k categories.
        """
        tf = (target_floor if (target_floor is not None
                               and self.global_grounded_ranges) else 0.0)
        p_pool = {
            "current":          (1 - p_replay) * (1 - tf),
            "prior":            p_replay,
            "global_grounded":  (1 - p_replay) * tf,
        }
        out: Dict[str, Dict[str, float]] = {}
        for pool_name, ranges in [
            ("current", self.current_ranges),
            ("prior", self.prior_ranges),
            ("global_grounded", self.global_grounded_ranges),
        ]:
            if not ranges or p_pool[pool_name] == 0.0:
                out[pool_name] = {}
                continue
            per_cat = p_pool[pool_name] / len(ranges)
            out[pool_name] = {r.name: per_cat for r in ranges}
        return out


def build_pool_plan(
    current_per_cat: Sequence[Tuple[str, int]],     # [(cat_name, n_samples), ...] for CURRENT
    prior_per_cat: Sequence[Tuple[str, int]],       # ditto for PRIOR (may be empty)
    global_grounded_per_cat: Sequence[Tuple[str, int]],  # ditto for GG (may be empty)
) -> PoolPlan:
    """Compose the flat-index layout: CURRENT block first (in given order),
    then PRIOR, then GLOBAL_GROUNDED. Per-cat sub-ranges record their start
    offset and size so the sampler can do two-stage uniform.

    Categories with `n == 0` are silently dropped from their pool's range
    list (they wouldn't be drawable anyway, and including them would skew
    the per-cat uniform pick).
    """
    plan = PoolPlan()
    offset = 0
    for cat, n in current_per_cat:
        if n > 0:
            plan.current_ranges.append(CategoryRange(cat, offset, n))
        offset += n
    for cat, n in prior_per_cat:
        if n > 0:
            plan.prior_ranges.append(CategoryRange(cat, offset, n))
        offset += n
    for cat, n in global_grounded_per_cat:
        if n > 0:
            plan.global_grounded_ranges.append(CategoryRange(cat, offset, n))
        offset += n
    return plan


# ---------------------------------------------------------------------------
# Composite sampler — two-stage uniform inside each pool
# ---------------------------------------------------------------------------
class CompositeSampler(Sampler[int]):
    """§5.2 composition + per-category uniform within each pool.

    Tracks per-(pool, cat) draw counts in `_exposure` so the trainer can
    dump them at end-of-training for acceptance verification (§9 +
    user-added acceptance: realized per-cat ratio == 1/N ± tolerance).
    """

    def __init__(
        self,
        plan: PoolPlan,
        target_floor: Optional[float],
        p_replay: float,
        num_samples: int,
        seed: int = 0,
    ):
        if num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, got {num_samples}")
        if not (0.0 <= p_replay < 1.0):
            raise ValueError(f"p_replay must be in [0, 1); got {p_replay}")
        if target_floor is not None and not (0.0 < target_floor < 1.0):
            raise ValueError(f"target_floor must be in (0, 1) or None; got {target_floor}")
        if p_replay > 0 and not plan.prior_ranges:
            raise ValueError("p_replay > 0 requires a non-empty PRIOR pool. "
                             "Did the orchestrator forget to pass --prior_data_paths "
                             "for a non-first stage?")
        if target_floor is not None and not plan.global_grounded_ranges:
            logger.warning("target_floor=%.3f but GLOBAL_GROUNDED pool is empty — "
                           "GF will have no effect (falls through to CURRENT).",
                           target_floor)
        if not plan.current_ranges:
            raise ValueError("CURRENT pool is empty — nothing to train on.")

        self.plan = plan
        self.target_floor = target_floor
        self.p_replay = p_replay
        self.num_samples = num_samples
        self.seed = seed

        # Per-(pool, cat) draw count, populated on __iter__.
        self._exposure: Dict[Tuple[str, str], int] = defaultdict(int)
        self._iter_count = 0

        logger.info(
            "CompositeSampler: |CURRENT|=%d (%d cats), |PRIOR|=%d (%d cats), "
            "|GLOBAL_GROUNDED|=%d (%d cats), target_floor=%r, p_replay=%.3f, "
            "num_samples=%d",
            plan.n_current, len(plan.current_ranges),
            plan.n_prior, len(plan.prior_ranges),
            plan.n_global_grounded, len(plan.global_grounded_ranges),
            target_floor, p_replay, num_samples,
        )

    def _ddp_rank_world(self) -> Tuple[int, int]:
        """Return (rank, world_size) using torch.distributed if initialized."""
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                return dist.get_rank(), dist.get_world_size()
        except ImportError:
            pass
        return 0, 1

    def _draw_from_ranges(self, ranges: List[CategoryRange], rng: random.Random,
                          pool_name: str) -> int:
        """Two-stage uniform: cat ~ Uniform(cats), idx ~ Uniform(0, n_cat).
        Records draw in exposure counter."""
        cat_idx = rng.randrange(len(ranges))
        r = ranges[cat_idx]
        sample_idx = rng.randrange(r.n)
        self._exposure[(pool_name, r.name)] += 1
        return r.start + sample_idx

    def __iter__(self) -> Iterator[int]:
        rank, world_size = self._ddp_rank_world()
        rng = random.Random(self.seed + rank + self._iter_count * 1009)
        self._iter_count += 1
        n_per_rank = self.num_samples // max(1, world_size)
        out: List[int] = []
        tf = (self.target_floor if (self.target_floor is not None
                                     and self.plan.global_grounded_ranges) else 0.0)

        for _ in range(n_per_rank):
            r1 = rng.random()
            if r1 < self.p_replay:
                out.append(self._draw_from_ranges(self.plan.prior_ranges, rng, "prior"))
                continue
            r2 = rng.random()
            if r2 < tf:
                out.append(self._draw_from_ranges(self.plan.global_grounded_ranges, rng, "global_grounded"))
            else:
                out.append(self._draw_from_ranges(self.plan.current_ranges, rng, "current"))
        return iter(out)

    def __len__(self) -> int:
        _, world_size = self._ddp_rank_world()
        return self.num_samples // max(1, world_size)

    # ---- Exposure dump for the manifest
    def exposure_summary(self) -> Dict:
        """Return a serializable dict of per-(pool, cat) draw counts +
        expected ratios. Suitable for direct json.dump."""
        rank, world_size = self._ddp_rank_world()
        total = sum(self._exposure.values())
        expected = self.plan.expected_weights(self.p_replay, self.target_floor)

        per_pool: Dict[str, Dict] = {}
        for pool_name in ("current", "prior", "global_grounded"):
            cat_counts = {}
            pool_total = sum(c for (p, _), c in self._exposure.items() if p == pool_name)
            for (p, cat), count in self._exposure.items():
                if p != pool_name:
                    continue
                cat_counts[cat] = {
                    "count": count,
                    "share_of_pool": (count / pool_total) if pool_total else 0.0,
                    "share_of_total": (count / total) if total else 0.0,
                    "expected_share_of_total": expected.get(pool_name, {}).get(cat, 0.0),
                }
            per_pool[pool_name] = {
                "pool_total": pool_total,
                "share_of_total": (pool_total / total) if total else 0.0,
                "categories": cat_counts,
            }
        return {
            "rank": rank,
            "world_size": world_size,
            "total_draws": total,
            "per_pool": per_pool,
        }


# ---------------------------------------------------------------------------
# Top-level factory the training script calls
# ---------------------------------------------------------------------------
def build_sampler(
    *,
    current_per_cat: Sequence[Tuple[str, int]],
    prior_per_cat: Sequence[Tuple[str, int]],
    global_grounded_per_cat: Sequence[Tuple[str, int]],
    target_floor: Optional[float],
    replay_fraction: Optional[float],
    num_training_samples: int,
    seed: int = 0,
):
    """Return (sampler, plan).

    B0 fast-path (sampler=None) requires BOTH: knobs off AND single-cat
    CURRENT. Multi-cat CURRENT always builds a sampler so per-category
    uniform applies — that's the §2.1 mixed-mode fix.
    """
    p_replay = float(replay_fraction) if replay_fraction else 0.0
    plan = build_pool_plan(
        current_per_cat=current_per_cat,
        prior_per_cat=prior_per_cat if p_replay > 0 else [],
        global_grounded_per_cat=global_grounded_per_cat if target_floor else [],
    )
    is_multi_cat_current = len(plan.current_ranges) > 1
    if target_floor is None and p_replay == 0 and not is_multi_cat_current:
        return None, plan
    return CompositeSampler(
        plan=plan,
        target_floor=target_floor,
        p_replay=p_replay,
        num_samples=num_training_samples,
        seed=seed,
    ), plan


__all__ = [
    "compute_grounding_flags",
    "CategoryRange",
    "PoolPlan",
    "build_pool_plan",
    "CompositeSampler",
    "build_sampler",
]
