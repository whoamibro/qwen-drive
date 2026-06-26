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
class ViewBucket:
    """T3 view-stratified GF: a (view, category) sub-pool with explicit
    absolute indices (NOT a contiguous range — a grounded sample with boxes
    in views 2 and 5 appears in BOTH bucket(view=2,cat) and bucket(view=5,cat),
    so the same combined-JSON index can occur in multiple buckets).
    """
    view: int                  # 1..6
    cat:  str
    indices: List[int]         # absolute offsets into the combined dataset

    @property
    def name(self) -> str:
        return f"v{self.view}/{self.cat}"

    @property
    def n(self) -> int:
        return len(self.indices)


@dataclass
class PoolPlan:
    """Three pools (CURRENT, PRIOR, GLOBAL_GROUNDED) each composed of one or
    more per-category sub-ranges within a single flat ConcatDataset.

    Two-stage uniform sampling: pick a category uniformly from a pool's
    sub-range list, then a sample uniformly within that category. Empty
    sub-ranges (cats with 0 samples in this pool) are dropped at plan-build
    time so they're not selected.

    T3: when `global_grounded_view_buckets` is non-empty, the GF pool uses
    THREE-stage stratified sampling (view -> cat -> sample-in-bucket) and the
    legacy `global_grounded_ranges` field is ignored for sampling. The
    per-view selection probabilities live in `global_grounded_view_weights`
    (a dict {view_int: weight}, sums to 1). With per-view uniform + view_cap
    safeguard applied at build time, every populated view gets the same
    weight unless cap-induced re-routing fires.
    """
    current_ranges: List[CategoryRange] = field(default_factory=list)
    prior_ranges: List[CategoryRange] = field(default_factory=list)
    global_grounded_ranges: List[CategoryRange] = field(default_factory=list)
    # T3 — stratified GF only. Empty list = unstratified GF (legacy path).
    global_grounded_view_buckets: List[ViewBucket] = field(default_factory=list)
    global_grounded_view_weights: Dict[int, float] = field(default_factory=dict)
    # Diagnostics: buckets dropped by min_bucket filter (informational).
    global_grounded_dropped_buckets: List[Tuple[int, str, int]] = field(default_factory=list)

    @property
    def n_current(self) -> int:
        return sum(r.n for r in self.current_ranges)

    @property
    def n_prior(self) -> int:
        return sum(r.n for r in self.prior_ranges)

    @property
    def n_global_grounded(self) -> int:
        # Unstratified path uses CategoryRanges; stratified path uses
        # ViewBuckets (which may double-count samples across views — that's
        # the credit-share semantic). Choose whichever is populated.
        if self.global_grounded_view_buckets:
            return sum(b.n for b in self.global_grounded_view_buckets)
        return sum(r.n for r in self.global_grounded_ranges)

    @property
    def total(self) -> int:
        return self.n_current + self.n_prior + self.n_global_grounded

    @property
    def stratified_gf(self) -> bool:
        return bool(self.global_grounded_view_buckets)

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
    global_grounded_per_sample_views: Optional[Sequence[Sequence[int]]] = None,
    view_stratified: bool = False,
    view_stratified_min_bucket: int = 50,
    view_stratified_view_cap: float = 2.0,
) -> PoolPlan:
    """Compose the flat-index layout: CURRENT block first (in given order),
    then PRIOR, then GLOBAL_GROUNDED. Per-cat sub-ranges record their start
    offset and size so the sampler can do two-stage uniform.

    Categories with `n == 0` are silently dropped from their pool's range
    list (they wouldn't be drawable anyway, and including them would skew
    the per-cat uniform pick).

    T3 — when `view_stratified=True`, additionally build per-view sub-pools
    of the GF pool. `global_grounded_per_sample_views[i]` is the iterable
    of view-ints (1..6) that the i-th sample in the *flattened* GG block
    contains a GT box for (a sample with 2-view grounding contributes to
    2 buckets). `view_stratified_min_bucket` drops (view, cat) buckets with
    too few samples; `view_stratified_view_cap` caps any view's selection
    probability at `cap / 6` and redistributes excess to uncapped views.
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
    gg_start_offset = offset
    for cat, n in global_grounded_per_cat:
        if n > 0:
            plan.global_grounded_ranges.append(CategoryRange(cat, offset, n))
        offset += n

    if view_stratified and plan.global_grounded_ranges:
        if global_grounded_per_sample_views is None:
            raise ValueError(
                "view_stratified=True requires global_grounded_per_sample_views "
                "(per-sample view-set lookup) to be passed."
            )
        plan.global_grounded_view_buckets, plan.global_grounded_dropped_buckets = (
            _build_view_buckets(
                global_grounded_ranges=plan.global_grounded_ranges,
                gg_start_offset=gg_start_offset,
                per_sample_views=global_grounded_per_sample_views,
                min_bucket=view_stratified_min_bucket,
            )
        )
        plan.global_grounded_view_weights = _compute_view_weights(
            populated_views=sorted({b.view for b in plan.global_grounded_view_buckets}),
            view_cap=view_stratified_view_cap,
            n_views_total=6,
        )
    return plan


# ---------------------------------------------------------------------------
# T3 — view-bucket construction + view-weight computation
# ---------------------------------------------------------------------------
def _build_view_buckets(
    *,
    global_grounded_ranges: List[CategoryRange],
    gg_start_offset: int,
    per_sample_views: Sequence[Sequence[int]],
    min_bucket: int,
) -> Tuple[List[ViewBucket], List[Tuple[int, str, int]]]:
    """Build (view, cat) sub-pools over the GG block.

    A single sample with grounding in views {2, 5} is credited to BOTH
    bucket(view=2, cat) and bucket(view=5, cat) — the buckets are CREDIT
    pools, not partitions. This is the "double-counting" caveat the
    verify_sampler report exposes.

    Returns (buckets, dropped_log). `dropped_log` is [(view, cat, n_dropped)].
    """
    n_gg = sum(r.n for r in global_grounded_ranges)
    if len(per_sample_views) != n_gg:
        raise ValueError(
            f"per_sample_views length {len(per_sample_views)} != GG block size {n_gg}."
        )

    # bucket_local: { (view, cat) -> list of local idx within GG block }
    bucket_local: Dict[Tuple[int, str], List[int]] = defaultdict(list)
    local_idx = 0
    for r in global_grounded_ranges:
        for _ in range(r.n):
            for v in per_sample_views[local_idx]:
                if isinstance(v, int) and 1 <= v <= 6:
                    bucket_local[(v, r.name)].append(local_idx)
            local_idx += 1

    buckets: List[ViewBucket] = []
    dropped: List[Tuple[int, str, int]] = []
    for (v, cat), local_idxs in sorted(bucket_local.items()):
        # absolute = local + gg_start_offset
        abs_idxs = [li + gg_start_offset for li in local_idxs]
        if len(abs_idxs) < max(1, int(min_bucket)):
            dropped.append((v, cat, len(abs_idxs)))
            continue
        buckets.append(ViewBucket(view=v, cat=cat, indices=abs_idxs))
    return buckets, dropped


def _compute_view_weights(
    populated_views: Sequence[int],
    view_cap: float,
    n_views_total: int = 6,
) -> Dict[int, float]:
    """Per-view selection probabilities for stratified GF.

    Start from per-view uniform among populated views (1/N_eff each). Cap
    each at `view_cap / n_views_total` (e.g. 2.0/6 = 0.333 with default cap).
    Redistribute excess across uncapped views proportionally. If the cap
    cannot be respected (so few populated views that even cap-share × N_eff
    < 1), log a warning and fall back to uniform-among-populated.
    """
    pop = sorted(populated_views)
    n_eff = len(pop)
    if n_eff == 0:
        return {}

    cap_share = view_cap / max(1, n_views_total)
    if n_eff * cap_share < 1.0 - 1e-9:
        logger.warning(
            "view_cap=%.3f cannot be respected: only %d populated views "
            "(need at least %d for cap × fair_share = %.3f to cover 1.0); "
            "falling back to uniform-among-populated (cap effectively ignored).",
            view_cap, n_eff, int(n_views_total / view_cap + 0.9999), cap_share,
        )
        return {v: 1.0 / n_eff for v in pop}

    weights = {v: 1.0 / n_eff for v in pop}
    for _ in range(n_views_total + 1):
        capped = [v for v in pop if weights[v] > cap_share + 1e-9]
        if not capped:
            break
        excess = sum(weights[v] - cap_share for v in capped)
        for v in capped:
            weights[v] = cap_share
        uncapped = [v for v in pop if v not in capped]
        if not uncapped:
            # Theoretically prevented by the early check above, but defend.
            logger.warning("view_cap re-distribution exhausted uncapped views.")
            return {v: 1.0 / n_eff for v in pop}
        per = excess / len(uncapped)
        for v in uncapped:
            weights[v] += per
    # Numerical clean-up: renormalize.
    s = sum(weights.values())
    if s > 0:
        weights = {v: w / s for v, w in weights.items()}
    return weights


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
        gf_present = bool(plan.global_grounded_ranges) or plan.stratified_gf
        if target_floor is not None and not gf_present:
            logger.warning("target_floor=%.3f but GLOBAL_GROUNDED pool is empty — "
                           "GF will have no effect (falls through to CURRENT).",
                           target_floor)
        if plan.stratified_gf and not plan.global_grounded_view_weights:
            raise ValueError("Stratified GF plan has buckets but no view weights — "
                             "build_pool_plan should have populated both.")
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

    def _draw_from_view_buckets(self, rng: random.Random) -> int:
        """T3 stratified GF: three-stage view -> cat-within-view -> sample.

        1. Pick view from `global_grounded_view_weights` (post-cap, post-drop).
        2. Pick a bucket within that view uniformly across the (possibly
           multiple) categories that have grounding in that view.
        3. Pick a sample uniformly from the bucket's `indices`.

        Exposure key is `f"v{v}/{cat}"` so the verify_sampler report can
        roll up per-view AND per-(view, cat).
        """
        buckets = self.plan.global_grounded_view_buckets
        weights = self.plan.global_grounded_view_weights

        # View pick (weighted).
        views = list(weights.keys())
        ws    = [weights[v] for v in views]
        v_pick = rng.choices(views, weights=ws, k=1)[0]

        # Cat pick within the chosen view (uniform among buckets with this view).
        candidate_buckets = [b for b in buckets if b.view == v_pick]
        b = candidate_buckets[rng.randrange(len(candidate_buckets))]

        # Sample pick within the bucket.
        sample_abs_idx = b.indices[rng.randrange(len(b.indices))]
        self._exposure[("global_grounded", b.name)] += 1
        return sample_abs_idx

    def __iter__(self) -> Iterator[int]:
        rank, world_size = self._ddp_rank_world()
        rng = random.Random(self.seed + rank + self._iter_count * 1009)
        self._iter_count += 1
        n_per_rank = self.num_samples // max(1, world_size)
        out: List[int] = []
        gf_pool_present = (
            self.plan.stratified_gf or bool(self.plan.global_grounded_ranges)
        )
        tf = (self.target_floor if (self.target_floor is not None
                                     and gf_pool_present) else 0.0)

        for _ in range(n_per_rank):
            r1 = rng.random()
            if r1 < self.p_replay:
                out.append(self._draw_from_ranges(self.plan.prior_ranges, rng, "prior"))
                continue
            r2 = rng.random()
            if r2 < tf:
                if self.plan.stratified_gf:
                    out.append(self._draw_from_view_buckets(rng))
                else:
                    out.append(self._draw_from_ranges(
                        self.plan.global_grounded_ranges, rng, "global_grounded"))
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
    # T3 — stratified GF (default off keeps existing call sites unchanged).
    global_grounded_per_sample_views: Optional[Sequence[Sequence[int]]] = None,
    view_stratified: bool = False,
    view_stratified_min_bucket: int = 50,
    view_stratified_view_cap: float = 2.0,
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
        global_grounded_per_sample_views=(
            global_grounded_per_sample_views if (target_floor and view_stratified) else None
        ),
        view_stratified=bool(view_stratified and target_floor),
        view_stratified_min_bucket=view_stratified_min_bucket,
        view_stratified_view_cap=view_stratified_view_cap,
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
    "ViewBucket",
    "PoolPlan",
    "build_pool_plan",
    "CompositeSampler",
    "build_sampler",
]
