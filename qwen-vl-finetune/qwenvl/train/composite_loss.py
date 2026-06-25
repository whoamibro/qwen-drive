"""
Composite SFT loss for grounded multi-task VQA (Qwen3-VL).

Implements the loss surface from `loss_improvement_design_v3.md`:

    L = L_ce                           # base causal LM CE (all assistant tokens)
      + w_ans   * L_ans                # answer-field weighting (all samples)
      + w_gate  * L_gate               # presence-gate (all samples)
      + lam_view * L_view              # 6-way view classification on image_idx
      + lam_iou  * L_iou               # Tier-2 IoU-aware coord-token CE
      + lam_klal * L_klal              # KL attention loss (scaffolded, off by default)

All auxiliary terms are normalized by the **relevant count** (box count,
grounded-coord count, grounded-sample count) — never by full batch — so the
effective lambda doesn't silently scale with batch composition (md Section
8). Empty-mask short-circuits return zeros (no NaN). Every aux term is
computed in fp32 even under bf16 training.

Token-position losses use logits[:, :-1] predicting labels[:, 1:]; masks are
taken in the shifted frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .token_role_masks import IGNORE_INDEX, VIEW_DIGIT_IDS, BOX_START_ID, BOX_END_ID


# ---------------------------------------------------------------------------
# Loss config
# ---------------------------------------------------------------------------
@dataclass
class CompositeLossConfig:
    w_ans: float = 0.0
    w_gate: float = 0.0
    lam_view: float = 0.0
    lam_iou: float = 0.0
    lam_klal: float = 0.0           # KLAL execution gated to 0 in this version
    klal_layers: Tuple[int, ...] = (-1,)
    # IoU-aware CE knobs
    iou_alpha: float = 1.0          # w_b = 1 + iou_alpha * max(0, 1 - GIoU_b)
    iou_pseudo_M: int = 0           # Tier-1 only; unused for Tier-2
    iou_giou_band: Tuple[float, float] = (0.5, 0.95)  # Tier-1 only
    # View-classification weights (optional, length 6)
    view_class_weight: Optional[List[float]] = None

    @property
    def any_aux_enabled(self) -> bool:
        return any(
            x > 0
            for x in (self.w_ans, self.w_gate, self.lam_view,
                      self.lam_iou, self.lam_klal)
        )

    @property
    def needs_output_attentions(self) -> bool:
        return self.lam_klal > 0


# ---------------------------------------------------------------------------
# Base CE — manual implementation that preserves num_items_in_batch
# ---------------------------------------------------------------------------
def base_ce(
    logits: torch.Tensor,           # (B, T, V)
    labels: torch.Tensor,           # (B, T)
    num_items_in_batch: Optional[int] = None,
) -> torch.Tensor:
    """Standard causal-LM CE, shift-by-one, ignore_index=-100, sum-reduction
    then divide by `num_items_in_batch` so gradient accumulation is exact.

    Mirrors HF `ForCausalLMLoss` semantics (transformers >=4.46).
    """
    sl = logits[:, :-1, :].contiguous()
    st = labels[:, 1:].contiguous()
    loss = F.cross_entropy(
        sl.view(-1, sl.size(-1)).float(),
        st.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )
    if num_items_in_batch is None or num_items_in_batch <= 0:
        denom = (st != IGNORE_INDEX).sum().clamp_min(1).to(loss.dtype)
    else:
        denom = torch.as_tensor(float(num_items_in_batch), device=loss.device, dtype=loss.dtype)
    return loss / denom


# ---------------------------------------------------------------------------
# Masked token CE — shared helper for answer & gate terms
# ---------------------------------------------------------------------------
def masked_token_ce(
    logits: torch.Tensor,           # (B, T, V)
    labels: torch.Tensor,           # (B, T)
    mask: torch.Tensor,             # (B, T) bool, in label frame
) -> torch.Tensor:
    """Mean CE over positions where `mask` is True. Mask is taken in the
    shifted (label) frame, so `mask[:, 1:]` picks the prediction targets."""
    if mask.sum() == 0:
        return logits.new_zeros((), dtype=torch.float32)
    sl = logits[:, :-1, :]
    st = labels[:, 1:]
    m = mask[:, 1:]
    if m.sum() == 0:
        return logits.new_zeros((), dtype=torch.float32)
    sel = sl[m].float()                  # (N_sel, V)
    tgt = st[m]                          # (N_sel,)
    return F.cross_entropy(sel, tgt, reduction="mean")


# ---------------------------------------------------------------------------
# View classification (6-way restricted softmax over digit-token IDs)
# ---------------------------------------------------------------------------
def view_classification_loss(
    logits: torch.Tensor,           # (B, T, V)
    image_idx_mask: torch.Tensor,   # (B, T) bool
    image_idx_target: torch.Tensor, # (B, T) long; class 0..5 at mask, else IGNORE
    view_token_ids: torch.Tensor,   # (6,) long — fixed digit-token IDs
    class_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """6-way CE on the image_idx position's logits restricted to the 6
    single-token view digits '1'..'6'. Sharper than full-vocab CE, supports
    inverse-frequency class weights for view imbalance."""
    m = image_idx_mask[:, 1:]
    if m.sum() == 0:
        return logits.new_zeros((), dtype=torch.float32)
    sl = logits[:, :-1, :]
    tgt = image_idx_target[:, 1:]

    sel = sl[m]                                   # (N_boxes, V)
    sel6 = sel[:, view_token_ids].float()         # (N_boxes, 6) fp32
    tgt_cls = tgt[m]                              # (N_boxes,) in 0..5
    # F.cross_entropy requires `weight` dtype to match input dtype (fp32 here).
    # Cast defensively — caller may pass it in bf16 if built from logits.dtype.
    if class_weight is not None:
        class_weight = class_weight.to(dtype=sel6.dtype)
    return F.cross_entropy(sel6, tgt_cls, weight=class_weight, reduction="mean")


# ---------------------------------------------------------------------------
# IoU-aware CE (Tier 2) — greedy-decode boxes, GIoU-weight coord-token CE
# ---------------------------------------------------------------------------
def parse_box_token_ids(
    tokenizer,
    coord_token_ids: Sequence[int],
) -> Optional[Tuple[int, int, int, int]]:
    """Decode a sequence of coordinate token IDs back to (x1, y1, x2, y2).

    Tier-2 uses argmax-decoded predicted IDs (not free generation), so the
    sequence is GT-length but values can be wrong. Returns None on parse
    failure — callers must treat parse-failure as "no IoU signal" (weight=1
    or skip).
    """
    if not coord_token_ids:
        return None
    text = tokenizer.decode(list(coord_token_ids))
    # Expected form: "(x1,y1),(x2,y2)" — strip spaces/newlines defensively.
    text = text.replace(" ", "").replace("\n", "")
    if not (text.startswith("(") and text.endswith(")")):
        return None
    try:
        inside = text[1:-1]                       # "x1,y1),(x2,y2"
        a, b = inside.split("),(")
        x1, y1 = a.split(",")
        x2, y2 = b.split(",")
        bx = (int(x1), int(y1), int(x2), int(y2))
    except (ValueError, IndexError):
        return None
    # Coords are 0..1000 normalized; sanity-clamp.
    if not all(0 <= v <= 1000 for v in bx):
        return None
    if bx[0] >= bx[2] or bx[1] >= bx[3]:
        return None
    return bx


def giou_xyxy(a: Tuple[float, float, float, float],
              b: Tuple[float, float, float, float]) -> float:
    """Generalized IoU on two boxes in (x1,y1,x2,y2). Returns scalar float
    in [-1, 1]. Pure-Python so it stays off the autograd path; callers
    .detach() before using as a CE weight."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    iou = inter / union
    enc_x1 = min(ax1, bx1); enc_y1 = min(ay1, by1)
    enc_x2 = max(ax2, bx2); enc_y2 = max(ay2, by2)
    enc_area = max(0.0, enc_x2 - enc_x1) * max(0.0, enc_y2 - enc_y1)
    if enc_area <= 0:
        return iou
    return iou - (enc_area - union) / enc_area


def iou_aware_ce_tier2(
    logits: torch.Tensor,           # (B, T, V)
    labels: torch.Tensor,           # (B, T)
    coord_token_mask: torch.Tensor, # (B, T) bool
    coord_box_id: torch.Tensor,     # (B, T) long; box idx at coord positions, else IGNORE
    gt_boxes: Sequence[Sequence[Tuple[int, int, int, int]]],  # per-sample list of (x1,y1,x2,y2)
    box_image_idx: Sequence[Sequence[int]],                   # per-sample list of 1..6
    tokenizer,
    iou_alpha: float = 1.0,
) -> torch.Tensor:
    """Tier-2 IoU-aware coord-token CE.

    For each sample with grounding:
      1. Group coord-token positions by `coord_box_id` -> per-box position list.
      2. At those positions, argmax `logits[:, :-1, :]` to get predicted token
         IDs (greedy decode under teacher forcing).
      3. Parse predicted IDs to a box; compute GIoU vs the GT box of the same
         box id. **Same image_idx** is guaranteed by construction (each
         coord_box_id maps to a single GT box which carries its own image_idx).
      4. Detach GIoU; weight that box's coord-token CE by
            w_b = 1 + iou_alpha * max(0, 1 - GIoU_b).
      5. Sum weighted box-CE across the batch, normalize by total
         grounded-coord token count (md Section 8).

    Returns 0 on no grounded coord tokens.
    """
    sl = logits[:, :-1, :]               # (B, T-1, V)
    st = labels[:, 1:]                   # (B, T-1)
    m  = coord_token_mask[:, 1:]         # (B, T-1)
    bid = coord_box_id[:, 1:]            # (B, T-1)

    total_coord = m.sum()
    if total_coord == 0:
        return logits.new_zeros((), dtype=torch.float32)

    # Per-position CE (no reduction), in fp32, only at coord positions.
    sel_logits = sl[m].float()                   # (N_coord, V)
    sel_targets = st[m]                          # (N_coord,)
    ce_per_pos = F.cross_entropy(sel_logits, sel_targets, reduction="none")  # (N_coord,)

    # Per-position predicted token (argmax under teacher forcing).
    pred_ids = sel_logits.argmax(dim=-1).tolist()          # (N_coord,)
    # Per-position (sample, box) ownership.
    pos_sample = m.nonzero(as_tuple=False)[:, 0].tolist()  # (N_coord,)
    pos_box = bid[m].tolist()                              # (N_coord,)

    # Group positions by (sample, box_id).
    from collections import defaultdict
    groups: dict[tuple, List[int]] = defaultdict(list)
    for i, (s_i, b_i) in enumerate(zip(pos_sample, pos_box)):
        if b_i < 0:
            continue
        groups[(s_i, b_i)].append(i)

    device = logits.device
    weighted_sum = torch.zeros((), device=device, dtype=torch.float32)
    for (s_i, b_i), positions in groups.items():
        # Bounds-check against the per-sample GT box list.
        try:
            gt_box = tuple(gt_boxes[s_i][b_i])
        except (IndexError, TypeError):
            w_b = 1.0   # no GT -> just standard CE weight
        else:
            pred_ids_b = [pred_ids[p] for p in positions]
            pred_box = parse_box_token_ids(tokenizer, pred_ids_b)
            if pred_box is None:
                # Parse failure: penalize moderately by treating GIoU = -1
                # (so w_b = 1 + 2*iou_alpha). Capped by the (1 - GIoU) form.
                giou = -1.0
            else:
                giou = giou_xyxy(pred_box, gt_box)
            w_b = 1.0 + iou_alpha * max(0.0, 1.0 - giou)

        # Sum CE of this box's positions, scaled by w_b (detached scalar).
        idx = torch.tensor(positions, device=device, dtype=torch.long)
        weighted_sum = weighted_sum + ce_per_pos[idx].sum() * float(w_b)

    return weighted_sum / total_coord.to(weighted_sum.dtype).clamp_min(1.0)


# ---------------------------------------------------------------------------
# KLAL — KL Attention Loss (scaffolded; execution disabled in this version)
# ---------------------------------------------------------------------------
def klal_loss_stub(
    attentions,
    klal_layers,
    gt_boxes,
    box_image_idx,
    has_grounding,
) -> torch.Tensor:
    """Placeholder for KLAL (md Section 2.6). Returning 0 means the loss
    surface is unchanged. Enable by:
      1. Wiring `output_attentions=True` in the forward pass (one-shot,
         DDP-safe; do not toggle data-dependently).
      2. Replacing this stub with the KL-divergence vs GT-attention map
         described in Esmaeilkhani & Latecki (WACV 2026).

    Kept off in this PR (lam_klal=0) so FlashAttention stays on.
    """
    return torch.zeros((), dtype=torch.float32)
