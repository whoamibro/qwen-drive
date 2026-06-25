"""
Token-role mask builders for the composite SFT loss (v2 training).

Given a tokenized assistant-turn token list (no padding), this module emits
per-token boolean masks aligned to that token list:

- answer_token_mask  : tokens inside the `"answer": "<value>"` JSON string body
- gate_token_mask    : the single token immediately after `"grounding":`,
                       which encodes empty (` [],\n` -> 10239) vs non-empty
                       (` [\n` -> 2278) presence
- coord_token_mask   : tokens strictly between each `<|box_start|>` /
                       `<|box_end|>` pair (the digit/comma coordinate tokens)
- image_idx_mask     : the digit token (one of '1'..'6') that gives the
                       camera view for each grounding entry
- image_idx_target   : long-tensor with class id (0..5 = view1..view6) at
                       `image_idx_mask` positions, IGNORE_INDEX elsewhere
- coord_box_id       : long-tensor with the 0-indexed box id at coord-token
                       positions, IGNORE_INDEX elsewhere (used by the IoU-aware
                       term to group coord tokens by box)

The masks/targets are built from token-id landmarks, NOT from re-decoding the
text, so they are robust to whitespace / formatting drift inside the
tokenizer's output. Landmarks are validated against the Qwen3-VL 8B Instruct
tokenizer (see analysis md / build verification).

If the tokenizer changes (different model, custom vocab), regenerate
LANDMARK_IDS by encoding the relevant substrings — see `verify_landmarks`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

IGNORE_INDEX = -100

# ---- Token IDs for Qwen3-VL 8B Instruct (verified 2025-12; see top docstring).
# Special tokens.
BOX_START_ID = 151648
BOX_END_ID = 151649
OBJ_REF_START_ID = 151646
OBJ_REF_END_ID = 151647
IM_END_ID = 151645

# Single-digit tokens for image_idx (' 1' .. ' 6' actually encode as digit-only
# when preceded by `: `; the preceding ' ' token is 220).
VIEW_DIGIT_IDS: Tuple[int, ...] = (16, 17, 18, 19, 20, 21)
SPACE_ID = 220  # ' '

# Landmark 3-grams / 4-grams (each entry produced by tokenizer.encode of the
# bracket-quoted substring with add_special_tokens=False; see verify_landmarks).
GROUNDING_LANDMARK: Tuple[int, ...] = (1951, 287, 788)        # ground + ing + ":"
ANSWER_LANDMARK:    Tuple[int, ...] = (9217, 788, 330)        # answer + ":" + ' "'
IMAGE_IDX_LANDMARK: Tuple[int, ...] = (1805, 7258, 788, 220)  # image + _idx + ":" + ' '

# Gate-target codebook: token-id that the model sees in `labels` at the
# gate position. The set is closed (verified across 800 mixed samples):
EMPTY_GROUNDING_TOKEN = 10239   # ' [],\n'
OPEN_GROUNDING_TOKEN  = 2278    # ' [\n'
GATE_TARGET_TOKENS: Tuple[int, ...] = (EMPTY_GROUNDING_TOKEN, OPEN_GROUNDING_TOKEN)


@dataclass
class TokenRoleMasks:
    """All masks/targets are length T == len(token_ids); image_idx_target
    and coord_box_id store IGNORE_INDEX at off-positions."""
    answer_token_mask: List[bool]
    gate_token_mask:   List[bool]
    coord_token_mask:  List[bool]
    image_idx_mask:    List[bool]
    image_idx_target:  List[int]   # 0..5 at mask positions, else IGNORE_INDEX
    coord_box_id:      List[int]   # 0..N-1 at mask positions, else IGNORE_INDEX


def _has_quote(decoded: str) -> bool:
    return '"' in decoded


def build_assistant_masks(
    token_ids: Sequence[int],
    tokenizer,
) -> TokenRoleMasks:
    """Construct token-role masks for a single assistant turn.

    Args:
        token_ids: full assistant-turn token IDs (includes leading
            `<|im_start|>assistant\\n` and trailing `<|im_end|>\\n`).
        tokenizer: HF tokenizer (used only to decode single tokens for the
            answer-span end check).

    Returns:
        TokenRoleMasks with each list of length len(token_ids).
    """
    T = len(token_ids)
    answer_mask  = [False] * T
    gate_mask    = [False] * T
    coord_mask   = [False] * T
    image_idx_mask    = [False] * T
    image_idx_target  = [IGNORE_INDEX] * T
    coord_box_id      = [IGNORE_INDEX] * T

    # --- coord_token_mask + coord_box_id ---
    in_box = False
    next_box_id = 0
    cur_box_id = -1
    for i, tid in enumerate(token_ids):
        if tid == BOX_START_ID:
            in_box = True
            cur_box_id = next_box_id
            next_box_id += 1
        elif tid == BOX_END_ID and in_box:
            in_box = False
            cur_box_id = -1
        elif in_box:
            coord_mask[i] = True
            coord_box_id[i] = cur_box_id

    # --- image_idx_mask + image_idx_target ---
    n_landmark = len(IMAGE_IDX_LANDMARK)
    for i in range(T - n_landmark):
        if tuple(token_ids[i:i + n_landmark]) == IMAGE_IDX_LANDMARK:
            j = i + n_landmark
            if j < T and token_ids[j] in VIEW_DIGIT_IDS:
                image_idx_mask[j] = True
                # '1' is class 0, '6' is class 5
                image_idx_target[j] = token_ids[j] - VIEW_DIGIT_IDS[0]

    # --- gate_token_mask ---
    g_n = len(GROUNDING_LANDMARK)
    for i in range(T - g_n):
        if tuple(token_ids[i:i + g_n]) == GROUNDING_LANDMARK:
            j = i + g_n
            if j < T and token_ids[j] in GATE_TARGET_TOKENS:
                gate_mask[j] = True

    # --- answer_token_mask ---
    # Walk forward from the landmark; mark tokens whose decoded text contains
    # no `"` (closing quote terminator). Robust to merged closing-quote tokens
    # like `'"\n'` or `'",\n'` — those contain `"` and stop the span.
    a_n = len(ANSWER_LANDMARK)
    for i in range(T - a_n):
        if tuple(token_ids[i:i + a_n]) == ANSWER_LANDMARK:
            j = i + a_n
            while j < T:
                tok_text = tokenizer.decode([token_ids[j]])
                if _has_quote(tok_text):
                    break
                answer_mask[j] = True
                j += 1
            break  # there is at most one answer field per assistant turn

    return TokenRoleMasks(
        answer_token_mask=answer_mask,
        gate_token_mask=gate_mask,
        coord_token_mask=coord_mask,
        image_idx_mask=image_idx_mask,
        image_idx_target=image_idx_target,
        coord_box_id=coord_box_id,
    )


def verify_landmarks(tokenizer) -> dict:
    """Sanity-check that the hard-coded landmark token IDs still match the
    tokenizer's current encoding. Returns a dict of {expected: actual}; call
    sites should `assert all(a == e for e, a in result.values())`.
    """
    out = {}

    def chk_suffix(name, expected, text):
        """Encode `text` and check whether `expected` is a suffix of the
        result. Standalone-encoding prepends a leading '"' token that does
        not occur inside the real conversation, so a suffix check matches
        the in-context landmarks correctly."""
        actual = tuple(tokenizer.encode(text, add_special_tokens=False))
        ok = len(actual) >= len(expected) and tuple(actual[-len(expected):]) == expected
        out[name] = (expected, actual if not ok else expected)

    chk_suffix("grounding_landmark", GROUNDING_LANDMARK, '"grounding":')
    chk_suffix("answer_landmark",    ANSWER_LANDMARK,    '"answer": "')
    # image_idx landmark starts at 'image' (1805), but encoded standalone its
    # prefix may differ; the 4-gram check below mirrors how we scan.
    img = tokenizer.encode('"image_idx": 1', add_special_tokens=False)
    # Strip leading quote tokens; look for the 5-gram ending with ' '
    # We just verify the suffix matches IMAGE_IDX_LANDMARK + space + digit.
    suffix_ok = (
        len(img) >= 5
        and tuple(img[-5:-1]) == IMAGE_IDX_LANDMARK
        and img[-1] == VIEW_DIGIT_IDS[0]
    )
    out["image_idx_landmark"] = (True, suffix_ok)

    for name, sid in [
        ("box_start", BOX_START_ID), ("box_end", BOX_END_ID),
        ("obj_ref_start", OBJ_REF_START_ID), ("obj_ref_end", OBJ_REF_END_ID),
    ]:
        actual = tokenizer.convert_tokens_to_ids(
            {"box_start": "<|box_start|>", "box_end": "<|box_end|>",
             "obj_ref_start": "<|object_ref_start|>",
             "obj_ref_end": "<|object_ref_end|>"}[name]
        )
        out[name] = (sid, actual)

    for c, expected in enumerate(VIEW_DIGIT_IDS):
        actual = tokenizer.encode(str(c + 1), add_special_tokens=False)
        out[f"digit_{c+1}"] = (expected, actual[0] if len(actual) == 1 else actual)

    return out
