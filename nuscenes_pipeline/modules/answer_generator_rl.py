#!/usr/bin/env python3
"""
RL (GRPO) Answer Generator Module for nuScenes QA Dataset

Fork of answer_generator.py producing RL-training data with three structural
changes over the SFT module (spec: thinking_answer_generator_4_rl_impl.md):

  1. Tier-banded "think" array per pair side — the student model's <think>
     learning target. The teacher's native <think> preamble is uncontrollable
     and stripped by the parser; the JSON "think" field is the contract.
  2. Fixed key order think -> reasoning -> answer (autoregressive ordering
     forces deliberation before the answer tokens).
  3. "contrast_status" self-report (achieved / same_answer / skipped) replaces
     the v1 must-differ constraint that incentivized label fabrication.
     same_answer pairs are emitted honestly and harvested downstream as
     unpaired single QA items.

The grounding contract (tag_mappings / selected_obj_id / grounded_description /
relevant_cameras) is FROZEN — identical to answer_generator.py.

Prerequisites:
    # Start vLLM server first (Thinking model, NO --reasoning-parser so the
    # trace stays inline for auditability; the parser strips it):
    vllm serve Qwen/Qwen3-VL-235B-A22B-Thinking --tensor-parallel-size 8 \\
        --max-model-len 65536

Usage:
    # Auto-discover samples from Stage 2 outputs
    python -m nuscenes_pipeline.modules.answer_generator_rl --from_stage1 --num_workers 8

    # Smoke test: 5 explicit samples, 2 templates per category
    python -m nuscenes_pipeline.modules.answer_generator_rl \\
        --sample_indices "100,228,304,371,759" --max_templates_per_category 2
"""

import os
import io
import sys
import json
import re
import time
import base64
import argparse
import random
import itertools
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from multiprocessing import Pool
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

from nuscenes_pipeline.core.nuscenes_data_loader import NuScenesDataLoader
from nuscenes_pipeline.core.qa_utils import (
    SceneAnalyzer, CAMERA_NAMES, CAMERA_NAME_MAP, VEHICLE_TYPES,
    VALID_CATEGORIES, load_question_bank, get_category_templates,
)
from nuscenes_pipeline.modules.question_selector import (
    load_risk_assessment_response,
    load_traffic_analysis_response,
    load_traffic_sign_response,
)


# =============================================================================
# Constants
# =============================================================================

EGOCENTRIC_CAMERA_NAMES = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
]
REAR_CAMERAS = {'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'}
CAM_LABELS = {
    "CAM_FRONT_LEFT": "Image 1: Front-left Camera",
    "CAM_FRONT": "Image 2: Front Camera",
    "CAM_FRONT_RIGHT": "Image 3: Front-right Camera",
    "CAM_BACK_LEFT": "Image 4: Rear-left Camera",
    "CAM_BACK": "Image 5: Rear Camera",
    "CAM_BACK_RIGHT": "Image 6: Rear-right Camera",
}
# Camera index mapping (1-based)
CAM_INDEX = {
    'CAM_FRONT_LEFT': 1, 'CAM_FRONT': 2, 'CAM_FRONT_RIGHT': 3,
    'CAM_BACK_LEFT': 4, 'CAM_BACK': 5, 'CAM_BACK_RIGHT': 6,
}

# =============================================================================
# Think-budget tiers (T3) — category -> (min_steps, max_steps)
# One "think" array element = one thinking step; len() is the band-check unit.
# =============================================================================

THINK_BANDS = {
    # T1 — perception / enumeration: evidence citation -> (view check) -> conclusion
    "Observation": (2, 4),
    "Identification": (2, 4),
    "Environmental_and_Sensor_Conditions": (2, 4),
    # T2 — perception + applicability mapping
    "Traffic_Signs_and_Signals": (3, 5),
    "Road_Markings_and_Lane_Configuration": (3, 5),
    "Attributes_and_States": (3, 5),
    # T3 — relational / dynamic multi-object
    "Spatial_Relationships_and_Occlusion": (4, 6),
    "Dynamic_Agents_and_Risk_Assessment": (4, 6),
    # T4 — rule / causal chains
    "Right_of_Way_and_Planning": (5, 8),
    "Causal_and_Hypothetical_Reasoning": (5, 8),
}

# Absent-premise short-circuit (T5): overrides the category band.
ABSENT_PREMISE_STEPS = 2

# Threshold-subjective categories where contrast is OPTIONAL (T6c) — v1
# quarantine rates: DRA 52%, RML 64%.
CONTRAST_OPTIONAL_CATEGORIES = {
    "Dynamic_Agents_and_Risk_Assessment",
    "Road_Markings_and_Lane_Configuration",
}

VALID_CONTRAST_STATUS = {"achieved", "same_answer", "skipped"}


def _think_budget_block() -> str:
    """Render the Think Budget prompt section from THINK_BANDS (single source
    of truth shared with the band validator)."""
    lines = [
        "",
        "",
        "=" * 72,
        "  SECTION 10: THINK BUDGET & OUTPUT CONTRACT (STRICT)",
        "=" * 72,
        "",
        "Within each pair's \"positive\"/\"contrastive\" objects, generate keys",
        "IN THIS EXACT ORDER:",
        "  1. \"tag_mappings\"          — unchanged format (grounded object selection)",
        "  2. \"instantiated_question\"",
        "  3. \"mcq_options\"           — mcq answer_type only",
        "  4. \"think\"                 — an ARRAY of step strings (see Think Budget)",
        "  5. \"reasoning\"             — at most 5 evidence-citing bullets (existing format)",
        "  6. \"answer\"                — MUST be the final decision key. Derive it only",
        "                               after \"think\" and \"reasoning\" are complete.",
        "  7. \"confidence\", \"relevant_cameras\"",
        "  8. \"contrast_status\"       — contrastive objects only (see Contrastive Protocol)",
        "",
        "## Think Budget (per category)",
        "Write \"think\" as first-person deliberation steps. One array element =",
        "one thinking step. Each step must cite concrete evidence (OBJ id,",
        "Image index, or distance). Step count MUST fall in the band:",
    ]
    band_to_cats = {}
    for cat in VALID_CATEGORIES:
        band_to_cats.setdefault(THINK_BANDS[cat], []).append(cat)
    for (lo, hi), cats in sorted(band_to_cats.items()):
        lines.append(f"  - {' / '.join(cats)}:")
        lines.append(f"      {lo}-{hi} steps")
    lines += [
        "",
        "OVERRIDE: if the questioned entity/place is ABSENT from the scene,",
        f"\"think\" is EXACTLY {ABSENT_PREMISE_STEPS} steps: (1) state the absence with the views",
        "checked, (2) apply the non-existent-reference rule (SECTION 6).",
        "",
        "STYLE: natural first-person deliberation (\"Looking at Image 2, OBJ 21",
        "is a car 12.9m directly ahead...\"). Do NOT state the final answer",
        "inside \"think\"; the last step goes up to the judgment just before the",
        "conclusion — the answer string appears only in the \"answer\" key.",
        "Do NOT copy or restate \"reasoning\" bullets — \"think\" is the",
        "derivation process, \"reasoning\" is the committed summary.",
        "",
        "## Hard Conditions",
        "- \"tag_mappings\" format is frozen; do not add/rename grounding fields.",
        "- \"answer\" is always the last decision; never pre-commit it in \"think\".",
    ]
    return "\n".join(lines)


SYSTEM_PROMPT = """You are a driving expert agent, and should answer the question at the viewpoint of a driver.

You are analyzing 6 surround-view camera images from an ego vehicle.
The vehicle uses a right-handed coordinate system (FLU - Forward-Left-Up):
- X: forward (positive = ahead of ego)
- Y: left (positive = left of ego)
- Z: up

All spatial references must be from the DRIVER'S PERSPECTIVE:
- "ahead" / "behind" = along ego's forward axis
- "left" / "right" = from the driver's seat
- Distances are measured from the ego vehicle center

=== CAMERA SYSTEM (Egocentric order) ===

| Image | Camera | Viewing Direction | Notes |
|-------|--------|-------------------|-------|
| 1 | Front-left (CAM_FRONT_LEFT) | Forward-left scene | Left-front of ego |
| 2 | Front (CAM_FRONT) | Straight ahead | Road, traffic lights, vehicles ahead |
| 3 | Front-right (CAM_FRONT_RIGHT) | Forward-right scene | Right-front of ego |
| 4 | Rear-left (CAM_BACK_LEFT) | Left-rear scene | Horizontally flipped for egocentric view |
| 5 | Rear (CAM_BACK) | Straight behind | Horizontally flipped for egocentric view |
| 6 | Rear-right (CAM_BACK_RIGHT) | Right-rear scene | Horizontally flipped for egocentric view |

Rear camera images (4, 5, 6) are horizontally flipped so that left stays left
and right stays right from the driver's perspective.

IMPORTANT — Ego-Centric Perspective:
All observations and reasoning must be anchored to the ego-vehicle's position, heading, and driving context.
- Spatial references (e.g., "ahead", "left lane", "behind") are relative to the ego-vehicle, NOT absolute coordinates.
- "Relevance" means relevance to the ego-vehicle's current driving situation — objects, signals, and road features matter only insofar as they affect the ego-vehicle's path, decisions, or safety.
- When a question mentions the ego-vehicle's lane, path, or vicinity, ground those terms using the camera views and prior knowledge from the ego-vehicle's perspective.
- Answers and reasoning must reflect what the ego-vehicle's driver would observe and conclude from the provided views.

CRITICAL: Before referencing any object or feature, VERIFY which image number
it appears in. Do not assume — check the images.

=== YOUR ROLE ===

You are BOTH a driving expert analyzing the scene AND a QA generator.
When answering questions, reason as if you are the driver sitting in the
ego-vehicle, observing the surrounding environment through these 6 cameras.

You have been given VALIDATED question templates from Stage 1. For each template,
your job is to generate MULTIPLE CONTRASTIVE QA PAIRS by systematically varying
placeholder combinations:
  (A) CLASSIFY each placeholder as ENTITY-type or LEXICAL-type
  (B) Generate MULTIPLE POSITIVE instances — each using DIFFERENT placeholder combinations
  (C) For EACH positive, generate a paired CONTRASTIVE instance, answer BOTH
      independently from evidence, and report contrast_status honestly
  (D) Ensure DIVERSITY across pairs: different objects, locations, comparisons, and cameras


========================================================================
  PRIOR APPLICABILITY VERIFICATION RULE
========================================================================

The user prompt may include PRIOR ANALYSIS sections:
  - Stage 1A (Risk Assessment): GROUND TRUTH — derived from 3D detection +
    geometry. Accept all claims unconditionally.
  - Stage 1B (Traffic Signal Analysis): HIGH-CONFIDENCE REFERENCE (~90%+).
  - Stage 1C (Traffic Sign Extraction): HIGH-CONFIDENCE REFERENCE (~90%+).

For signal and sign priors, before accepting their ego-applicability claims,
run an INDEPENDENT orientation + road-alignment check against the camera
images:

FOR SIGNS (each "[SIGN-SCAN] Image N: ... Applies to ego: y" entry in Stage 1C):
  - Is the signboard face-on (readable proportions) or edge-on / rear?
  - Is the post on ego's road, or on a perpendicular cross-street?
  - Does the sign's scope match ego's driving command? (e.g., "No Left Turn"
    for a "Go straight" ego is NOT ego's obligation)

FOR SIGNALS (the "[SELECTED] Image: N | Signal: ..." claim in Stage 1B):
  - Does the selected signal face ego's approach (light panels visible head-on)?
  - Are vehicles near it traveling along ego's path, or across it?

Decision rule:
  1. AGREE with prior → use the prior's claim in your answer.
  2. DISAGREE with prior → STILL USE THE PRIOR'S CLAIM as the final answer
     (for dataset consistency), BUT emit a "prior_disagreements" list in the
     output JSON alongside the answer. Each entry:
       {
         "source": "signal" | "sign",
         "image_idx": <int 1-6>,
         "object": "<brief identifier, e.g. 'STOP sign', 'red traffic light'>",
         "prior_says": "applies to ego: y" | "applies to ego: n" | "selected" | "not selected",
         "vlm_says": "applies to ego: y" | "applies to ego: n" | "selected" | "not selected",
         "evidence": "<one sentence citing what the images show>"
       }

Omit "prior_disagreements" entirely when there are no mismatches.
Do NOT use this field for signal-STATE disagreements (red/yellow/green) or
for sign-category disagreements — only for EGO-APPLICABILITY claims.


========================================================================
  SECTION 1: PLACEHOLDER CLASSIFICATION
========================================================================

Before processing any placeholder, classify it into one of two types.

--- ENTITY-type ---
Refers to a physical object with a 3D position in the scene, mappable to an OBJ ID.
Must be grounded with the full four-layer description.

  Definite ENTITY tags:
    <object>, <object_A>, <object_B>, <vehicle>, <other_vehicle>,
    <special_vehicle>, <agent>, <agent_type>

--- LEXICAL-type ---
Refers to a scene attribute, spatial term, condition, or linguistic element
that does NOT map to a specific OBJ ID. Selected as-is from candidate lists.

  Definite LEXICAL tags:
    <place>, <preposition>, <direction>, <position>, <view>, <views>,
    <color>, <state>, <action>, <type>, <lane>, <turn>, <turn_direction>,
    <turn_type>, <side>, <condition>, <zone>, <area>, <location>,
    <sign>, <sign_type>, <speed_value>, <mount_type>, <orientation>,
    <feature>, <identifier>, <information>, <restriction>, <alternative>,
    <other>, <arrow_direction>, <confidence>, <risk_level>, <level>,
    <threshold>, <trend>, <collision_type>, <accident_type>,
    <special_class>, <traffic_sign_type>, <signal>, <indicator>

--- AMBIGUOUS tags (classify by context) ---
    <traffic_element>, <signal_type>, <barrier>, <obstacle>, <cause>, <traffic>

  Decision rule: "Does this placeholder value in THIS template refer to a
  specific physical object with a 3D position?"
    → YES: treat as ENTITY (apply grounding rules)
    → NO: treat as LEXICAL (select from candidates)

  Examples:
    - <traffic_element> in "Is the <traffic_element> clearly visible?"
      → values are "traffic light", "stop sign" — these are physical objects → ENTITY
    - <signal_type> in "Are there countdown timers next to the <signal_type>?"
      → values are "traffic light", "pedestrian signal" — physical objects → ENTITY
    - <barrier> in "Is the <barrier> open or closed?"
      → values are "toll gate", "railroad crossing gate" — physical objects → ENTITY
    - <obstacle> in "Is the <agent> <action> because of the <obstacle>?"
      → values are "red light", "stop sign", "pedestrian" — can be physical → ENTITY
    - <cause> in "Is the intersection blocked by <cause>?"
      → values are "gridlock", "accident", "construction" — scene conditions → LEXICAL
    - <traffic> in "Is the ego yielding because of the <traffic>?"
      → values are "oncoming traffic", "cross traffic" — abstract conditions → LEXICAL


========================================================================
  SECTION 2: ENTITY GROUNDING RULES
========================================================================

For ENTITY-type placeholders, produce a four-layer description:

1. **OBJ ID**: from spatial data (e.g., "OBJ 18")
2. **Visual Appearance**: at least one distinguishing visual attribute
   (e.g., "white sedan", "person in dark jacket")
3. **Spatial Localization**: ego-centric position
   (e.g., "7.4m ahead", "5.1m behind and 1.9m to the left")
4. **Camera Reference**: most clearly visible camera
   (e.g., "visible in Image 2 (Front)")

Format: `OBJ {id} ({visual_description}, {distance_and_direction}, {camera_reference})`

Examples:
  - `OBJ 18 (white sedan, 7.4m directly ahead, visible in Image 2 (Front))`
  - `OBJ 5 (pedestrian in dark clothing, 46.6m ahead and 23.5m to the left, visible in Image 1 (Front-left))`

Constraints:
  - Prefer closer, more clearly visible objects for meaningful questions.
  - For two-object templates (<object_A> vs <object_B>), select two DIFFERENT objects
    with non-trivial differences (avoid near-identical distances).
  - For no-placeholder templates, reference relevant objects in your reasoning.


========================================================================
  SECTION 3: LEXICAL SELECTION RULES
========================================================================

For LEXICAL-type placeholders, select a value as-is (no OBJ grounding).

Value sources (in priority order):

  **For POSITIVE instances:**
    1. `valid_placeholders` — values confirmed present in this scene by Stage 1

  **For CONTRASTIVE instances:**
    1. `original_placeholders` \\ `valid_placeholders` — values from the template bank
       that Stage 1 confirmed are ABSENT from this scene (highest confidence for "no")
    2. `original_placeholders` (remaining) — other values if set difference is empty
    3. **Fallback**: propose a plausible value clearly absent from the scene,
       semantically consistent with the placeholder type

Constraints:
  - NEVER replace a lexical placeholder with an OBJ ID or grounded description.
  - The selected value must produce a grammatically correct, meaningful question.


========================================================================
  SECTION 4: MULTI-PAIR CONTRASTIVE QA GENERATION
========================================================================

For EVERY template, generate MULTIPLE contrastive QA pairs by systematically
varying placeholder combinations. Target: MINIMUM 2 PAIRS per template
where the scene supports it.

--- 4.1 Pair Generation Strategy ---

To maximize the number of valid pairs, systematically expand across these dimensions:

**For templates WITH placeholders:**

  a) ENTITY-type expansion:
     - Cycle through DIFFERENT OBJ IDs across pairs
       (e.g., if 8 vehicles exist, use different vehicles in different pairs)
     - Vary by proximity: near objects, mid-range objects, far objects
     - Vary by camera view: front, rear, left, right objects
     - For two-entity templates: use different object PAIRS, not just swapped order

  b) LEXICAL-type expansion:
     - Cycle through ALL values in `valid_placeholders` for positive instances
     - Cycle through ALL values in `original_placeholders` \\ `valid_placeholders`
       for contrastive instances
     - For multi-placeholder templates: use the CARTESIAN PRODUCT of values,
       filtered to meaningful and non-redundant combinations

  c) Combined expansion:
     - When a template has both ENTITY and LEXICAL placeholders,
       vary BOTH across pairs to maximize diversity

**For templates WITHOUT placeholders:**
  - Typically only 1 pair is achievable (or 0 contrastive if the answer is fixed).
  - For no-placeholder templates that ask about a general scene condition
    (e.g., "Is the vehicle facing towards or away?"), you MAY generate
    multiple pairs by applying the question to DIFFERENT objects in the scene,
    even though the template has no explicit placeholder.
    In such cases, prepend the object reference to the question for clarity.
  - If only 1 pair is possible, output that pair and explain in `max_pairs_note`.

--- 4.2 Diversity Requirements ---

Across all pairs generated for a single template, ensure:

  1. **Object diversity**: Use different OBJ IDs — do not repeat the same object
     in the same role across pairs. Spread across near/far, front/rear, left/right.
  2. **Location diversity**: Vary <place>, <direction>, <view> values across pairs.
  3. **Answer diversity**: Achieve answer variety through QUESTION construction,
     never by adjusting an evidence-based answer.
     - For y_or_n: vary which values you instantiate as positive (e.g., use an
       originally-absent value as the positive so its honest answer is "no").
     - For mcq: vary the correct option letter (A-E) across pairs; use different
       distractors drawn from different aspects of the scene.
  4. **Difficulty diversity**: Mix easy pairs (large distance gaps, obvious presence/absence)
     with harder pairs (smaller margins, subtle distinctions).
  5. **Camera diversity**: Spread object selections across all 6 camera views,
     not only the front camera.

--- 4.3 Contrastive Construction Strategies (question construction only) ---

These strategies describe how to CONSTRUCT a contrastive QUESTION. They say
nothing about what its answer must be — the answer always comes from scene
evidence alone (see 4.4 Contrastive Protocol).

**y_or_n** (141 templates):
  Strategies:
    a) Swap <object>/<agent> to an absent category
       (e.g., "pedestrians" → "cyclists" when no cyclists exist)
    b) Swap <place>/<location> to a location where the object is absent
    c) Swap <state>/<condition> to a state not observed in the scene
    d) Swap <direction>/<view> to a view where the condition is absent
    e) For ENTITY placeholders: select a different OBJ
    f) For no-placeholder templates: if no meaningful variation exists,
       set contrastive to null (contrast_status "skipped").

**mcq** (68 templates — 5 options A-E, EXACTLY ONE correct; see SECTION 7
  for the full option-construction rules):
  Strategies:
    a) Swap placeholder to a different object/state
    b) For comparison templates: swap object order or swap one object
    c) For view-specific templates: change <direction>/<view>
    d) If only one unique valid question variant exists, set contrastive
       to null.
  Note: build each question's 5 options independently and honestly. The
  contrastive's correct letter may or may not differ from the positive's —
  report which via contrast_status.

**open_ended** (21 templates):
  Strategies:
    a) Change <agent>/<object> to a different entity
    b) Change <location>/<view> to a different area
    c) Change <action> to a different action
    d) For no-placeholder templates with only one valid interpretation,
       set contrastive to null.

**num_count** (8 templates):
  Strategies:
    a) Swap <object> to an absent category
    b) Swap <direction>/<view> to a view with a different population

**distance** (3 templates):
  Strategies:
    a) Change target object to a different object

--- 4.4 CONTRASTIVE PROTOCOL (STRICT — read before answering) ---

1. ANSWER-FIRST ORDERING: Answer the positive and the contrastive
   questions INDEPENDENTLY, each based only on scene evidence. Only
   after both answers are committed, evaluate whether they differ.
2. NEVER adjust, reverse, or soften an evidence-based answer to
   manufacture a contrast. A fabricated flip is a critical failure;
   a same-answer pair is NOT a failure.
3. Set "contrast_status" on every contrastive object:
   - "achieved":    the two answers genuinely differ based on evidence
   - "same_answer": the contrastive question is valid but yields the
                    same answer — output it fully and honestly with
                    this flag; it will be used as an unpaired QA item
   - "skipped":     no valid contrastive question can be constructed;
                    set "contrastive": null and give
                    "contrastive_skip_reason" at the pair level
4. For risk/clarity judgment templates (Dynamic_Agents_and_Risk_Assessment,
   Road_Markings_and_Lane_Configuration): contrast is OPTIONAL. When the
   judgment threshold is subjective and a flip is uncertain, prefer
   "same_answer" or "skipped" over a marginal, forced flip.
5. If the contrastive swaps in an entity/place ABSENT from the scene,
   the 2-step think override applies (see SECTION 10 Think Budget).

--- 4.5 Contrastive Generation Constraints ---

1. **Minimal perturbation**: Change as FEW placeholders as possible (ideally ONE).
2. **Factual grounding**: Every answer (positive and contrastive) MUST be
   verifiably correct against the scene. Never output an answer you cannot
   ground in evidence.
3. **Semantic coherence**: The contrastive question must be grammatically
   correct and semantically meaningful.
4. **Non-trivial difficulty**: Avoid trivially obvious contrasts where possible.
5. **No duplicate questions**: Each pair's positive question must be UNIQUE
   across all pairs for this template.
6. Set contrastive to null (contrast_status "skipped") when:
   - The template has NO placeholders and no meaningful variation exists
   - Altering any placeholder would produce an ambiguous or unverifiable
     question


========================================================================
  SECTION 5: MOTION STATE DETERMINATION
========================================================================

When describing whether an object is moving, parked, stopped, or stationary,
you MUST use the velocity data provided in the object list — do NOT guess
motion state from the image alone.

RULE: An object's motion state is determined by its speed (magnitude of velocity):
  - speed < 0.1 m/s  →  STATIONARY (parked, stopped, idle, at rest)
  - speed >= 0.1 m/s  →  MOVING (in motion, driving, riding, traveling)

Each object's velocity is shown as:
  velocity=[vx=..., vy=...] m/s, speed=... m/s, motion=stationary|moving

EXAMPLES:
  - velocity=[vx=0.01, vy=0.04], speed=0.04 → stationary
    USE: "parked", "stopped", "stationary", "at rest"
    DO NOT USE: "moving", "driving", "riding", "in motion", "actively riding"

  - velocity=[vx=7.73, vy=0.21], speed=7.74 → moving
    USE: "moving", "driving", "in motion", "traveling"
    DO NOT USE: "parked", "stopped", "stationary", "pulled over", "stalled"

CRITICAL: Never describe a stationary object (speed < 0.1) as "moving",
"driving", "riding", or "in motion". Never describe a moving object
(speed >= 0.1) as "parked", "stopped", "stationary", or "pulled over".
This is the most common error in QA generation — always check the velocity.


========================================================================
  SECTION 6: NON-EXISTENT REFERENCE HANDLING
========================================================================

When a question references a location, object, or condition that does NOT
exist in the scene, apply the following CONSISTENT rule:

**RULE: If the referenced location or entity does not exist in the scene,
the answer is always "no" for y_or_n questions.**

This applies regardless of how the question is phrased:
  - "Are there any X in the <location>?"  → "no" (location doesn't exist)
  - "Is the <location> clear of X?"       → "no" (location doesn't exist)
  - "Is the <location> free of X?"        → "no" (location doesn't exist)
  - "Can you see X at the <location>?"    → "no" (location doesn't exist)

REASONING PATTERN: Always state that the location/entity does not exist
in the scene first, then conclude "no" because the question's premise
is not met.

EXAMPLE (correct):
  Q: "Is the intersection clear of parked cars?"
  Reasoning: "No intersection is visible in any of the six camera views.
  Since the referenced location does not exist in this scene, the
  question cannot be affirmed. The answer is no."
  A: "no"

EXAMPLE (incorrect — DO NOT use vacuous truth):
  Q: "Is the intersection clear of parked cars?"
  Reasoning: "Since there is no intersection, it is trivially clear..."
  ← This uses vacuous truth logic, which produces inconsistent answers.
  A: "yes"  ← WRONG

**Why this matters:** Vacuous truth ("X is trivially true because Y
doesn't exist") creates contradictions. For the same scene:
  - "Are there vehicles in the intersection?" → "no" (no intersection)
  - "Is the intersection clear of vehicles?" → "yes" (vacuously true)
These are logically contradictory for a training dataset. Always
answer "no" when the premise doesn't hold.

For other answer types (open_ended, num_count, distance, mcq),
explicitly state that the referenced entity/location is absent and
provide the most factually grounded response possible.


========================================================================
  SECTION 7: ANSWER GENERATION RULES
========================================================================

All answers MUST be SHORT-ANSWER format unless the answer_type is open_ended.
Short-answer = the minimal factual response with no extra elaboration.
All detail and justification goes in the "reasoning" field, not in "answer".

1. **y_or_n**: Strictly "yes" or "no". Nothing else.
2. **mcq**: The correct option LETTER (always exactly one).
   - Answer is a single letter, e.g. `"B"`. Never a list / never comma-separated.
   - Construct exactly 5 options labeled (A) through (E), with EXACTLY ONE that is
     factually correct of the scene. The other 4 are plausible distractors.
   - If more than one option appears arguably true, refine the question (or pick
     the single most-specific true option) so that only ONE is correct. Never
     assert a false option as correct to hit a target count.
   - Distractors: draw from `expected_answers` when available; generate
     semantically consistent alternatives when the pool is too small.
   - Include the full options list in `mcq_options`:
     {"A": "option text", "B": "option text", "C": "option text",
      "D": "option text", "E": "option text"}
   - Randomize which position (A-E) holds the correct option across pairs.
3. **distance**: A single value with unit (e.g., "approximately 7.4m"). No sentence.
4. **num_count**: A single integer (e.g., "3"). No sentence.
5. **open_ended** (EXCEPTION — longer answers allowed):
   Provide a descriptive answer in 2-4 sentences.
   Reference specific objects and spatial relationships directly in the answer.
   This is the ONLY answer_type where the answer field may contain full sentences.


========================================================================
  SECTION 8: REASONING REQUIREMENTS
========================================================================

Reasoning MUST be written as BULLET POINTS (not prose).
Each bullet is one piece of evidence or one logical step.
Use 3-5 bullets per answer (at most 5).
"reasoning" is the committed summary AFTER "think" — do not copy think
steps verbatim; condense them into evidence-citing bullets.

Format:
  "reasoning": "- <bullet 1>\n- <bullet 2>\n- <bullet 3>"

Required bullet content:
  - At least one bullet referencing a specific OBJ ID and its spatial data
  - At least one bullet citing which camera Image number(s) were used
  - At least one bullet stating the logical conclusion from evidence to answer
  - For spatial questions, include a bullet with distance/position data

Example (y_or_n):
  "reasoning": "- OBJ 18 (car) is 7.4m directly ahead, visible in Image 2 (Front)\n- OBJ 23 (car) is 12.5m ahead in the same lane, visible in Image 2\n- Both vehicles are in the ego-vehicle's current lane, confirming presence\n- Answer: yes"

Example (mcq, single-correct contract):
  "reasoning": "- Image 2 (Front) shows a traffic light ahead of the ego-vehicle\n- The light displays a solid red circle, verified in Image 2\n- No green or yellow signal is visible for the ego-vehicle's lane\n- Correct option: (C) red"
  "answer": "C"

For CONTRASTIVE reasoning, additionally include:
  - A bullet stating which placeholder(s) were changed and the value source
  - A bullet explaining why the change produces a different answer


========================================================================
  SECTION 9: DRIVING CONTEXT
========================================================================

The ego-vehicle's current driving command provides context for scene dynamics.
Use it for right-of-way, risk, and planning questions.
It does NOT change factual spatial relationships.""" + _think_budget_block()


# =============================================================================
# Helper functions
# =============================================================================

def image_to_base64_data_uri(img_pil: Image.Image, format: str = "JPEG") -> str:
    """Convert a PIL Image to a base64 data URI string."""
    buffer = io.BytesIO()
    img_pil.save(buffer, format=format)
    base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime_type = "image/jpeg" if format.upper() == "JPEG" else f"image/{format.lower()}"
    return f"data:{mime_type};base64,{base64_str}"


def parse_json_response(response: str) -> Optional[object]:
    """
    Parse JSON from the model response. Handles both dict and list responses.
    """
    # Thinking models emit chain-of-thought before the final payload, closed by
    # </think>. The thinking text routinely contains incidental JSON fragments
    # (lists, schema sketches) that must never be mistaken for the answer, so
    # only the text after the LAST </think> is parsed. Instruct responses have
    # no </think> and pass through unchanged.
    if '</think>' in response:
        response = response.rsplit('</think>', 1)[-1]

    # Try to parse the entire response as JSON
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code blocks
    json_patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

    # Try to find a JSON object BEFORE trying arrays: the expected payload is
    # always an object, while stray arrays (e.g. inside a truncated thinking
    # trace that never reached </think>) are far more likely to be noise.
    brace_count = 0
    start_idx = None
    for i, char in enumerate(response):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx is not None:
                try:
                    return json.loads(response[start_idx:i+1])
                except json.JSONDecodeError:
                    start_idx = None

    # Try to find a JSON array
    bracket_count = 0
    start_idx = None
    for i, char in enumerate(response):
        if char == '[':
            if bracket_count == 0:
                start_idx = i
            bracket_count += 1
        elif char == ']':
            bracket_count -= 1
            if bracket_count == 0 and start_idx is not None:
                try:
                    return json.loads(response[start_idx:i+1])
                except json.JSONDecodeError:
                    start_idx = None

    return None


_NUM_WORDS = {
    'zero', 'one', 'two', 'three', 'four', 'five',
    'six', 'seven', 'eight', 'nine', 'ten',
}
# MCQ is single-correct: answer must be EXACTLY ONE letter A-E. Multi-letter
# (comma-separated) responses are rejected at the parser layer so the pipeline
# can never silently emit multi-correct MCQs again.
_MCQ_LETTER_RE = re.compile(r'^[A-E]$')


def _block_matches_contract(blk: Dict, expected_at: str) -> Tuple[bool, str]:
    """Check a single Q-block (positive / contrastive / vlm_proposed entry)
    against the template's expected answer_type. Returns (ok, reason)."""
    if not isinstance(blk, dict):
        return False, "block is not a dict"

    ans = blk.get('answer')
    if not isinstance(ans, str):
        return False, "answer missing or not a string"
    ans_norm = ans.strip()

    if expected_at == 'mcq':
        if not isinstance(blk.get('mcq_options'), dict):
            return False, "mcq template missing mcq_options dict"
        if not _MCQ_LETTER_RE.match(ans_norm):
            return False, f"mcq answer not in letter format: {ans_norm!r}"
        return True, ""

    # Non-MCQ: mcq_options must be absent/empty
    if blk.get('mcq_options'):
        return False, f"mcq_options leaked into non-mcq ({expected_at}) block"

    if expected_at == 'y_or_n':
        if ans_norm.lower() not in ('yes', 'no'):
            return False, f"y_or_n answer not yes/no: {ans_norm!r}"
        return True, ""

    if expected_at == 'num_count':
        if not (ans_norm.isdigit() or ans_norm.lower() in _NUM_WORDS):
            return False, f"num_count answer not numeric: {ans_norm!r}"
        return True, ""

    if expected_at == 'distance':
        if not re.search(r'\d', ans_norm):
            return False, f"distance answer has no number: {ans_norm!r}"
        return True, ""

    if expected_at == 'open_ended':
        if len(ans_norm) < 2:
            return False, "open_ended answer too short"
        return True, ""

    # Unknown answer_type — fall back to non-empty check
    if not ans_norm:
        return False, "empty answer"
    return True, ""


def validate_result_contract(result: Dict, expected_at: str) -> Tuple[bool, List[str]]:
    """Validate every Q-block in a parsed VLM result against the template's
    expected answer_type. Strict: any single block failure rejects the result."""
    failures: List[str] = []
    pairs = result.get('pairs') or []
    if not isinstance(pairs, list) or not pairs:
        return False, ["no pairs in result"]

    for p_idx, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            failures.append(f"pair[{p_idx}] not a dict")
            continue
        for role in ('positive', 'contrastive'):
            blk = pair.get(role)
            if blk is None:
                continue  # contrastive may legitimately be absent
            if not isinstance(blk, dict):
                failures.append(f"pair[{p_idx}].{role} not a dict")
                continue
            ok, reason = _block_matches_contract(blk, expected_at)
            if not ok:
                failures.append(f"pair[{p_idx}].{role}: {reason}")
        for vp_idx, vp in enumerate(pair.get('vlm_proposed_contrastives') or []):
            if not isinstance(vp, dict):
                failures.append(f"pair[{p_idx}].vlm_proposed[{vp_idx}] not a dict")
                continue
            ok, reason = _block_matches_contract(vp, expected_at)
            if not ok:
                failures.append(f"pair[{p_idx}].vlm_proposed[{vp_idx}]: {reason}")

    return (len(failures) == 0), failures


# =============================================================================
# RL-specific validation (T7): think band, contrast_status, status_mismatch
# =============================================================================

# Flags that trigger ONE re-inference retry; after a failed retry the result
# is kept with the flags attached (no silent drops).
RETRY_FLAGS = {
    'malformed_think', 'band_violation', 'absent_band_violation',
    'missing_contrast_status',
}
# Flags that never trigger a retry (kept for downstream routing only).
AUDIT_FLAGS = {'status_mismatch', 'contract_violation', 'answer_null_downgraded'}


def _norm_answer_value(ans) -> str:
    """Normalize an answer value for cross-side equality comparison."""
    if isinstance(ans, str):
        return ans.strip().lower()
    if isinstance(ans, (list, dict)):
        return json.dumps(ans, sort_keys=True)
    return str(ans)


def _comparable_answer(block: Dict, answer_type: str) -> Optional[str]:
    """Return the comparison key for a side's answer, or None if this
    answer_type is exempt from status_mismatch checking.

    mcq letters index DIFFERENT option sets on each side, so the selected
    option TEXT is compared instead of the letter. open_ended is exempt
    (wording variance makes equality meaningless)."""
    if answer_type == 'open_ended':
        return None
    ans = block.get('answer')
    if ans is None:
        return None
    if answer_type == 'mcq':
        opts = block.get('mcq_options')
        if isinstance(opts, dict) and isinstance(ans, str):
            text = opts.get(ans.strip().upper())
            if text is not None:
                return _norm_answer_value(text)
        return None  # cannot resolve option text — skip the check
    return _norm_answer_value(ans)


def _is_absent_swap(pre_pair: Optional[Dict]) -> bool:
    """True when the pre-instantiated contrastive swapped in a value known to
    be absent from the scene (pre_instantiate_pairs marks the strategy)."""
    if not isinstance(pre_pair, dict):
        return False
    return 'absent from scene' in (pre_pair.get('contrastive_strategy') or '')


def _check_think(block: Dict, category: str, expect_absent_2: bool) -> List[str]:
    """Validate one side's think field. Returns a list of flags."""
    flags = []
    think = block.get('think')
    if (not isinstance(think, list) or not think
            or not all(isinstance(s, str) and s.strip() for s in think)):
        return ['malformed_think']
    n = len(think)
    lo, hi = THINK_BANDS.get(category, (2, 8))
    # ABSENT_PREMISE_STEPS is always legal: the absent-premise short-circuit
    # is the model's judgment and code cannot always verify scene absence.
    if not (n == ABSENT_PREMISE_STEPS or lo <= n <= hi):
        flags.append('band_violation')
    # Where code KNOWS the premise is absent (absent-pool swap), enforce
    # the exact 2-step short-circuit.
    if expect_absent_2 and n > ABSENT_PREMISE_STEPS:
        flags.append('absent_band_violation')
    return flags


def validate_rl_result(
    result: Dict,
    answer_type: str,
    category: str,
    pre_pairs: List[Dict],
) -> Tuple[List[str], bool]:
    """RL contract validation for one template result (runs AFTER the v1
    answer-type contract). Mutates result in place:
      - downgrades null-answer contrastives to skipped
      - stamps pair-level 'contrast_status'
    Returns (flags, retry_needed)."""
    flags: List[str] = []

    for idx, pair in enumerate(result.get('pairs') or []):
        if not isinstance(pair, dict):
            continue
        pre_pair = pre_pairs[idx] if idx < len(pre_pairs) else None

        pos = pair.get('positive')
        if isinstance(pos, dict):
            flags += _check_think(pos, category, expect_absent_2=False)

        con = pair.get('contrastive')

        # T7: null answer on the contrastive side → downgrade that side only.
        if isinstance(con, dict) and con.get('answer') is None:
            pair['contrastive'] = None
            pair['contrastive_skip_reason'] = (
                'downgraded: contrastive answer was null'
            )
            pair['contrast_status'] = 'skipped'
            flags.append('answer_null_downgraded')
            con = None

        if con is None:
            pair.setdefault('contrast_status', 'skipped')
            continue
        if not isinstance(con, dict):
            continue

        flags += _check_think(con, category,
                              expect_absent_2=_is_absent_swap(pre_pair))

        status = con.get('contrast_status')
        if status == 'skipped':
            # 'skipped' belongs to null contrastives; a fully-answered object
            # self-reporting skipped is a status error → fall through to
            # missing-status handling.
            status = None
        if status not in VALID_CONTRAST_STATUS:
            flags.append('missing_contrast_status')
            pair['contrast_status'] = None
        else:
            pair['contrast_status'] = status
            # Trust but verify (T7): compare committed answers.
            pa = _comparable_answer(pos if isinstance(pos, dict) else {}, answer_type)
            ca = _comparable_answer(con, answer_type)
            if pa is not None and ca is not None:
                answers_equal = (pa == ca)
                if (status == 'achieved' and answers_equal) or \
                   (status == 'same_answer' and not answers_equal):
                    flags.append('status_mismatch')

    flags = sorted(set(flags))
    retry_needed = any(f in RETRY_FLAGS for f in flags)
    return flags, retry_needed


def collect_template_stats(result: Dict, category: str, flags: List[str],
                           stats: Dict) -> None:
    """Accumulate per-template RL metrics into the per-sample stats dict
    (consumed by the run-end report, T8)."""
    for f in flags:
        stats['flag_counts'][f] = stats['flag_counts'].get(f, 0) + 1
    band = stats['band'].setdefault(category, {'sides_ok': 0, 'sides_violation': 0})
    lo, hi = THINK_BANDS.get(category, (2, 8))
    for pair in (result.get('pairs') or []):
        if not isinstance(pair, dict):
            continue
        for role in ('positive', 'contrastive'):
            side = pair.get(role)
            if not isinstance(side, dict):
                continue
            think = side.get('think')
            if isinstance(think, list) and (
                    len(think) == ABSENT_PREMISE_STEPS or lo <= len(think) <= hi):
                band['sides_ok'] += 1
            else:
                band['sides_violation'] += 1
        status = pair.get('contrast_status')
        key = status if status in VALID_CONTRAST_STATUS else 'invalid'
        stats['contrast_status'][key] = stats['contrast_status'].get(key, 0) + 1


def stamp_provenance(result: Dict, sample_idx: int, template_num: int) -> None:
    """T9: qa_id on every pair side — format s0710_t002_p1_pos."""
    for pair in (result.get('pairs') or []):
        if not isinstance(pair, dict):
            continue
        pid = pair.get('pair_id')
        base = f"s{sample_idx:04d}_t{template_num:03d}_p{pid}"
        if isinstance(pair.get('positive'), dict):
            pair['positive']['qa_id'] = f"{base}_pos"
        if isinstance(pair.get('contrastive'), dict):
            pair['contrastive']['qa_id'] = f"{base}_con"


def load_stage1_results(stage1_dir: str, sample_idx: int, categories: Optional[List[str]] = None) -> Dict:
    """
    Load Stage 1 applicable questions for a sample.

    Supports two directory layouts produced by question_selector:
      1. Per-category subdirs:  stage1_dir/<Category>/sample_X_applicable_questions.json
         (created when question_selector runs with --category <Category>)
      2. All-category subdir:   stage1_dir/all/sample_X_applicable_questions.json
         (created when question_selector runs with --category all)

    Both layouts can coexist; results are merged with per-category taking precedence.
    If categories is None, loads all. Otherwise filters to the specified categories.
    Returns merged dict: {category: {q_01: {...}, q_02: {...}}}
    """
    merged = {}

    if not os.path.isdir(stage1_dir):
        return merged

    filename = f"sample_{sample_idx}_applicable_questions.json"

    for entry in os.listdir(stage1_dir):
        cat_dir = os.path.join(stage1_dir, entry)
        if not os.path.isdir(cat_dir):
            continue

        filepath = os.path.join(cat_dir, filename)
        if not os.path.isfile(filepath):
            continue

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        sample_key = str(sample_idx)
        if sample_key not in data:
            continue

        is_category_dir = entry in VALID_CATEGORIES
        is_all_dir = entry == "all"

        if is_category_dir:
            # Per-category subdir: directory name is the category
            if categories is not None and entry not in categories:
                continue
            for category, questions in data[sample_key].items():
                if questions:
                    merged[category] = questions
        elif is_all_dir:
            # All-category subdir: file contains multiple categories
            for category, questions in data[sample_key].items():
                if categories is not None and category not in categories:
                    continue
                if questions and category not in merged:
                    merged[category] = questions
        # Other subdirs (e.g., legacy "all_v3") are skipped — they may carry
        # outputs from older question banks with answer_types that no longer
        # match the active contract.

    return merged


def discover_stage1_sample_indices(stage1_dir: str) -> List[int]:
    """
    Scan stage1_dir for applicable_questions files and return a sorted list of
    sample indices with Stage 1 results. Only the per-category subdirs (in
    VALID_CATEGORIES) and the "all" subdir are considered — other subdirs
    (e.g., legacy "all_v3") are skipped to keep this in sync with
    load_stage1_results.
    """
    indices = set()
    if not os.path.isdir(stage1_dir):
        return []

    pattern = re.compile(r'^sample_(\d+)_applicable_questions\.json$')

    for entry in os.listdir(stage1_dir):
        sub = os.path.join(stage1_dir, entry)
        if not os.path.isdir(sub):
            continue
        if entry != "all" and entry not in VALID_CATEGORIES:
            continue
        for fname in os.listdir(sub):
            m = pattern.match(fname)
            if m:
                indices.add(int(m.group(1)))

    return sorted(indices)


def format_object_positions(scene_data: Dict) -> str:
    """
    Format scene_data['objects'] for user prompt.
    Uses sequential 1-based indexing to match risk assessment and Stage 1 prompts.
    """
    objects = scene_data.get('objects', [])
    if not objects:
        return "  (No objects detected within filter distance)"

    lines = []
    for idx, obj in enumerate(objects, 1):
        category = obj['category']
        distance = obj['distance']
        pos = obj['position_ego']  # [x, y, z] in ego frame (FLU: forward-left-up)
        pos_x = pos[0]
        pos_y = pos[1]

        x_label = "ahead" if pos_x >= 0 else "behind"
        if pos_y > 0:
            y_label = "left"
        elif pos_y < 0:
            y_label = "right"
        else:
            y_label = "center"

        visible_cams = obj.get('visible_cameras', [])
        cam_strs = []
        for cam in visible_cams:
            cam_idx = CAM_INDEX.get(cam, '?')
            cam_readable = CAMERA_NAME_MAP.get(cam, cam)
            cam_strs.append(f"Image {cam_idx} ({cam_readable})")
        cam_list = ", ".join(cam_strs) if cam_strs else "not visible in any camera"

        vel = obj.get('velocity_ego', [0, 0])
        speed = (vel[0]**2 + vel[1]**2) ** 0.5
        motion_state = "stationary" if speed < 0.1 else "moving"

        line = (
            f"  OBJ {idx}: {category}, distance={distance:.1f}m, "
            f"position=[x={abs(pos_x):.1f}m {x_label}, y={abs(pos_y):.1f}m {y_label}], "
            f"velocity=[vx={vel[0]:.2f}, vy={vel[1]:.2f}] m/s, speed={speed:.2f} m/s, "
            f"motion={motion_state}, "
            f"visible in {cam_list}"
        )
        lines.append(line)

    return "\n".join(lines)


def format_closest_objects(scene_data: Dict) -> str:
    """
    Find and format the closest object in each direction (ahead, behind, left, right).
    Uses sequential 1-based indexing to match risk assessment and Stage 1 prompts.
    """
    objects = scene_data.get('objects', [])
    if not objects:
        return "  (No objects detected)"

    # Build map from raw obj to sequential 1-based index
    obj_to_seq = {id(obj): seq_idx for seq_idx, obj in enumerate(objects, 1)}

    closest = {'ahead': None, 'behind': None, 'left': None, 'right': None}
    closest_dist = {'ahead': float('inf'), 'behind': float('inf'),
                    'left': float('inf'), 'right': float('inf')}

    for obj in objects:
        pos = obj['position_ego']
        pos_x, pos_y = pos[0], pos[1]
        dist = obj['distance']

        if pos_x > 0 and abs(pos_y) < abs(pos_x):
            if dist < closest_dist['ahead']:
                closest_dist['ahead'] = dist
                closest['ahead'] = obj
        if pos_x < 0 and abs(pos_y) < abs(pos_x):
            if dist < closest_dist['behind']:
                closest_dist['behind'] = dist
                closest['behind'] = obj
        if pos_y > 0 and abs(pos_y) > abs(pos_x):
            if dist < closest_dist['left']:
                closest_dist['left'] = dist
                closest['left'] = obj
        if pos_y < 0 and abs(pos_y) > abs(pos_x):
            if dist < closest_dist['right']:
                closest_dist['right'] = dist
                closest['right'] = obj

    lines = []
    for direction in ['ahead', 'behind', 'left', 'right']:
        obj = closest[direction]
        if obj:
            seq_idx = obj_to_seq[id(obj)]
            lines.append(f"  - {direction.capitalize()}: OBJ {seq_idx} ({obj['category']}) at {obj['distance']:.1f}m")
        else:
            lines.append(f"  - {direction.capitalize()}: (none detected)")

    return "\n".join(lines)


def format_placeholders_block(placeholders: Dict, label: str = "") -> str:
    """Format a placeholders dict for the prompt. Handles both <tag> and tag keys."""
    if not placeholders:
        return f"    (no placeholders)"

    lines = []
    for tag, values in placeholders.items():
        lines.append(f"    {tag}: {json.dumps(values)}")
    return "\n".join(lines)


def format_answer_type_contract(answer_type: str) -> str:
    """Per-template strict contract injected into the user prompt.

    The downstream parser rejects any result that violates this contract
    (see validate_result_contract), so the model must follow it exactly.
    Aligned with question_bank_v4_static.json's five answer_types:
    y_or_n, mcq, num_count, distance, open_ended.
    """
    lines = [
        "--- ANSWER FORMAT CONTRACT (STRICT — violations cause result rejection) ---",
        f"This template's answer_type is: {answer_type}",
        "",
        "Hard rules:",
        f"  1. The 'answer_type' field in your JSON output MUST be exactly '{answer_type}'.",
    ]
    if answer_type == "mcq":
        lines += [
            "  2. You MUST include 'mcq_options' as a 5-key dict (A-E) with full option text.",
            "  3. The 'answer' field MUST be EXACTLY ONE option letter (e.g. 'B').",
            "     Never a list, never comma-separated, never multiple letters.",
            "  4. Construct the 5 options so that EXACTLY ONE is correct of the scene",
            "     and the other 4 are plausible distractors. If more than one option",
            "     appears arguably true, refine the question (or pick the single most",
            "     specific true option) so that only one is correct.",
        ]
    else:
        # Non-MCQ rule shared by all other answer_types.
        lines += [
            "  2. DO NOT include 'mcq_options' anywhere in the response. The 'mcq_options'",
            "     field is reserved for answer_type='mcq' only.",
            "  3. DO NOT use option-letter form ('A', 'B,C', etc.) for the 'answer' field —",
            "     option letters are reserved for answer_type='mcq' only.",
        ]
        if answer_type == "y_or_n":
            lines += [
                "  4. The 'answer' field MUST be exactly 'yes' or 'no' (lowercase, no extra text).",
            ]
        elif answer_type == "num_count":
            lines += [
                "  4. The 'answer' field MUST be a single integer like '3' (digits only,",
                "     no units, no sentence).",
            ]
        elif answer_type == "distance":
            lines += [
                "  4. The 'answer' field MUST be a single value with unit, e.g. 'approximately",
                "     7.4m'. The string MUST contain a digit. No multi-sentence text.",
            ]
        elif answer_type == "open_ended":
            lines += [
                "  4. The 'answer' field MUST be a 2-4 sentence descriptive answer that",
                "     references specific objects and spatial relationships.",
            ]
        else:
            # Defensive fallback (the pin step overwrites answer_type to the
            # template's v4_static value, so this branch should be unreachable).
            lines += [
                f"  4. The 'answer' field MUST be in the natural form for answer_type '{answer_type}'.",
            ]
    lines.append("")
    return "\n".join(lines)


# =============================================================================
# Pre-sampling and Pre-instantiation (Options 1 + 3)
# =============================================================================

# ENTITY tags that should be grounded to OBJ IDs
ENTITY_TAGS = {
    '<object>', '<object_a>', '<object_b>', '<vehicle>', '<other_vehicle>',
    '<special_vehicle>', '<agent>', '<agent_type>', '<traffic_element>',
    '<signal_type>', '<barrier>', '<obstacle>',
}


def _normalize_tag(tag: str) -> str:
    """Ensure tag has <> brackets."""
    if not tag.startswith('<'):
        return f"<{tag}>"
    return tag


def _match_objects_by_category(category_value: str, objects: List[Dict]) -> List[Dict]:
    """Find scene objects whose category matches a placeholder value."""
    val = category_value.lower().rstrip('s')  # "pedestrians" -> "pedestrian"
    matches = []
    for obj in objects:
        obj_cat = obj['category'].lower()
        if val in obj_cat or obj_cat in val or val == obj_cat.rstrip('s'):
            matches.append(obj)
    return matches


def pre_instantiate_pairs(
    template: Dict,
    scene_data: Dict,
    max_pairs: int = 3,
    seed: int = 0,
) -> List[Dict]:
    """
    Pre-sample placeholder values and instantiate QA pairs programmatically.

    For each pair, produces:
      - positive: template filled with valid_placeholder values (present in scene)
      - contrastive: 1-2 placeholders swapped to absent values
      - VLM-proposed contrastive: placeholder for VLM to add more

    Returns list of dicts with keys:
        'positive_question', 'contrastive_question', 'positive_mapping',
        'contrastive_mapping', 'contrastive_strategy'
    """
    rng = random.Random(seed)

    tmpl_str = template.get('template', '')
    valid_ph = template.get('valid_placeholders', {})
    original_ph = template.get('original_placeholders', {})
    answer_type = template.get('answer_type', '')
    is_mcq = answer_type == 'mcq'

    def _sample_mcq_correct_counts():
        """Single-correct MCQ contract: exactly ONE correct option per question.

        Earlier versions randomly sampled num_correct in [1, 5] before the VLM
        had seen the scene, which forced the model to assert N options correct
        regardless of how many were actually true of the scene. Hard-coding to
        1 removes the impossible-count failure mode entirely; the contrastive
        pair already differs from the positive because its question has an
        altered placeholder."""
        return 1, 1

    # Normalize tag keys
    valid_ph = {_normalize_tag(k): v for k, v in valid_ph.items()}
    original_ph = {_normalize_tag(k): v for k, v in original_ph.items()}

    # Find all placeholder tags in the template
    tags_in_template = re.findall(r'<[^>]+>', tmpl_str)

    if not tags_in_template:
        # No-placeholder template — single pair, no pre-instantiation needed
        pair = {
            'positive_question': tmpl_str,
            'contrastive_question': None,
            'positive_mapping': {},
            'contrastive_mapping': {},
            'contrastive_strategy': 'no_placeholders',
        }
        if is_mcq:
            n_pos, n_ctr = _sample_mcq_correct_counts()
            pair['num_correct_positive'] = n_pos
            pair['num_correct_contrastive'] = n_ctr
        return [pair]

    # Separate ENTITY vs LEXICAL tags
    entity_tags = [t for t in tags_in_template if t.lower() in ENTITY_TAGS]
    lexical_tags = [t for t in tags_in_template if t.lower() not in ENTITY_TAGS]

    # Build candidate pools for each tag
    objects = scene_data.get('objects', [])
    positive_pools = {}  # tag -> list of values
    absent_pools = {}    # tag -> list of values not in scene (for contrastive)

    for tag in tags_in_template:
        tag_lower = tag.lower()
        valid_vals = valid_ph.get(tag, [])
        orig_vals = original_ph.get(tag, [])

        if isinstance(valid_vals, str):
            valid_vals = [valid_vals]
        if isinstance(orig_vals, str):
            orig_vals = [orig_vals]

        if tag_lower in ENTITY_TAGS and objects:
            # For ENTITY tags: use valid_vals but also match to actual objects
            positive_pools[tag] = valid_vals if valid_vals else [objects[0]['category']]
            absent_vals = [v for v in orig_vals if v not in valid_vals]
            absent_pools[tag] = absent_vals if absent_vals else ['unknown_entity']
        else:
            # LEXICAL tags
            positive_pools[tag] = valid_vals if valid_vals else ['unknown']
            absent_vals = [v for v in orig_vals if v not in valid_vals]
            absent_pools[tag] = absent_vals if absent_vals else []

    # Generate positive combinations (cartesian product, then sample)
    tag_order = tags_in_template
    # De-duplicate tags that appear multiple times
    seen_tags = set()
    unique_tags = []
    for t in tag_order:
        if t not in seen_tags:
            seen_tags.add(t)
            unique_tags.append(t)
    tag_order = unique_tags

    pos_value_lists = [positive_pools.get(t, ['unknown']) for t in tag_order]
    all_pos_combos = list(itertools.product(*pos_value_lists))
    rng.shuffle(all_pos_combos)
    selected_pos = all_pos_combos[:max_pairs]

    pairs = []
    for combo in selected_pos:
        mapping = dict(zip(tag_order, combo))

        # Build positive question
        pos_q = tmpl_str
        for tag, val in mapping.items():
            pos_q = pos_q.replace(tag, val, 1)

        # Build contrastive: change ONE tag (prefer lexical, then entity)
        # Pick the tag with the most absent alternatives
        best_tag = None
        best_absent = []
        for tag in tag_order:
            absent = absent_pools.get(tag, [])
            if len(absent) > len(best_absent):
                best_tag = tag
                best_absent = absent

        cont_q = None
        cont_mapping = dict(mapping)
        cont_strategy = None

        if best_tag and best_absent:
            swap_val = rng.choice(best_absent)
            cont_mapping[best_tag] = swap_val
            cont_q = tmpl_str
            for tag in tag_order:
                cont_q = cont_q.replace(tag, cont_mapping[tag], 1)
            cont_strategy = f"Swapped {best_tag} from '{mapping[best_tag]}' to '{swap_val}' (absent from scene)"
        else:
            cont_strategy = "no_absent_values_available"

        pair = {
            'positive_question': pos_q,
            'contrastive_question': cont_q,
            'positive_mapping': mapping,
            'contrastive_mapping': cont_mapping,
            'contrastive_strategy': cont_strategy,
        }
        if is_mcq:
            n_pos, n_ctr = _sample_mcq_correct_counts()
            pair['num_correct_positive'] = n_pos
            pair['num_correct_contrastive'] = n_ctr
        pairs.append(pair)

    return pairs


def build_verification_prompt(
    driving_command: str,
    object_positions_text: str,
    closest_objects_text: str,
    template: Dict,
    pre_pairs: List[Dict],
    template_num: int,
    total_templates: int,
) -> str:
    """
    Build a prompt for the VLM to VERIFY pre-instantiated QA pairs.

    The output contract is FIXED (no answer_mode variants): per pair side,
    key order is think -> reasoning -> answer, with contrast_status
    self-reporting on contrastive objects.

    The VLM's job:
      1. Answer each pre-instantiated positive + contrastive question
         INDEPENDENTLY from scene evidence
      2. Emit tier-banded "think" steps, then condensed "reasoning" bullets,
         then the derived "answer"
      3. Report contrast_status honestly (achieved / same_answer / skipped)
    """
    parts = []

    # Scene context
    parts.append("--- Ego-Vehicle Driving Command ---")
    parts.append(f"  Current Command: {driving_command}")
    parts.append("")
    parts.append("--- Prior Knowledge: Spatial Relationships ---")
    parts.append("")
    parts.append("** Object Positions (Ego-centric) **")
    parts.append(object_positions_text)
    parts.append("")
    parts.append("** Closest Objects by Direction **")
    parts.append(closest_objects_text)
    parts.append("")

    # Template info
    t = template
    parts.append(f"--- Template ({template_num}/{total_templates}) ---")
    parts.append(f"  Category: {t['category']}")
    parts.append(f"  Template: \"{t['template']}\"")
    parts.append(f"  Answer Type: {t['answer_type']}")
    ea = t.get('expected_answers')
    parts.append(f"  Expected Answers: {json.dumps(ea) if ea else 'null'}")
    ego_note = t.get('ego_centric_note')
    if ego_note:
        parts.append(f"  Ego-Centric Guidance: {ego_note}")
    if t['category'] in CONTRAST_OPTIONAL_CATEGORIES:
        parts.append("  NOTE: Contrast is OPTIONAL for this category — when the judgment")
        parts.append("  threshold is subjective, prefer honest same_answer/skipped over a")
        parts.append("  marginal flip (see Contrastive Protocol).")
    band_lo, band_hi = THINK_BANDS.get(t['category'], (2, 8))
    parts.append(f"  Think Budget for this template: {band_lo}-{band_hi} steps "
                 f"(exactly 2 steps if the questioned entity/place is absent)")

    # Show available placeholder pools for VLM-proposed contrasts
    valid_ph = t.get('valid_placeholders', {})
    original_ph = t.get('original_placeholders', {})
    if valid_ph:
        parts.append(f"  Valid Placeholders (present in scene):")
        parts.append(format_placeholders_block(valid_ph))
    if original_ph:
        parts.append(f"  Original Placeholders (full candidate bank):")
        parts.append(format_placeholders_block(original_ph))
    parts.append("")

    # Per-template strict answer-format contract (parser-enforced)
    parts.append(format_answer_type_contract(t.get('answer_type', '')))

    # Pre-instantiated pairs
    parts.append("--- PRE-INSTANTIATED QA PAIRS (verify and answer these) ---")
    parts.append("")
    for i, pair in enumerate(pre_pairs, 1):
        parts.append(f"  Pair {i}:")
        parts.append(f"    Positive Question: \"{pair['positive_question']}\"")
        if 'num_correct_positive' in pair:
            parts.append(f"    [MCQ] Build 5 options A-E with EXACTLY ONE correct of the scene; answer is a single letter.")
        if pair['contrastive_question']:
            parts.append(f"    Contrastive Question: \"{pair['contrastive_question']}\"")
            parts.append(f"    Contrastive Strategy: {pair['contrastive_strategy']}")
            if 'num_correct_contrastive' in pair:
                parts.append(f"    [MCQ] Same single-correct rule; build the options independently of the positive and report contrast_status honestly.")
        else:
            parts.append(f"    Contrastive Question: (none pre-generated — you must propose one)")
        parts.append("")

    # Task instructions
    parts.append("--- YOUR TASK ---")
    parts.append("")

    parts.append("Follow the FIXED output contract (SECTION 10): for every question,")
    parts.append("deliberate in tier-banded \"think\" steps FIRST, condense into")
    parts.append("\"reasoning\" bullets, and only then derive \"answer\" — the answer is")
    parts.append("the natural conclusion of the deliberation, never a pre-chosen value")
    parts.append("you then justify.")
    parts.append("")
    parts.append("For EACH pre-instantiated pair above:")
    parts.append("")
    parts.append("1. **THINK then ANSWER** the positive question:")
    parts.append("   a. \"think\": first-person deliberation steps, one array element per")
    parts.append("      step, each citing concrete evidence (OBJ id, Image index, distance);")
    parts.append("      step count within the category band (see Think Budget)")
    parts.append("   b. \"reasoning\": at most 5 condensed evidence-citing bullets")
    parts.append("   c. \"answer\": derived from the above; stated ONLY in the answer key")
    parts.append("")
    parts.append("2. **THINK then ANSWER** the contrastive question (if provided),")
    parts.append("   INDEPENDENTLY of the positive — same think -> reasoning -> answer")
    parts.append("   process. Then set \"contrast_status\" honestly (see Contrastive")
    parts.append("   Protocol): \"achieved\" or \"same_answer\". NEVER adjust either answer")
    parts.append("   to force a contrast.")
    parts.append("")
    parts.append("3. For ENTITY-type placeholders, use four-layer grounding:")
    parts.append("   OBJ {id} ({visual_description}, {distance_direction}, {camera_ref})")
    parts.append("")

    # Output format — FIXED key order: think -> reasoning -> answer
    parts.append("--- OUTPUT FORMAT ---")
    parts.append("")

    order_note = ("IMPORTANT: within each positive/contrastive object, keys MUST appear "
                  "in the exact order: tag_mappings, instantiated_question, (mcq_options), "
                  "think, reasoning, answer, confidence, relevant_cameras, "
                  "(contrast_status — contrastive only). 'answer' is written LAST, "
                  "derived from 'think' and 'reasoning' — never chosen first.")
    schema = """Respond with a JSON object:
{
    "template_idx": <int>,
    "category": "<string>",
    "answer_type": "<y_or_n|mcq|distance|open_ended|num_count>",
    "placeholder_classification": {
        "<tag>": "<entity|lexical>"
    },
    "total_pairs": <int>,
    "max_pairs_note": <null or "explanation if fewer pairs">,
    "pairs": [
        {
            "pair_id": 1,
            "positive": {
                "tag_mappings": {
                    "<tag>": {
                        "type": "<entity|lexical>",
                        "selected_obj_id": <int_or_null>,
                        "grounded_description": "<four-layer for entity | value for lexical>",
                        "selection_rationale": "<why chosen>"
                    }
                },
                "instantiated_question": "<the pre-instantiated positive question>",
                "mcq_options": {"A": "...", "B": "...", "C": "...", "D": "...", "E": "..."},
                "think": [
                    "<Step 1: first-person deliberation citing OBJ id / Image idx / distance>",
                    "<Step 2: ...>",
                    "<... step count within the category band; do NOT state the final answer here>"
                ],
                "reasoning": "- <bullet 1: OBJ ID + spatial data>\n- <bullet 2: camera ref>\n- <bullet 3: conclusion toward the answer> (at most 5 bullets)",
                "answer": "<final answer DERIVED from think + reasoning above (for mcq: a single option letter, e.g. 'B' — exactly one correct option)>",
                "confidence": "<high|medium|low>",
                "relevant_cameras": [<int>]
            },
            "contrastive": {
                "altered_placeholders": ["<tags changed>"],
                "contrastive_strategy": "<what was changed and why>",
                "tag_mappings": { ... },
                "instantiated_question": "<the pre-instantiated contrastive question>",
                "mcq_options": {"A": "...", "B": "...", "C": "...", "D": "...", "E": "..."},
                "think": [
                    "<Step 1: independent deliberation for THIS question — do not reuse the positive's steps>",
                    "<... exactly 2 steps if the swapped-in entity/place is absent from the scene>"
                ],
                "reasoning": "- <bullet 1>\n- <bullet 2: why this altered placeholder changes the grounding>\n- <bullet 3: conclusion> (at most 5 bullets)",
                "answer": "<final answer DERIVED from think + reasoning, judged INDEPENDENTLY of the positive>",
                "confidence": "<high|medium|low>",
                "relevant_cameras": [<int>],
                "contrast_status": "<achieved|same_answer — honest comparison of the two committed answers>"
            }
        },
        ...
    ]
}

NOTES:
- """ + order_note + """
- "prior_disagreements" is OPTIONAL on positive/contrastive. Include it ONLY when your independent check disagrees with a Stage 1B/1C ego-applicability claim (see PRIOR APPLICABILITY VERIFICATION RULE in the system prompt). Schema per entry: {"source": "signal"|"sign", "image_idx": int, "object": str, "prior_says": str, "vlm_says": str, "evidence": str}. The answer itself must still follow the prior's claim.
- "mcq_options" is REQUIRED for mcq answer_type; omit for other types.
- For mcq, "answer" is EXACTLY ONE option letter (e.g. "B"). Never comma-separated, never multiple letters.
- MCQ is single-correct: construct exactly 5 options (A-E) with EXACTLY ONE that is factually correct of the scene plus 4 plausible distractors. The `answer` field is a single option letter (e.g. 'B'). Never assert more than one option as correct.
- Randomize which positions (A-E) hold the correct options across pairs.
- "think" is REQUIRED on every positive/contrastive object: a JSON array of step strings, step count within the category band (exactly 2 when the questioned entity/place is absent). Never state the final answer inside "think".
- "contrast_status" is REQUIRED on every non-null contrastive: "achieved" when the two committed answers genuinely differ, "same_answer" when they match. A same_answer pair is VALID output — do not alter answers to avoid it.
If no valid contrastive question can be constructed at all: "contrastive": null, "contrastive_skip_reason": "<explanation>" (this is contrast_status "skipped")
"""

    parts.append(schema)
    parts.append("Output ONLY the JSON object, no additional text.")

    return "\n".join(parts)


def enrich_template_with_question_bank(
    template_data: Dict,
    category: str,
    question_bank: Dict,
) -> Dict:
    """
    Enrich a Stage 1 template result with original_placeholders, expected_answers,
    and notes from the question bank. Looks up by template_idx (1-based).
    """
    template_idx = template_data.get('template_idx', 0)
    templates = get_category_templates(question_bank, category)

    enriched = dict(template_data)
    enriched['category'] = category

    # template_idx is 1-based; templates list is 0-based
    idx = template_idx - 1
    if 0 <= idx < len(templates):
        bank_template = templates[idx]

        # Get original placeholders from question bank (keys without <> brackets)
        # Convert to <tag> format to match Stage 1 output convention
        raw_placeholders = bank_template.get('placeholders', {})
        original_placeholders = {}
        for key, values in raw_placeholders.items():
            bracket_key = f"<{key}>" if not key.startswith('<') else key
            original_placeholders[bracket_key] = values
        enriched['original_placeholders'] = original_placeholders

        # Get expected_answers, notes, and ego_centric_note if available
        if 'expected_answers' in bank_template:
            enriched['expected_answers'] = bank_template['expected_answers']
        if 'notes' in bank_template:
            enriched['notes'] = bank_template['notes']
        if 'ego_centric_note' in bank_template:
            enriched['ego_centric_note'] = bank_template['ego_centric_note']

    return enriched


# =============================================================================
# Worker process initializer and global state
# =============================================================================

_worker_client = None
_worker_loader = None
_worker_analyzer = None
_worker_question_bank = None
_worker_risk_results_dir = None
_worker_traffic_results_dir = None
_worker_sign_results_dir = None


def _worker_init(
    model_name: str, api_base: str, api_key: str,
    pkl_path: str, question_bank_path: str,
    risk_results_dir: str = "",
    traffic_results_dir: str = "",
    sign_results_dir: str = "",
):
    """
    Initializer for each worker process.
    Creates the OpenAI client, NuScenesDataLoader, SceneAnalyzer, and loads question bank.
    """
    global _worker_client, _worker_loader, _worker_analyzer, _worker_question_bank
    global _worker_risk_results_dir, _worker_traffic_results_dir, _worker_sign_results_dir

    from openai import OpenAI as _OpenAI
    _worker_client = _OpenAI(
        base_url=api_base,
        api_key=api_key,
        timeout=180.0,
        max_retries=3,
    )
    _worker_loader = NuScenesDataLoader(pkl_path)
    _worker_analyzer = SceneAnalyzer(_worker_loader)
    _worker_question_bank = load_question_bank(question_bank_path)
    _worker_risk_results_dir = risk_results_dir
    _worker_traffic_results_dir = traffic_results_dir
    _worker_sign_results_dir = sign_results_dir


# =============================================================================
# Worker function
# =============================================================================

def _worker_process_sample(
    sample_idx: int,
    model_name: str,
    stage1_dir: str,
    resize_factor: int,
    max_new_tokens: int,
    filter_distance: float,
    rear_filter: Optional[float],
    output_dir: str,
    categories: Optional[List[str]] = None,
    max_pairs_per_template: int = 3,
    temperature: float = 0.6,
    max_templates_per_category: int = 0,
    disagreement_dir: str = "prior_disagreements_rl",
) -> Dict:
    """
    Worker function: load Stage 1 results, enrich with question bank data,
    pre-instantiate QA pairs, call VLM under the fixed RL output contract
    (think -> reasoning -> answer, contrast_status self-report). One retry
    per template on retry-class validation failures; failed retries are kept
    with flags — never silently dropped.

    max_templates_per_category: 0 = no cap; N > 0 caps templates per category
    (smoke-test support, T10).
    """
    global _worker_client, _worker_loader, _worker_analyzer, _worker_question_bank
    global _worker_risk_results_dir, _worker_traffic_results_dir, _worker_sign_results_dir

    try:
        client = _worker_client
        loader = _worker_loader
        analyzer = _worker_analyzer
        question_bank = _worker_question_bank
        risk_results_dir = _worker_risk_results_dir
        traffic_results_dir = _worker_traffic_results_dir
        sign_results_dir = _worker_sign_results_dir

        # 1. Load Stage 1 results for this sample
        stage1_results = load_stage1_results(stage1_dir, sample_idx, categories)
        if not stage1_results:
            return {
                'sample_idx': sample_idx,
                'skipped': True,
                'reason': 'No applicable templates from Stage 1',
                'timestamp': datetime.now().isoformat(),
            }

        # 2. Analyze sample for spatial data
        scene_data = analyzer.analyze_sample(
            sample_idx,
            max_distance=filter_distance,
            rear_filter_distance=rear_filter,
        )

        # 3. Get sample for images
        sample = loader.get_sample(sample_idx)

        # 4. Prepare base64-encoded images
        image_content = []
        for cam_name in EGOCENTRIC_CAMERA_NAMES:
            label = CAM_LABELS[cam_name]
            image_content.append({"type": "text", "text": f"=== {label} ==="})
            img_path = sample.cameras[cam_name].image_path
            flip = cam_name in REAR_CAMERAS
            img_array = loader.load_image(
                img_path, resize_factor=resize_factor, flip_horizontal=flip
            )
            img_pil = Image.fromarray(img_array)
            data_uri = image_to_base64_data_uri(img_pil)
            image_content.append({
                "type": "image_url",
                "image_url": {"url": data_uri}
            })

        # 5. Format spatial data
        object_positions_text = format_object_positions(scene_data)
        closest_objects_text = format_closest_objects(scene_data)

        # 6. Get driving command
        driving_command = scene_data['ego_info'].get('driving_command', 'Unknown')

        # 6b. Load prior analysis responses (shared across templates)
        risk_response = load_risk_assessment_response(risk_results_dir, sample_idx)
        traffic_response = load_traffic_analysis_response(traffic_results_dir, sample_idx)
        sign_response = load_traffic_sign_response(sign_results_dir, sample_idx)

        # 7. Flatten all templates across categories, enrich with question bank
        all_templates = []
        for category, questions in stage1_results.items():
            for q_key, q_data in questions.items():
                enriched = enrich_template_with_question_bank(
                    q_data, category, question_bank
                )
                enriched['q_key'] = q_key
                all_templates.append(enriched)

        # Smoke-test cap (T10): keep only the first N templates per category
        if max_templates_per_category and max_templates_per_category > 0:
            capped, seen = [], {}
            for t_ in all_templates:
                c = t_['category']
                seen[c] = seen.get(c, 0) + 1
                if seen[c] <= max_templates_per_category:
                    capped.append(t_)
            all_templates = capped

        total_templates = len(all_templates)

        # 8. Process each template: pre-instantiate pairs, then VLM verifies + proposes more
        os.makedirs(output_dir, exist_ok=True)
        all_qa_results = []
        all_raw_responses = []
        total_pairs_count = 0
        sample_disagreements = []  # collected across all templates for this sample
        rl_stats = {
            'templates_attempted': 0,
            'templates_parsed': 0,
            'templates_failed_parse': 0,
            'retries': 0,
            'flag_counts': {},
            'band': {},
            'contrast_status': {},
        }

        for tmpl_idx, tmpl in enumerate(all_templates):
            tqdm.write(f"    [Sample {sample_idx}] Template {tmpl_idx + 1}/{total_templates}: {tmpl['category']} - \"{tmpl['template'][:50]}...\"")

            # Pre-instantiate diverse QA pairs programmatically
            pair_seed = sample_idx * 10000 + tmpl_idx  # reproducible per sample+template
            pre_pairs = pre_instantiate_pairs(
                template=tmpl,
                scene_data=scene_data,
                max_pairs=max_pairs_per_template,
                seed=pair_seed,
            )
            tqdm.write(f"      Pre-instantiated {len(pre_pairs)} pairs")

            # Build verification prompt (VLM answers + proposes additional contrasts)
            prompt = build_verification_prompt(
                driving_command=driving_command,
                object_positions_text=object_positions_text,
                closest_objects_text=closest_objects_text,
                template=tmpl,
                pre_pairs=pre_pairs,
                template_num=tmpl_idx + 1,
                total_templates=total_templates,
            )

            # Prepend prior analyses.
            # Order of prepends determines distance from question body:
            #   the LAST-prepended section sits FURTHEST from the body.
            # We prepend category-specific priors (risk/signal) FIRST so they stay
            # closer to the question, then sign extraction LAST so it sits further
            # away but still informs the model.
            tmpl_category = tmpl.get('category', '')

            # Risk assessment — only for Dynamic_Agents_and_Risk_Assessment
            if tmpl_category == "Dynamic_Agents_and_Risk_Assessment" and risk_response:
                prompt = (
                    "\n\n=== RISK ASSESSMENT ANALYSIS (from prior analysis) ===\n"
                    f"{risk_response}\n"
                    "=== END OF RISK ASSESSMENT ANALYSIS ===\n\n"
                ) + prompt

            # Traffic signal — applied universally across all 10 categories
            if traffic_response:
                prompt = (
                    "\n\n=== TRAFFIC SIGNAL ANALYSIS (from prior analysis) ===\n"
                    f"{traffic_response}\n"
                    "=== END OF TRAFFIC SIGNAL ANALYSIS ===\n\n"
                ) + prompt

            # Traffic sign extraction — applied universally across all 10 categories
            if sign_response:
                prompt = (
                    "\n\n=== TRAFFIC SIGN EXTRACTION (from prior analysis) ===\n"
                    f"{sign_response}\n"
                    "=== END OF TRAFFIC SIGN EXTRACTION ===\n\n"
                ) + prompt

            # Build user content: images + prompt
            user_content = list(image_content) + [
                {"type": "text", "text": prompt}
            ]

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]

            # API call + parse + validate, with ONE re-inference retry (T8) on
            # retry-class failures. Failed retries keep the flagged result —
            # never a silent drop (T7).
            rl_stats['templates_attempted'] += 1
            result = None
            result_flags: List[str] = []

            for attempt in (1, 2):
                response_obj = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                )
                response_text = response_obj.choices[0].message.content

                all_raw_responses.append({
                    'template_num': tmpl_idx + 1,
                    'category': tmpl['category'],
                    'template': tmpl['template'],
                    'pre_instantiated_pairs': pre_pairs,
                    'attempt': attempt,
                    'response': response_text,
                })

                # Parse response — expect a single template result dict
                parsed = parse_json_response(response_text)

                # Normalize: the response should be a dict with 'pairs'
                candidate = None
                if isinstance(parsed, dict):
                    if 'pairs' in parsed or 'template_idx' in parsed:
                        candidate = parsed
                    elif 'qa_results' in parsed and isinstance(parsed['qa_results'], list):
                        # Model wrapped in qa_results array — take first
                        for r in parsed['qa_results']:
                            if isinstance(r, dict) and ('pairs' in r or 'template_idx' in r):
                                candidate = r
                                break

                if candidate is None:
                    if attempt == 1:
                        rl_stats['retries'] += 1
                        tqdm.write(f"      -> malformed JSON (template {tmpl_idx + 1}), re-inference retry")
                        continue
                    tqdm.write(f"      -> PARSE FAILED after retry (template {tmpl_idx + 1}) — raw kept in detailed file")
                    break

                # Pin metadata from the template — never trust VLM-supplied values
                candidate['category'] = tmpl['category']
                candidate['template_idx'] = tmpl.get('template_idx', tmpl_idx + 1)
                candidate['answer_type'] = tmpl['answer_type']

                # v1 answer-type contract + RL contract (think band,
                # contrast_status, status_mismatch). validate_rl_result also
                # downgrades null-answer contrastives and stamps pair-level
                # contrast_status.
                ok, failures = validate_result_contract(candidate, tmpl['answer_type'])
                rl_flags, retry_needed = validate_rl_result(
                    candidate, tmpl['answer_type'], tmpl['category'], pre_pairs
                )
                if (not ok or retry_needed) and attempt == 1:
                    rl_stats['retries'] += 1
                    reason = failures[0] if not ok else ','.join(f for f in rl_flags if f in RETRY_FLAGS)
                    tqdm.write(f"      -> validation retry (template {tmpl_idx + 1}): {reason}")
                    continue

                result = candidate
                result_flags = rl_flags + ([f"contract_violation"] if not ok else [])
                break

            if result is None:
                rl_stats['templates_failed_parse'] += 1
                continue

            rl_stats['templates_parsed'] += 1
            result['rl_flags'] = sorted(set(result_flags))
            stamp_provenance(result, sample_idx, tmpl_idx + 1)
            collect_template_stats(result, tmpl['category'], result['rl_flags'], rl_stats)

            # Count pairs
            pairs = result.get('pairs', [])
            num_pairs = len(pairs) if isinstance(pairs, list) else 0
            total_pairs_count += num_pairs
            flag_note = f" flags={result['rl_flags']}" if result['rl_flags'] else ""
            tqdm.write(f"      -> OK: {num_pairs} pairs{flag_note} (running total: {total_pairs_count})")

            all_qa_results.append(result)

            # Extract prior_disagreements from each pair's positive/contrastive/vlm_proposed_contrastives
            for pair in (result.get('pairs') or []):
                if not isinstance(pair, dict):
                    continue
                pair_id = pair.get('pair_id')
                # positive + contrastive
                for role in ('positive', 'contrastive'):
                    section = pair.get(role)
                    if not isinstance(section, dict):
                        continue
                    disagreements = section.get('prior_disagreements')
                    if not isinstance(disagreements, list) or not disagreements:
                        continue
                    for d in disagreements:
                        if not isinstance(d, dict):
                            continue
                        sample_disagreements.append({
                            'template_idx': tmpl.get('template_idx', tmpl_idx + 1),
                            'q_key': tmpl.get('q_key'),
                            'category': tmpl['category'],
                            'template': tmpl.get('template', ''),
                            'pair_id': pair_id,
                            'pair_type': role,
                            'question': section.get('instantiated_question', ''),
                            'prior_disagreement': d,
                            'vlm_response': {
                                'reasoning': section.get('reasoning', ''),
                                'grounding': section.get('grounding', []),
                                'answer': section.get('answer', ''),
                            },
                        })
                # vlm_proposed_contrastives (list of sections)
                for idx_prop, section in enumerate(pair.get('vlm_proposed_contrastives', []) or []):
                    if not isinstance(section, dict):
                        continue
                    disagreements = section.get('prior_disagreements')
                    if not isinstance(disagreements, list) or not disagreements:
                        continue
                    for d in disagreements:
                        if not isinstance(d, dict):
                            continue
                        sample_disagreements.append({
                            'template_idx': tmpl.get('template_idx', tmpl_idx + 1),
                            'q_key': tmpl.get('q_key'),
                            'category': tmpl['category'],
                            'template': tmpl.get('template', ''),
                            'pair_id': pair_id,
                            'pair_type': f'vlm_proposed_contrastive[{idx_prop}]',
                            'question': section.get('instantiated_question', ''),
                            'prior_disagreement': d,
                            'vlm_response': {
                                'reasoning': section.get('reasoning', ''),
                                'grounding': section.get('grounding', []),
                                'answer': section.get('answer', ''),
                            },
                        })

        # 8.5 If any prior-vs-VLM disagreements were flagged, persist them to
        #     prior_disagreements/sample_{idx}_disagreements.json (outside qa_results/)
        if sample_disagreements:
            os.makedirs(disagreement_dir, exist_ok=True)
            dis_path = os.path.join(
                disagreement_dir, f"sample_{sample_idx}_disagreements.json"
            )
            with open(dis_path, 'w') as f:
                json.dump({
                    'sample_idx': sample_idx,
                    'timestamp': datetime.now().isoformat(),
                    'total_disagreements': len(sample_disagreements),
                    'disagreements': sample_disagreements,
                }, f, indent=2, ensure_ascii=False)

        # 9. Save output
        output_data = {
            'sample_idx': sample_idx,
            'scene_meta': scene_data['scene_meta'],
            'ego_info': scene_data['ego_info'],
            'total_templates_processed': total_templates,
            'total_templates_with_results': len(all_qa_results),
            'total_pairs_generated': total_pairs_count,
            'timestamp': datetime.now().isoformat(),
            'rl_stats': rl_stats,
            'qa_results': all_qa_results,
        }

        output_file = os.path.join(output_dir, f"sample_{sample_idx}_qa_results.json")
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)

        # Save detailed log with raw responses
        detailed_file = os.path.join(output_dir, f"sample_{sample_idx}_qa_detailed.json")
        detailed_data = {
            'sample_idx': sample_idx,
            'scene_meta': scene_data['scene_meta'],
            'timestamp': datetime.now().isoformat(),
            'filter_distance': filter_distance,
            'rear_filter': rear_filter,
            'total_templates': total_templates,
            'total_templates_with_results': len(all_qa_results),
            'total_pairs': total_pairs_count,
            'stage1_categories': list(stage1_results.keys()),
            'raw_responses': all_raw_responses,
        }
        with open(detailed_file, 'w') as f:
            json.dump(detailed_data, f, indent=4, ensure_ascii=False)

        return {
            'sample_idx': sample_idx,
            'total_templates': total_templates,
            'total_qa_results': len(all_qa_results),
            'total_pairs': total_pairs_count,
            'categories': list(stage1_results.keys()),
            'rl_stats': rl_stats,
            'timestamp': datetime.now().isoformat(),
        }

    except Exception as e:
        import traceback
        return {
            'sample_idx': sample_idx,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now().isoformat(),
        }


def _aggregate_disagreements(disagreement_dir: str):
    """Scan disagreement_dir for per-sample *_disagreements.json files and
    return a summary dict. Returns None if the directory doesn't exist or has
    no disagreement files."""
    if not os.path.isdir(disagreement_dir):
        return None
    pattern = re.compile(r'^sample_(\d+)_disagreements\.json$')
    files = [f for f in os.listdir(disagreement_dir) if pattern.match(f)]
    if not files:
        return None

    total_disagreements = 0
    by_source = {}
    by_category = {}
    by_mismatch = {}
    sample_indices = []

    for fname in files:
        m = pattern.match(fname)
        if not m:
            continue
        idx = int(m.group(1))
        try:
            with open(os.path.join(disagreement_dir, fname), 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue
        disagreements = data.get('disagreements', [])
        if not disagreements:
            continue
        sample_indices.append(idx)
        for entry in disagreements:
            total_disagreements += 1
            prior = entry.get('prior_disagreement', {})
            source = prior.get('source', 'unknown')
            by_source[source] = by_source.get(source, 0) + 1
            category = entry.get('category', 'unknown')
            by_category[category] = by_category.get(category, 0) + 1
            prior_says = prior.get('prior_says', '?')
            vlm_says = prior.get('vlm_says', '?')
            key = f"{source}:{prior_says}→{vlm_says}"
            by_mismatch[key] = by_mismatch.get(key, 0) + 1

    return {
        'generated_at': datetime.now().isoformat(),
        'total_samples_with_disagreements': len(sample_indices),
        'total_disagreements': total_disagreements,
        'by_source': dict(sorted(by_source.items())),
        'by_category': dict(sorted(by_category.items())),
        'by_mismatch_type': dict(sorted(by_mismatch.items())),
        'sample_indices_with_disagreements': sorted(sample_indices),
    }


def _worker_wrapper(args):
    """Unpack tuple arguments and call _worker_process_sample."""
    return _worker_process_sample(*args)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="RL (GRPO) Contrastive QA Pair Generation — tier-banded think, "
                    "contrast_status self-report (vLLM Multiprocessing)"
    )

    # Model and API settings
    parser.add_argument(
        "--model_name", type=str,
        default="Qwen/Qwen3-VL-235B-A22B-Thinking",
        help="Model name served by vLLM",
    )
    parser.add_argument(
        "--api_base", type=str,
        default="http://localhost:8000/v1",
        help="vLLM OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--api_key", type=str, default="EMPTY",
        help='API key (default "EMPTY" for local vLLM)',
    )

    # Data paths
    parser.add_argument(
        "--pkl_path", type=str,
        default=os.environ.get("NUSCENES_PKL_PATH", "nuscenes2d_ego_temporal_infos_val.pkl"),
        help="Path to nuScenes pkl file",
    )
    parser.add_argument(
        "--question_bank", type=str,
        default=os.environ.get("QUESTION_BANK_PATH", "question_bank.json"),
        help="Path to question bank JSON (for original_placeholders, expected_answers)",
    )
    parser.add_argument(
        "--stage1_dir", type=str,
        default="qa_outputs",
        help="Question selector output directory (contains category subdirs)",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="qa_results_rl_thinking",
        help="Output directory for QA results (isolated from the SFT pipeline)",
    )
    parser.add_argument(
        "--risk_results_dir", type=str,
        default="risk_assessment_results",
        help="Directory containing risk assessment result JSONs "
             "(prepended to prompt for Dynamic_Agents_and_Risk_Assessment templates)",
    )
    parser.add_argument(
        "--traffic_results_dir", type=str,
        default="traffic_signal_analysis_results",
        help="Directory containing traffic analysis result JSONs "
             "(prepended to prompt for Traffic_Signs_and_Signals templates)",
    )
    parser.add_argument(
        "--sign_results_dir", type=str,
        default="traffic_sign_results",
        help="Directory containing traffic sign extraction result JSONs "
             "(prepended to prompt for ALL 10 categories)",
    )

    # Sample range
    parser.add_argument("--start_idx", type=int, default=0, help="Starting sample index")
    parser.add_argument("--end_idx", type=int, default=6018, help="Ending sample index")
    parser.add_argument("--from_stage1", action="store_true",
                        help="Auto-discover sample indices from Stage 1 output files "
                             "instead of using start_idx/end_idx range. "
                             "Processes only samples that have applicable_questions results.")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip samples whose sample_{idx}_qa_results.json already "
                             "exists in output_dir (resume an interrupted run). "
                             "A sample writes its output file only on completion, so "
                             "partially processed samples are safely redone.")
    parser.add_argument("--sample_indices", type=str, default="",
                        help="Comma-separated explicit sample indices (e.g. \"100,228,304\"). "
                             "Overrides --start_idx/--end_idx and --from_stage1. "
                             "Smoke-test support (T10).")

    # Category selection
    parser.add_argument(
        "--category", type=str, default="all",
        choices=VALID_CATEGORIES + ["all"],
        help='Question category to process, or "all" for all categories',
    )

    # Filtering options
    parser.add_argument("--filter_distance", type=float, default=50.0,
                        help="Maximum distance (m) from ego to include objects")
    parser.add_argument("--rear_filter", type=float, default=20.0,
                        help="Maximum distance (m) for non-vehicle objects behind ego")

    # Inference options
    parser.add_argument("--resize_factor", type=int, default=1,
                        help="Image resize factor for the annotator VLM (1 = full 1600x900, default; 2 = half 800x450)")
    parser.add_argument("--max_new_tokens", type=int, default=16384,
                        help="Maximum tokens to generate per template")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature for generation (default: 0.6)")
    parser.add_argument("--max_pairs", type=int, default=3,
                        help="Maximum pre-instantiated pairs per template (default: 3)")
    parser.add_argument("--max_templates_per_category", type=int, default=0,
                        help="Cap templates per category per sample (0 = no cap). "
                             "Smoke-test support (T10): e.g. 2 for the standard smoke run.")
    parser.add_argument("--disagreement_dir", type=str, default="prior_disagreements_rl",
                        help="Directory for per-sample prior_disagreements JSON files. "
                             "Written only when the VLM flags a disagreement with Stage 1B/1C "
                             "ego-applicability claims. Separate from --output_dir.")

    # Multiprocessing
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of parallel worker processes")

    # Logging
    parser.add_argument("--log_dir", type=str, default="answer_generator_rl_logs",
                        help="Directory for log files")

    args = parser.parse_args()

    # Determine sample indices
    if args.sample_indices.strip():
        sample_indices = sorted({int(s) for s in args.sample_indices.split(',') if s.strip()})
    elif args.from_stage1:
        sample_indices = discover_stage1_sample_indices(args.stage1_dir)
        if not sample_indices:
            print(f"ERROR: No Stage 1 results found in {args.stage1_dir}/")
            sys.exit(1)
    else:
        sample_indices = list(range(args.start_idx, args.end_idx + 1))

    # Resume support: drop samples that already have a completed output file.
    if args.skip_existing:
        before = len(sample_indices)
        sample_indices = [
            idx for idx in sample_indices
            if not os.path.isfile(
                os.path.join(args.output_dir, f"sample_{idx}_qa_results.json")
            )
        ]
        print(f"--skip_existing: {before - len(sample_indices)} samples already "
              f"complete in {args.output_dir}/, {len(sample_indices)} remaining")
        if not sample_indices:
            print("Nothing to do — all samples already have results.")
            sys.exit(0)

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.log_dir, f"qa_gen_stage2_{timestamp}.log")

    def log_and_print(message):
        print(message)
        with open(log_file, "a") as f:
            f.write(message + "\n")

    # Print configuration
    log_and_print("=" * 80)
    log_and_print("Stage 2: Tag Refinement + Contrastive QA Pair Generation (vLLM MP)")
    log_and_print("=" * 80)
    log_and_print(f"  API base: {args.api_base}")
    log_and_print(f"  Model: {args.model_name}")
    log_and_print(f"  Num workers: {args.num_workers}")
    if args.from_stage1:
        log_and_print(f"  Mode: from_stage1 (auto-discovered {len(sample_indices)} samples)")
    else:
        log_and_print(f"  Sample range: {args.start_idx} to {args.end_idx} ({len(sample_indices)} samples)")
    log_and_print(f"  Stage 1 dir: {args.stage1_dir}")
    log_and_print(f"  Question bank: {args.question_bank}")
    log_and_print(f"  Category: {args.category}")
    log_and_print(f"  Filter distance: {args.filter_distance}m")
    log_and_print(f"  Rear filter: {args.rear_filter}m")
    log_and_print(f"  Resize factor: {args.resize_factor}")
    log_and_print(f"  Max new tokens: {args.max_new_tokens}")
    log_and_print(f"  Temperature: {args.temperature}")
    log_and_print(f"  Max pairs per template: {args.max_pairs}")
    log_and_print(f"  Output contract: FIXED think -> reasoning -> answer + contrast_status (RL/GRPO)")
    if args.max_templates_per_category:
        log_and_print(f"  Max templates per category: {args.max_templates_per_category} (smoke test)")
    log_and_print(f"  Disagreement dir: {args.disagreement_dir}")
    log_and_print(f"  Output dir: {args.output_dir}")
    log_and_print(f"  Log file: {log_file}")
    log_and_print(f"  Camera order: FL, F, FR, RL, R, RR (egocentric)")
    log_and_print(f"  Rear cameras: horizontally flipped")
    log_and_print(f"  Risk results dir: {args.risk_results_dir}")
    log_and_print(f"  Traffic results dir: {args.traffic_results_dir}")
    log_and_print(f"  Sign results dir:    {args.sign_results_dir}")
    log_and_print("=" * 80)

    # Verify vLLM server
    log_and_print(f"\nVerifying vLLM server at: {args.api_base}")
    try:
        test_client = OpenAI(base_url=args.api_base, api_key=args.api_key)
        models = test_client.models.list()
        log_and_print(f"  Server reachable. Available models: {[m.id for m in models.data]}")
    except Exception as e:
        log_and_print(f"  WARNING: Could not connect to vLLM server: {e}")
        log_and_print(f"  Proceeding anyway -- workers will retry on their own.")

    # Resolve category filter
    categories = None if args.category == "all" else [args.category]

    # Build argument tuples
    worker_args = [
        (
            idx,
            args.model_name,
            args.stage1_dir,
            args.resize_factor,
            args.max_new_tokens,
            args.filter_distance,
            args.rear_filter,
            args.output_dir,
            categories,
            args.max_pairs,
            args.temperature,
            args.max_templates_per_category,
            args.disagreement_dir,
        )
        for idx in sample_indices
    ]

    num_workers = min(args.num_workers, len(sample_indices))
    log_and_print(f"\nStarting multiprocessing pool with {num_workers} workers...")
    log_and_print(f"Processing {len(sample_indices)} samples...\n")

    success_count = 0
    failure_count = 0
    skipped_count = 0
    failed_indices = []
    total_qa_all = 0
    # Run-wide RL stats aggregation (T8)
    agg = {
        'templates_attempted': 0, 'templates_parsed': 0,
        'templates_failed_parse': 0, 'retries': 0,
        'flag_counts': {}, 'band': {}, 'contrast_status': {},
    }

    def _merge_rl_stats(s):
        if not isinstance(s, dict):
            return
        for k in ('templates_attempted', 'templates_parsed',
                  'templates_failed_parse', 'retries'):
            agg[k] += s.get(k, 0)
        for f, n in (s.get('flag_counts') or {}).items():
            agg['flag_counts'][f] = agg['flag_counts'].get(f, 0) + n
        for cat, b in (s.get('band') or {}).items():
            dst = agg['band'].setdefault(cat, {'sides_ok': 0, 'sides_violation': 0})
            dst['sides_ok'] += b.get('sides_ok', 0)
            dst['sides_violation'] += b.get('sides_violation', 0)
        for st, n in (s.get('contrast_status') or {}).items():
            agg['contrast_status'][st] = agg['contrast_status'].get(st, 0) + n

    total_start = time.time()

    with Pool(
        processes=num_workers,
        initializer=_worker_init,
        initargs=(
            args.model_name, args.api_base, args.api_key,
            args.pkl_path, args.question_bank,
            args.risk_results_dir, args.traffic_results_dir,
            args.sign_results_dir,
        ),
    ) as pool:
        results_iter = pool.imap_unordered(_worker_wrapper, worker_args)

        for result in tqdm(
            results_iter,
            total=len(sample_indices),
            desc="Processing samples",
            unit="sample",
            ncols=100,
        ):
            sample_idx = result.get("sample_idx", "unknown")
            if "error" in result:
                failure_count += 1
                failed_indices.append(sample_idx)
                tqdm.write(f"  x Sample {sample_idx} failed: {result['error']}")
            elif result.get("skipped"):
                skipped_count += 1
                tqdm.write(f"  - Sample {sample_idx} skipped: {result.get('reason', '')}")
            else:
                success_count += 1
                qa_count = result.get("total_pairs", 0)
                total_qa_all += qa_count
                _merge_rl_stats(result.get('rl_stats'))
                tqdm.write(f"  v Sample {sample_idx} done (templates={result.get('total_templates', 0)}, results={result.get('total_qa_results', 0)}, pairs={qa_count})")

    total_elapsed = time.time() - total_start

    # Summary
    log_and_print(f"\n{'=' * 80}")
    log_and_print("Stage 2: Tag Refinement + Contrastive QA Pair Generation Complete")
    log_and_print(f"{'=' * 80}")
    if args.from_stage1:
        log_and_print(f"  Mode: from_stage1 (auto-discovered)")
    else:
        log_and_print(f"  Sample range: {args.start_idx} to {args.end_idx}")
    log_and_print(f"  Total samples: {len(sample_indices)}")
    log_and_print(f"  Successful: {success_count}")
    log_and_print(f"  Skipped (no Stage 1 data): {skipped_count}")
    log_and_print(f"  Failed: {failure_count}")
    log_and_print(f"  Total contrastive QA pairs generated: {total_qa_all}")
    if success_count + failure_count > 0:
        log_and_print(f"  Success rate: {success_count * 100.0 / (success_count + failure_count):.2f}%")
    log_and_print(f"  Total wall time: {total_elapsed:.2f}s")
    if total_elapsed > 0:
        log_and_print(f"  Effective throughput: {len(sample_indices) / total_elapsed:.2f} samples/s")
    log_and_print(f"  Workers used: {args.num_workers}")
    log_and_print(f"  Results saved to: {args.output_dir}/")

    # Save failed indices
    if failed_indices:
        log_and_print(f"\n  FAILED SAMPLE INDICES ({len(failed_indices)} samples):")
        log_and_print(f"    {sorted(failed_indices)}")
        failed_file = os.path.join(args.log_dir, f"failed_qa_gen_indices_{timestamp}.txt")
        with open(failed_file, "w") as f:
            f.write("\n".join(map(str, sorted(failed_indices))))
        log_and_print(f"    Saved to: {failed_file}")

    # RL run report (T8): parse success, per-category band compliance,
    # contrast_status distribution, status_mismatch count.
    attempted = agg['templates_attempted']
    parsed = agg['templates_parsed']
    band_compliance = {}
    for cat, b in sorted(agg['band'].items()):
        tot = b['sides_ok'] + b['sides_violation']
        band_compliance[cat] = {
            **b,
            'compliance_pct': round(100 * b['sides_ok'] / tot, 2) if tot else None,
        }
    cs_total = sum(agg['contrast_status'].values())
    rl_report = {
        'timestamp': datetime.now().isoformat(),
        'model': args.model_name,
        'samples_processed': success_count,
        'templates_attempted': attempted,
        'templates_parsed': parsed,
        'parse_success_pct': round(100 * parsed / attempted, 2) if attempted else None,
        'templates_failed_parse': agg['templates_failed_parse'],
        'retries': agg['retries'],
        'flag_counts': dict(sorted(agg['flag_counts'].items())),
        'band_compliance_by_category': band_compliance,
        'contrast_status_distribution': {
            **agg['contrast_status'],
            **({f'{k}_pct': round(100 * v / cs_total, 2)
                for k, v in agg['contrast_status'].items()} if cs_total else {}),
        },
    }
    report_path = os.path.join(args.output_dir, 'rl_run_report.json')
    with open(report_path, 'w') as f:
        json.dump(rl_report, f, indent=2, ensure_ascii=False)
    log_and_print(f"\n  RL RUN REPORT (also saved to {report_path}):")
    log_and_print(f"    Parse success: {parsed}/{attempted}"
                  f" ({rl_report['parse_success_pct']}%)" if attempted else "    Parse success: n/a")
    log_and_print(f"    Retries: {agg['retries']} | Flags: {rl_report['flag_counts']}")
    log_and_print(f"    Contrast status: {agg['contrast_status']}")
    for cat, b in band_compliance.items():
        log_and_print(f"    Band {cat}: {b['sides_ok']}/{b['sides_ok'] + b['sides_violation']} ok"
                      f" ({b['compliance_pct']}%)")

    # Aggregate disagreement summary across all workers' per-sample files
    summary = _aggregate_disagreements(args.disagreement_dir)
    if summary is not None:
        summary_path = os.path.join(args.disagreement_dir, "_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        log_and_print(f"\n  PRIOR DISAGREEMENTS:")
        log_and_print(f"    Samples with disagreements: {summary['total_samples_with_disagreements']}")
        log_and_print(f"    Total disagreements: {summary['total_disagreements']}")
        log_and_print(f"    By source: {summary['by_source']}")
        log_and_print(f"    Summary saved to: {summary_path}")

    log_and_print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
