#!/usr/bin/env python3
"""
Answer Generator Module for nuScenes QA Dataset

Takes question selector outputs (Stage 1), classifies placeholders as ENTITY/LEXICAL,
grounds entities to specific objects, generates positive QA pairs, and produces
contrastive QA pairs with altered placeholders for training data diversity.

Prerequisites:
    # Start vLLM server first:
    vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct --tensor-parallel-size 8

Usage:
    # Process samples 0 to 100 with 4 workers
    python -m nuscenes_pipeline.modules.answer_generator --start_idx 0 --end_idx 100 --num_workers 4

    # Single sample test
    python -m nuscenes_pipeline.modules.answer_generator --start_idx 0 --end_idx 0 --num_workers 1

    # Specific category only
    python -m nuscenes_pipeline.modules.answer_generator --start_idx 0 --end_idx 100 --category Observation
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
  (C) For EACH positive, generate a paired CONTRASTIVE instance with a different answer
  (D) Ensure DIVERSITY across pairs: different objects, locations, comparisons, and cameras


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
  3. **Answer diversity**: Include a balanced mix of answers.
     - For y_or_n: not all positives should be "yes" — also include pairs where
       the positive is "no" and the contrastive is "yes" (by using originally-absent
       values as positive, confirming absence, then swapping to present values).
     - For mcq: vary the correct option letter (A-E) across pairs; use different
       distractors drawn from different aspects of the scene.
  4. **Difficulty diversity**: Mix easy pairs (large distance gaps, obvious presence/absence)
     with harder pairs (smaller margins, subtle distinctions).
  5. **Camera diversity**: Spread object selections across all 6 camera views,
     not only the front camera.

--- 4.3 Contrastive Strategies by Answer Type ---

**y_or_n** (141 templates):
  POSITIVE → "yes"  |  CONTRASTIVE → "no"
  (Also generate reverse pairs: POSITIVE → "no"  |  CONTRASTIVE → "yes")
  Strategies:
    a) Swap <object>/<agent> to an absent category
       (e.g., "pedestrians" → "cyclists" when no cyclists exist)
    b) Swap <place>/<location> to a location where the object is absent
       (e.g., "sidewalk" → "crosswalk" when no pedestrians are in the crosswalk)
    c) Swap <state>/<condition> to a false state
       (e.g., traffic light "green" → "red" when light is actually green)
    d) Swap <direction>/<view> to a view where the condition is absent
    e) For ENTITY placeholders: select a different OBJ that changes the answer
    f) For no-placeholder templates: if the answer is fixed, contrastive is null.

**mcq** (68 templates — multiple choice with 5 options A-E):
  Each MCQ question MUST have exactly 5 options labeled (A) through (E).
  The correct answer is the option letter (e.g., "B").

  Option construction rules:
    1. **Correct option**: The factually correct answer grounded in the scene.
    2. **Distractors (4 options)**: Plausible but incorrect alternatives.
       - Draw from `expected_answers` when available (these are the candidate pool).
       - If `expected_answers` has fewer than 5 values, generate plausible
         distractors that are semantically consistent with the question type
         (e.g., for a color question: other colors; for a vehicle type: other types).
       - If `expected_answers` has more than 5 values, select the 4 most
         plausible distractors from the pool.
    3. **Distractor quality**:
       - At least 1 distractor should be a "close miss" (plausible for the scene).
       - At least 1 distractor should be clearly wrong (to set a difficulty range).
       - Avoid absurd or semantically incoherent options.
    4. **Randomize position**: The correct answer should NOT always be option (A).
       Vary the position across pairs.

  POSITIVE → correct option letter  |  CONTRASTIVE → different correct option letter
  Strategies:
    a) For object/state templates: swap placeholder to a different object/state
       that produces a different correct answer and different MCQ options
    b) For comparison templates: swap object order or swap one object to shift
       which option wins
    c) For view-specific templates: change <direction>/<view> to get a different
       set of visible objects → different correct answer
    d) If only one valid answer exists in the scene, set contrastive to null.

**open_ended** (21 templates):
  POSITIVE → factual descriptive answer  |  CONTRASTIVE → different factual answer
  Strategies:
    a) Change <agent>/<object> to a different entity → different description
    b) Change <location>/<view> to a different area → different hazards/conditions
    c) Change <action> to a different action → different causal explanation
    d) For no-placeholder templates: if only one valid answer exists,
       set contrastive to null.

**num_count** (8 templates):
  POSITIVE → count N (where N > 0)  |  CONTRASTIVE → count M (where M ≠ N, ideally 0)
  Strategies:
    a) Swap <object> to an absent category → count = 0
    b) Swap <direction>/<view> to a view with a different count

**distance** (3 templates):
  POSITIVE → distance to object X  |  CONTRASTIVE → distance to object Y (notably different)
  Strategies:
    a) Change target object to one at a significantly different distance

--- 4.4 Contrastive Generation Constraints ---

1. **Minimal perturbation**: Change as FEW placeholders as possible (ideally ONE)
   to flip the answer.
2. **Factual grounding**: The contrastive answer MUST be verifiably correct.
   Never generate ambiguous or uncertain contrastive answers.
3. **Semantic coherence**: The contrastive question must be grammatically correct
   and semantically meaningful.
4. **Non-trivial difficulty**: Avoid trivially obvious contrasts.
   The contrastive question should still require scene analysis.
5. **No duplicate questions**: Each pair's positive question must be UNIQUE
   across all pairs for this template. Do not generate the same question twice
   with different contrastive partners.
6. **Graceful fallback**: If no valid contrastive instance is possible for a
   specific pair, set that pair's `contrastive` to null with explanation.

--- 4.5 When Contrastive is NOT Possible ---

Set contrastive to null when:
  - The template has NO placeholders AND the answer is uniquely determined
    (e.g., "Are there any speed bumps on this road?" — if yes, there is no way
    to rephrase the same template to get "no")
  - ALL `original_placeholders` values are confirmed present in the scene
    (no absent value to swap in)
  - The answer type is open_ended and only one meaningful interpretation exists
  - Altering any placeholder would produce an ambiguous or unverifiable answer


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
  A: "no"
  Reasoning: "No intersection is visible in any of the six camera views.
  Since the referenced location does not exist in this scene, the
  question cannot be affirmed. The answer is no."

EXAMPLE (incorrect — DO NOT use vacuous truth):
  Q: "Is the intersection clear of parked cars?"
  A: "yes"  ← WRONG
  Reasoning: "Since there is no intersection, it is trivially clear..."
  ← This uses vacuous truth logic, which produces inconsistent answers.

**Why this matters:** Vacuous truth ("X is trivially true because Y
doesn't exist") creates contradictions. For the same scene:
  - "Are there vehicles in the intersection?" → "no" (no intersection)
  - "Is the intersection clear of vehicles?" → "yes" (vacuously true)
These are logically contradictory for a training dataset. Always
answer "no" when the premise doesn't hold.

For other answer types (categorical, open_ended, num_count, etc.),
explicitly state that the referenced entity/location is absent and
provide the most factually grounded response possible.


========================================================================
  SECTION 7: ANSWER GENERATION RULES
========================================================================

All answers MUST be SHORT-ANSWER format unless the answer_type is open_ended.
Short-answer = the minimal factual response with no extra elaboration.
All detail and justification goes in the "reasoning" field, not in "answer".

1. **y_or_n**: Strictly "yes" or "no". Nothing else.
2. **mcq**: The correct option LETTER only (e.g., "B"). Nothing else.
   - Construct exactly 5 options labeled (A) through (E): 1 correct + 4 plausible distractors.
   - Draw distractors from `expected_answers` pool when available;
     generate semantically consistent alternatives when the pool is too small.
   - Include the full options list in `mcq_options`:
     {"A": "option text", "B": "option text", "C": "option text",
      "D": "option text", "E": "option text"}
   - Randomize the position of the correct answer across pairs.
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
Use 3-6 bullets per answer.

Format:
  "reasoning": "- <bullet 1>\n- <bullet 2>\n- <bullet 3>"

Required bullet content:
  - At least one bullet referencing a specific OBJ ID and its spatial data
  - At least one bullet citing which camera Image number(s) were used
  - At least one bullet stating the logical conclusion from evidence to answer
  - For spatial questions, include a bullet with distance/position data

Example (y_or_n):
  "reasoning": "- OBJ 18 (car) is 7.4m directly ahead, visible in Image 2 (Front)\n- OBJ 23 (car) is 12.5m ahead in the same lane, visible in Image 2\n- Both vehicles are in the ego-vehicle's current lane, confirming presence\n- Answer: yes"

Example (mcq):
  "reasoning": "- Image 2 (Front) shows a traffic light ahead of the ego-vehicle\n- The light displays a solid red circle, verified in Image 2\n- No green or yellow signal is visible for the ego-vehicle's lane\n- Correct option: (C) red"

For CONTRASTIVE reasoning, additionally include:
  - A bullet stating which placeholder(s) were changed and the value source
  - A bullet explaining why the change produces a different answer


========================================================================
  SECTION 9: DRIVING CONTEXT
========================================================================

The ego-vehicle's current driving command provides context for scene dynamics.
Use it for right-of-way, risk, and planning questions.
It does NOT change factual spatial relationships."""


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

    # Try to find a JSON object
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

    return None


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

        if is_category_dir:
            # Per-category subdir: directory name is the category
            if categories is not None and entry not in categories:
                continue
            for category, questions in data[sample_key].items():
                if questions:
                    merged[category] = questions
        else:
            # All-category subdir (e.g., "all"): file contains multiple categories
            for category, questions in data[sample_key].items():
                if categories is not None and category not in categories:
                    continue
                if questions and category not in merged:
                    merged[category] = questions

    return merged


def discover_stage1_sample_indices(stage1_dir: str) -> List[int]:
    """
    Scan all subdirectories of stage1_dir for applicable_questions files
    and return a sorted list of sample indices that have Stage 1 results.
    """
    indices = set()
    if not os.path.isdir(stage1_dir):
        return []

    pattern = re.compile(r'^sample_(\d+)_applicable_questions\.json$')

    for entry in os.listdir(stage1_dir):
        sub = os.path.join(stage1_dir, entry)
        if not os.path.isdir(sub):
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

    # Normalize tag keys
    valid_ph = {_normalize_tag(k): v for k, v in valid_ph.items()}
    original_ph = {_normalize_tag(k): v for k, v in original_ph.items()}

    # Find all placeholder tags in the template
    tags_in_template = re.findall(r'<[^>]+>', tmpl_str)

    if not tags_in_template:
        # No-placeholder template — single pair, no pre-instantiation needed
        return [{
            'positive_question': tmpl_str,
            'contrastive_question': None,
            'positive_mapping': {},
            'contrastive_mapping': {},
            'contrastive_strategy': 'no_placeholders',
        }]

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

        pairs.append({
            'positive_question': pos_q,
            'contrastive_question': cont_q,
            'positive_mapping': mapping,
            'contrastive_mapping': cont_mapping,
            'contrastive_strategy': cont_strategy,
        })

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
    Build a prompt for the VLM to VERIFY pre-instantiated QA pairs and
    propose additional contrastive variations.

    The VLM's job:
      1. Answer each pre-instantiated positive + contrastive question
      2. Provide grounded reasoning (OBJ IDs, camera refs, spatial data)
      3. Propose additional creative contrastive questions for more diversity
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

    # Pre-instantiated pairs
    parts.append("--- PRE-INSTANTIATED QA PAIRS (verify and answer these) ---")
    parts.append("")
    for i, pair in enumerate(pre_pairs, 1):
        parts.append(f"  Pair {i}:")
        parts.append(f"    Positive Question: \"{pair['positive_question']}\"")
        if pair['contrastive_question']:
            parts.append(f"    Contrastive Question: \"{pair['contrastive_question']}\"")
            parts.append(f"    Contrastive Strategy: {pair['contrastive_strategy']}")
        else:
            parts.append(f"    Contrastive Question: (none pre-generated — you must propose one)")
        parts.append("")

    # Task instructions
    parts.append("--- YOUR TASK ---")
    parts.append("")
    parts.append("For EACH pre-instantiated pair above:")
    parts.append("")
    parts.append("1. **ANSWER** the positive question with grounded reasoning")
    parts.append("   (reference OBJ IDs, camera Image numbers, spatial data)")
    parts.append("")
    parts.append("2. **ANSWER** the contrastive question (if provided) with grounded reasoning")
    parts.append("")
    parts.append("3. **PROPOSE ADDITIONAL CONTRASTIVE QUESTIONS** (1-2 per pair):")
    parts.append("   For each pair, suggest creative alternative contrastive questions that:")
    parts.append("   - Change different placeholder(s) than the pre-generated contrastive")
    parts.append("   - Target different objects, locations, or conditions in the scene")
    parts.append("   - Produce a different answer from the positive with high confidence")
    parts.append("   - Are non-trivial (require actual scene analysis)")
    parts.append("   Include the answer and reasoning for each proposed contrastive.")
    parts.append("")
    parts.append("4. For ENTITY-type placeholders, use four-layer grounding:")
    parts.append("   OBJ {id} ({visual_description}, {distance_direction}, {camera_ref})")
    parts.append("")

    # Output format
    parts.append("--- OUTPUT FORMAT ---")
    parts.append("")
    parts.append("""Respond with a JSON object:
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
                "answer": "<answer (option letter for mcq, e.g. 'B')>",
                "reasoning": "- <bullet 1: OBJ ID + spatial data>\n- <bullet 2: camera ref>\n- <bullet 3: conclusion>",
                "confidence": "<high|medium|low>",
                "relevant_cameras": [<int>]
            },
            "contrastive": {
                "altered_placeholders": ["<tags changed>"],
                "contrastive_strategy": "<what was changed and why>",
                "tag_mappings": { ... },
                "instantiated_question": "<the pre-instantiated contrastive question>",
                "mcq_options": {"A": "...", "B": "...", "C": "...", "D": "...", "E": "..."},
                "answer": "<different answer (different option letter for mcq)>",
                "reasoning": "- <bullet 1>\n- <bullet 2>\n- <bullet 3>",
                "confidence": "<high|medium|low>",
                "relevant_cameras": [<int>]
            },
            "vlm_proposed_contrastives": [
                {
                    "altered_placeholders": ["<tags changed>"],
                    "contrastive_strategy": "<what was changed and why>",
                    "instantiated_question": "<your proposed contrastive question>",
                    "mcq_options": {"A": "...", "B": "...", "C": "...", "D": "...", "E": "..."},
                    "answer": "<answer>",
                    "reasoning": "- <bullet 1>\n- <bullet 2>\n- <bullet 3>",
                    "confidence": "<high|medium|low>",
                    "relevant_cameras": [<int>]
                }
            ]
        },
        ...
    ]
}

NOTES:
- "mcq_options" is REQUIRED for mcq answer_type; omit for other types.
- For mcq, "answer" must be the correct option LETTER (A/B/C/D/E).
- Construct exactly 5 options: 1 correct + 4 plausible distractors.
- Randomize the correct answer position across pairs.
If contrastive is impossible: "contrastive": null, "contrastive_skip_reason": "<explanation>"
""")
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


def build_qa_generation_prompt(
    driving_command: str,
    object_positions_text: str,
    closest_objects_text: str,
    template: Dict,
    template_num: int,
    total_templates: int,
) -> str:
    """
    Build user prompt for a SINGLE template: tag refinement + contrastive QA pair generation.
    """
    parts = []

    # Driving command
    parts.append("--- Ego-Vehicle Driving Command ---")
    parts.append(f"  Current Command: {driving_command}")
    parts.append("")

    # Spatial data
    parts.append("--- Prior Knowledge: Spatial Relationships ---")
    parts.append("")
    parts.append("** Object Positions (Ego-centric) **")
    parts.append(object_positions_text)
    parts.append("")
    parts.append("** Closest Objects by Direction **")
    parts.append(closest_objects_text)
    parts.append("")
    parts.append("Note: Occlusion relationships require visual analysis of camera images.")
    parts.append("")

    # Single template
    t = template
    parts.append(f"--- Question Template ({template_num}/{total_templates}, Validated from Stage 1) ---")
    parts.append("")
    parts.append(f"  Category: {t['category']}")
    parts.append(f"  Template: \"{t['template']}\"")
    parts.append(f"  Answer Type: {t['answer_type']}")

    ea = t.get('expected_answers')
    parts.append(f"  Expected Answers: {json.dumps(ea) if ea else 'null'}")

    notes = t.get('notes')
    parts.append(f"  Notes: {notes if notes else 'null'}")

    ego_note = t.get('ego_centric_note')
    if ego_note:
        parts.append(f"  Ego-Centric Guidance: {ego_note}")

    parts.append(f"  Original Placeholders (full template bank candidates):")
    parts.append(format_placeholders_block(t.get('original_placeholders', {})))

    parts.append(f"  Valid Placeholders (Stage 1 — confirmed present in this scene):")
    parts.append(format_placeholders_block(t.get('valid_placeholders', {})))
    parts.append("")

    # Task instructions
    parts.append("--- YOUR TASK ---")
    parts.append("")
    parts.append("Generate MULTIPLE CONTRASTIVE QA PAIRS for this template (minimum 2 pairs")
    parts.append("where the scene supports it):")
    parts.append("")
    parts.append("1. **PLACEHOLDER CLASSIFICATION**: Classify each tag as ENTITY or LEXICAL.")
    parts.append("")
    parts.append("2. **FOR EACH PAIR, generate**:")
    parts.append("")
    parts.append("   a. **POSITIVE INSTANCE**:")
    parts.append("      - ENTITY tags → ground to a specific OBJ with four-layer description")
    parts.append("      - LEXICAL tags → select from available candidates")
    parts.append("      - Write the instantiated question")
    parts.append("      - Provide answer + reasoning")
    parts.append("")
    parts.append("   b. **CONTRASTIVE INSTANCE**:")
    parts.append("      - Alter placeholder(s) with minimal perturbation to change the answer")
    parts.append("      - LEXICAL changes: prefer `original_placeholders` \\ `valid_placeholders`;")
    parts.append("        fallback to plausible absent values")
    parts.append("      - ENTITY changes: select different object, swap order, or use absent category")
    parts.append("      - Write the altered question")
    parts.append("      - Provide the different answer + reasoning")
    parts.append("      - If impossible for this combination, set contrastive to null")
    parts.append("")
    parts.append("3. **DIVERSITY REQUIREMENTS**:")
    parts.append("   - Each pair MUST use a DIFFERENT combination of placeholder values")
    parts.append("   - Vary objects (different OBJ IDs), locations, directions, difficulty levels")
    parts.append("   - Spread selections across all 6 camera views")
    parts.append('   - Include BOTH "yes→no" AND "no→yes" pairs for y_or_n templates')
    parts.append("   - For no-placeholder templates: apply the question to different objects")
    parts.append("     in the scene to generate multiple pairs; if only 1 pair is possible,")
    parts.append("     output that pair and explain in max_pairs_note")
    parts.append("")

    # Output format — single template, direct structure
    parts.append("--- OUTPUT FORMAT ---")
    parts.append("")
    parts.append("""Respond with a JSON object:
{
    "template_idx": <int>,
    "category": "<string>",
    "answer_type": "<y_or_n|mcq|distance|open_ended|num_count>",
    "placeholder_classification": {
        "<tag>": "<entity|lexical>"
    },
    "total_pairs": <int>,
    "max_pairs_note": <null or "explanation if fewer than 3 pairs">,
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
                "instantiated_question": "<final question>",
                "mcq_options": {"A": "...", "B": "...", "C": "...", "D": "...", "E": "..."},
                "answer": "<answer (option letter for mcq, e.g. 'B')>",
                "reasoning": "- <bullet 1: OBJ ID + spatial data>\n- <bullet 2: camera ref>\n- <bullet 3: conclusion>",
                "confidence": "<high|medium|low>",
                "relevant_cameras": [<int>]
            },
            "contrastive": {
                "altered_placeholders": ["<tags changed>"],
                "contrastive_strategy": "<what was changed and why>",
                "tag_mappings": {
                    "<tag>": {
                        "type": "<entity|lexical>",
                        "selected_obj_id": <int_or_null>,
                        "grounded_description": "<value used>",
                        "selection_rationale": "<why chosen>",
                        "value_source": "<valid_placeholders|original_placeholders|fallback>"
                    }
                },
                "instantiated_question": "<altered question>",
                "mcq_options": {"A": "...", "B": "...", "C": "...", "D": "...", "E": "..."},
                "answer": "<different answer (different option letter for mcq)>",
                "reasoning": "- <bullet 1>\n- <bullet 2: why answer differs>\n- <bullet 3>",
                "confidence": "<high|medium|low>",
                "relevant_cameras": [<int>]
            }
        },
        {
            "pair_id": 2,
            "positive": { ... },
            "contrastive": { ... }
        },
        ...
    ]
}

NOTES:
- "mcq_options" is REQUIRED for mcq answer_type; omit for other types.
- For mcq, "answer" must be the correct option LETTER (A/B/C/D/E).
- Construct exactly 5 options: 1 correct + 4 plausible distractors.
- Randomize the correct answer position across pairs.
If contrastive is impossible for a specific pair:
    "contrastive": null,
    "contrastive_skip_reason": "<explanation>"
""")
    parts.append("Output ONLY the JSON object, no additional text.")

    return "\n".join(parts)


# =============================================================================
# Worker process initializer and global state
# =============================================================================

_worker_client = None
_worker_loader = None
_worker_analyzer = None
_worker_question_bank = None
_worker_risk_results_dir = None
_worker_traffic_results_dir = None


def _worker_init(
    model_name: str, api_base: str, api_key: str,
    pkl_path: str, question_bank_path: str,
    risk_results_dir: str = "",
    traffic_results_dir: str = "",
):
    """
    Initializer for each worker process.
    Creates the OpenAI client, NuScenesDataLoader, SceneAnalyzer, and loads question bank.
    """
    global _worker_client, _worker_loader, _worker_analyzer, _worker_question_bank
    global _worker_risk_results_dir, _worker_traffic_results_dir

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
) -> Dict:
    """
    Worker function: load Stage 1 results, enrich with question bank data,
    pre-instantiate QA pairs, call VLM for verification + additional contrasts.
    """
    global _worker_client, _worker_loader, _worker_analyzer, _worker_question_bank
    global _worker_risk_results_dir, _worker_traffic_results_dir

    try:
        client = _worker_client
        loader = _worker_loader
        analyzer = _worker_analyzer
        question_bank = _worker_question_bank
        risk_results_dir = _worker_risk_results_dir
        traffic_results_dir = _worker_traffic_results_dir

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

        # 7. Flatten all templates across categories, enrich with question bank
        all_templates = []
        for category, questions in stage1_results.items():
            for q_key, q_data in questions.items():
                enriched = enrich_template_with_question_bank(
                    q_data, category, question_bank
                )
                enriched['q_key'] = q_key
                all_templates.append(enriched)

        total_templates = len(all_templates)

        # 8. Process each template: pre-instantiate pairs, then VLM verifies + proposes more
        os.makedirs(output_dir, exist_ok=True)
        all_qa_results = []
        all_raw_responses = []
        total_pairs_count = 0

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

            # Prepend prior analysis for risk/traffic categories
            tmpl_category = tmpl.get('category', '')
            if tmpl_category == "Dynamic_Agents_and_Risk_Assessment" and risk_response:
                prompt = (
                    "\n\n=== RISK ASSESSMENT ANALYSIS (from prior analysis) ===\n"
                    f"{risk_response}\n"
                    "=== END OF RISK ASSESSMENT ANALYSIS ===\n\n"
                ) + prompt

            if tmpl_category == "Traffic_Signs_and_Signals" and traffic_response:
                prompt = (
                    "\n\n=== TRAFFIC SIGNAL ANALYSIS (from prior analysis) ===\n"
                    f"{traffic_response}\n"
                    "=== END OF TRAFFIC SIGNAL ANALYSIS ===\n\n"
                ) + prompt

            # Build user content: images + prompt
            user_content = list(image_content) + [
                {"type": "text", "text": prompt}
            ]

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]

            # API call
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
                'response': response_text,
            })

            # Parse response — expect a single template result dict
            parsed = parse_json_response(response_text)

            if parsed is None:
                tqdm.write(f"      -> FAILED to parse JSON (template {tmpl_idx + 1})")
                continue

            # Normalize: the response should be a dict with 'pairs'
            result = None
            if isinstance(parsed, dict):
                if 'pairs' in parsed or 'template_idx' in parsed:
                    result = parsed
                elif 'qa_results' in parsed and isinstance(parsed['qa_results'], list):
                    # Model wrapped in qa_results array — take first
                    for r in parsed['qa_results']:
                        if isinstance(r, dict) and ('pairs' in r or 'template_idx' in r):
                            result = r
                            break

            if result is None:
                tqdm.write(f"      -> No valid result structure (template {tmpl_idx + 1})")
                continue

            # Ensure category is attached
            if 'category' not in result:
                result['category'] = tmpl['category']
            if 'template_idx' not in result:
                result['template_idx'] = tmpl.get('template_idx', tmpl_idx + 1)

            # Count pairs (including VLM-proposed contrastives)
            pairs = result.get('pairs', [])
            num_pairs = len(pairs) if isinstance(pairs, list) else 0
            # Count VLM-proposed extras
            num_vlm_proposed = 0
            if isinstance(pairs, list):
                for p in pairs:
                    vlm_extras = p.get('vlm_proposed_contrastives', [])
                    if isinstance(vlm_extras, list):
                        num_vlm_proposed += len(vlm_extras)
            total_pairs_count += num_pairs
            tqdm.write(f"      -> OK: {num_pairs} pairs + {num_vlm_proposed} VLM-proposed contrasts (running total: {total_pairs_count})")

            all_qa_results.append(result)

        # 9. Save output
        output_data = {
            'sample_idx': sample_idx,
            'scene_meta': scene_data['scene_meta'],
            'ego_info': scene_data['ego_info'],
            'total_templates_processed': total_templates,
            'total_templates_with_results': len(all_qa_results),
            'total_pairs_generated': total_pairs_count,
            'timestamp': datetime.now().isoformat(),
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


def _worker_wrapper(args):
    """Unpack tuple arguments and call _worker_process_sample."""
    return _worker_process_sample(*args)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: Tag Refinement + Contrastive QA Pair Generation (vLLM Multiprocessing)"
    )

    # Model and API settings
    parser.add_argument(
        "--model_name", type=str,
        default="Qwen/Qwen3-VL-235B-A22B-Instruct",
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
        default="qa_results",
        help="Output directory for QA results",
    )
    parser.add_argument(
        "--risk_results_dir", type=str,
        default="risk_assessment_results",
        help="Directory containing risk assessment result JSONs "
             "(prepended to prompt for Dynamic_Agents_and_Risk_Assessment templates)",
    )
    parser.add_argument(
        "--traffic_results_dir", type=str,
        default="traffic_analysis_results",
        help="Directory containing traffic analysis result JSONs "
             "(prepended to prompt for Traffic_Signs_and_Signals templates)",
    )

    # Sample range
    parser.add_argument("--start_idx", type=int, default=0, help="Starting sample index")
    parser.add_argument("--end_idx", type=int, default=6018, help="Ending sample index")
    parser.add_argument("--from_stage1", action="store_true",
                        help="Auto-discover sample indices from Stage 1 output files "
                             "instead of using start_idx/end_idx range. "
                             "Processes only samples that have applicable_questions results.")

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
    parser.add_argument("--resize_factor", type=int, default=2,
                        help="Image resize factor (1/n of original size)")
    parser.add_argument("--max_new_tokens", type=int, default=16384,
                        help="Maximum tokens to generate per template")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature for generation (default: 0.6)")
    parser.add_argument("--max_pairs", type=int, default=3,
                        help="Maximum pre-instantiated pairs per template (default: 3)")

    # Multiprocessing
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of parallel worker processes")

    # Logging
    parser.add_argument("--log_dir", type=str, default="qa_gen_logs",
                        help="Directory for log files")

    args = parser.parse_args()

    # Determine sample indices
    if args.from_stage1:
        sample_indices = discover_stage1_sample_indices(args.stage1_dir)
        if not sample_indices:
            print(f"ERROR: No Stage 1 results found in {args.stage1_dir}/")
            sys.exit(1)
    else:
        sample_indices = list(range(args.start_idx, args.end_idx + 1))

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
    log_and_print(f"  Output dir: {args.output_dir}")
    log_and_print(f"  Log file: {log_file}")
    log_and_print(f"  Camera order: FL, F, FR, RL, R, RR (egocentric)")
    log_and_print(f"  Rear cameras: horizontally flipped")
    log_and_print(f"  Risk results dir: {args.risk_results_dir}")
    log_and_print(f"  Traffic results dir: {args.traffic_results_dir}")
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

    total_start = time.time()

    with Pool(
        processes=num_workers,
        initializer=_worker_init,
        initargs=(
            args.model_name, args.api_base, args.api_key,
            args.pkl_path, args.question_bank,
            args.risk_results_dir, args.traffic_results_dir,
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

    log_and_print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
