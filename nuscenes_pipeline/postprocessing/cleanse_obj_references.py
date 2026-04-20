"""
Cleanse OBJ-related references from SFT dataset JSON files.

Implements three rounds of cleansing to remove leaked 3D object list artifacts
so the trained VLM relies on visual perception, not memorized object IDs.

Round 1 — Leaked OBJ refs (direct OBJ N patterns):
  Removes patterns like "OBJ 36", "(OBJ 36)", "OBJ 36 (description)" from
  both question and answer fields.

Round 2 — Meta-references (sentence-level removal):
  Removes entire sentences that reference the object list system, e.g. sentences
  containing: "spatial data", "OBJ ID" / "OBJ IDs", "object list" / "object lists",
  "does not list", "pre-computed", "prior knowledge".

Round 3 — Remaining edge cases (phrase-level replacement):
  - "no object with a 3D position or OBJ ID(s)" -> "no object"
  - "(the) OBJ ID(s) (and their descriptions)" -> "" (removed)
  - "spatial database" / "spatial dataset" / "spatial data" -> "scene"
  - "object list" / "object lists" -> "scene"

Usage:
    # Cleanse the SFT dataset files
    python -m nuscenes_pipeline.postprocessing.cleanse_obj_references \\
        --input sft_train_no_objlist.json --output sft_train_no_objlist.json

    # Dry run (report counts without modifying)
    python -m nuscenes_pipeline.postprocessing.cleanse_obj_references \\
        --input sft_train_no_objlist.json --dry_run
"""

import argparse
import json
import re
import os


# ---------------------------------------------------------------------------
# Round 1: Leaked OBJ ref patterns
# ---------------------------------------------------------------------------

# OBJ N (description) — e.g. "OBJ 36 (pedestrian, 19.6m ahead, visible in Image 2)"
_RE_OBJ_WITH_DESC = re.compile(
    r'\bOBJ\s+\d+\s*\([^)]*\)',
    re.IGNORECASE,
)

# (OBJ N) or (OBJ N, OBJ M, ...) — parenthesized OBJ groups
_RE_OBJ_PAREN_GROUP = re.compile(
    r'\(\s*OBJ\s+\d+(?:\s*,\s*(?:OBJ\s+)?\d+)*\s*\)',
    re.IGNORECASE,
)

# Standalone OBJ N — e.g. "OBJ 36", "OBJs 2, 4, 5"
_RE_OBJ_STANDALONE = re.compile(
    r'\bOBJs?\s+\d+(?:\s*,\s*(?:OBJ\s+)?\d+)*',
    re.IGNORECASE,
)


def round1_clean_obj_refs(text: str) -> tuple[str, int]:
    """Remove direct OBJ N patterns. Returns (cleaned_text, num_replacements)."""
    count = 0

    # Order matters: most specific first
    text, n = _RE_OBJ_WITH_DESC.subn('', text)
    count += n

    text, n = _RE_OBJ_PAREN_GROUP.subn('', text)
    count += n

    text, n = _RE_OBJ_STANDALONE.subn('', text)
    count += n

    # Clean up artifacts: double spaces, empty parens, orphaned commas
    text = re.sub(r'\(\s*\)', '', text)
    text = re.sub(r',\s*,', ',', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = text.strip()

    return text, count


# ---------------------------------------------------------------------------
# Round 2: Meta-reference sentence removal
# ---------------------------------------------------------------------------

_META_KEYWORDS = [
    'spatial data',
    'OBJ IDs',       # plural first (longest match priority)
    'OBJ ids',
    'OBJ ID',
    'OBJ id',
    'object lists',  # plural
    'object list',
    'does not list',
    'pre-computed',
    'precomputed',
    'prior knowledge',
    '3D detection',
    '3D object detection',
    'provided object data',
    'object detection data',
]

# Build a single regex that matches any sentence containing a meta-keyword
_META_PATTERN = re.compile(
    r'[^.!?\n]*\b(?:' +
    '|'.join(re.escape(kw) for kw in _META_KEYWORDS) +
    r')\b[^.!?\n]*[.!?]?\s*',
    re.IGNORECASE,
)


def round2_remove_meta_sentences(text: str) -> tuple[str, int]:
    """Remove sentences containing meta-references. Returns (cleaned_text, num_sentences_removed)."""
    matches = _META_PATTERN.findall(text)
    count = len(matches)
    if count > 0:
        text = _META_PATTERN.sub('', text)
        text = re.sub(r'\s{2,}', ' ', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        text = text.strip()
    return text, count


# ---------------------------------------------------------------------------
# Round 3: Edge-case phrase replacements
# ---------------------------------------------------------------------------

_PHRASE_REPLACEMENTS = [
    # Full phrase replacements (order matters: most specific first)
    (re.compile(r'no objects? with (?:a 3D position or )?OBJ IDs?', re.IGNORECASE), 'no object'),
    (re.compile(r'no objects? with an? OBJ IDs?', re.IGNORECASE), 'no object'),
    (re.compile(r'(?:the\s+)?OBJ IDs?(?:\s+and\s+(?:their\s+)?descriptions?)?', re.IGNORECASE), ''),
    (re.compile(r'spatial database', re.IGNORECASE), 'scene'),
    (re.compile(r'spatial dataset', re.IGNORECASE), 'scene'),
    (re.compile(r'spatial data', re.IGNORECASE), 'scene'),
    (re.compile(r'object lists?', re.IGNORECASE), 'scene'),
]


def round3_replace_edge_cases(text: str) -> tuple[str, int]:
    """Apply phrase-level replacements. Returns (cleaned_text, num_replacements)."""
    count = 0
    for pattern, replacement in _PHRASE_REPLACEMENTS:
        text, n = pattern.subn(replacement, text)
        count += n
    return text, count


# ---------------------------------------------------------------------------
# Combined cleansing
# ---------------------------------------------------------------------------

def cleanse_text(text: str) -> tuple[str, dict]:
    """Apply all 3 rounds of cleansing. Returns (cleaned_text, stats_dict)."""
    if not text or not isinstance(text, str):
        return text, {'r1': 0, 'r2': 0, 'r3': 0}

    text, r1 = round1_clean_obj_refs(text)
    text, r2 = round2_remove_meta_sentences(text)
    text, r3 = round3_replace_edge_cases(text)

    return text, {'r1': r1, 'r2': r2, 'r3': r3}


def cleanse_conversation(conv: dict) -> dict:
    """Cleanse a single SFT conversation entry (has 'conversations' list)."""
    stats = {'r1': 0, 'r2': 0, 'r3': 0}

    if 'conversations' not in conv:
        return stats

    for turn in conv['conversations']:
        if turn.get('from') in ('human', 'gpt'):
            text = turn.get('value', '')
            cleaned, s = cleanse_text(text)
            if cleaned != text:
                turn['value'] = cleaned
                for k in stats:
                    stats[k] += s[k]

    return stats


def cleanse_dataset(data: list) -> dict:
    """Cleanse an entire SFT dataset (list of conversation entries)."""
    totals = {'r1': 0, 'r2': 0, 'r3': 0, 'samples_modified': 0}

    for entry in data:
        stats = cleanse_conversation(entry)
        if any(v > 0 for v in stats.values()):
            totals['samples_modified'] += 1
            for k in ('r1', 'r2', 'r3'):
                totals[k] += stats[k]

    return totals


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cleanse OBJ-related references from SFT dataset JSON files",
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Input SFT JSON file path")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: overwrite input)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Report counts without modifying files")
    args = parser.parse_args()

    output_path = args.output or args.input

    print(f"Loading {args.input}...")
    with open(args.input, 'r') as f:
        data = json.load(f)

    print(f"  {len(data)} samples loaded")
    print(f"Cleansing (Rounds 1-3)...")

    totals = cleanse_dataset(data)

    print(f"\nResults:")
    print(f"  Round 1 (OBJ refs removed):         {totals['r1']}")
    print(f"  Round 2 (meta sentences removed):    {totals['r2']}")
    print(f"  Round 3 (edge case replacements):    {totals['r3']}")
    print(f"  Total samples modified:              {totals['samples_modified']}/{len(data)}")

    if args.dry_run:
        print("\n[DRY RUN] No files modified.")
    else:
        print(f"\nSaving to {output_path}...")
        with open(output_path, 'w') as f:
            json.dump(data, f, ensure_ascii=False)
        print("Done.")


if __name__ == "__main__":
    main()
