"""
Offline re-parse + trap audit for Thinking-model answer-generator outputs.

Why this exists
---------------
The Thinking model emits its chain-of-thought inline before the final JSON
(closed by </think>). The original parse_json_response() scanned for JSON
arrays before objects and could latch onto incidental list fragments inside
the thinking text, discarding ~99% of otherwise-valid template responses as
"No valid result structure". The raw responses were still persisted in
sample_{idx}_qa_detailed.json, so everything is recoverable without
re-inference. This script replays the full post-parse chain over those saved
responses:

    parse (patched) -> normalize -> metadata pin -> validate_result_contract

and, in the same pass (the only point where the raw thinking trace is read),
runs contrastive-trap detection:

  (a) templates whose thinking contains trap markers ("same answer",
      "can't create a contrastive", ...) get trap_flag / trap_markers /
      trap_resolution fields attached;
  (b) trap templates resolved WITHOUT a contrastive skip (suspected label
      fabrication) are diverted to a 'quarantined_results' key — kept out of
      'qa_results' so downstream SFT postprocessing never consumes them —
      and listed in a quarantine manifest;
  (c) quarantined templates are sub-classified by comparing final answers:
      pairs where positive == contrastive get trap_pair_class 'honest_same'
      (the model stayed truthful and violated the must-differ constraint —
      the positive is salvageable), pairs where they differ get
      'contrast_achieved' (contrast under trap conditions — the label-
      fabrication suspects). Template-level 'trap_subclass' is honest_same /
      contrast_achieved / mixed. Caveat: equality is exact (normalized)
      string comparison, so open_ended answers that differ only in wording
      are counted as contrast_achieved — weigh that answer_type separately
      downstream.

Outputs (written to --output_dir, originals untouched):
  sample_{idx}_qa_results.json   rebuilt results (+ quarantined_results key)
  quarantine_manifest.json       every quarantined template with markers
  reparse_report.json            aggregate parse / trap / resolution stats
  sample_{idx}_disagreements.json  (in --disagreement_dir) prior disagreements

Usage:
  python -m nuscenes_pipeline.postprocessing.reparse_thinking_qa \
      --input_dir qa_results_thinking \
      --output_dir qa_results_thinking_reparsed \
      --stage1_dir qa_outputs
"""

import argparse
import glob
import json
import os
import re
import time
from datetime import datetime

from nuscenes_pipeline.modules.answer_generator import (
    parse_json_response,
    validate_result_contract,
    load_stage1_results,
)


# ---------------------------------------------------------------------------
# Trap-marker detection
# ---------------------------------------------------------------------------
# Applied case-insensitively to the thinking trace (text before </think>).
# Each marker signals the model noticed the pre-instantiated contrastive does
# not actually flip the answer — the precondition for label fabrication.
TRAP_MARKER_PATTERNS = [
    r"same answer",
    r"same as the positive",
    r"answer (?:would|will|does)(?: not|n't) (?:change|differ|flip)",
    r"(?:can(?:'t|not)|unable to|not possible to|hard to|difficult to)"
    r"[^.\n]{0,30}(?:create|come up with|make|find|construct|produce)"
    r"[^.\n]{0,40}contrastive",
    r"no valid contrastive",
    r"contrastive[^.\n]{0,40}(?:would|will) (?:also )?(?:be|have) the same",
    r"both (?:answers?|questions?|would be|are '?(?:yes|no)'?)",
    r"have to go with",
    r"forced to (?:say|answer|pick|choose)",
    r"the positive answer will be '?no'? for all",
]
_TRAP_RES = [re.compile(p, re.IGNORECASE) for p in TRAP_MARKER_PATTERNS]


def detect_trap_markers(thinking_text: str):
    """Return the list of marker pattern strings that fire on the trace."""
    hits = []
    for pat, rx in zip(TRAP_MARKER_PATTERNS, _TRAP_RES):
        if rx.search(thinking_text):
            hits.append(pat)
    return hits


def _norm_answer(ans):
    """Normalize an answer value for equality comparison across pair roles."""
    if isinstance(ans, str):
        return ans.strip().lower()
    if isinstance(ans, (list, dict)):
        return json.dumps(ans, sort_keys=True)
    return str(ans)


def classify_trap_pairs(result: dict):
    """Annotate each pair of a trap-flagged template with trap_pair_class:
    'honest_same' (positive answer == contrastive answer — truthful constraint
    violation), 'contrast_achieved' (answers differ — fabrication suspect), or
    'skipped' (no contrastive section). Returns (n_honest, n_contrast, n_skipped)."""
    honest = contrast = skipped = 0
    for pair in (result.get('pairs') or []):
        if not isinstance(pair, dict):
            continue
        con = pair.get('contrastive')
        if not isinstance(con, dict):
            pair['trap_pair_class'] = 'skipped'
            skipped += 1
            continue
        pos = pair.get('positive')
        pa = _norm_answer((pos or {}).get('answer'))
        ca = _norm_answer(con.get('answer'))
        if pa == ca:
            pair['trap_pair_class'] = 'honest_same'
            honest += 1
        else:
            pair['trap_pair_class'] = 'contrast_achieved'
            contrast += 1
    return honest, contrast, skipped


def trap_subclass_of(n_honest: int, n_contrast: int) -> str:
    """Template-level subclass from pair-level counts."""
    if n_contrast and not n_honest:
        return 'contrast_achieved'
    if n_honest and not n_contrast:
        return 'honest_same'
    if n_honest and n_contrast:
        return 'mixed'
    return 'no_comparable_pairs'


def has_contrastive_skip(result: dict) -> bool:
    """True if any pair resolved via the escape hatch (contrastive: null)."""
    for pair in (result.get('pairs') or []):
        if not isinstance(pair, dict):
            continue
        if pair.get('contrastive') is None and 'contrastive' in pair:
            return True
        if pair.get('contrastive_skip_reason'):
            return True
    return False


# ---------------------------------------------------------------------------
# Per-template chain (mirrors the worker in answer_generator.py)
# ---------------------------------------------------------------------------

def normalize_parsed(parsed):
    """Replicates the worker's normalization: expect dict with pairs /
    template_idx, or a qa_results-wrapped list of such dicts."""
    if not isinstance(parsed, dict):
        return None
    if 'pairs' in parsed or 'template_idx' in parsed:
        return parsed
    if 'qa_results' in parsed and isinstance(parsed['qa_results'], list):
        for r in parsed['qa_results']:
            if isinstance(r, dict) and ('pairs' in r or 'template_idx' in r):
                return r
    return None


def build_template_meta_map(stage1_dir: str, sample_idx: int):
    """(category, template_text) -> {'template_idx', 'answer_type'} from the
    Stage 2 applicable-questions file (the same source the worker used)."""
    meta = {}
    stage1 = load_stage1_results(stage1_dir, sample_idx)
    for category, questions in stage1.items():
        for q_data in questions.values():
            if not isinstance(q_data, dict):
                continue
            key = (category, q_data.get('template', ''))
            meta[key] = {
                'template_idx': q_data.get('template_idx'),
                'answer_type': q_data.get('answer_type'),
            }
    return meta


def extract_prior_disagreements(result: dict, tmpl_meta: dict, raw: dict):
    """Mirrors worker step 8-extract: collect prior_disagreements entries."""
    out = []
    for pair in (result.get('pairs') or []):
        if not isinstance(pair, dict):
            continue
        pair_id = pair.get('pair_id')
        sections = [('positive', pair.get('positive')),
                    ('contrastive', pair.get('contrastive'))]
        for idx_prop, sec in enumerate(pair.get('vlm_proposed_contrastives') or []):
            sections.append((f'vlm_proposed_contrastive[{idx_prop}]', sec))
        for role, section in sections:
            if not isinstance(section, dict):
                continue
            disagreements = section.get('prior_disagreements')
            if not isinstance(disagreements, list) or not disagreements:
                continue
            for d in disagreements:
                if not isinstance(d, dict):
                    continue
                out.append({
                    'template_idx': result.get('template_idx'),
                    'category': result.get('category'),
                    'template': raw.get('template', ''),
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
    return out


def process_sample(detailed_path: str, stage1_dir: str, stats: dict):
    """Re-run the chain for one sample. Returns (output_data, quarantine_rows,
    disagreements) or None if the detailed file is unreadable."""
    try:
        with open(detailed_path) as f:
            detailed = json.load(f)
    except (json.JSONDecodeError, IOError):
        stats['files_unreadable'] += 1
        return None

    sample_idx = detailed.get('sample_idx')
    meta_map = build_template_meta_map(stage1_dir, sample_idx)

    # ego_info only exists in the original results file — carry it over.
    ego_info = None
    orig_results_path = detailed_path.replace('_qa_detailed.json', '_qa_results.json')
    if os.path.isfile(orig_results_path):
        try:
            with open(orig_results_path) as f:
                ego_info = json.load(f).get('ego_info')
        except (json.JSONDecodeError, IOError):
            pass

    accepted, quarantined, quarantine_rows, disagreements = [], [], [], []
    total_pairs = 0

    for raw in detailed.get('raw_responses', []):
        stats['templates_total'] += 1
        response = raw.get('response') or ''
        thinking, sep, _ = response.rpartition('</think>')
        if not sep:
            stats['no_think_close'] += 1  # truncated / malformed trace

        parsed = parse_json_response(response)
        if parsed is None:
            stats['parse_fail'] += 1
            continue
        result = normalize_parsed(parsed)
        if result is None:
            stats['no_valid_structure'] += 1
            continue

        # Metadata pin — never trust VLM-supplied values.
        key = (raw.get('category', ''), raw.get('template', ''))
        tmpl_meta = meta_map.get(key)
        if tmpl_meta is None or not tmpl_meta.get('answer_type'):
            stats['meta_missing'] += 1
            continue
        result['category'] = raw.get('category', '')
        result['template_idx'] = tmpl_meta['template_idx']
        result['answer_type'] = tmpl_meta['answer_type']

        ok, failures = validate_result_contract(result, result['answer_type'])
        if not ok:
            stats['contract_rejected'] += 1
            continue

        stats['templates_valid'] += 1

        # Trap audit — the one pass where the raw trace is in hand.
        markers = detect_trap_markers(thinking) if sep else []
        skip = has_contrastive_skip(result)
        result['trap_flag'] = bool(markers)
        result['trap_markers'] = markers
        if markers:
            stats['trap_templates'] += 1
            result['trap_resolution'] = 'skip' if skip else 'flip_suspect'
            stats['trap_resolved_skip' if skip else 'trap_resolved_flip'] += 1
        else:
            result['trap_resolution'] = None

        disagreements.extend(extract_prior_disagreements(result, tmpl_meta, raw))

        pairs = result.get('pairs') or []
        if result['trap_resolution'] == 'flip_suspect':
            n_honest, n_contrast, n_skipped = classify_trap_pairs(result)
            subclass = trap_subclass_of(n_honest, n_contrast)
            result['trap_subclass'] = subclass
            stats[f'quarantine_subclass_{subclass}'] = \
                stats.get(f'quarantine_subclass_{subclass}', 0) + 1
            stats['quarantine_pairs_honest_same'] += n_honest
            stats['quarantine_pairs_contrast_achieved'] += n_contrast
            quarantined.append(result)
            quarantine_rows.append({
                'sample_idx': sample_idx,
                'template_idx': result['template_idx'],
                'category': result['category'],
                'answer_type': result['answer_type'],
                'template': raw.get('template', ''),
                'template_num': raw.get('template_num'),
                'trap_markers': markers,
                'trap_subclass': subclass,
                'pairs_honest_same': n_honest,
                'pairs_contrast_achieved': n_contrast,
                'num_pairs': len(pairs) if isinstance(pairs, list) else 0,
            })
        else:
            accepted.append(result)
            total_pairs += len(pairs) if isinstance(pairs, list) else 0

    output_data = {
        'sample_idx': sample_idx,
        'scene_meta': detailed.get('scene_meta'),
        'ego_info': ego_info,
        'total_templates_processed': detailed.get('total_templates'),
        'total_templates_with_results': len(accepted),
        'total_pairs_generated': total_pairs,
        'timestamp': datetime.now().isoformat(),
        'reparsed_from': os.path.basename(detailed_path),
        'qa_results': accepted,
        'quarantined_results': quarantined,
    }
    return output_data, quarantine_rows, disagreements


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--input_dir', default='qa_results_thinking',
                        help='Dir with sample_*_qa_detailed.json from the Thinking run')
    parser.add_argument('--output_dir', default='qa_results_thinking_reparsed',
                        help='Dir for rebuilt qa_results (originals untouched)')
    parser.add_argument('--stage1_dir', default='qa_outputs',
                        help='Stage 2 output dir (metadata source for pinning)')
    parser.add_argument('--disagreement_dir', default='prior_disagreements_thinking_reparsed',
                        help='Dir for re-extracted prior-disagreement files')
    parser.add_argument('--min_age_sec', type=int, default=120,
                        help='Skip detailed files modified more recently than this '
                             '(avoids reading files the live run is still writing)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    stats = {k: 0 for k in [
        'files_scanned', 'files_skipped_fresh', 'files_unreadable',
        'templates_total', 'parse_fail', 'no_valid_structure', 'meta_missing',
        'contract_rejected', 'templates_valid', 'no_think_close',
        'trap_templates', 'trap_resolved_skip', 'trap_resolved_flip',
        'quarantine_pairs_honest_same', 'quarantine_pairs_contrast_achieved',
    ]}
    all_quarantine, samples_written = [], 0
    now = time.time()

    files = sorted(glob.glob(os.path.join(args.input_dir, 'sample_*_qa_detailed.json')))
    for path in files:
        if now - os.path.getmtime(path) < args.min_age_sec:
            stats['files_skipped_fresh'] += 1
            continue
        stats['files_scanned'] += 1
        out = process_sample(path, args.stage1_dir, stats)
        if out is None:
            continue
        output_data, quarantine_rows, disagreements = out
        sample_idx = output_data['sample_idx']

        out_path = os.path.join(args.output_dir, f'sample_{sample_idx}_qa_results.json')
        with open(out_path, 'w') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        samples_written += 1
        all_quarantine.extend(quarantine_rows)

        if disagreements:
            os.makedirs(args.disagreement_dir, exist_ok=True)
            dis_path = os.path.join(args.disagreement_dir,
                                    f'sample_{sample_idx}_disagreements.json')
            with open(dis_path, 'w') as f:
                json.dump({
                    'sample_idx': sample_idx,
                    'timestamp': datetime.now().isoformat(),
                    'total_disagreements': len(disagreements),
                    'disagreements': disagreements,
                }, f, indent=2, ensure_ascii=False)

    manifest_path = os.path.join(args.output_dir, 'quarantine_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_quarantined': len(all_quarantine),
            'trap_marker_patterns': TRAP_MARKER_PATTERNS,
            'templates': all_quarantine,
        }, f, indent=2, ensure_ascii=False)

    t = stats
    parsed_ok = t['templates_valid']
    parse_attempted = t['templates_total']
    trap_rate = 100 * t['trap_templates'] / max(parsed_ok, 1)
    report = {
        'timestamp': datetime.now().isoformat(),
        'input_dir': args.input_dir,
        'samples_written': samples_written,
        'stats': stats,
        'rates': {
            'parse_success_pct': round(100 * parsed_ok / max(parse_attempted, 1), 2),
            'trap_rate_pct_of_valid': round(trap_rate, 2),
            'trap_skip_vs_flip': f"{t['trap_resolved_skip']}:{t['trap_resolved_flip']}",
            'quarantine_subclass_templates': {
                k.replace('quarantine_subclass_', ''): v
                for k, v in sorted(stats.items())
                if k.startswith('quarantine_subclass_')
            },
            'quarantine_pairs_honest_same_pct': round(
                100 * t['quarantine_pairs_honest_same']
                / max(t['quarantine_pairs_honest_same']
                      + t['quarantine_pairs_contrast_achieved'], 1), 2),
        },
    }
    report_path = os.path.join(args.output_dir, 'reparse_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2))
    print(f"\nRebuilt results:      {args.output_dir}/sample_*_qa_results.json")
    print(f"Quarantine manifest:  {manifest_path}")
    print(f"Report:               {report_path}")


if __name__ == '__main__':
    main()
