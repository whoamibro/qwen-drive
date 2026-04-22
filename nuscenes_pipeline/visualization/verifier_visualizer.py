"""
Verifier Visualizer — inspect Stage 1 seed-data outputs.

Shows, per sample, all three Stage 1 autolabel analyses side-by-side:
  - Stage 1A: risk_assessment_results/*_single_frame.json
  - Stage 1B: traffic_signal_analysis_results/*_traffic_signal.json
  - Stage 1C: traffic_sign_results/*_sign.json

Layout:
  Top: 6-view panoramic with OBJ bboxes (extracted from the risk_assessment prompt)
  + BEV overlay with ego/object motion
  Bottom: 3 scrollable columns showing the parsed analysis response blocks.

Usage:
  python -m nuscenes_pipeline.visualization.verifier_visualizer \\
      --risk_dir risk_assessment_results \\
      --signal_dir traffic_signal_analysis_results \\
      --sign_dir traffic_sign_results \\
      --pkl_path ./data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl \\
      --port 6061

  # Or use the shell script:
  bash nuscenes_pipeline/scripts/run_verifier_visualizer.sh 6061
"""

import os
import re
import sys
import json
import argparse

from flask import Flask, render_template_string, request, jsonify

from nuscenes_pipeline.core.nuscenes_data_loader import NuScenesDataLoader
from nuscenes_pipeline.visualization._shared import (
    parse_bboxes_seed_format, generate_bev, build_panoramic,
    detect_bbox_coord_width,
)


# ---------------------------------------------------------------------------
# Directory / file patterns
# ---------------------------------------------------------------------------

RISK_SUFFIX = "single_frame"
SIGNAL_SUFFIX = "traffic_signal"
SIGN_SUFFIX = "sign"


def _find_file(dir_path: str, sample_idx: int, suffix: str) -> str | None:
    """Find {idx:04d}_*_{suffix}.json under dir_path. Returns absolute path or None."""
    if not os.path.isdir(dir_path):
        return None
    prefix = f"{sample_idx:04d}_"
    for fname in os.listdir(dir_path):
        if fname.startswith(prefix) and fname.endswith(f"_{suffix}.json"):
            return os.path.join(dir_path, fname)
    return None


def load_analysis(dir_path: str, sample_idx: int, suffix: str) -> dict | None:
    """Load a single seed-data JSON by sample index and suffix."""
    path = _find_file(dir_path, sample_idx, suffix)
    if not path:
        return None
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Response parsers: turn free-form analysis text into structured blocks
# ---------------------------------------------------------------------------

def parse_risk_response(text: str) -> list[tuple[str, str]]:
    """Split risk_assessment response into (header, body) pairs.

    Headers to detect: '1. Immediate Risks', '2. Potential Risks',
    '3. Recommended Actions', '4. Overall Risk Level'.
    Returns list of (header, body). If no headers found, returns [("Response", text)].
    """
    if not text:
        return []
    pattern = re.compile(r'(?m)^\s*(\d+\.\s+[A-Za-z][^\n]*)')
    matches = list(pattern.finditer(text))
    if not matches:
        return [("Response", text.strip())]
    blocks = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        header = m.group(1).strip()
        body = text[start:end].strip()
        blocks.append((header, body))
    return blocks


def parse_bracketed_response(text: str) -> list[tuple[str, str]]:
    """Split '[BLOCK-TAG] ...' style responses (signal + sign analyses)
    into (tag, body) pairs.

    Also strips the === HEADER === and === END === delimiters.
    Returns list of (tag, body) or [("Response", text)] if no blocks found.
    """
    if not text:
        return []
    # Strip delimiters
    cleaned = re.sub(r'={3,}\s*[A-Z ]+={3,}', '', text)
    cleaned = re.sub(r'={3,}\s*END\s*={3,}', '', cleaned, flags=re.IGNORECASE)

    pattern = re.compile(r'\[([A-Z][A-Z\-]*)\]', re.MULTILINE)
    matches = list(pattern.finditer(cleaned))
    if not matches:
        return [("Response", cleaned.strip())]
    blocks = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(cleaned)
        tag = m.group(1)
        body = cleaned[start:end].strip()
        blocks.append((tag, body))
    return blocks


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)
_loader = None
_config = {
    "risk_dir": "risk_assessment_results",
    "signal_dir": "traffic_signal_analysis_results",
    "sign_dir": "traffic_sign_results",
    "pkl_path": "nuscenes2d_ego_temporal_infos_val.pkl",
    "resize_factor": 2,  # seed data bboxes are in width/2 x height/2 space
    "bev_range": 50.0,
}


def get_loader():
    global _loader
    if _loader is None:
        _loader = NuScenesDataLoader(pkl_path=_config["pkl_path"])
    return _loader


HTML_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Verifier Visualizer</title>
<style>
  body { margin:0; background:#0f0f1a; color:#e0e0e0; font-family: "Segoe UI", sans-serif; }
  header { padding: 8px 16px; background:#1a1a2e; display:flex; gap:16px; align-items:center; border-bottom:1px solid #2a3f5f; }
  header h1 { font-size:16px; margin:0; color:#00ff88; }
  .nav { display:flex; gap:8px; align-items:center; }
  .nav input { width:80px; padding:4px; background:#0f0f1a; color:#e0e0e0; border:1px solid #3a5a8f; border-radius:3px; }
  .nav button { padding:4px 12px; background:#2a3f5f; color:#e0e0e0; border:none; cursor:pointer; border-radius:3px; }
  .nav button:hover { background:#3a5a8f; }
  .meta { font-size:12px; color:#aaa; }
  main { padding: 8px; }
  .visual-row { display:flex; gap:8px; margin-bottom:8px; }
  .pano { flex:3; display:grid; grid-template-columns:repeat(3,1fr); gap:4px; }
  .pano img { width:100%; border:1px solid #2a3f5f; display:block; }
  .pano .cam-label { font-size:10px; color:#6a8abf; text-align:center; margin-top:2px; }
  .bev { flex:1; }
  .bev img { width:100%; border:1px solid #2a3f5f; }
  .analysis-row { display:flex; gap:8px; }
  .panel { flex:1; background:#16213e; border:1px solid #2a3f5f; border-radius:4px; padding:8px; max-height:600px; overflow-y:auto; }
  .panel h2 { font-size:14px; margin:0 0 8px 0; padding-bottom:4px; border-bottom:1px solid #2a3f5f; }
  .panel .risk h2 { color:#ff6b6b; }
  .panel .signal h2 { color:#ffa502; }
  .panel .sign h2 { color:#00ff88; }
  .block { margin-bottom:8px; }
  .block-header { font-size:11px; font-weight:bold; color:#6a8abf; text-transform:uppercase; margin-bottom:2px; }
  .block-body { font-size:12px; white-space:pre-wrap; color:#ddd; line-height:1.4; }
  .missing { color:#888; font-style:italic; padding:16px; text-align:center; }
  kbd { background:#2a3f5f; padding:1px 6px; border-radius:3px; font-family:monospace; font-size:11px; }
</style>
</head>
<body>
<header>
  <h1>Seed-Data Verifier</h1>
  <div class="nav">
    <button id="prev">◀ Prev</button>
    <input type="number" id="idx" value="0" min="0">
    <button id="go">Go</button>
    <button id="next">Next ▶</button>
    <button id="rand">Random</button>
  </div>
  <div class="meta" id="meta"></div>
  <div class="meta" style="margin-left:auto;">Keys: <kbd>←</kbd>/<kbd>→</kbd> navigate, <kbd>r</kbd> random</div>
</header>
<main>
  <div class="visual-row">
    <div class="pano" id="pano"></div>
    <div class="bev"><img id="bev"></div>
  </div>
  <div class="analysis-row">
    <div class="panel risk">
      <h2>🟥 Risk Assessment (Stage 1A)</h2>
      <div id="risk"></div>
    </div>
    <div class="panel signal">
      <h2>🟧 Traffic Signal (Stage 1B)</h2>
      <div id="signal"></div>
    </div>
    <div class="panel sign">
      <h2>🟩 Traffic Sign (Stage 1C)</h2>
      <div id="sign"></div>
    </div>
  </div>
</main>
<script>
const CAM_LABELS = ['Front-Left', 'Front', 'Front-Right', 'Rear-Left', 'Rear', 'Rear-Right'];
const STATE = { idx: 0, total: 0 };

async function fetchSample(idx) {
  const r = await fetch(`/api/sample?idx=${idx}`);
  return r.json();
}

async function fetchInfo() {
  const r = await fetch('/api/info');
  return r.json();
}

function renderBlocks(target, blocks) {
  const el = document.getElementById(target);
  if (!blocks || blocks.length === 0) {
    el.innerHTML = '<div class="missing">(no analysis found)</div>';
    return;
  }
  el.innerHTML = blocks.map(([h, b]) =>
    `<div class="block"><div class="block-header">${h}</div><div class="block-body">${b}</div></div>`
  ).join('');
}

async function load(idx) {
  const data = await fetchSample(idx);
  STATE.idx = data.idx;
  document.getElementById('idx').value = data.idx;
  document.getElementById('meta').textContent =
    `Sample ${data.idx}/${STATE.total - 1} | Scene: ${data.scene || '?'} | ${data.location || '?'} | Command: ${data.command || '?'}`;
  // Panoramic
  const pano = document.getElementById('pano');
  pano.innerHTML = data.images.map((src, i) =>
    `<div><img src="${src}"><div class="cam-label">${CAM_LABELS[i]} (Image ${i+1})</div></div>`
  ).join('');
  document.getElementById('bev').src = data.bev;
  renderBlocks('risk', data.risk);
  renderBlocks('signal', data.signal);
  renderBlocks('sign', data.sign);
}

async function init() {
  const info = await fetchInfo();
  STATE.total = info.total;
  document.getElementById('idx').max = info.total - 1;
  document.getElementById('prev').onclick = () => load(Math.max(0, STATE.idx - 1));
  document.getElementById('next').onclick = () => load(Math.min(STATE.total - 1, STATE.idx + 1));
  document.getElementById('go').onclick = () => load(parseInt(document.getElementById('idx').value) || 0);
  document.getElementById('rand').onclick = () => load(Math.floor(Math.random() * STATE.total));
  document.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT') return;
    if (e.key === 'ArrowLeft') document.getElementById('prev').click();
    else if (e.key === 'ArrowRight') document.getElementById('next').click();
    else if (e.key === 'r') document.getElementById('rand').click();
  });
  load(0);
}
init();
</script>
</body>
</html>"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/info')
def api_info():
    loader = get_loader()
    return jsonify({"total": len(loader.infos)})


@app.route('/api/sample')
def api_sample():
    try:
        idx = int(request.args.get('idx', 0))
    except ValueError:
        idx = 0
    loader = get_loader()
    idx = max(0, min(len(loader.infos) - 1, idx))
    sample = loader.get_sample(idx)

    risk = load_analysis(_config["risk_dir"], idx, RISK_SUFFIX)
    signal = load_analysis(_config["signal_dir"], idx, SIGNAL_SUFFIX)
    sign = load_analysis(_config["sign_dir"], idx, SIGN_SUFFIX)

    # Metadata (prefer values from the risk analysis which contains driving command)
    scene = None
    location = None
    command = None
    for d in (risk, signal, sign):
        if not d:
            continue
        scene = scene or d.get('description')
        location = location or d.get('location')
        # Extract driving command from prompt if available
        if not command and 'prompt' in d:
            m = re.search(r'Driving Command:\s*([^\n]+)', d['prompt'])
            if m:
                command = m.group(1).strip()

    # Extract OBJ bboxes from risk_assessment prompt (same bbox set for all three)
    obj_bboxes = []
    if risk and 'prompt' in risk:
        obj_bboxes = parse_bboxes_seed_format(risk['prompt'])

    # Group bboxes by image number for panoramic overlay
    bbox_by_img = {i: [] for i in range(1, 7)}
    for img_num, x1, y1, x2, y2, label in obj_bboxes:
        if 1 <= img_num <= 6:
            bbox_by_img[img_num].append((x1, y1, x2, y2, label))
    bbox_overlays = {img_num: [(bboxes, '#00ffff')] for img_num, bboxes in bbox_by_img.items() if bboxes}

    # Detect the coord space the bboxes live in (e.g., 400 if seed data was
    # generated with resize_factor=4 — which is the module default).
    bbox_coord_width = detect_bbox_coord_width(obj_bboxes)

    # Build panoramic + BEV
    panoramic = build_panoramic(sample, loader, bbox_overlays=bbox_overlays,
                                 resize_factor=_config["resize_factor"],
                                 bbox_coord_width=bbox_coord_width)
    bev = generate_bev(sample, loader, bev_range=_config["bev_range"])

    # Parse analysis responses into blocks
    risk_blocks = parse_risk_response(risk.get('response', '') if risk else '')
    signal_blocks = parse_bracketed_response(signal.get('response', '') if signal else '')
    sign_blocks = parse_bracketed_response(sign.get('response', '') if sign else '')

    return jsonify({
        'idx': idx,
        'scene': scene,
        'location': location,
        'command': command,
        'images': panoramic,
        'bev': bev,
        'risk': risk_blocks,
        'signal': signal_blocks,
        'sign': sign_blocks,
    })


def main():
    parser = argparse.ArgumentParser(description="Verifier visualizer for Stage 1 seed data")
    parser.add_argument('--risk_dir', type=str, default='risk_assessment_results')
    parser.add_argument('--signal_dir', type=str, default='traffic_signal_analysis_results')
    parser.add_argument('--sign_dir', type=str, default='traffic_sign_results')
    parser.add_argument('--pkl_path', type=str,
                        default=os.environ.get("NUSCENES_PKL_PATH", "nuscenes2d_ego_temporal_infos_val.pkl"))
    parser.add_argument('--resize_factor', type=int, default=2,
                        help='Image resize factor (must match the seed-data resize_factor, default 2)')
    parser.add_argument('--bev_range', type=float, default=50.0)
    parser.add_argument('--port', type=int, default=6061)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    args = parser.parse_args()

    _config["risk_dir"] = args.risk_dir
    _config["signal_dir"] = args.signal_dir
    _config["sign_dir"] = args.sign_dir
    _config["pkl_path"] = args.pkl_path
    _config["resize_factor"] = args.resize_factor
    _config["bev_range"] = args.bev_range

    print(f"Risk dir:    {args.risk_dir}")
    print(f"Signal dir:  {args.signal_dir}")
    print(f"Sign dir:    {args.sign_dir}")
    print(f"PKL path:    {args.pkl_path}")
    print(f"Starting server on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
