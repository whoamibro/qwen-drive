"""
Driving Command Labeler — browser tool to hand-label the GT driving command.

Shows, per sample, ONLY the 3 forward camera views
(CAM_FRONT_LEFT / CAM_FRONT / CAM_FRONT_RIGHT) and lets you assign one of the
7 driving-command labels:

  0 Turn left | 1 Turn right | 2 Go straight | 3 Follow lane
  4 Change lane to left | 5 Change lane to right | 6 U-Turn

Labeling flow:
  - Click one of the 7 command buttons, press keys 0-6, or type the command
    index into the input box and press Enter.
  - Each label is saved immediately (autosave) and the view auto-advances to
    the next sample in the scene.
  - Re-labeling: pressing a different command on an already-labeled sample
    overwrites it and STAYS on the sample (status shows "updated N->M") so the
    correction is visible; pressing the same command just advances.
  - Backspace/Delete (or the Clear button) removes the current sample's label.

Output (one JSON per scene, written into --output_dir):
  {
    "scene_id": "<scene_token>",
    "scene_index": "0001",
    "sample_labels": {
      "<sample_token>": {"<global_sample_idx>": <command_label_index>},
      ...
    }
  }

Existing label files in --output_dir are loaded on startup, so labeling can
be resumed across server restarts.

Usage:
  python -m nuscenes_pipeline.visualization.driving_command_labeler \\
      --pkl_path ./data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl \\
      --output_dir driving_command_labels \\
      --port 6062

  # Or use the shell script:
  bash nuscenes_pipeline/scripts/run_command_labeler.sh 6062
"""

import os
import json
import argparse
import threading

from flask import Flask, render_template_string, request, jsonify
from PIL import Image

from nuscenes_pipeline.core.nuscenes_data_loader import NuScenesDataLoader
from nuscenes_pipeline.visualization._shared import image_to_base64, generate_bev


FRONT_CAMERAS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT']

COMMAND_LABELS = {
    0: "Turn left",
    1: "Turn right",
    2: "Go straight",
    3: "Follow lane",
    4: "Change lane to left",
    5: "Change lane to right",
    6: "U-Turn",
}


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)
_loader = None
_config = {
    "pkl_path": "nuscenes2d_ego_temporal_infos_val.pkl",
    "output_dir": "driving_command_labels",
    "resize_factor": 2,
    "bev_range": 50.0,
}

# Scene index built from the pkl at startup:
#   _scenes[i] = {scene_token, scene_name, start, count}
_scenes = []
# _labels[scene_idx] = {global_sample_idx(int): label(int)}
_labels = {}
_save_lock = threading.Lock()


def get_loader():
    global _loader
    if _loader is None:
        _loader = NuScenesDataLoader(pkl_path=_config["pkl_path"])
    return _loader


def build_scene_index():
    """Group contiguous pkl samples by scene_token (pkl is scene-ordered)."""
    loader = get_loader()
    _scenes.clear()
    for i, info in enumerate(loader.infos):
        token = info.get('scene_token')
        if _scenes and _scenes[-1]['scene_token'] == token:
            _scenes[-1]['count'] += 1
        else:
            _scenes.append({
                'scene_token': token,
                'scene_name': info.get('scene_name', ''),
                'start': i,
                'count': 1,
            })


def scene_of_sample(sample_idx: int) -> int:
    """Binary search: global sample idx -> scene idx."""
    lo, hi = 0, len(_scenes) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _scenes[mid]['start'] <= sample_idx:
            lo = mid
        else:
            hi = mid - 1
    return lo


def label_file_path(scene_idx: int) -> str:
    scene = _scenes[scene_idx]
    name = scene['scene_name'] or scene['scene_token'][:8]
    return os.path.join(_config["output_dir"], f"scene_{scene_idx:04d}_{name}.json")


def save_scene_labels(scene_idx: int):
    """Write one scene's labels as the requested JSON format (autosave)."""
    loader = get_loader()
    scene = _scenes[scene_idx]
    labels = _labels.get(scene_idx, {})
    sample_labels = {}
    for gidx in sorted(labels):
        sample_token = loader.infos[gidx].get('token', str(gidx))
        sample_labels[sample_token] = {str(gidx): labels[gidx]}
    out = {
        'scene_id': scene['scene_token'],
        'scene_index': f"{scene_idx:04d}",
        'sample_labels': sample_labels,
    }
    os.makedirs(_config["output_dir"], exist_ok=True)
    path = label_file_path(scene_idx)
    with _save_lock:
        tmp = path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(out, f, indent=2)
        os.replace(tmp, path)


def load_existing_labels():
    """Resume: read any scene_*.json already present in output_dir."""
    out_dir = _config["output_dir"]
    if not os.path.isdir(out_dir):
        return 0
    token_to_scene = {s['scene_token']: i for i, s in enumerate(_scenes)}
    n = 0
    for fname in sorted(os.listdir(out_dir)):
        if not (fname.startswith('scene_') and fname.endswith('.json')):
            continue
        try:
            with open(os.path.join(out_dir, fname)) as f:
                data = json.load(f)
        except Exception:
            continue
        scene_idx = token_to_scene.get(data.get('scene_id'))
        if scene_idx is None:
            continue
        dst = _labels.setdefault(scene_idx, {})
        for _token, idx_map in data.get('sample_labels', {}).items():
            for gidx_str, label in idx_map.items():
                dst[int(gidx_str)] = int(label)
                n += 1
    return n


def build_front_views(sample_idx: int) -> list:
    """3 forward views as base64 data URIs (no bbox overlays)."""
    loader = get_loader()
    sample = loader.get_sample(sample_idx)
    rf = _config["resize_factor"]
    uris = []
    for cam_name in FRONT_CAMERAS:
        cam = sample.cameras[cam_name]
        try:
            img = Image.open(cam.image_path).convert('RGB')
        except Exception:
            img = Image.open(cam.image_path.lstrip('./')).convert('RGB')
        if rf > 1:
            w, h = img.size
            img = img.resize((w // rf, h // rf), Image.LANCZOS)
        uris.append(image_to_base64(img))
    return uris


HTML_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Driving Command Labeler</title>
<style>
  body { margin:0; background:#0f0f1a; color:#e0e0e0; font-family: "Segoe UI", sans-serif; }
  header { padding: 8px 16px; background:#1a1a2e; display:flex; gap:16px; align-items:center; border-bottom:1px solid #2a3f5f; flex-wrap:wrap; }
  header h1 { font-size:16px; margin:0; color:#00ff88; }
  .nav { display:flex; gap:8px; align-items:center; }
  .nav .nav-label { font-size:11px; color:#6a8abf; text-transform:uppercase; }
  .nav input { width:64px; padding:4px; background:#0f0f1a; color:#e0e0e0; border:1px solid #3a5a8f; border-radius:3px; }
  .nav button { padding:4px 12px; background:#2a3f5f; color:#e0e0e0; border:none; cursor:pointer; border-radius:3px; }
  .nav button:hover { background:#3a5a8f; }
  .meta { font-size:12px; color:#aaa; }
  main { padding: 8px; }
  .pano { display:grid; grid-template-columns:repeat(3,1fr); gap:4px; margin-bottom:8px; }
  .pano img { width:100%; border:1px solid #2a3f5f; display:block; }
  .pano .cam-label { font-size:11px; color:#6a8abf; text-align:center; margin-top:2px; }
  .label-bar { display:flex; gap:8px; align-items:stretch; flex-wrap:wrap; margin-bottom:8px; }
  .cmd-btn { flex:1; min-width:120px; padding:12px 8px; background:#16213e; color:#e0e0e0;
             border:1px solid #2a3f5f; border-radius:4px; cursor:pointer; text-align:center; }
  .cmd-btn:hover { background:#2a3f5f; }
  .cmd-btn .key { display:block; font-size:18px; font-weight:bold; color:#6a8abf; }
  .cmd-btn .name { display:block; font-size:12px; margin-top:4px; }
  .cmd-btn.selected { background:#0a3d2a; border-color:#00ff88; }
  .cmd-btn.selected .key { color:#00ff88; }
  .entry-bar { display:flex; gap:16px; align-items:center; background:#16213e; border:1px solid #2a3f5f;
               border-radius:4px; padding:8px 12px; flex-wrap:wrap; }
  .entry-bar input { width:64px; padding:6px; background:#0f0f1a; color:#e0e0e0; border:1px solid #3a5a8f;
                     border-radius:3px; font-size:16px; text-align:center; }
  .entry-bar .hint { font-size:12px; color:#888; }
  .progress { font-size:12px; color:#aaa; }
  .progress .done { color:#00ff88; }
  .progress-track { width:220px; height:8px; background:#0f0f1a; border:1px solid #2a3f5f; border-radius:4px; overflow:hidden; }
  .progress-fill { height:100%; background:#00ff88; width:0%; }
  .current-label { font-size:13px; }
  .current-label b { color:#00ff88; }
  .current-label .none { color:#888; font-style:italic; }
  .gt { color:#ffa502; }
  .bev-row { display:flex; justify-content:center; margin-top:8px; }
  .bev-row .bev-box { text-align:center; }
  .bev-row img { width:440px; max-width:70vw; border:1px solid #2a3f5f; border-radius:4px; display:block; }
  .bev-row .bev-label { font-size:11px; color:#6a8abf; margin-top:2px; }
  .bev-row .bev-loading { width:440px; max-width:70vw; padding:40px 0; text-align:center;
                          color:#555; font-size:12px; border:1px dashed #2a3f5f; border-radius:4px; }
  .save-status { font-size:12px; color:#888; min-width:80px; }
  .save-status.saved { color:#00ff88; }
  .save-status.error { color:#ff6b6b; }
  kbd { background:#2a3f5f; padding:1px 6px; border-radius:3px; font-family:monospace; font-size:11px; }
</style>
</head>
<body>
<header>
  <h1>Driving Command Labeler</h1>
  <div class="nav">
    <span class="nav-label">Scene</span>
    <button id="prev-scene">◀</button>
    <input type="number" id="scene-idx" value="0" min="0">
    <button id="next-scene">▶</button>
  </div>
  <div class="nav">
    <span class="nav-label">Sample</span>
    <button id="prev">◀ Prev</button>
    <input type="number" id="idx" value="0" min="0">
    <button id="go">Go</button>
    <button id="next">Next ▶</button>
  </div>
  <div class="meta" id="meta"></div>
  <div class="meta" style="margin-left:auto;">
    Keys: <kbd>0</kbd>-<kbd>6</kbd> label, <kbd>⌫</kbd> clear, <kbd>←</kbd>/<kbd>→</kbd> sample,
    <kbd>[</kbd>/<kbd>]</kbd> scene, <kbd>g</kbd> GT
  </div>
</header>
<main>
  <div class="pano" id="pano"></div>
  <div class="label-bar" id="label-bar"></div>
  <div class="entry-bar">
    <span class="hint">Command index:</span>
    <input type="number" id="cmd-input" min="0" max="6" placeholder="0-6">
    <span class="hint">press <kbd>Enter</kbd> to label &amp; advance</span>
    <button class="nav-btn" id="clear-label" style="padding:6px 12px; background:#3d1a1a; color:#ff9b9b; border:1px solid #5f2a2a; border-radius:3px; cursor:pointer;">Clear label</button>
    <span class="current-label" id="current-label"></span>
    <span class="save-status" id="save-status"></span>
    <span style="margin-left:auto;" class="progress" id="scene-progress"></span>
    <div class="progress-track"><div class="progress-fill" id="progress-fill"></div></div>
    <span class="progress" id="total-progress"></span>
  </div>
  <div class="bev-row"><div class="bev-box" id="bev-box"></div></div>
</main>
<script>
const CAM_LABELS = ['Front-Left', 'Front', 'Front-Right'];
const COMMANDS = {{ commands_json }};
const STATE = {
  idx: 0, total: 0, totalScenes: 0,
  sceneIdx: 0, sceneStart: 0, sceneCount: 0, sceneName: '',
  labels: {},          // global idx -> label, for the CURRENT scene
  totalLabeled: 0,
  showGT: false, gtCommand: null,
  bevSeq: 0,
};

async function loadBev(idx) {
  const seq = ++STATE.bevSeq;
  const box = document.getElementById('bev-box');
  box.innerHTML = '<div class="bev-loading">loading BEV…</div>';
  try {
    const data = await api(`/api/bev?idx=${idx}`);
    if (seq !== STATE.bevSeq) return;  // a newer sample was loaded meanwhile
    box.innerHTML = `<img src="${data.bev}"><div class="bev-label">BEV (ego-centered)</div>`;
  } catch (err) {
    if (seq === STATE.bevSeq) box.innerHTML = '<div class="bev-loading">BEV unavailable</div>';
  }
}

async function api(path) { const r = await fetch(path); return r.json(); }

function renderLabelBar() {
  const cur = STATE.labels[STATE.idx];
  document.getElementById('label-bar').innerHTML = Object.entries(COMMANDS).map(([k, name]) =>
    `<button class="cmd-btn ${parseInt(k) === cur ? 'selected' : ''}" data-cmd="${k}">
       <span class="key">${k}</span><span class="name">${name}</span></button>`
  ).join('');
  document.querySelectorAll('.cmd-btn').forEach(b =>
    b.onclick = () => { b.blur(); submitLabel(parseInt(b.dataset.cmd)); });
  const cl = document.getElementById('current-label');
  cl.innerHTML = cur === undefined
    ? 'Label: <span class="none">(unlabeled)</span>'
    : `Label: <b>${cur} — ${COMMANDS[cur]}</b>`;
}

function renderProgress() {
  const labeled = Object.keys(STATE.labels).length;
  const sp = document.getElementById('scene-progress');
  sp.innerHTML = `Scene: <span class="done">${labeled}</span>/${STATE.sceneCount} labeled`;
  document.getElementById('progress-fill').style.width =
    STATE.sceneCount ? `${100 * labeled / STATE.sceneCount}%` : '0%';
  document.getElementById('total-progress').innerHTML =
    `Total: <span class="done">${STATE.totalLabeled}</span>/${STATE.total}`;
}

function renderMeta(data) {
  const inScene = STATE.idx - STATE.sceneStart;
  const gt = STATE.showGT
    ? ` | <span class="gt">GT: ${data.gt_command ?? '?'}</span>` : '';
  document.getElementById('meta').innerHTML =
    `Scene ${STATE.sceneIdx}/${STATE.totalScenes - 1} (${STATE.sceneName})` +
    ` | Sample ${inScene + 1}/${STATE.sceneCount} (global ${STATE.idx})` +
    ` | ${data.location || '?'}${gt}`;
}

async function loadScene(sceneIdx) {
  const s = await api(`/api/scene?scene_idx=${sceneIdx}`);
  STATE.sceneIdx = s.scene_idx; STATE.sceneStart = s.start;
  STATE.sceneCount = s.count; STATE.sceneName = s.scene_name;
  STATE.labels = {};
  for (const [k, v] of Object.entries(s.labels)) STATE.labels[parseInt(k)] = v;
  document.getElementById('scene-idx').value = s.scene_idx;
}

async function load(idx, forceScene) {
  idx = Math.max(0, Math.min(STATE.total - 1, idx));
  const data = await api(`/api/sample?idx=${idx}`);
  STATE.idx = data.idx;
  STATE.gtCommand = data.gt_command;
  if (forceScene || data.scene_idx !== STATE.sceneIdx) await loadScene(data.scene_idx);
  document.getElementById('idx').value = data.idx;
  renderMeta(data);
  document.getElementById('pano').innerHTML = data.images.map((src, i) =>
    `<div><img src="${src}"><div class="cam-label">${CAM_LABELS[i]}</div></div>`
  ).join('');
  renderLabelBar();
  renderProgress();
  loadBev(data.idx);  // async — labeling stays responsive while BEV renders
}

async function submitLabel(label) {
  if (!(label >= 0 && label <= 6)) return;
  const status = document.getElementById('save-status');
  status.textContent = 'saving…'; status.className = 'save-status';
  const r = await fetch('/api/label', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({sample_idx: STATE.idx, label: label}),
  });
  const res = await r.json();
  if (!res.ok) { status.textContent = 'save failed'; status.className = 'save-status error'; return; }
  const prev = STATE.labels[STATE.idx];
  if (prev === undefined) STATE.totalLabeled += 1;
  STATE.labels[STATE.idx] = label;
  status.className = 'save-status saved';
  document.getElementById('cmd-input').value = '';
  if (prev !== undefined && prev !== label) {
    // Correction of an existing label: stay on the sample so the change is visible.
    status.textContent = `updated ${prev} → ${label} ✓`;
    renderLabelBar(); renderProgress();
    return;
  }
  status.textContent = `saved ✓ (${res.file})`;
  const sceneEnd = STATE.sceneStart + STATE.sceneCount - 1;
  if (STATE.idx < sceneEnd) load(STATE.idx + 1);
  else { renderLabelBar(); renderProgress(); status.textContent += ' — scene complete'; }
}

async function clearLabel() {
  if (!(STATE.idx in STATE.labels)) return;
  const status = document.getElementById('save-status');
  status.textContent = 'clearing…'; status.className = 'save-status';
  const r = await fetch('/api/label', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({sample_idx: STATE.idx, label: null}),
  });
  const res = await r.json();
  if (!res.ok) { status.textContent = 'clear failed'; status.className = 'save-status error'; return; }
  delete STATE.labels[STATE.idx];
  STATE.totalLabeled -= 1;
  status.textContent = 'label cleared ✓'; status.className = 'save-status saved';
  renderLabelBar(); renderProgress();
}

async function init() {
  const info = await api('/api/info');
  STATE.total = info.total; STATE.totalScenes = info.total_scenes;
  STATE.totalLabeled = info.total_labeled;
  document.getElementById('idx').max = info.total - 1;
  document.getElementById('scene-idx').max = info.total_scenes - 1;

  document.getElementById('prev').onclick = () => load(STATE.idx - 1);
  document.getElementById('next').onclick = () => load(STATE.idx + 1);
  document.getElementById('go').onclick = () =>
    load(parseInt(document.getElementById('idx').value) || 0);
  document.getElementById('prev-scene').onclick = () => gotoScene(STATE.sceneIdx - 1);
  document.getElementById('next-scene').onclick = () => gotoScene(STATE.sceneIdx + 1);
  document.getElementById('clear-label').onclick = () => clearLabel();
  // Release focus after Enter so subsequent 0-6 keypresses label globally
  // instead of typing into the box.
  document.getElementById('scene-idx').addEventListener('keydown', e => {
    if (e.key === 'Enter') { gotoScene(parseInt(e.target.value) || 0); e.target.blur(); }
  });
  document.getElementById('idx').addEventListener('keydown', e => {
    if (e.key === 'Enter') { load(parseInt(e.target.value) || 0); e.target.blur(); }
  });
  document.getElementById('cmd-input').addEventListener('keydown', e => {
    if (e.key === 'Enter') { submitLabel(parseInt(e.target.value)); e.target.blur(); }
  });

  document.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT') return;
    if (e.key >= '0' && e.key <= '6') submitLabel(parseInt(e.key));
    else if (e.key === 'Backspace' || e.key === 'Delete') clearLabel();
    else if (e.key === 'ArrowLeft') load(STATE.idx - 1);
    else if (e.key === 'ArrowRight') load(STATE.idx + 1);
    else if (e.key === '[') gotoScene(STATE.sceneIdx - 1);
    else if (e.key === ']') gotoScene(STATE.sceneIdx + 1);
    else if (e.key === 'g') { STATE.showGT = !STATE.showGT; load(STATE.idx); }
  });
  load(0, true);
}

async function gotoScene(sceneIdx) {
  sceneIdx = Math.max(0, Math.min(STATE.totalScenes - 1, sceneIdx));
  const s = await api(`/api/scene?scene_idx=${sceneIdx}`);
  // Jump to the first unlabeled sample in the scene (or its start).
  let target = s.start;
  for (let i = s.start; i < s.start + s.count; i++) {
    if (!(String(i) in s.labels)) { target = i; break; }
  }
  load(target, true);
}
init();
</script>
</body>
</html>"""


@app.route('/')
def index():
    return render_template_string(
        HTML_TEMPLATE.replace('{{ commands_json }}', json.dumps(COMMAND_LABELS)))


@app.route('/api/info')
def api_info():
    loader = get_loader()
    total_labeled = sum(len(v) for v in _labels.values())
    return jsonify({
        'total': len(loader.infos),
        'total_scenes': len(_scenes),
        'total_labeled': total_labeled,
    })


@app.route('/api/scene')
def api_scene():
    try:
        scene_idx = int(request.args.get('scene_idx', 0))
    except ValueError:
        scene_idx = 0
    scene_idx = max(0, min(len(_scenes) - 1, scene_idx))
    scene = _scenes[scene_idx]
    return jsonify({
        'scene_idx': scene_idx,
        'scene_token': scene['scene_token'],
        'scene_name': scene['scene_name'],
        'start': scene['start'],
        'count': scene['count'],
        'labels': {str(k): v for k, v in _labels.get(scene_idx, {}).items()},
    })


@app.route('/api/sample')
def api_sample():
    try:
        idx = int(request.args.get('idx', 0))
    except ValueError:
        idx = 0
    loader = get_loader()
    idx = max(0, min(len(loader.infos) - 1, idx))
    info = loader.infos[idx]
    scene_idx = scene_of_sample(idx)

    gt_cmd = info.get('gt_navigation_command')
    if gt_cmd is None:
        gt_cmd = info.get('gt_planning_command')
    gt_str = (f"{int(gt_cmd)} — {COMMAND_LABELS.get(int(gt_cmd), '?')}"
              if gt_cmd is not None else None)

    return jsonify({
        'idx': idx,
        'scene_idx': scene_idx,
        'token': info.get('token'),
        'location': info.get('location'),
        'gt_command': gt_str,
        'images': build_front_views(idx),
    })


@app.route('/api/bev')
def api_bev():
    try:
        idx = int(request.args.get('idx', 0))
    except ValueError:
        idx = 0
    loader = get_loader()
    idx = max(0, min(len(loader.infos) - 1, idx))
    sample = loader.get_sample(idx)
    return jsonify({'idx': idx, 'bev': generate_bev(sample, loader, bev_range=_config["bev_range"])})


@app.route('/api/label', methods=['POST'])
def api_label():
    data = request.get_json(force=True)
    try:
        sample_idx = int(data['sample_idx'])
        label = data['label']
        if label is not None:
            label = int(label)
    except (KeyError, TypeError, ValueError):
        return jsonify({'ok': False, 'error': 'sample_idx and label (int or null) required'}), 400
    loader = get_loader()
    if not (0 <= sample_idx < len(loader.infos)):
        return jsonify({'ok': False, 'error': 'sample_idx out of range'}), 400
    if label is not None and label not in COMMAND_LABELS:
        return jsonify({'ok': False, 'error': f'label must be one of {sorted(COMMAND_LABELS)} or null to clear'}), 400

    scene_idx = scene_of_sample(sample_idx)
    if label is None:
        _labels.get(scene_idx, {}).pop(sample_idx, None)
    else:
        _labels.setdefault(scene_idx, {})[sample_idx] = label
    save_scene_labels(scene_idx)
    return jsonify({
        'ok': True,
        'file': os.path.basename(label_file_path(scene_idx)),
        'scene_labeled': len(_labels[scene_idx]),
        'scene_total': _scenes[scene_idx]['count'],
    })


def main():
    parser = argparse.ArgumentParser(description="Web labeling tool for GT driving commands")
    parser.add_argument('--pkl_path', type=str,
                        default=os.environ.get("NUSCENES_PKL_PATH",
                                               "./data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl"))
    parser.add_argument('--output_dir', type=str, default='driving_command_labels',
                        help='Directory for per-scene label JSON files (autosaved)')
    parser.add_argument('--resize_factor', type=int, default=2,
                        help='Display resize factor for the 3 forward views')
    parser.add_argument('--bev_range', type=float, default=50.0,
                        help='BEV map half-range in meters (default 50)')
    parser.add_argument('--port', type=int, default=6062)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    args = parser.parse_args()

    _config["pkl_path"] = args.pkl_path
    _config["output_dir"] = args.output_dir
    _config["resize_factor"] = args.resize_factor
    _config["bev_range"] = args.bev_range

    build_scene_index()
    n = load_existing_labels()

    print(f"PKL path:    {args.pkl_path}")
    print(f"Output dir:  {args.output_dir}")
    print(f"Scenes:      {len(_scenes)}  |  Samples: {len(get_loader().infos)}")
    print(f"Resumed:     {n} existing labels")
    print(f"Starting server on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
