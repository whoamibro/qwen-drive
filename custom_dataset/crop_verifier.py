"""
Crop Verifier — browser tool to verify the custom-dataset → nuScenes virtual
camera crops before running the batch conversion.

Shows, per frame, the six used cameras in three ways:

  Grid mode      — the cropped+resized 1600x900 outputs arranged in the
                   training pipeline's 2x3 egocentric layout (FL / F / FR on
                   top, RL / R / RR on bottom, rear views horizontally
                   flipped). Optional boundary guide lines, hover magnifier,
                   and an adjacent-edge strip row for continuity checks.
  Overlay mode   — each ORIGINAL full-resolution image with the computed
                   crop rectangle + principal point drawn on top, alongside
                   per-camera intrinsics / crop metadata. Per-view (and
                   global) before/after toggle to the cropped result
                   (shown UNFLIPPED to stay comparable with the original).
  Processed mode — same card format as overlay, but showing the FINAL
                   processed images (crop + resize + pitch leveling, rear
                   views flipped) — exactly what the model receives.

All crop math comes from `custom_dataset.virtual_camera` — this module only
renders. Crops are computed once per clip x camera at startup from
maps/calib_tuned/*.yaml; if any crop box would exceed its source image, the
server prints the violating clip/camera and the minimum feasible f_t, then
exits instead of serving wrong geometry.

Images are processed on demand (nothing pre-generated) and cached in a small
LRU keyed by (clip, frame, camera, mode).

Usage:
  python -m custom_dataset.crop_verifier \\
      --dataset_root /yjj_rnd_home/data/vlm_dataset/demo_dataset \\
      --port 6063

Keys:  ←/→ frame | [/] clip | g grid | o overlay | b before/after
       l guides | e edge strips | m magnifier | k keyframes(2Hz)/all
"""

import argparse
import io
import json
import os
import re
import sys
import threading
from collections import OrderedDict

from flask import Flask, Response, jsonify, render_template_string, request
from PIL import Image, ImageDraw

from custom_dataset.virtual_camera import (
    CAMERA_ORDER,
    DEFAULT_TARGET,
    ORIENTATION_JSON,
    NUSCENES_TO_CHANNEL,
    REAR_CAMERAS,
    TargetCamera,
    convert_image,
    crop_metadata,
    axis_elevation_after,
    clamp_extra_pitch,
    load_camera_elevations,
    load_clip_calibs,
    min_feasible_focal,
    source_quad,
    target_for_camera,
)

# Microsecond timestamp embedded in every image filename,
# e.g. GER_MACHET18_20260414_192304_1776187384099211_2160p_h120_front.jpg
_TS_RE = re.compile(r'_(\d{13,17})_')

CAM_DISPLAY = {
    'CAM_FRONT_LEFT':  'Front-left',
    'CAM_FRONT':       'Front',
    'CAM_FRONT_RIGHT': 'Front-right',
    'CAM_BACK_LEFT':   'Rear-left',
    'CAM_BACK':        'Rear',
    'CAM_BACK_RIGHT':  'Rear-right',
}

IMG_MODES = ('grid', 'crop', 'orig')


# ---------------------------------------------------------------------------
# Flask app + state
# ---------------------------------------------------------------------------

app = Flask(__name__)
_config = {
    'dataset_root': '',
    'jpeg_quality': 90,
    'orig_display_width': 1152,   # originals are 8MP; downscale for display
    'cache_size': 240,
    'target': DEFAULT_TARGET,
    'keyframe_stride': 5,         # 10 Hz -> 2 Hz default sampling
    # Measured axis elevation per camera (deg, + = up) for pitch leveling;
    # {} disables the correction.
    'elevations': {},
    # Explicit CAM_BACK focal; None = nuScenes back/front ratio (0.639 x base).
    'back_focal': None,
    # Extra up-pitch (deg) per camera BEYOND auto-leveling. Unclamped: past
    # the data boundary the top of the view renders black. Live-adjustable
    # from the UI (POST /api/pitch_offset) to find the right value visually.
    'pitch_offsets': {},
    # Keystone bottom-edge scale per camera (<1 shrinks the footprint's
    # bottom edge symmetrically, keeping the top edge). Live-adjustable
    # (POST /api/bottom_scale).
    'bottom_scales': {},
    # Cameras whose footprint top edge is pinned to the source top border.
    'pin_tops': set(),
}


def _elev(cam: str) -> float:
    return float(_config['elevations'].get(cam, 0.0))


def _pitch(cam: str) -> float:
    return float(_config['pitch_offsets'].get(cam, 0.0))


def _bscale(cam: str) -> float:
    return float(_config['bottom_scales'].get(cam, 1.0))


def _pintop(cam: str) -> bool:
    return cam in _config['pin_tops']


def _target(cam: str) -> TargetCamera:
    """Per-camera virtual target (CAM_BACK is wider, nuScenes-style)."""
    return target_for_camera(cam, _config['target'], _config['back_focal'])

# _clips[i] = {name, path, calibs: {cam: CameraCalib}, meta: {cam: dict},
#              files: {cam: {ts: abspath}}, timestamps: [ts, ...]}
_clips = []

_cache = OrderedDict()            # (clip, frame, cam, mode) -> jpeg bytes
_cache_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Startup: clip discovery + crop feasibility
# ---------------------------------------------------------------------------

def discover_clips(root: str):
    """Find clip dirs (must contain images/ and maps/calib_tuned/), sorted."""
    clips = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        if (os.path.isdir(os.path.join(path, 'images'))
                and os.path.isdir(os.path.join(path, 'maps', 'calib_tuned'))):
            clips.append((name, path))
    return clips


def build_clip(name: str, path: str) -> dict:
    """Load calibs, compute crop metadata, and index frames by timestamp."""
    calibs = load_clip_calibs(path)
    meta = {cam: crop_metadata(calibs[cam], _target(cam), _elev(cam),
                               _pitch(cam), _bscale(cam), _pintop(cam))
            for cam in CAMERA_ORDER}

    files = {}
    ts_sets = []
    for cam in CAMERA_ORDER:
        channel = NUSCENES_TO_CHANNEL[cam]
        cam_dir = os.path.join(path, 'images', channel)
        per_ts = {}
        for fname in os.listdir(cam_dir):
            m = _TS_RE.search(fname)
            if m and fname.lower().endswith('.jpg'):
                per_ts[m.group(1)] = os.path.join(cam_dir, fname)
        files[cam] = per_ts
        ts_sets.append(set(per_ts))

    # Frames = timestamps present in ALL six channels, temporally sorted.
    common = sorted(set.intersection(*ts_sets), key=int)
    n_dropped = max(len(s) for s in ts_sets) - len(common)
    if n_dropped:
        print(f'  [{name}] {n_dropped} frame(s) missing from at least one '
              f'channel — using the {len(common)} complete frames')
    return {'name': name, 'path': path, 'calibs': calibs, 'meta': meta,
            'files': files, 'timestamps': common}


def verify_feasibility():
    """Fail fast (no stack trace) if any clip x camera crop exceeds bounds."""
    violations = []
    for clip in _clips:
        for cam in CAMERA_ORDER:
            m = clip['meta'][cam]
            if not m['in_bounds']:
                violations.append(
                    f"  clip={clip['name']}  cam={cam} ({m['channel']}): "
                    f"crop {m['crop_box']} exceeds source "
                    f"{m['source_size'][0]}x{m['source_size'][1]} — "
                    f"minimum feasible f_t = {m['min_feasible_ft']}")
    if violations:
        t = _config['target']
        overall = max(min_feasible_focal(clip['calibs'][cam], _target(cam),
                                         _elev(cam))
                      for clip in _clips for cam in CAMERA_ORDER)
        print('ERROR: infeasible crop box(es) at '
              f'target {t.width}x{t.height} f_t={t.focal}:', file=sys.stderr)
        for v in violations:
            print(v, file=sys.stderr)
        print(f'Smallest f_t feasible for ALL cameras: {overall:.1f}',
              file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Image rendering (on demand, LRU-cached)
# ---------------------------------------------------------------------------

def _cache_get(key):
    with _cache_lock:
        data = _cache.get(key)
        if data is not None:
            _cache.move_to_end(key)
        return data


def _cache_put(key, data):
    with _cache_lock:
        _cache[key] = data
        _cache.move_to_end(key)
        while len(_cache) > _config['cache_size']:
            _cache.popitem(last=False)


def _encode_jpeg(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=_config['jpeg_quality'])
    return buf.getvalue()


def _render_orig_overlay(img: Image.Image, calib, quad) -> Image.Image:
    """Downscaled original with the warp footprint (trapezoid for rotated
    views, rectangle at elevation 0), principal point cross, image-center
    dot, and the virtual optical-axis line (= leveled horizon)."""
    disp_w = _config['orig_display_width']
    scale = disp_w / img.width
    disp = img.resize((disp_w, max(1, round(img.height * scale))),
                      Image.BILINEAR)
    draw = ImageDraw.Draw(disp)
    pts = [(x * scale, y * scale) for x, y in quad]   # TL, TR, BR, BL
    draw.polygon(pts, outline=(0, 255, 136), width=2)
    # Virtual optical-axis row: midpoints of the left/right footprint edges
    # (= where the leveled horizon crosses the view): dashed cyan line.
    (tlx, tly), (trx, try_), (brx, bry), (blx, bly) = pts
    lx, ly = (tlx + blx) / 2.0, (tly + bly) / 2.0
    rx, ry = (trx + brx) / 2.0, (try_ + bry) / 2.0
    n_seg = 24
    for i in range(0, n_seg, 2):
        t0, t1 = i / n_seg, (i + 0.5) / n_seg
        draw.line([lx + (rx - lx) * t0, ly + (ry - ly) * t0,
                   lx + (rx - lx) * t1, ly + (ry - ly) * t1],
                  fill=(0, 200, 255), width=2)
    # Principal point: orange cross
    px, py = calib.cx * scale, calib.cy * scale
    r = 10
    draw.line([px - r, py, px + r, py], fill=(255, 165, 2), width=2)
    draw.line([px, py - r, px, py + r], fill=(255, 165, 2), width=2)
    # Geometric image center: small grey circle
    gx, gy = disp.width / 2.0, disp.height / 2.0
    draw.ellipse([gx - 4, gy - 4, gx + 4, gy + 4], outline=(160, 160, 160),
                 width=2)
    return disp


def render_image(clip_idx: int, frame_idx: int, cam: str, mode: str) -> bytes:
    key = (clip_idx, frame_idx, cam, mode)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    clip = _clips[clip_idx]
    ts = clip['timestamps'][frame_idx]
    img = Image.open(clip['files'][cam][ts]).convert('RGB')
    calib = clip['calibs'][cam]
    target = _target(cam)

    if mode == 'grid':      # cropped output, rear views flipped (egocentric)
        out = convert_image(img, calib, target, flip=cam in REAR_CAMERAS,
                            elevation_deg=_elev(cam),
                            extra_pitch_deg=_pitch(cam),
                            bottom_scale=_bscale(cam), pin_top=_pintop(cam))
    elif mode == 'crop':    # cropped output, no flip (for before/after)
        out = convert_image(img, calib, target, flip=False,
                            elevation_deg=_elev(cam),
                            extra_pitch_deg=_pitch(cam),
                            bottom_scale=_bscale(cam), pin_top=_pintop(cam))
    else:                   # 'orig': full frame + warp footprint + pp markers
        out = _render_orig_overlay(
            img, calib, source_quad(calib, target, _elev(cam), _pitch(cam),
                                    _bscale(cam), _pintop(cam)))

    data = _encode_jpeg(out)
    _cache_put(key, data)
    return data


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Crop Verifier</title>
<style>
  body { margin:0; background:#0f0f1a; color:#e0e0e0; font-family: "Segoe UI", sans-serif; }
  header { padding: 8px 16px; background:#1a1a2e; display:flex; gap:16px; align-items:center; border-bottom:1px solid #2a3f5f; flex-wrap:wrap; }
  header h1 { font-size:16px; margin:0; color:#00ff88; }
  .nav { display:flex; gap:8px; align-items:center; }
  .nav .nav-label { font-size:11px; color:#6a8abf; text-transform:uppercase; }
  .nav input { width:72px; padding:4px; background:#0f0f1a; color:#e0e0e0; border:1px solid #3a5a8f; border-radius:3px; }
  .nav select { padding:4px; background:#0f0f1a; color:#e0e0e0; border:1px solid #3a5a8f; border-radius:3px; max-width:260px; }
  .nav button { padding:4px 12px; background:#2a3f5f; color:#e0e0e0; border:none; cursor:pointer; border-radius:3px; }
  .nav button:hover { background:#3a5a8f; }
  .nav button.active { background:#0a3d2a; outline:1px solid #00ff88; }
  .nav label.chk { font-size:12px; color:#aaa; display:flex; gap:4px; align-items:center; cursor:pointer; }
  .meta { font-size:12px; color:#aaa; }
  .meta b { color:#00ff88; }
  kbd { background:#2a3f5f; padding:1px 6px; border-radius:3px; font-family:monospace; font-size:11px; }
  main { padding: 8px; }

  /* --- grid mode --- */
  .pano6 { display:grid; grid-template-columns:repeat(3,1fr); gap:4px; }
  .cell { position:relative; }
  .cell img { width:100%; border:1px solid #2a3f5f; display:block; }
  .cell .cam-label { font-size:11px; color:#6a8abf; text-align:center; margin-top:2px; }
  .cell .flip-tag { color:#ffa502; }
  .guides .cell .guide-v1, .guides .cell .guide-v2 {
    position:absolute; top:0; bottom:18px; width:0; border-left:1px dashed rgba(0,255,136,.65); pointer-events:none; }
  .guides .cell .guide-v1 { left:2.5%; }
  .guides .cell .guide-v2 { right:2.5%; }
  .guides .cell .guide-h {
    position:absolute; left:0; right:0; top:calc((100% - 18px)/2); height:0;
    border-top:1px dashed rgba(255,165,2,.65); pointer-events:none; }
  .lens { position:absolute; width:220px; height:220px; border:2px solid #00ff88; border-radius:4px;
          background-repeat:no-repeat; pointer-events:none; display:none; z-index:10;
          box-shadow:0 0 12px rgba(0,0,0,.8); }

  /* --- edge strips --- */
  .strips { margin-top:10px; display:flex; gap:24px; flex-wrap:wrap; }
  .strip-pair { text-align:center; }
  .strip-row { display:flex; gap:2px; position:relative; }
  .slice { width:170px; height:260px; overflow:hidden; position:relative; border:1px solid #2a3f5f; }
  .slice img { position:absolute; height:100%; width:auto; max-width:none; top:0; }
  .slice.edge-r img { right:0; }
  .slice.edge-l img { left:0; }
  .strip-row::after { content:''; position:absolute; left:0; right:0; top:50%; height:0;
                      border-top:1px dashed rgba(255,165,2,.7); pointer-events:none; }
  .strip-label { font-size:11px; color:#6a8abf; margin-top:2px; }

  /* --- overlay mode --- */
  .cards { display:grid; grid-template-columns:repeat(2,1fr); gap:8px; }
  /* processed mode: 3 columns x 2 rows, matching the egocentric layout
     (FL | F | FR on top, RL | R | RR below) */
  .cards.cols3 { grid-template-columns:repeat(3,1fr); }
  .card { background:#16213e; border:1px solid #2a3f5f; border-radius:4px; padding:8px; }
  .card .card-head { display:flex; align-items:center; gap:8px; margin-bottom:6px; }
  .card .card-head h3 { font-size:13px; margin:0; color:#00ff88; }
  .card .card-head .chan { font-size:11px; color:#6a8abf; }
  .card .card-head button { margin-left:auto; padding:3px 10px; background:#2a3f5f; color:#e0e0e0;
                            border:none; cursor:pointer; border-radius:3px; font-size:11px; }
  .card .card-head .ab-state { font-size:11px; color:#ffa502; min-width:120px; text-align:right; }
  .card img { width:100%; display:block; border:1px solid #2a3f5f; cursor:pointer; }
  .card table { width:100%; font-size:11px; color:#aaa; border-collapse:collapse; margin-top:6px; }
  .card td { padding:1px 6px 1px 0; vertical-align:top; }
  .card td.k { color:#6a8abf; white-space:nowrap; width:130px; }
  .card td.v { font-family:monospace; }
  .legend { font-size:11px; color:#888; margin:6px 2px; }
  .legend .g { color:#00ff88; } .legend .o { color:#ffa502; } .legend .grey { color:#aaa; }
</style>
</head>
<body>
<header>
  <h1>Crop Verifier</h1>
  <div class="nav">
    <span class="nav-label">Clip</span>
    <button id="prev-clip">◀</button>
    <select id="clip-select"></select>
    <button id="next-clip">▶</button>
  </div>
  <div class="nav">
    <span class="nav-label">Frame</span>
    <button id="prev">◀ Prev</button>
    <input type="number" id="frame-idx" value="0" min="0">
    <button id="go">Go</button>
    <button id="next">Next ▶</button>
    <label class="chk"><input type="checkbox" id="keyframes" checked> 2 Hz keyframes</label>
  </div>
  <div class="nav">
    <button id="mode-grid" class="active">Grid</button>
    <button id="mode-overlay">Overlay</button>
    <button id="mode-processed">Processed</button>
    <label class="chk" id="chk-guides-wrap"><input type="checkbox" id="chk-guides"> guides</label>
    <label class="chk" id="chk-strips-wrap"><input type="checkbox" id="chk-strips"> edge strips</label>
    <label class="chk" id="chk-magnify-wrap"><input type="checkbox" id="chk-magnify"> magnifier</label>
    <button id="btn-ab" style="display:none;">Before / After</button>
  </div>
  <div class="nav" title="Extra pitch (+ = up) beyond auto-leveling for the selected rear camera. Past the data boundary the view gains a black border. 'Align sides' tilts rear-left/right down to match rear-center's axis (clamped, no black).">
    <span class="nav-label">Pitch+</span>
    <select id="pitch-cam" style="max-width:110px;">
      <option value="CAM_BACK">Rear</option>
      <option value="CAM_BACK_LEFT">Rear-left</option>
      <option value="CAM_BACK_RIGHT">Rear-right</option>
    </select>
    <button id="pitch-dn">−</button>
    <input type="number" id="pitch-deg" value="0" step="1" min="-30" max="30" style="width:56px;">
    <button id="pitch-up">+</button>
    <button id="pitch-align" title="Tilt rear-left/right down to rear-center's axis">Align sides</button>
    <button id="pitch-reset">Reset</button>
    <span class="meta" id="pitch-info"></span>
  </div>
  <div class="meta" id="meta"></div>
  <div class="meta" style="margin-left:auto;">
    Keys: <kbd>←</kbd>/<kbd>→</kbd> frame, <kbd>[</kbd>/<kbd>]</kbd> clip, <kbd>g</kbd>/<kbd>o</kbd>/<kbd>p</kbd> mode,
    <kbd>b</kbd> A/B, <kbd>l</kbd> guides, <kbd>e</kbd> strips, <kbd>m</kbd> magnify, <kbd>k</kbd> stride
  </div>
</header>
<main>
  <div id="grid-view"></div>
  <div id="overlay-view" style="display:none;"></div>
  <div id="processed-view" style="display:none;"></div>
</main>
<script>
const CAMERA_ORDER = {{ camera_order }};
const REAR = new Set({{ rear_cams }});
const CAM_DISPLAY = {{ cam_display }};
const KEYFRAME_STRIDE = {{ keyframe_stride }};
// Display-adjacent view pairs for the edge-strip continuity check
// (indices into CAMERA_ORDER: FL|F, F|FR on top; RL|R, R|RR on bottom).
const STRIP_PAIRS = [[0,1],[1,2],[3,4],[4,5]];
const MAGNIFY_ZOOM = 3;

const STATE = {
  clips: [], target: null, version: '', pitchOffsets: {},
  clip: 0, frame: 0,
  mode: 'grid',            // 'grid' | 'overlay'
  guides: false, strips: false, magnify: false,
  ab: 'before',            // overlay global toggle
  cardAB: {},              // per-card override: cam -> 'before'|'after'
};

async function api(path) { const r = await fetch(path); return r.json(); }
// version param busts the browser cache when the crop geometry changes
const imgUrl = (cam, mode) =>
  `/img/${STATE.clip}/${STATE.frame}/${cam}/${mode}?v=${STATE.version}`;
const nFrames = () => STATE.clips[STATE.clip].n_frames;
const stride = () => document.getElementById('keyframes').checked ? KEYFRAME_STRIDE : 1;

function camTitle(cam, i, withFlipTag) {
  // The rear flip is applied ONLY in grid mode (and at inference); overlay
  // mode shows unflipped images, so it must not carry the "(flipped)" tag.
  const flip = withFlipTag && REAR.has(cam)
    ? ' <span class="flip-tag">(flipped)</span>' : '';
  return `Image ${i + 1}: ${CAM_DISPLAY[cam]} (${cam})${flip}`;
}

// ---------------- grid mode ----------------
function renderGrid() {
  const cells = CAMERA_ORDER.map((cam, i) => `
    <div class="cell" data-cam="${cam}">
      <img src="${imgUrl(cam, 'grid')}" draggable="false">
      <div class="guide-v1"></div><div class="guide-v2"></div><div class="guide-h"></div>
      <div class="lens"></div>
      <div class="cam-label">${camTitle(cam, i, true)}</div>
    </div>`).join('');

  let strips = '';
  if (STATE.strips) {
    strips = '<div class="strips">' + STRIP_PAIRS.map(([a, b]) => `
      <div class="strip-pair">
        <div class="strip-row">
          <div class="slice edge-r"><img src="${imgUrl(CAMERA_ORDER[a], 'grid')}"></div>
          <div class="slice edge-l"><img src="${imgUrl(CAMERA_ORDER[b], 'grid')}"></div>
        </div>
        <div class="strip-label">Image ${a + 1} right edge ▸|◂ Image ${b + 1} left edge</div>
      </div>`).join('') + `</div>
      <div class="legend">Objects crossing a boundary should sit at the same height and scale on
      both sides of each strip (dashed line = vertical center reference).</div>`;
  }

  const view = document.getElementById('grid-view');
  view.innerHTML =
    `<div class="pano6 ${STATE.guides ? 'guides' : ''}">${cells}</div>` + strips;

  if (STATE.magnify) attachMagnifiers();
}

function attachMagnifiers() {
  document.querySelectorAll('#grid-view .cell').forEach(cell => {
    const img = cell.querySelector('img');
    const lens = cell.querySelector('.lens');
    cell.addEventListener('mousemove', e => {
      const r = img.getBoundingClientRect();
      const x = e.clientX - r.left, y = e.clientY - r.top;
      if (x < 0 || y < 0 || x > r.width || y > r.height) { lens.style.display = 'none'; return; }
      lens.style.display = 'block';
      const lw = lens.offsetWidth, lh = lens.offsetHeight;
      lens.style.left = `${Math.min(Math.max(x - lw / 2, 0), r.width - lw)}px`;
      lens.style.top  = `${Math.min(Math.max(y - lh / 2, 0), r.height - lh)}px`;
      lens.style.backgroundImage = `url(${img.src})`;
      lens.style.backgroundSize = `${r.width * MAGNIFY_ZOOM}px ${r.height * MAGNIFY_ZOOM}px`;
      lens.style.backgroundPosition =
        `${-(x * MAGNIFY_ZOOM - lw / 2)}px ${-(y * MAGNIFY_ZOOM - lh / 2)}px`;
    });
    cell.addEventListener('mouseleave', () => { lens.style.display = 'none'; });
  });
}

// ---------------- overlay mode ----------------
function cardState(cam) { return STATE.cardAB[cam] || STATE.ab; }

function metaTable(m) {
  const residual = Math.abs(m.residual_tilt_deg) > 0.05
    ? ` (residual ${m.residual_tilt_deg}°)` : '';
  const blackBits = [];
  if (m.black_top_rows) blackBits.push(`TOP ${m.black_top_rows}px`);
  if (m.black_bottom_rows) blackBits.push(`BOTTOM ${m.black_bottom_rows}px`);
  const extra = m.extra_pitch_deg
    ? ` + extra ${m.extra_pitch_deg}°` +
      (blackBits.length ? ` → BLACK ${blackBits.join(', ')}` : '') : '';
  const rows = [
    ['source', `${m.source_size[0]} × ${m.source_size[1]}  (HFOV ${m.source_hfov_deg}°, VFOV ${m.source_vfov_deg}°)`],
    ['fx / fy', `${m.fx} / ${m.fy}`],
    ['cx / cy', `${m.cx} / ${m.cy}`],
    ['axis elevation (measured)', `${m.axis_elevation_deg}° ${m.axis_elevation_deg > 0 ? '(up)' : m.axis_elevation_deg < 0 ? '(down)' : '(level)'}`],
    ['pitch correction applied', `${m.pitch_correction_applied_deg}°${extra}${residual}`],
    ['crop box (x1,y1,x2,y2)', `[${m.crop_box.join(', ')}]`],
    ['crop size', `${m.crop_size[0]} × ${m.crop_size[1]}`],
    ['retained ratio (x, y)', `${m.retained_ratio[0]} / ${m.retained_ratio[1]}`],
    ['effective FOV after crop', `${m.effective_hfov_deg}° × ${m.effective_vfov_deg}°`],
    ['min feasible f_t (centered / leveled)', `${m.min_feasible_ft} / ${m.min_feasible_ft_leveled}`],
  ];
  return '<table>' + rows.map(([k, v]) =>
    `<tr><td class="k">${k}</td><td class="v">${v}</td></tr>`).join('') + '</table>';
}

function renderOverlay() {
  const meta = STATE.clips[STATE.clip].cams;
  document.getElementById('overlay-view').innerHTML =
    `<div class="legend"><span class="g">green</span> crop box &nbsp;|&nbsp;
     <span style="color:#00c8ff">cyan dashes</span> crop vertical center = leveled horizon &nbsp;|&nbsp;
     <span class="o">orange cross</span> principal point (cx, cy) &nbsp;|&nbsp;
     <span class="grey">grey circle</span> image center &nbsp;—&nbsp;
     the crop center is shifted vertically by fy·tan(axis elevation) to level
     each virtual camera; "after" shows the cropped 1600×900 result UNFLIPPED
     (rear flip is applied only in the grid / at inference).</div>
    <div class="cards">` + CAMERA_ORDER.map((cam, i) => {
      const st = cardState(cam);
      const mode = st === 'before' ? 'orig' : 'crop';
      return `
      <div class="card" data-cam="${cam}">
        <div class="card-head">
          <h3>${camTitle(cam, i, false)}</h3><span class="chan">${meta[cam].channel}</span>
          <button data-cam="${cam}" class="ab-btn">A/B</button>
          <span class="ab-state">${st === 'before' ? 'BEFORE (orig + box)' : 'AFTER (cropped, no flip)'}</span>
        </div>
        <img src="${imgUrl(cam, mode)}" title="click to toggle before/after" draggable="false">
        ${metaTable(meta[cam])}
      </div>`;
    }).join('') + '</div>';

  document.querySelectorAll('#overlay-view .ab-btn').forEach(b =>
    b.onclick = () => toggleCard(b.dataset.cam));
  document.querySelectorAll('#overlay-view .card img').forEach(img =>
    img.onclick = () => toggleCard(img.closest('.card').dataset.cam));
}

function toggleCard(cam) {
  STATE.cardAB[cam] = cardState(cam) === 'before' ? 'after' : 'before';
  renderOverlay();
}

// ---------------- processed mode ----------------
// Same card format as overlay, but showing the FINAL processed images —
// cropped + resized + rear views flipped — i.e., exactly what the model
// (and the grid tiles) receive.
function renderProcessed() {
  const meta = STATE.clips[STATE.clip].cams;
  document.getElementById('processed-view').innerHTML =
    `<div class="legend">Final processed ${STATE.target.width}×${STATE.target.height}
     outputs as fed to the model: crop + resize + pitch leveling, rear views
     <span class="flip-tag" style="color:#ffa502;">horizontally flipped</span>
     (egocentric convention, same as the grid tiles).</div>
    <div class="cards cols3">` + CAMERA_ORDER.map((cam, i) => `
      <div class="card" data-cam="${cam}">
        <div class="card-head">
          <h3>${camTitle(cam, i, true)}</h3><span class="chan">${meta[cam].channel}</span>
        </div>
        <img src="${imgUrl(cam, 'grid')}" draggable="false" style="cursor:default;">
        ${metaTable(meta[cam])}
      </div>`).join('') + '</div>';
}

function toggleABAll() {
  STATE.ab = STATE.ab === 'before' ? 'after' : 'before';
  STATE.cardAB = {};
  if (STATE.mode === 'overlay') renderOverlay();
}

// ---------------- navigation / meta ----------------
async function renderMeta() {
  const clip = STATE.clips[STATE.clip];
  const t = STATE.target;
  const f = await api(`/api/frame?clip=${STATE.clip}&frame=${STATE.frame}`);
  document.getElementById('meta').innerHTML =
    `<b>${clip.name}</b> | frame ${STATE.frame}/${clip.n_frames - 1}` +
    ` | ts ${f.timestamp} (t=+${f.t_offset_s.toFixed(1)}s)` +
    ` | target ${t.width}×${t.height} f_t=${t.focal} (${t.hfov_deg}°×${t.vfov_deg}°),` +
    ` CAM_BACK f_t=${t.back_focal} (${t.back_hfov_deg}°, nuScenes-style wide)`;
}

function render() {
  for (const m of ['grid', 'overlay', 'processed']) {
    document.getElementById(m + '-view').style.display =
      STATE.mode === m ? '' : 'none';
    document.getElementById('mode-' + m).classList.toggle('active', STATE.mode === m);
  }
  document.getElementById('btn-ab').style.display =
    STATE.mode === 'overlay' ? '' : 'none';
  ['guides', 'strips', 'magnify'].forEach(k =>
    document.getElementById('chk-' + k + '-wrap').style.display =
      STATE.mode === 'grid' ? '' : 'none');
  document.getElementById('frame-idx').value = STATE.frame;
  if (STATE.mode === 'grid') renderGrid();
  else if (STATE.mode === 'overlay') renderOverlay();
  else renderProcessed();
  renderPitchInfo();
  renderMeta();
}

function loadFrame(idx) {
  STATE.frame = Math.max(0, Math.min(nFrames() - 1, idx));
  STATE.cardAB = {};
  render();
}

function loadClip(idx) {
  STATE.clip = Math.max(0, Math.min(STATE.clips.length - 1, idx));
  STATE.frame = 0;
  STATE.cardAB = {};
  document.getElementById('clip-select').value = STATE.clip;
  document.getElementById('frame-idx').max = nFrames() - 1;
  render();
}

function setMode(mode) { STATE.mode = mode; render(); }

// ---------------- rear extra-pitch controls ----------------
const pitchCam = () => document.getElementById('pitch-cam').value;

function renderPitchInfo() {
  const cams = STATE.clips[STATE.clip].cams;
  const bits = [];
  for (const cam of ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']) {
    const m = cams[cam];
    const black = m.black_top_rows + m.black_bottom_rows;
    bits.push(`${cam.replace('CAM_BACK', 'B').replace('B_', 'B')}` +
              ` ${m.axis_elevation_after_deg}°` +
              (black ? ` (black ${black}px)` : ''));
  }
  const el = document.getElementById('pitch-info');
  el.textContent = 'axes: ' + bits.join(' | ');
  const anyBlack = ['CAM_BACK_LEFT','CAM_BACK','CAM_BACK_RIGHT'].some(c =>
    cams[c].black_top_rows + cams[c].black_bottom_rows > 0);
  el.style.color = anyBlack ? '#ff6b6b' : '#888';
  document.getElementById('pitch-deg').value =
    STATE.pitchOffsets[pitchCam()] || 0;
}

function applyGeometry(res) {
  STATE.version = res.version;              // busts image URLs
  STATE.pitchOffsets = res.pitch_offsets || {};
  for (const c of res.clips) STATE.clips[c.idx].cams = c.cams;
  render();
}

async function postJson(path, body) {
  const r = await fetch(path, {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body || {}),
  });
  return r.json();
}

async function setPitch(deg) {
  deg = Math.max(-30, Math.min(30, deg));
  const res = await postJson('/api/pitch_offset', {cam: pitchCam(), deg: deg});
  if (res.ok) applyGeometry(res);
}

async function alignRear() {
  const res = await postJson('/api/align_rear');
  if (res.ok) applyGeometry(res);
}

async function resetPitch() {
  const res = await postJson('/api/reset_pitch');
  if (res.ok) applyGeometry(res);
}

async function init() {
  const info = await api('/api/info');
  STATE.clips = info.clips;
  STATE.target = info.target;
  STATE.version = info.version;

  const sel = document.getElementById('clip-select');
  sel.innerHTML = STATE.clips.map((c, i) =>
    `<option value="${i}">${c.name} (${c.n_frames} frames)</option>`).join('');
  sel.onchange = () => loadClip(parseInt(sel.value));

  document.getElementById('prev-clip').onclick = () => loadClip(STATE.clip - 1);
  document.getElementById('next-clip').onclick = () => loadClip(STATE.clip + 1);
  document.getElementById('prev').onclick = () => loadFrame(STATE.frame - stride());
  document.getElementById('next').onclick = () => loadFrame(STATE.frame + stride());
  document.getElementById('go').onclick = () =>
    loadFrame(parseInt(document.getElementById('frame-idx').value) || 0);
  document.getElementById('frame-idx').addEventListener('keydown', e => {
    if (e.key === 'Enter') { loadFrame(parseInt(e.target.value) || 0); e.target.blur(); }
  });
  document.getElementById('mode-grid').onclick = () => setMode('grid');
  document.getElementById('mode-overlay').onclick = () => setMode('overlay');
  document.getElementById('mode-processed').onclick = () => setMode('processed');
  STATE.pitchOffsets = info.pitch_offsets || {};
  const pitchInput = document.getElementById('pitch-deg');
  pitchInput.value = STATE.pitchOffsets[pitchCam()] || 0;
  document.getElementById('pitch-up').onclick = () => setPitch(parseFloat(pitchInput.value || 0) + 1);
  document.getElementById('pitch-dn').onclick = () => setPitch(parseFloat(pitchInput.value || 0) - 1);
  document.getElementById('pitch-align').onclick = alignRear;
  document.getElementById('pitch-reset').onclick = resetPitch;
  document.getElementById('pitch-cam').onchange = () => {
    pitchInput.value = STATE.pitchOffsets[pitchCam()] || 0;
  };
  pitchInput.addEventListener('keydown', e => {
    if (e.key === 'Enter') { setPitch(parseFloat(e.target.value) || 0); e.target.blur(); }
  });
  document.getElementById('btn-ab').onclick = toggleABAll;
  document.getElementById('chk-guides').onchange = e => { STATE.guides = e.target.checked; render(); };
  document.getElementById('chk-strips').onchange = e => { STATE.strips = e.target.checked; render(); };
  document.getElementById('chk-magnify').onchange = e => { STATE.magnify = e.target.checked; render(); };

  document.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    if (e.key === 'ArrowLeft') loadFrame(STATE.frame - stride());
    else if (e.key === 'ArrowRight') loadFrame(STATE.frame + stride());
    else if (e.key === '[') loadClip(STATE.clip - 1);
    else if (e.key === ']') loadClip(STATE.clip + 1);
    else if (e.key === 'g') setMode('grid');
    else if (e.key === 'o') setMode('overlay');
    else if (e.key === 'p') setMode('processed');
    else if (e.key === 'b') toggleABAll();
    else if (e.key === 'l') { const c = document.getElementById('chk-guides'); c.checked = !c.checked; STATE.guides = c.checked; render(); }
    else if (e.key === 'e') { const c = document.getElementById('chk-strips'); c.checked = !c.checked; STATE.strips = c.checked; render(); }
    else if (e.key === 'm') { const c = document.getElementById('chk-magnify'); c.checked = !c.checked; STATE.magnify = c.checked; render(); }
    else if (e.key === 'k') { const c = document.getElementById('keyframes'); c.checked = !c.checked; }
  });

  loadClip(0);
}
init();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template_string(
        HTML_TEMPLATE
        .replace('{{ camera_order }}', json.dumps(CAMERA_ORDER))
        .replace('{{ rear_cams }}', json.dumps(sorted(REAR_CAMERAS)))
        .replace('{{ cam_display }}', json.dumps(CAM_DISPLAY))
        .replace('{{ keyframe_stride }}', str(_config['keyframe_stride'])))


def _geometry_version() -> str:
    """Short hash of everything that affects rendered pixels. Appended to
    image URLs (?v=...) so the browser cache is busted whenever the target
    camera or the pitch-leveling values change — otherwise tiles cached
    under an older geometry silently mix with fresh ones."""
    import hashlib
    t = _config['target']
    key = json.dumps(['warp-v2-homography',   # bump on algorithm changes
                      t.width, t.height, t.focal,
                      _target('CAM_BACK').focal,
                      sorted(_config['elevations'].items()),
                      sorted(_config['pitch_offsets'].items()),
                      sorted(_config['bottom_scales'].items()),
                      sorted(_config['pin_tops']),
                      _config['jpeg_quality'],
                      _config['orig_display_width']], sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()[:10]


@app.route('/api/info')
def api_info():
    t = _config['target']
    return jsonify({
        'version': _geometry_version(),
        'target': {
            'width': t.width, 'height': t.height, 'focal': t.focal,
            'hfov_deg': round(t.hfov_deg, 1), 'vfov_deg': round(t.vfov_deg, 1),
            'back_focal': round(_target('CAM_BACK').focal, 1),
            'back_hfov_deg': round(_target('CAM_BACK').hfov_deg, 1),
        },
        'pitch_offsets': _config['pitch_offsets'],
        'bottom_scales': _config['bottom_scales'],
        'pin_tops': sorted(_config['pin_tops']),
        'pin_tops': sorted(_config['pin_tops']),
        'clips': [{
            'idx': i,
            'name': c['name'],
            'n_frames': len(c['timestamps']),
            'cams': c['meta'],
        } for i, c in enumerate(_clips)],
    })


@app.route('/api/pitch_offset', methods=['POST'])
def api_pitch_offset():
    """Live-set a camera's extra up-pitch (deg). Recomputes crop metadata,
    clears the render cache, and returns the new geometry version + metadata
    so the UI can refresh with the changed images."""
    data = request.get_json(force=True)
    cam = data.get('cam')
    try:
        deg = float(data.get('deg', 0.0))
    except (TypeError, ValueError):
        return jsonify({'ok': False, 'error': 'deg must be a number'}), 400
    if cam not in CAMERA_ORDER:
        return jsonify({'ok': False, 'error': f'cam must be one of {CAMERA_ORDER}'}), 400
    deg = max(-30.0, min(30.0, deg))
    if deg:
        _config['pitch_offsets'][cam] = deg
    else:
        _config['pitch_offsets'].pop(cam, None)
    return jsonify({'ok': True, 'cam': cam, 'deg': deg,
                    **_apply_geometry_change()})


def _apply_geometry_change() -> dict:
    """Recompute per-clip metadata, drop cached renders, and return the
    payload common to geometry-mutating endpoints."""
    for clip in _clips:
        clip['meta'] = {c: crop_metadata(clip['calibs'][c], _target(c),
                                         _elev(c), _pitch(c), _bscale(c),
                                         _pintop(c))
                        for c in CAMERA_ORDER}
    with _cache_lock:
        _cache.clear()
    return {
        'version': _geometry_version(),
        'pitch_offsets': _config['pitch_offsets'],
        'bottom_scales': _config['bottom_scales'],
        'pin_tops': sorted(_config['pin_tops']),
        'bottom_scales': _config['bottom_scales'],
        'pin_tops': sorted(_config['pin_tops']),
        'clips': [{'idx': i, 'name': c['name'],
                   'n_frames': len(c['timestamps']), 'cams': c['meta']}
                  for i, c in enumerate(_clips)],
    }


@app.route('/api/bottom_scale', methods=['POST'])
def api_bottom_scale():
    """Live-set a camera's keystone bottom-edge scale (0.5..1.0)."""
    data = request.get_json(force=True)
    cam = data.get('cam')
    try:
        scale = float(data.get('scale', 1.0))
    except (TypeError, ValueError):
        return jsonify({'ok': False, 'error': 'scale must be a number'}), 400
    if cam not in CAMERA_ORDER:
        return jsonify({'ok': False, 'error': f'cam must be one of {CAMERA_ORDER}'}), 400
    scale = max(0.5, min(1.0, scale))
    if abs(scale - 1.0) > 1e-6:
        _config['bottom_scales'][cam] = scale
    else:
        _config['bottom_scales'].pop(cam, None)
    return jsonify({'ok': True, 'cam': cam, 'scale': scale,
                    **_apply_geometry_change()})


@app.route('/api/pin_top', methods=['POST'])
def api_pin_top():
    """Live-toggle pinning a camera's footprint top edge to the source top
    border (corners (0,0)-(W,0)); part of the keystone warp."""
    data = request.get_json(force=True)
    cam = data.get('cam')
    on = bool(data.get('on', True))
    if cam not in CAMERA_ORDER:
        return jsonify({'ok': False, 'error': f'cam must be one of {CAMERA_ORDER}'}), 400
    if on:
        _config['pin_tops'].add(cam)
    else:
        _config['pin_tops'].discard(cam)
    return jsonify({'ok': True, 'cam': cam, 'on': on,
                    **_apply_geometry_change()})


@app.route('/api/align_rear', methods=['POST'])
def api_align_rear():
    """Tilt CAM_BACK_LEFT / CAM_BACK_RIGHT down to match CAM_BACK's current
    virtual-axis elevation (the user's 'align sides to rear-center' option).
    Each side's extra pitch is CLAMPED to its own data boundary, so the
    sides never gain black borders; any leftover mismatch is reported."""
    calibs = _clips[0]['calibs']   # calibs are rig-constant across clips
    back_axis = axis_elevation_after(
        calibs['CAM_BACK'], _target('CAM_BACK'),
        _elev('CAM_BACK'), _pitch('CAM_BACK'))
    result = {}
    for cam in ('CAM_BACK_LEFT', 'CAM_BACK_RIGHT'):
        # After auto-leveling the side sits at axis_elevation_after(extra=0);
        # the extra needed to reach the rear-center's axis is the difference.
        base_axis = axis_elevation_after(calibs[cam], _target(cam),
                                         _elev(cam), 0.0)
        desired = back_axis - base_axis
        applied = clamp_extra_pitch(calibs[cam], _target(cam), _elev(cam),
                                    desired)
        if abs(applied) > 1e-6:
            _config['pitch_offsets'][cam] = round(applied, 2)
        else:
            _config['pitch_offsets'].pop(cam, None)
        result[cam] = {'desired_deg': round(desired, 2),
                       'applied_deg': round(applied, 2),
                       'mismatch_deg': round(desired - applied, 2)}
    return jsonify({'ok': True, 'back_axis_deg': round(back_axis, 2),
                    'aligned': result, **_apply_geometry_change()})


@app.route('/api/reset_pitch', methods=['POST'])
def api_reset_pitch():
    """Clear every extra pitch offset (back to pure auto-leveling)."""
    _config['pitch_offsets'].clear()
    return jsonify({'ok': True, **_apply_geometry_change()})


@app.route('/api/frame')
def api_frame():
    try:
        clip_idx = int(request.args.get('clip', 0))
        frame_idx = int(request.args.get('frame', 0))
    except ValueError:
        clip_idx, frame_idx = 0, 0
    clip_idx = max(0, min(len(_clips) - 1, clip_idx))
    clip = _clips[clip_idx]
    frame_idx = max(0, min(len(clip['timestamps']) - 1, frame_idx))
    ts = clip['timestamps'][frame_idx]
    return jsonify({
        'clip': clip_idx,
        'frame': frame_idx,
        'timestamp': ts,
        't_offset_s': (int(ts) - int(clip['timestamps'][0])) / 1e6,
    })


@app.route('/img/<int:clip_idx>/<int:frame_idx>/<cam>/<mode>')
def img_route(clip_idx: int, frame_idx: int, cam: str, mode: str):
    if not (0 <= clip_idx < len(_clips)):
        return jsonify({'error': 'clip out of range'}), 404
    clip = _clips[clip_idx]
    if not (0 <= frame_idx < len(clip['timestamps'])):
        return jsonify({'error': 'frame out of range'}), 404
    if cam not in CAMERA_ORDER or mode not in IMG_MODES:
        return jsonify({'error': f'cam must be one of {CAMERA_ORDER}, '
                                 f'mode one of {IMG_MODES}'}), 404
    data = render_image(clip_idx, frame_idx, cam, mode)
    # Content per URL is immutable (crops are fixed at startup) — let the
    # browser cache aggressively so navigation back/forth is instant.
    return Response(data, mimetype='image/jpeg',
                    headers={'Cache-Control': 'public, max-age=86400'})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Web visualizer to verify virtual-camera crops for the '
                    'custom demo dataset')
    parser.add_argument('--dataset_root', type=str,
                        default='/yjj_rnd_home/data/vlm_dataset/demo_dataset')
    parser.add_argument('--target_width', type=int, default=DEFAULT_TARGET.width)
    parser.add_argument('--target_height', type=int, default=DEFAULT_TARGET.height)
    parser.add_argument('--target_focal', type=float, default=DEFAULT_TARGET.focal)
    parser.add_argument('--back_focal', type=float, default=None,
                        help='Explicit CAM_BACK virtual focal. Default: '
                             'target_focal x 0.639 (the nuScenes back/front '
                             'ratio -> wide rear view). Pass the same value '
                             'as --target_focal for a uniform-FOV rig.')
    parser.add_argument('--back_pitch_offset', type=float, default=0.0,
                        help='Startup extra up-pitch (deg) for CAM_BACK '
                             'beyond auto-leveling; also live-adjustable in '
                             'the UI. Past the data boundary (+13 deg '
                             'elevation) the top of the view renders black.')
    parser.add_argument('--keyframe_stride', type=int, default=5,
                        help='Default frame step in the UI (10 Hz / 5 = 2 Hz)')
    parser.add_argument('--orientation_json', type=str, default=ORIENTATION_JSON,
                        help='camera_orientation.json with measured axis '
                             'elevations for pitch leveling (from '
                             'measure_camera_orientation.py); pass a '
                             'non-existent path to disable the correction')
    parser.add_argument('--jpeg_quality', type=int, default=90)
    parser.add_argument('--cache_size', type=int, default=240,
                        help='Max processed images kept in the LRU cache')
    parser.add_argument('--port', type=int, default=6063)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    args = parser.parse_args()

    _config['dataset_root'] = args.dataset_root
    _config['jpeg_quality'] = args.jpeg_quality
    _config['cache_size'] = args.cache_size
    _config['keyframe_stride'] = args.keyframe_stride
    _config['target'] = TargetCamera(
        width=args.target_width, height=args.target_height,
        focal=args.target_focal)
    _config['back_focal'] = args.back_focal
    # Frozen rig: when custom_dataset/virtual_rig.json exists, its
    # frozen_config becomes the startup state (CLI flags still override).
    rig_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'virtual_rig.json')
    if os.path.isfile(rig_path):
        with open(rig_path) as f:
            rig = json.load(f).get('frozen_config', {})
        _config['pitch_offsets'].update(rig.get('pitch_offsets_deg', {}))
        _config['bottom_scales'].update(rig.get('bottom_scales', {}))
        _config['pin_tops'].update(rig.get('pin_tops', []))
        if _config['back_focal'] is None:
            _config['back_focal'] = rig.get('back_focal')
        print(f'Loaded frozen rig from {rig_path}: '
              f'pitch={_config["pitch_offsets"]} '
              f'bottom_scales={_config["bottom_scales"]} '
              f'pin_tops={sorted(_config["pin_tops"])}')
    if args.back_pitch_offset:
        _config['pitch_offsets']['CAM_BACK'] = args.back_pitch_offset
    _config['elevations'] = load_camera_elevations(args.orientation_json)
    if _config['elevations']:
        print(f'Pitch leveling ON (from {args.orientation_json}): '
              + ', '.join(f'{c}={e:+.1f}°'
                          for c, e in _config['elevations'].items()))
    else:
        print('Pitch leveling OFF — no camera_orientation.json found; run '
              'python -m custom_dataset.measure_camera_orientation to enable')

    found = discover_clips(args.dataset_root)
    if not found:
        print(f'ERROR: no clip directories (with images/ and maps/calib_tuned/) '
              f'under {args.dataset_root}', file=sys.stderr)
        sys.exit(1)
    print(f'Found {len(found)} clip(s) under {args.dataset_root}')
    for name, path in found:
        _clips.append(build_clip(name, path))

    verify_feasibility()

    t = _config['target']
    tb = _target('CAM_BACK')
    print(f'Target virtual camera: {t.width}x{t.height}  f_t={t.focal}  '
          f'({t.hfov_deg:.1f}° x {t.vfov_deg:.1f}°)')
    print(f'  CAM_BACK (nuScenes-style wide): f_t={tb.focal:.1f}  '
          f'({tb.hfov_deg:.1f}° x {tb.vfov_deg:.1f}°)')
    for clip in _clips:
        print(f"[{clip['name']}]  {len(clip['timestamps'])} frames")
        for cam in CAMERA_ORDER:
            m = clip['meta'][cam]
            tilt = (f"  pitch {m['pitch_correction_applied_deg']:+.1f}°"
                    + (f" (residual {m['residual_tilt_deg']:+.1f}°)"
                       if abs(m['residual_tilt_deg']) > 0.05 else ''))
            print(f"  {cam:<16s} {m['channel']:<28s} "
                  f"crop {m['crop_size'][0]:.0f}x{m['crop_size'][1]:.0f} "
                  f"@ [{m['crop_box'][0]:.0f},{m['crop_box'][1]:.0f},"
                  f"{m['crop_box'][2]:.0f},{m['crop_box'][3]:.0f}]  "
                  f"retained {m['retained_ratio'][0]:.2f}/{m['retained_ratio'][1]:.2f}"
                  f"{tilt}")
    print(f'Starting server on http://{args.host}:{args.port}')
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
