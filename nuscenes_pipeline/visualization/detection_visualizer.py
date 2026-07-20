"""
Traffic Light & Pole Detection Visualizer

Web-based dashboard (based on qa_visualizer.py) for verifying the Stage-1D
traffic_light_pole_detection outputs against the other Stage-1 prior analyses.
Browses by nuScenes sample_idx and provides on/off switches for every input:

  - Traffic light boxes : detected 2D bboxes drawn on the 6-view images
                          (colored by detected state) + 3D boxes on the BEV
  - Poles               : detected pole/mast-arm line segments drawn on the
                          6-view images + base positions on the BEV
  - Risk analysis       : Stage 1A response text (risk_assessment_results/)
  - Traffic signal      : Stage 1B response text (traffic_signal_analysis_results/)
  - Traffic sign        : Stage 1C response text (traffic_sign_results/)

Rear-camera images are flipped for display (consistent with the other tools);
detection coordinates, which live in the unflipped image space, are mirrored
before drawing so the overlays stay correct.

Usage:
    python -m nuscenes_pipeline.visualization.detection_visualizer --port 6061
    # Then open http://<server-ip>:6061 in browser
"""

import os
import io
import json
import glob
import base64
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
from flask import Flask, render_template_string, request, jsonify
from PIL import Image, ImageDraw, ImageFont

from nuscenes_pipeline.core.nuscenes_data_loader import NuScenesDataLoader
from nuscenes_pipeline.visualization.qa_visualizer import generate_bev, image_to_base64
from nuscenes_pipeline.visualization.detection_drawing import (
    CAMERA_ORDER, REAR_CAMERAS, STATE_COLORS, POLE_COLORS,
    draw_traffic_lights, draw_poles, make_detection_overlay_fn,
)

PKL_PATH = os.environ.get("NUSCENES_PKL_PATH", "nuscenes2d_ego_temporal_infos_val.pkl")
DETECTION_DIR = "traffic_light_pole_3d_results"
RISK_DIR = "risk_assessment_results"
SIGNAL_DIR = "traffic_signal_analysis_results"
SIGN_DIR = "traffic_sign_results"


# ===========================================================================
# Result-file loading
# ===========================================================================

def load_detection_result(sample_idx):
    """Load Stage-1D detection JSON: {idx:04d}_{token}.json"""
    files = sorted(glob.glob(os.path.join(DETECTION_DIR, f"{sample_idx:04d}_*.json")))
    if not files:
        return None
    with open(files[0]) as f:
        return json.load(f)


def load_prior_response(results_dir, sample_idx, suffix):
    """Load a Stage-1A/1B/1C prior response: {idx:04d}_*_{suffix}.json -> response text."""
    files = sorted(glob.glob(os.path.join(results_dir, f"{sample_idx:04d}_*{suffix}.json")))
    if not files:
        return None
    try:
        with open(files[0]) as f:
            return json.load(f).get('response')
    except Exception as e:
        return f"(failed to read {files[0]}: {e})"


# ===========================================================================
# Flask app
# ===========================================================================

app = Flask(__name__)
_loader = None


def get_loader():
    global _loader
    if _loader is None:
        _loader = NuScenesDataLoader(pkl_path=PKL_PATH)
    return _loader


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/info')
def api_info():
    loader = get_loader()
    return jsonify({"total": len(loader)})


@app.route('/api/sample')
def api_sample():
    sample_idx = int(request.args.get('idx', 0))
    show_lights = request.args.get('lights', '1') == '1'
    show_poles = request.args.get('poles', '1') == '1'

    loader = get_loader()
    if sample_idx < 0 or sample_idx >= len(loader):
        return jsonify({"error": f"Invalid sample_idx {sample_idx} (0..{len(loader)-1})"})

    sample = loader.get_sample(sample_idx)
    detection = load_detection_result(sample_idx)
    lights = (detection or {}).get('traffic_lights', [])
    poles = (detection or {}).get('poles', [])

    lights_by_cam, poles_by_cam = {}, {}
    for tl in lights:
        lights_by_cam.setdefault(tl.get('camera'), []).append(tl)
    for p in poles:
        poles_by_cam.setdefault(p.get('camera'), []).append(p)

    images_b64 = []
    for cam_name in CAMERA_ORDER:
        try:
            img = Image.open(sample.cameras[cam_name].image_path).convert('RGB')
            full_w, full_h = img.size
            img = img.resize((full_w // 2, full_h // 2), Image.LANCZOS)
            is_rear = cam_name in REAR_CAMERAS
            if is_rear:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            scale = img.size[0] / full_w
            mirror_w = full_w if is_rear else None
            if show_lights and cam_name in lights_by_cam:
                img = draw_traffic_lights(img, lights_by_cam[cam_name], scale, mirror_w)
            if show_poles and cam_name in poles_by_cam:
                img = draw_poles(img, poles_by_cam[cam_name], scale, mirror_w)
            images_b64.append(image_to_base64(img))
        except Exception as e:
            ph = Image.new('RGB', (400, 225), (40, 40, 40))
            ImageDraw.Draw(ph).text((10, 100), str(e)[:60], fill=(255, 0, 0))
            images_b64.append(image_to_base64(ph))

    # BEV with detection overlay
    bev_b64 = ""
    try:
        overlay = make_detection_overlay_fn(lights, poles, show_lights, show_poles) \
            if detection else None
        bev_b64 = generate_bev(sample, loader, bev_range=50.0, overlay_fn=overlay)
    except Exception as e:
        print(f"BEV error: {e}")

    # Detection summary table rows
    det_rows = []
    for tl in lights:
        b3d = tl.get('bbox_3d') or {}
        det_rows.append({
            "id": tl.get('id'), "kind": "light", "camera": tl.get('camera'),
            "state": tl.get('state'), "facing": tl.get('facing'),
            "center": b3d.get('center'), "size": b3d.get('size'),
            "depth": b3d.get('depth_m'), "source": b3d.get('depth_source'),
        })
    for p in poles:
        l3d = p.get('line_3d') or {}
        det_rows.append({
            "id": p.get('id'), "kind": p.get('segment_type', 'pole'), "camera": p.get('camera'),
            "state": "-", "facing": "-",
            "center": l3d.get('bottom'), "size": l3d.get('top'),
            "depth": l3d.get('depth_m'), "source": l3d.get('depth_source'),
        })

    return jsonify({
        "images": images_b64,
        "bev": bev_b64,
        "token": sample.token,
        "scene_token": sample.scene_token,
        "location": sample.location,
        "description": sample.description,
        "detection_found": detection is not None,
        "num_lights": len(lights),
        "num_poles": len(poles),
        "det_rows": det_rows,
        "risk_text": load_prior_response(RISK_DIR, sample_idx, "_single_frame"),
        "signal_text": load_prior_response(SIGNAL_DIR, sample_idx, "_traffic_signal"),
        "sign_text": load_prior_response(SIGN_DIR, sample_idx, "_sign"),
    })


# ===========================================================================
# HTML Template
# ===========================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Traffic Light & Pole Detection Visualizer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, sans-serif; background: #1a1a2e; color: #e0e0e0; }
        .header {
            background: #16213e; padding: 10px 20px;
            display: flex; align-items: center; gap: 16px;
            border-bottom: 2px solid #0f3460; flex-wrap: wrap;
        }
        .header h1 { font-size: 17px; color: #53d8fb; }
        .controls { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
        .controls select, .controls input[type=number], .controls button {
            padding: 5px 10px; border-radius: 4px; border: 1px solid #0f3460;
            background: #1a1a2e; color: #e0e0e0; font-size: 13px;
        }
        .controls button { background: #0f3460; cursor: pointer; font-weight: bold; }
        .controls button:hover { background: #53d8fb; color: #1a1a2e; }
        .controls label { font-size: 12px; color: #aaa; }
        .toggles { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
        .toggle {
            display: flex; align-items: center; gap: 4px; font-size: 12px;
            background: #0f3460; padding: 3px 8px; border-radius: 10px; cursor: pointer;
            user-select: none;
        }
        .toggle input { accent-color: #53d8fb; cursor: pointer; }
        .info-badge { background: #0f3460; padding: 3px 8px; border-radius: 10px; font-size: 11px; color: #53d8fb; }
        .main { display: flex; height: calc(100vh - 52px); }

        /* Left: images */
        .left-panel { flex: 3; padding: 6px; display: flex; flex-direction: column; gap: 3px; }
        .image-grid {
            display: grid; grid-template-columns: 1fr 1fr 1fr; grid-template-rows: 1fr 1fr;
            gap: 3px; flex: 1;
        }
        .image-cell { position: relative; background: #111; border-radius: 3px; overflow: hidden; }
        .image-cell img { width: 100%; height: 100%; object-fit: contain; }
        .image-cell .cam-label {
            position: absolute; top: 3px; left: 3px; background: rgba(0,0,0,0.75);
            color: #53d8fb; padding: 1px 6px; border-radius: 2px; font-size: 11px; font-weight: bold;
        }

        /* Right: panels */
        .right-panel { flex: 2; display: flex; flex-direction: column; padding: 6px; gap: 6px; overflow-y: auto; }
        .panel {
            background: #16213e; border-radius: 5px; padding: 10px;
            border: 1px solid #0f3460; overflow-y: auto;
        }
        .panel h3 { font-size: 13px; margin-bottom: 6px; display: flex; align-items: center; gap: 6px; }
        .badge { color: #fff; padding: 1px 6px; border-radius: 3px; font-size: 10px; }
        .badge-det { background: #3498db; } .badge-risk { background: #e67e22; }
        .badge-sig { background: #2ecc71; } .badge-sign { background: #9b59b6; }
        .badge-bev { background: #34495e; }
        .panel-text { font-size: 12px; line-height: 1.5; white-space: pre-wrap; word-break: break-word;
                      max-height: 260px; overflow-y: auto; }
        .panel.hidden { display: none; }
        .missing { color: #777; font-style: italic; font-size: 12px; }
        table.det { width: 100%; border-collapse: collapse; font-size: 11px; }
        table.det th, table.det td { border: 1px solid #0f3460; padding: 2px 5px; text-align: left; }
        table.det th { color: #53d8fb; background: #101a33; }
        .bev-section { display: flex; flex-direction: column; align-items: center; }
        .bev-section img { max-width: 100%; object-fit: contain; }
        .legend { display: flex; gap: 10px; font-size: 11px; flex-wrap: wrap; }
        .legend-item { display: flex; align-items: center; gap: 3px; }
        .legend-box { width: 12px; height: 12px; border-radius: 2px; }
        .loading { color: #555; font-style: italic; }
        .meta { font-size: 11px; color: #888; }
    </style>
</head>
<body>
    <div class="header">
        <h1>TL &amp; Pole Detection Visualizer</h1>
        <div class="controls">
            <span id="total-badge" class="info-badge">--</span>
            <label>Sample idx:</label>
            <input type="number" id="sample-id" value="0" min="0" style="width:80px"
                   onkeypress="if(event.key==='Enter') loadSample()">
            <button onclick="loadSample()">Go</button>
            <button onclick="navigate(-10)">&laquo;</button>
            <button onclick="navigate(-1)">&larr;</button>
            <button onclick="navigate(1)">&rarr;</button>
            <button onclick="navigate(10)">&raquo;</button>
            <button onclick="randomSample()" style="background:#533d8f">Rand</button>
        </div>
        <div class="toggles">
            <label class="toggle"><input type="checkbox" id="tg-lights" checked onchange="loadSample()"> TL boxes</label>
            <label class="toggle"><input type="checkbox" id="tg-poles" checked onchange="loadSample()"> Poles</label>
            <label class="toggle"><input type="checkbox" id="tg-risk" checked onchange="applyPanelToggles()"> Risk</label>
            <label class="toggle"><input type="checkbox" id="tg-signal" checked onchange="applyPanelToggles()"> Signal</label>
            <label class="toggle"><input type="checkbox" id="tg-sign" checked onchange="applyPanelToggles()"> Sign</label>
        </div>
        <div class="legend">
            <div class="legend-item"><div class="legend-box" style="background:#e74c3c"></div>red</div>
            <div class="legend-item"><div class="legend-box" style="background:#f1c40f"></div>yellow</div>
            <div class="legend-item"><div class="legend-box" style="background:#2ecc71"></div>green</div>
            <div class="legend-item"><div class="legend-box" style="background:#3498db"></div>unknown</div>
            <div class="legend-item"><div class="legend-box" style="background:#ffffff"></div>pole</div>
            <div class="legend-item"><div class="legend-box" style="background:#ff00ff"></div>mast arm</div>
            <span id="det-counts" class="meta"></span>
        </div>
    </div>
    <div class="main">
        <div class="left-panel">
            <div class="image-grid">
                <div class="image-cell"><div class="cam-label">1: Front-left</div><img id="img-0"></div>
                <div class="image-cell"><div class="cam-label">2: Front</div><img id="img-1"></div>
                <div class="image-cell"><div class="cam-label">3: Front-right</div><img id="img-2"></div>
                <div class="image-cell"><div class="cam-label">4: Rear-left (flip)</div><img id="img-3"></div>
                <div class="image-cell"><div class="cam-label">5: Rear (flip)</div><img id="img-4"></div>
                <div class="image-cell"><div class="cam-label">6: Rear-right (flip)</div><img id="img-5"></div>
            </div>
            <div class="meta" id="sample-meta"></div>
        </div>
        <div class="right-panel">
            <div class="panel" id="panel-det">
                <h3><span class="badge badge-det">DET</span> Detections (3D, ego FLU)</h3>
                <div id="det-table"><span class="loading">Select a sample...</span></div>
            </div>
            <div class="panel bev-section" id="panel-bev">
                <h3><span class="badge badge-bev">BEV</span> Bird's Eye View (GT + detections)</h3>
                <img id="bev-img">
            </div>
            <div class="panel" id="panel-risk">
                <h3><span class="badge badge-risk">1A</span> Risk Assessment</h3>
                <div class="panel-text" id="risk-text"></div>
            </div>
            <div class="panel" id="panel-signal">
                <h3><span class="badge badge-sig">1B</span> Traffic Signal Analysis</h3>
                <div class="panel-text" id="signal-text"></div>
            </div>
            <div class="panel" id="panel-sign">
                <h3><span class="badge badge-sign">1C</span> Traffic Sign Extraction</h3>
                <div class="panel-text" id="sign-text"></div>
            </div>
        </div>
    </div>
    <script>
        let totalSamples = 0;
        async function init() {
            const r = await fetch('/api/info');
            const d = await r.json();
            totalSamples = d.total;
            document.getElementById('total-badge').textContent = totalSamples.toLocaleString()+' samples';
            document.getElementById('sample-id').max = totalSamples - 1;
            loadSample();
        }
        function applyPanelToggles() {
            document.getElementById('panel-risk').classList.toggle('hidden', !document.getElementById('tg-risk').checked);
            document.getElementById('panel-signal').classList.toggle('hidden', !document.getElementById('tg-signal').checked);
            document.getElementById('panel-sign').classList.toggle('hidden', !document.getElementById('tg-sign').checked);
        }
        function esc(s) {
            return (s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
        }
        async function loadSample() {
            const id = parseInt(document.getElementById('sample-id').value);
            if (isNaN(id) || id<0 || id>=totalSamples) return;
            const lights = document.getElementById('tg-lights').checked ? 1 : 0;
            const poles = document.getElementById('tg-poles').checked ? 1 : 0;
            document.getElementById('det-table').innerHTML = '<span class="loading">Loading...</span>';
            const r = await fetch(`/api/sample?idx=${id}&lights=${lights}&poles=${poles}`);
            const d = await r.json();
            if (d.error) { document.getElementById('det-table').textContent = d.error; return; }
            for (let i=0;i<6;i++) document.getElementById('img-'+i).src = d.images[i];
            if (d.bev) document.getElementById('bev-img').src = d.bev;
            document.getElementById('sample-meta').textContent =
                `token=${d.token} | scene=${d.scene_token} | ${d.location} | ${d.description ?? ''}`;
            document.getElementById('det-counts').textContent =
                d.detection_found ? `lights: ${d.num_lights} | poles: ${d.num_poles}` : 'no detection file';

            if (!d.detection_found) {
                document.getElementById('det-table').innerHTML =
                    '<span class="missing">No detection result for this sample — run traffic_light_pole_detection first.</span>';
            } else if (d.det_rows.length === 0) {
                document.getElementById('det-table').innerHTML = '<span class="missing">No traffic lights / poles detected.</span>';
            } else {
                let html = '<table class="det"><tr><th>id</th><th>kind</th><th>camera</th><th>state</th>' +
                           '<th>center/bottom [x,y,z]</th><th>size/top</th><th>depth</th><th>source</th></tr>';
                for (const row of d.det_rows) {
                    const fmt = v => Array.isArray(v) ? '['+v.map(x=>(+x).toFixed(1)).join(', ')+']' : (v ?? '-');
                    html += `<tr><td>${esc(row.id)}</td><td>${esc(row.kind)}</td><td>${esc((row.camera||'').replace('CAM_',''))}</td>` +
                            `<td>${esc(row.state)}</td><td>${fmt(row.center)}</td><td>${fmt(row.size)}</td>` +
                            `<td>${row.depth ?? '-'}</td><td>${esc(row.source ?? '-')}</td></tr>`;
                }
                document.getElementById('det-table').innerHTML = html + '</table>';
            }

            const setText = (elId, text) => {
                document.getElementById(elId).innerHTML =
                    text ? esc(text) : '<span class="missing">not found for this sample</span>';
            };
            setText('risk-text', d.risk_text);
            setText('signal-text', d.signal_text);
            setText('sign-text', d.sign_text);
            applyPanelToggles();
        }
        function navigate(delta) {
            const inp = document.getElementById('sample-id');
            let id = parseInt(inp.value)+delta;
            if (id<0) id=totalSamples-1; if (id>=totalSamples) id=0;
            inp.value=id; loadSample();
        }
        function randomSample() {
            document.getElementById('sample-id').value = Math.floor(Math.random()*totalSamples);
            loadSample();
        }
        document.addEventListener('keydown', e => {
            if (e.target.tagName==='INPUT') return;
            if (e.key==='ArrowLeft') navigate(-1);
            if (e.key==='ArrowRight') navigate(1);
            if (e.key==='r') randomSample();
        });
        init();
    </script>
</body>
</html>
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Traffic Light & Pole Detection Visualizer")
    parser.add_argument('--port', type=int, default=6061)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--pkl_path', type=str, default=PKL_PATH)
    parser.add_argument('--detection_dir', type=str, default=DETECTION_DIR,
                        help="Stage 1D output dir (traffic_light_pole_detection)")
    parser.add_argument('--risk_dir', type=str, default=RISK_DIR, help="Stage 1A output dir")
    parser.add_argument('--signal_dir', type=str, default=SIGNAL_DIR, help="Stage 1B output dir")
    parser.add_argument('--sign_dir', type=str, default=SIGN_DIR, help="Stage 1C output dir")
    args = parser.parse_args()
    PKL_PATH = args.pkl_path
    DETECTION_DIR = args.detection_dir
    RISK_DIR = args.risk_dir
    SIGNAL_DIR = args.signal_dir
    SIGN_DIR = args.sign_dir

    print("Traffic Light & Pole Detection Visualizer")
    print(f"  pkl:           {PKL_PATH}")
    print(f"  detection dir: {DETECTION_DIR}")
    print(f"  risk dir:      {RISK_DIR}")
    print(f"  signal dir:    {SIGNAL_DIR}")
    print(f"  sign dir:      {SIGN_DIR}")
    print("  Loading nuScenes...")
    get_loader()
    print(f"  Ready! Open http://<server-ip>:{args.port}")
    print("  Keys: Left/Right arrows, 'r' for random")
    app.run(host=args.host, port=args.port, debug=False)
