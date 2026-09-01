"""
Traffic Signal Status Visualizer

Web dashboard (based on detection_visualizer.py) for verifying the VLM
traffic-signal status labels stored in the infos pkls by
add_traffic_lights_to_infos.py + apply_traffic_signal_status.py.

Everything shown is read from the pkl itself (tl_bboxes2d / tl_pole_bboxes2d /
tl_pole_idx2d / tl_signal_type2d / tl_light_observable2d / tl_light_color2d /
tl_lit_shape2d / tl_status_conf2d) — so what you see is exactly what training
would consume. The per-signal table also shows the crop image the classifier
graded (from cropped_ts_p/), for eyeballing wrong labels.

  - Signal boxes  : colored by identified light_color (red/yellow/green/off),
                    gray = light not observable, dim magenta = not_a_signal
  - Pole boxes    : blue; optional signal->pole association lines
  - Split switch  : train / val pkl in one server
  - Rear cameras  : flipped for display (like the other tools); box coords are
                    mirrored accordingly

Usage:
    python -m nuscenes_pipeline.visualization.traffic_signal_status_visualizer --port 6060
    # Then open http://<server-ip>:6060 in browser
"""

import argparse
import os
import pickle

from flask import Flask, render_template_string, request, jsonify
from PIL import Image, ImageDraw

from nuscenes_pipeline.visualization.qa_visualizer import image_to_base64
from nuscenes_pipeline.visualization.detection_drawing import CAMERA_ORDER, REAR_CAMERAS

PKL_PATHS = {
    "train": "data/nuscenes/nuscenes2d_ego_temporal_infos_train.pkl",
    "val": "data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl",
}
CROPS_DIR = "./cropped_ts_p"

COLOR_BY_LIGHT = {          # box color by identified light_color (observable)
    "red": (231, 76, 60),
    "yellow": (241, 196, 15),
    "green": (46, 204, 113),
    "off": (149, 165, 166),
}
COLOR_UNOBSERVABLE = (120, 144, 156)   # gray-blue
COLOR_NOT_A_SIGNAL = (155, 89, 182)    # dim magenta
COLOR_POLE = (52, 152, 219)            # blue
TYPE_PREFIX = {"vehicle": "V", "pedestrian": "P", "other": "O", "not_a_signal": "N", "unknown": "?"}

app = Flask(__name__)
_infos = {}


def get_infos(split):
    if split not in _infos:
        with open(PKL_PATHS[split], "rb") as f:
            _infos[split] = pickle.load(f)["infos"]
    return _infos[split]


def signal_color(sig_type, observable, color):
    if sig_type == "not_a_signal":
        return COLOR_NOT_A_SIGNAL
    if not observable:
        return COLOR_UNOBSERVABLE
    return COLOR_BY_LIGHT.get(color, COLOR_UNOBSERVABLE)


def mirror_box(box, w):
    x1, y1, x2, y2 = box
    return [w - x2, y1, w - x1, y2]


def draw_boxes(img, cam_rows, pole_boxes, scale, mirror_w, show_poles, show_assoc):
    d = ImageDraw.Draw(img)
    pole_centers = {}
    if show_poles:
        for pi, p in enumerate(pole_boxes):
            b = mirror_box(p, mirror_w) if mirror_w else list(p)
            b = [v * scale for v in b]
            d.rectangle(b, outline=COLOR_POLE, width=2)
            d.text((b[0] + 2, b[1] + 2), f"P{pi}", fill=COLOR_POLE)
            pole_centers[pi] = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    for r in cam_rows:
        b = mirror_box(r["bbox"], mirror_w) if mirror_w else list(r["bbox"])
        b = [v * scale for v in b]
        col = signal_color(r["signal_type"], r["observable"], r["color"])
        d.rectangle(b, outline=col, width=2)
        label = f"S{r['sig_idx']} {TYPE_PREFIX.get(r['signal_type'], '?')}"
        label += f" {r['color']}" if r["observable"] else " unobs"
        d.text((b[0] + 2, max(b[1] - 12, 1)), label, fill=col)
        if show_assoc and r["pole_idx"] >= 0 and r["pole_idx"] in pole_centers:
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            d.line([(cx, cy), pole_centers[r["pole_idx"]]], fill=(241, 196, 15), width=1)
    return img


def crop_b64(split, sample_idx, token, cam, sig_idx):
    path = os.path.join(CROPS_DIR, split, f"{split}_{sample_idx:05d}_{token}_{cam}_s{sig_idx:02d}.jpg")
    if not os.path.exists(path):
        return None
    img = Image.open(path).convert("RGB")
    if img.height > 72:
        img = img.resize((max(1, round(img.width * 72 / img.height)), 72))
    return image_to_base64(img)


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/info")
def api_info():
    """Total for one split (?split=...). Loads only that split's pkl, so the
    956MB train pkl is not pulled in until the user switches to train."""
    split = request.args.get("split", "val")
    if split not in PKL_PATHS:
        return jsonify({"error": f"unknown split {split}"})
    return jsonify({split: len(get_infos(split))})


@app.route("/api/sample")
def api_sample():
    split = request.args.get("split", "val")
    if split not in PKL_PATHS:
        return jsonify({"error": f"unknown split {split}"})
    infos = get_infos(split)
    idx = int(request.args.get("idx", 0))
    if idx < 0 or idx >= len(infos):
        return jsonify({"error": f"Invalid sample_idx {idx} (0..{len(infos)-1})"})
    show_poles = request.args.get("poles", "1") == "1"
    show_assoc = request.args.get("assoc", "1") == "1"
    show_nas = request.args.get("nas", "1") == "1"
    with_crops = request.args.get("crops", "1") == "1"

    info = infos[idx]
    if "tl_bboxes2d" not in info:
        return jsonify({"error": "pkl has no tl_bboxes2d — run add_traffic_lights_to_infos first"})
    has_status = "tl_light_color2d" in info
    cam_names = list(info["cams"].keys())

    rows_by_cam, table_rows = {}, []
    for ci, cam in enumerate(cam_names):
        rows = []
        n = len(info["tl_bboxes2d"][ci])
        for si in range(n):
            r = {
                "sig_idx": si, "camera": cam,
                "bbox": [float(v) for v in info["tl_bboxes2d"][ci][si]],
                "det_score": round(float(info["tl_scores2d"][ci][si]), 3),
                "pole_idx": int(info["tl_pole_idx2d"][ci][si]),
                "signal_type": str(info["tl_signal_type2d"][ci][si]) if has_status else "unknown",
                "observable": bool(info["tl_light_observable2d"][ci][si]) if has_status else False,
                "color": str(info["tl_light_color2d"][ci][si]) if has_status else "unknown",
                "shape": str(info["tl_lit_shape2d"][ci][si]) if has_status else "unknown",
                "conf": round(float(info["tl_status_conf2d"][ci][si]), 2) if has_status else -1.0,
            }
            if not show_nas and r["signal_type"] == "not_a_signal":
                continue
            rows.append(r)
            table_rows.append({**r,
                "crop": crop_b64(split, idx, info["token"], cam, si) if with_crops else None})
        rows_by_cam[cam] = rows

    images_b64 = []
    for cam in CAMERA_ORDER:
        try:
            ci = cam_names.index(cam)
            img = Image.open(info["cams"][cam]["data_path"]).convert("RGB")
            full_w = img.width
            img = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
            is_rear = cam in REAR_CAMERAS
            if is_rear:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img = draw_boxes(img, rows_by_cam[cam],
                             [list(map(float, p)) for p in info["tl_pole_bboxes2d"][ci]],
                             img.width / full_w, full_w if is_rear else None,
                             show_poles, show_assoc)
            images_b64.append(image_to_base64(img))
        except Exception as e:  # noqa: BLE001 — always render 6 tiles
            ph = Image.new("RGB", (400, 225), (40, 40, 40))
            ImageDraw.Draw(ph).text((10, 100), str(e)[:60], fill=(255, 0, 0))
            images_b64.append(image_to_base64(ph))

    n_sig = sum(len(v) for v in info["tl_bboxes2d"])
    n_pole = sum(len(v) for v in info["tl_pole_bboxes2d"])
    return jsonify({
        "images": images_b64, "token": info["token"], "scene": info.get("scene_name"),
        "location": info.get("location"), "description": info.get("description"),
        "num_signals": n_sig, "num_poles": n_pole, "has_status": has_status,
        "rows": table_rows,
    })


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Traffic Signal Status Visualizer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, sans-serif; background: #1a1a2e; color: #e0e0e0; }
        .header { background: #16213e; padding: 10px 20px; display: flex; align-items: center;
                  gap: 14px; border-bottom: 2px solid #0f3460; flex-wrap: wrap; }
        .header h1 { font-size: 17px; color: #53d8fb; }
        .controls { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
        .controls select, .controls input[type=number], .controls button {
            padding: 5px 10px; border-radius: 4px; border: 1px solid #0f3460;
            background: #1a1a2e; color: #e0e0e0; font-size: 13px; }
        .controls button { background: #0f3460; cursor: pointer; font-weight: bold; }
        .controls button:hover { background: #53d8fb; color: #1a1a2e; }
        .toggle { display: flex; align-items: center; gap: 4px; font-size: 12px;
                  background: #0f3460; padding: 3px 8px; border-radius: 10px; cursor: pointer; user-select: none; }
        .toggle input { accent-color: #53d8fb; cursor: pointer; }
        .info-badge { background: #0f3460; padding: 3px 8px; border-radius: 10px; font-size: 11px; color: #53d8fb; }
        .legend { display: flex; gap: 10px; font-size: 11px; flex-wrap: wrap; align-items: center; }
        .legend-item { display: flex; align-items: center; gap: 3px; }
        .legend-box { width: 12px; height: 12px; border-radius: 2px; }
        .main { display: flex; height: calc(100vh - 52px); }
        .left-panel { flex: 3; padding: 6px; display: flex; flex-direction: column; gap: 3px; }
        .image-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; grid-template-rows: 1fr 1fr;
                      gap: 3px; flex: 1; }
        .image-cell { position: relative; background: #111; border-radius: 3px; overflow: hidden; }
        .image-cell img { width: 100%; height: 100%; object-fit: contain; }
        .image-cell .cam-label { position: absolute; top: 3px; left: 3px; background: rgba(0,0,0,0.75);
            color: #53d8fb; padding: 1px 6px; border-radius: 2px; font-size: 11px; font-weight: bold; }
        .right-panel { flex: 2; display: flex; flex-direction: column; padding: 6px; gap: 6px; overflow-y: auto; }
        .panel { background: #16213e; border-radius: 5px; padding: 10px; border: 1px solid #0f3460; overflow-y: auto; }
        .panel h3 { font-size: 13px; margin-bottom: 6px; }
        table.det { width: 100%; border-collapse: collapse; font-size: 11px; }
        table.det th, table.det td { border: 1px solid #0f3460; padding: 2px 5px; text-align: left; vertical-align: middle; }
        table.det th { color: #53d8fb; background: #101a33; position: sticky; top: -10px; }
        table.det img { max-height: 60px; display: block; }
        .missing { color: #777; font-style: italic; font-size: 12px; }
        .meta { font-size: 11px; color: #888; }
        .pill { padding: 0 6px; border-radius: 8px; color: #111; font-weight: bold; }
    </style>
</head>
<body>
    <div class="header">
        <h1>TS Status Visualizer</h1>
        <div class="controls">
            <select id="split" onchange="splitChanged()">
                <option value="val">val</option>
                <option value="train">train</option>
            </select>
            <span id="total-badge" class="info-badge">--</span>
            <label class="meta">idx:</label>
            <input type="number" id="sample-id" value="0" min="0" style="width:80px"
                   onkeypress="if(event.key==='Enter') loadSample()">
            <button onclick="loadSample()">Go</button>
            <button onclick="navigate(-10)">&laquo;</button>
            <button onclick="navigate(-1)">&larr;</button>
            <button onclick="navigate(1)">&rarr;</button>
            <button onclick="navigate(10)">&raquo;</button>
            <button onclick="randomSample()" style="background:#533d8f">Rand</button>
            <button onclick="nextWithSignals()" style="background:#2d6a4f" title="next sample with >=1 signal">Next w/ signals</button>
        </div>
        <label class="toggle"><input type="checkbox" id="tg-poles" checked onchange="loadSample()"> Poles</label>
        <label class="toggle"><input type="checkbox" id="tg-assoc" checked onchange="loadSample()"> Assoc</label>
        <label class="toggle"><input type="checkbox" id="tg-nas" checked onchange="loadSample()"> not_a_signal</label>
        <label class="toggle"><input type="checkbox" id="tg-crops" checked onchange="loadSample()"> Crops</label>
        <div class="legend">
            <div class="legend-item"><div class="legend-box" style="background:#e74c3c"></div>red</div>
            <div class="legend-item"><div class="legend-box" style="background:#f1c40f"></div>yellow</div>
            <div class="legend-item"><div class="legend-box" style="background:#2ecc71"></div>green</div>
            <div class="legend-item"><div class="legend-box" style="background:#95a5a6"></div>off</div>
            <div class="legend-item"><div class="legend-box" style="background:#789098"></div>unobservable</div>
            <div class="legend-item"><div class="legend-box" style="background:#9b59b6"></div>not_a_signal</div>
            <div class="legend-item"><div class="legend-box" style="background:#3498db"></div>pole</div>
            <span id="counts" class="meta"></span>
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
            <div class="panel" style="flex:1">
                <h3>Signals (from pkl tl_* arrays)</h3>
                <div id="det-table"><span class="missing">Select a sample...</span></div>
            </div>
        </div>
    </div>
    <script>
        let totals = {};
        const colorPill = {red:'#e74c3c', yellow:'#f1c40f', green:'#2ecc71', off:'#95a5a6', unknown:'#789098'};
        async function init() { splitChanged(); }
        function curSplit() { return document.getElementById('split').value; }
        async function splitChanged() {
            const s = curSplit();
            if (!(s in totals)) {
                document.getElementById('total-badge').textContent = 'loading '+s+' pkl...';
                const d = await (await fetch('/api/info?split='+s)).json();
                totals[s] = d[s] || 0;
            }
            const t = totals[s] || 0;
            document.getElementById('total-badge').textContent = t.toLocaleString()+' samples';
            document.getElementById('sample-id').max = t-1;
            loadSample();
        }
        function esc(s) { return (s ?? '').toString().replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
        async function fetchSample(id) {
            const q = `split=${curSplit()}&idx=${id}` +
                `&poles=${document.getElementById('tg-poles').checked?1:0}` +
                `&assoc=${document.getElementById('tg-assoc').checked?1:0}` +
                `&nas=${document.getElementById('tg-nas').checked?1:0}` +
                `&crops=${document.getElementById('tg-crops').checked?1:0}`;
            return await (await fetch('/api/sample?'+q)).json();
        }
        async function loadSample() {
            const id = parseInt(document.getElementById('sample-id').value);
            const total = totals[curSplit()] || 0;
            if (isNaN(id) || id<0 || id>=total) return;
            document.getElementById('det-table').innerHTML = '<span class="missing">Loading...</span>';
            const d = await fetchSample(id);
            if (d.error) { document.getElementById('det-table').textContent = d.error; return; }
            for (let i=0;i<6;i++) document.getElementById('img-'+i).src = d.images[i];
            document.getElementById('sample-meta').textContent =
                `token=${d.token} | ${d.scene ?? ''} | ${d.location ?? ''} | ${d.description ?? ''}`;
            document.getElementById('counts').textContent =
                `signals: ${d.num_signals} | poles: ${d.num_poles}` + (d.has_status ? '' : ' | NO STATUS in pkl');
            if (d.rows.length === 0) {
                document.getElementById('det-table').innerHTML = '<span class="missing">No signal boxes in this sample.</span>';
                return;
            }
            let html = '<table class="det"><tr><th>crop</th><th>cam</th><th>#</th><th>type</th>' +
                       '<th>light</th><th>shape</th><th>conf</th><th>det</th><th>pole</th></tr>';
            for (const r of d.rows) {
                const light = r.observable ? r.color : 'unobs';
                const pill = `<span class="pill" style="background:${colorPill[r.color] ?? '#789098'}">${esc(light)}</span>`;
                html += `<tr><td>${r.crop ? `<img src="${r.crop}">` : '-'}</td>` +
                        `<td>${esc(r.camera.replace('CAM_',''))}</td><td>S${r.sig_idx}</td>` +
                        `<td>${esc(r.signal_type)}</td><td>${pill}</td><td>${esc(r.shape)}</td>` +
                        `<td>${r.conf}</td><td>${r.det_score}</td>` +
                        `<td>${r.pole_idx>=0 ? 'P'+r.pole_idx : '-'}</td></tr>`;
            }
            document.getElementById('det-table').innerHTML = html + '</table>';
        }
        function navigate(delta) {
            const total = totals[curSplit()] || 0;
            const inp = document.getElementById('sample-id');
            let id = parseInt(inp.value)+delta;
            if (id<0) id=total-1; if (id>=total) id=0;
            inp.value=id; loadSample();
        }
        function randomSample() {
            document.getElementById('sample-id').value = Math.floor(Math.random()*(totals[curSplit()]||1));
            loadSample();
        }
        async function nextWithSignals() {
            const total = totals[curSplit()] || 0;
            let id = parseInt(document.getElementById('sample-id').value);
            document.getElementById('det-table').innerHTML = '<span class="missing">Searching...</span>';
            for (let step=1; step<=total; step++) {
                const probe = (id+step) % total;
                const q = `split=${curSplit()}&idx=${probe}&poles=0&assoc=0&nas=1&crops=0`;
                const d = await (await fetch('/api/sample?'+q)).json();
                if ((d.num_signals ?? 0) > 0) {
                    document.getElementById('sample-id').value = probe; loadSample(); return;
                }
            }
        }
        document.addEventListener('keydown', e => {
            if (e.target.tagName==='INPUT' || e.target.tagName==='SELECT') return;
            if (e.key==='ArrowLeft') navigate(-1);
            if (e.key==='ArrowRight') navigate(1);
            if (e.key==='r') randomSample();
            if (e.key==='n') nextWithSignals();
        });
        init();
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic Signal Status Visualizer")
    parser.add_argument("--port", type=int, default=6060)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--train_pkl", type=str, default=PKL_PATHS["train"])
    parser.add_argument("--val_pkl", type=str, default=PKL_PATHS["val"])
    parser.add_argument("--crops_dir", type=str, default=CROPS_DIR)
    args = parser.parse_args()
    PKL_PATHS["train"] = args.train_pkl
    PKL_PATHS["val"] = args.val_pkl
    CROPS_DIR = args.crops_dir

    print("Traffic Signal Status Visualizer")
    print(f"  train pkl: {PKL_PATHS['train']}")
    print(f"  val pkl:   {PKL_PATHS['val']}")
    print(f"  crops dir: {CROPS_DIR}")
    print("  Loading val pkl...")
    get_infos("val")
    print(f"  Ready! Open http://<server-ip>:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
