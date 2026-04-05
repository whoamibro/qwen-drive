"""
QA Dataset Visualization Tool

Web-based dashboard for verifying the generated SFT QA dataset.
- Draws bboxes from questions (green) and answers (pink) on 6-view panoramic images
- BEV (Bird's Eye View) visualization showing all GT objects with velocities

Usage:
    python qa_visualizer.py --port 6060
    # Then open http://<server-ip>:6060 in browser
"""

import os
import sys
import io
import re
import json
import base64
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import matplotlib.patches as mpatches
from flask import Flask, render_template_string, request, jsonify
from PIL import Image, ImageDraw, ImageFont

from nuscenes_pipeline.core.nuscenes_data_loader import NuScenesDataLoader

DATA_DIR = os.environ.get("QA_DATASET_DIR", "qa_dataset")
PKL_PATH = os.environ.get("NUSCENES_PKL_PATH", "nuscenes2d_ego_temporal_infos_val.pkl")

CAMERA_ORDER = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT',
]
REAR_INDICES = {3, 4, 5}
IMG_NUM_TO_IDX = {i + 1: i for i in range(6)}

# BEV color map
OBJ_COLORS = {
    'car': '#ff6b6b', 'truck': '#ffa502', 'bus': '#ff7f50',
    'trailer': '#cd853f', 'construction_vehicle': '#daa520',
    'pedestrian': '#ffff00', 'motorcycle': '#ff69b4',
    'bicycle': '#ee82ee', 'traffic_cone': '#ff4500', 'barrier': '#808080',
}


def parse_bboxes(text):
    """Extract bboxes from text: category (Image N (CamName) bbox[x1,y1,x2,y2])"""
    pattern = re.compile(
        r'(\w[\w_]*)\s*\(Image\s+(\d+)\s*\([^)]*\)\s*bbox\[(\d+),(\d+),(\d+),(\d+)\]\)'
    )
    results = []
    found_spans = set()
    for match in pattern.finditer(text):
        results.append((int(match.group(2)), int(match.group(3)), int(match.group(4)),
                         int(match.group(5)), int(match.group(6)), match.group(1)))
        found_spans.add((match.start(), match.end()))

    pattern2 = re.compile(r'Image\s+(\d+)\s*\([^)]*\)\s*bbox\[(\d+),(\d+),(\d+),(\d+)\]')
    for match in pattern2.finditer(text):
        if not any(match.start() >= s and match.end() <= e for s, e in found_spans):
            results.append((int(match.group(1)), int(match.group(2)), int(match.group(3)),
                             int(match.group(4)), int(match.group(5)), "object"))
    return results


def draw_bboxes_on_image(img, bboxes, color):
    """Draw bboxes on PIL image (before flipping)."""
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except:
        font = ImageFont.load_default()

    for x1, y1, x2, y2, label in bboxes:
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        tb = draw.textbbox((x1, y1 - 16), label, font=font)
        draw.rectangle([tb[0]-1, tb[1]-1, tb[2]+1, tb[3]+1], fill=color)
        draw.text((x1, y1 - 16), label, fill=(255, 255, 255), font=font)
    return img


def image_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=85)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


def generate_bev(sample, loader, bev_range=50.0):
    """Generate BEV visualization as base64 PNG. Referencing vqa_test_egocentric.py."""
    first_cam = sample.cameras[loader.CAMERA_NAMES[0]]
    ego_pos_global = np.array(first_cam.ego2global_translation[:2])
    ego_quat = first_cam.ego2global_rotation
    qw, qx, qy, qz = ego_quat
    ego_yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))

    cos_yaw, sin_yaw = np.cos(ego_yaw), np.sin(ego_yaw)
    R_ego2global = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])

    rot_angle = np.pi/2 - ego_yaw
    cos_rot, sin_rot = np.cos(rot_angle), np.sin(rot_angle)
    R_scene = np.array([[cos_rot, -sin_rot], [sin_rot, cos_rot]])

    ego_length, ego_width = 4.7, 1.9
    EGO_REAR_TO_CENTER = 2.35

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), facecolor='#1a1a2e')
    ax.set_facecolor('#16213e')
    ax.set_xlim(-bev_range, bev_range)
    ax.set_ylim(-bev_range, bev_range)
    ax.set_aspect('equal')

    # Grid and range circles
    for i in range(-int(bev_range), int(bev_range)+1, 10):
        ax.axhline(y=i, color='#2a3f5f', linewidth=0.5, alpha=0.5)
        ax.axvline(x=i, color='#2a3f5f', linewidth=0.5, alpha=0.5)
    for r in [10, 20, 30, 40, 50]:
        if r <= bev_range:
            ax.add_patch(plt.Circle((0,0), r, fill=False, color='#3a5a8f', linewidth=0.8, linestyle='--', alpha=0.6))
            ax.text(r*0.707, r*0.707, f'{r}m', fontsize=7, color='#6a8abf', alpha=0.8)

    # Ego vehicle
    ego_rect = Rectangle((-ego_width/2, 0), ego_width, ego_length, linewidth=2,
                          edgecolor='#00ff88', facecolor='#00ff8844', zorder=10)
    ax.add_patch(ego_rect)
    ax.annotate('', xy=(0, ego_length+2), xytext=(0, EGO_REAR_TO_CENTER),
                arrowprops=dict(arrowstyle='->', color='#00ff88', lw=2), zorder=11)
    ax.text(0, -2, 'EGO', fontsize=9, color='#00ff88', ha='center', va='top', fontweight='bold')

    # Ego velocity
    ego_vel_global = np.array([0.0, 0.0])
    try:
        next_s = loader.get_sample(sample.sample_idx + 1)
        if next_s.scene_token == sample.scene_token:
            next_pos = np.array(next_s.cameras[loader.CAMERA_NAMES[0]].ego2global_translation[:2])
            dt = (next_s.timestamp - sample.timestamp) / 1e6
            if dt > 0:
                ego_vel_global = (next_pos - ego_pos_global) / dt
    except:
        pass

    vel_scale = 2.0
    if np.linalg.norm(ego_vel_global) > 0.5:
        ev_rot = R_scene @ ego_vel_global
        ve = np.array([0, EGO_REAR_TO_CENTER]) + ev_rot * vel_scale
        ax.annotate('', xy=(ve[0], ve[1]), xytext=(0, EGO_REAR_TO_CENTER),
                    arrowprops=dict(arrowstyle='->', color='#00ffff', lw=2, linestyle='--'), zorder=9)

    # Draw objects
    drawn_types = set()
    if len(sample.gt_boxes) > 0:
        for obj_idx in range(len(sample.gt_boxes)):
            bbox = sample.gt_boxes[obj_idx]
            velocity = sample.gt_velocity[obj_idx]
            obj_name = sample.gt_names[obj_idx] if sample.gt_names is not None else "object"

            obj_pos_ego = np.array([bbox[0], bbox[1]])
            if abs(obj_pos_ego[0]) > bev_range or abs(obj_pos_ego[1]) > bev_range:
                continue

            obj_pos_global = R_ego2global @ obj_pos_ego + ego_pos_global
            obj_pos_centered = obj_pos_global - ego_pos_global
            obj_pos_plot = R_scene @ obj_pos_centered

            obj_yaw_global = bbox[6] + ego_yaw
            obj_yaw_plot = obj_yaw_global + rot_angle
            obj_l, obj_w = bbox[3], bbox[4]
            distance = np.linalg.norm(obj_pos_ego)

            color = OBJ_COLORS.get(obj_name.lower(), '#ffffff')
            drawn_types.add(obj_name.lower())

            # Object rectangle
            corners = np.array([[-obj_l/2,-obj_w/2],[obj_l/2,-obj_w/2],[obj_l/2,obj_w/2],[-obj_l/2,obj_w/2]])
            cos_o, sin_o = np.cos(obj_yaw_plot), np.sin(obj_yaw_plot)
            R_obj = np.array([[cos_o,-sin_o],[sin_o,cos_o]])
            corners_plot = (R_obj @ corners.T).T + obj_pos_plot
            ax.add_patch(Polygon(corners_plot, closed=True, linewidth=1.5,
                                  edgecolor=color, facecolor=color+'44', zorder=5))

            # Heading arrow
            h_len = min(obj_l, 3.0)
            h_dir = np.array([np.cos(obj_yaw_plot), np.sin(obj_yaw_plot)]) * h_len
            ax.annotate('', xy=obj_pos_plot + h_dir, xytext=obj_pos_plot,
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.5), zorder=6)

            # Velocity vector
            vel_ego = np.array([velocity[0], velocity[1]])
            vel_mag = np.linalg.norm(vel_ego)
            if vel_mag > 0.5:
                vel_global = R_ego2global @ vel_ego
                vel_rot = R_scene @ vel_global
                ve = obj_pos_plot + vel_rot * vel_scale
                ax.annotate('', xy=ve, xytext=obj_pos_plot,
                            arrowprops=dict(arrowstyle='->', color='#00ffff', lw=1, linestyle='--'), zorder=4)

            # Label with motion state
            motion = "S" if vel_mag < 0.1 else f"{vel_mag:.1f}"
            ax.text(obj_pos_plot[0], obj_pos_plot[1] + obj_w/2 + 1.2,
                    f"{obj_name}\n{distance:.0f}m|{motion}", fontsize=6, color=color,
                    ha='center', va='bottom', alpha=0.9)

    # Coordinate axes
    ax.annotate('', xy=(0,10), xytext=(0,0), arrowprops=dict(arrowstyle='->', color='#55ff55', lw=2), zorder=2)
    ax.text(1, 10, 'Fwd', fontsize=8, color='#55ff55', va='center')
    ax.annotate('', xy=(10,0), xytext=(0,0), arrowprops=dict(arrowstyle='->', color='#ff5555', lw=2), zorder=2)
    ax.text(10, -1.5, 'Right', fontsize=8, color='#ff5555', ha='center')

    # Legend
    legend_els = [
        mpatches.Patch(facecolor='#00ff8844', edgecolor='#00ff88', lw=2, label='Ego'),
        plt.Line2D([0],[0], color='#00ffff', lw=2, linestyle='--', label='Velocity'),
    ]
    for t in sorted(drawn_types):
        legend_els.append(mpatches.Patch(facecolor=OBJ_COLORS.get(t,'#fff')+'44',
                                          edgecolor=OBJ_COLORS.get(t,'#fff'), lw=1.5, label=t))
    ax.legend(handles=legend_els, loc='upper right', fontsize=7,
              facecolor='#1a1a2e', edgecolor='#3a5a8f', labelcolor='white')

    ego_speed = np.linalg.norm(ego_vel_global)
    ax.set_title(f'BEV | Sample {sample.sample_idx} | Speed: {ego_speed:.1f} m/s',
                 fontsize=10, color='white', pad=8)
    ax.text(0.02, 0.02, f"S = stationary (<0.1 m/s) | number = speed (m/s)",
            transform=ax.transAxes, fontsize=7, color='#aaa', va='bottom')
    ax.tick_params(colors='#666', labelsize=7)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, facecolor='#1a1a2e', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


# ===========================================================================
# Flask app
# ===========================================================================

app = Flask(__name__)
_cache = {}
_loader = None
_img_index = None


def get_loader():
    global _loader, _img_index
    if _loader is None:
        _loader = NuScenesDataLoader(pkl_path=PKL_PATH)
        _img_index = {}
        for idx in range(len(_loader.infos)):
            cam_path = _loader.infos[idx]['cams']['CAM_FRONT']['data_path']
            _img_index[os.path.basename(cam_path)] = idx
    return _loader, _img_index


def get_dataset(split):
    if split not in _cache:
        fpath = os.path.join(DATA_DIR, f"sft_{split}_no_objlist.json")
        if os.path.exists(fpath):
            with open(fpath) as f:
                _cache[split] = json.load(f)
        else:
            _cache[split] = None
    return _cache[split]


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/info')
def api_info():
    split = request.args.get('split', 'train')
    data = get_dataset(split)
    return jsonify({"total": len(data) if data else 0})


@app.route('/api/sample')
def api_sample():
    split = request.args.get('split', 'train')
    sample_id = int(request.args.get('id', 0))

    data = get_dataset(split)
    if not data or sample_id < 0 or sample_id >= len(data):
        return jsonify({"error": f"Invalid: split={split}, id={sample_id}"})

    sample_data = data[sample_id]
    image_paths = sample_data['image']
    convs = sample_data['conversations']

    user_msg = convs[1]['value']
    gpt_msg = convs[2]['value']
    question = user_msg.split('TASK:\n')[1].strip() if 'TASK:\n' in user_msg else user_msg

    q_bboxes = parse_bboxes(question)
    a_bboxes = parse_bboxes(gpt_msg)

    q_by_img, a_by_img = {}, {}
    for img_num, x1, y1, x2, y2, label in q_bboxes:
        idx = IMG_NUM_TO_IDX.get(img_num)
        if idx is not None:
            q_by_img.setdefault(idx, []).append((x1, y1, x2, y2, label))
    for img_num, x1, y1, x2, y2, label in a_bboxes:
        idx = IMG_NUM_TO_IDX.get(img_num)
        if idx is not None:
            a_by_img.setdefault(idx, []).append((x1, y1, x2, y2, label))

    images_b64 = []
    for i, img_path in enumerate(image_paths):
        try:
            img = Image.open(img_path).convert('RGB')
            w, h = img.size
            img = img.resize((w // 2, h // 2), Image.LANCZOS)
            if i in q_by_img:
                img = draw_bboxes_on_image(img, q_by_img[i], color=(46, 204, 113))
            if i in a_by_img:
                img = draw_bboxes_on_image(img, a_by_img[i], color=(233, 30, 140))
            if i in REAR_INDICES:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            images_b64.append(image_to_base64(img))
        except Exception as e:
            ph = Image.new('RGB', (400, 225), (40, 40, 40))
            ImageDraw.Draw(ph).text((10, 100), str(e)[:40], fill=(255, 0, 0))
            images_b64.append(image_to_base64(ph))

    # Generate BEV
    bev_b64 = ""
    try:
        loader, img_index = get_loader()
        front_basename = os.path.basename(image_paths[1])
        sample_idx = img_index.get(front_basename)
        if sample_idx is not None:
            ns_sample = loader.get_sample(sample_idx)
            bev_b64 = generate_bev(ns_sample, loader, bev_range=50.0)
    except Exception as e:
        print(f"BEV error: {e}")

    def highlight(text, cls):
        pat = re.compile(r'(\w[\w_]*\s*\(Image\s+\d+\s*\([^)]*\)\s*bbox\[\d+,\d+,\d+,\d+\]\))')
        return pat.sub(rf'<span class="{cls}">\1</span>', text)

    return jsonify({
        "images": images_b64,
        "bev": bev_b64,
        "question_html": highlight(question, 'bbox-highlight-green'),
        "answer_html": highlight(gpt_msg, 'bbox-highlight-pink'),
        "q_bbox_count": len(q_bboxes),
        "a_bbox_count": len(a_bboxes),
    })


# ===========================================================================
# HTML Template
# ===========================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>QA Dataset Visualizer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, sans-serif; background: #1a1a2e; color: #e0e0e0; }
        .header {
            background: #16213e; padding: 10px 20px;
            display: flex; align-items: center; gap: 16px;
            border-bottom: 2px solid #0f3460; flex-wrap: wrap;
        }
        .header h1 { font-size: 18px; color: #53d8fb; }
        .controls { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
        .controls select, .controls input, .controls button {
            padding: 5px 10px; border-radius: 4px; border: 1px solid #0f3460;
            background: #1a1a2e; color: #e0e0e0; font-size: 13px;
        }
        .controls button { background: #0f3460; cursor: pointer; font-weight: bold; }
        .controls button:hover { background: #53d8fb; color: #1a1a2e; }
        .controls label { font-size: 12px; color: #aaa; }
        .info-badge { background: #0f3460; padding: 3px 8px; border-radius: 10px; font-size: 11px; color: #53d8fb; }
        .bbox-count { font-size: 11px; color: #888; }
        .main { display: flex; height: calc(100vh - 50px); }

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

        /* Right: Q&A + BEV */
        .right-panel { flex: 2; display: flex; flex-direction: column; padding: 6px; gap: 6px; }
        .qa-section {
            flex: 1; background: #16213e; border-radius: 5px; padding: 10px;
            overflow-y: auto; border: 1px solid #0f3460;
        }
        .qa-section h3 { font-size: 13px; margin-bottom: 6px; display: flex; align-items: center; gap: 6px; }
        .badge-q { background: #2ecc71; color: #fff; padding: 1px 6px; border-radius: 3px; font-size: 10px; }
        .badge-a { background: #e91e8c; color: #fff; padding: 1px 6px; border-radius: 3px; font-size: 10px; }
        .badge-bev { background: #3498db; color: #fff; padding: 1px 6px; border-radius: 3px; font-size: 10px; }
        .qa-text { font-size: 13px; line-height: 1.6; white-space: pre-wrap; word-break: break-word; }
        .bbox-highlight-green { background: rgba(46,204,113,0.2); border: 1px solid #2ecc71; border-radius: 2px; padding: 0 2px; font-size: 11px; }
        .bbox-highlight-pink { background: rgba(233,30,140,0.2); border: 1px solid #e91e8c; border-radius: 2px; padding: 0 2px; font-size: 11px; }

        .bev-section {
            flex: 1.2; background: #16213e; border-radius: 5px; padding: 6px;
            border: 1px solid #0f3460; display: flex; flex-direction: column; align-items: center;
        }
        .bev-section img { max-width: 100%; max-height: 100%; object-fit: contain; }

        .nav-buttons { display: flex; gap: 5px; }
        .legend { display: flex; gap: 12px; font-size: 11px; }
        .legend-item { display: flex; align-items: center; gap: 3px; }
        .legend-box { width: 12px; height: 12px; border-radius: 2px; }
        .loading { color: #555; font-style: italic; }
    </style>
</head>
<body>
    <div class="header">
        <h1>QA Dataset Visualizer</h1>
        <div class="controls">
            <label>Dataset:</label>
            <select id="split" onchange="loadDataset()">
                <option value="train">Train</option>
                <option value="val">Val</option>
            </select>
            <span id="total-badge" class="info-badge">--</span>
            <label>ID:</label>
            <input type="number" id="sample-id" value="0" min="0" style="width:70px"
                   onkeypress="if(event.key==='Enter') loadSample()">
            <button onclick="loadSample()">Go</button>
            <div class="nav-buttons">
                <button onclick="navigate(-10)">&laquo;</button>
                <button onclick="navigate(-1)">&larr;</button>
                <button onclick="navigate(1)">&rarr;</button>
                <button onclick="navigate(10)">&raquo;</button>
            </div>
            <button onclick="randomSample()" style="background:#533d8f">Rand</button>
        </div>
        <div class="legend">
            <div class="legend-item"><div class="legend-box" style="background:#2ecc71"></div> Q bbox</div>
            <div class="legend-item"><div class="legend-box" style="background:#e91e8c"></div> A bbox</div>
            <span id="bbox-counts" class="bbox-count"></span>
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
        </div>
        <div class="right-panel">
            <div class="qa-section" style="flex:0.6">
                <h3><span class="badge-q">Q</span> Question</h3>
                <div class="qa-text" id="question-text"><span class="loading">Select a sample...</span></div>
            </div>
            <div class="qa-section" style="flex:1">
                <h3><span class="badge-a">A</span> Answer</h3>
                <div class="qa-text" id="answer-text"><span class="loading">Select a sample...</span></div>
            </div>
            <div class="bev-section">
                <h3><span class="badge-bev">BEV</span> Bird's Eye View (GT Objects)</h3>
                <img id="bev-img">
            </div>
        </div>
    </div>
    <script>
        let totalSamples = 0;
        async function loadDataset() {
            const split = document.getElementById('split').value;
            const r = await fetch('/api/info?split='+split);
            const d = await r.json();
            totalSamples = d.total;
            document.getElementById('total-badge').textContent = totalSamples.toLocaleString()+' samples';
            document.getElementById('sample-id').max = totalSamples - 1;
            document.getElementById('sample-id').value = 0;
            loadSample();
        }
        async function loadSample() {
            const split = document.getElementById('split').value;
            const id = parseInt(document.getElementById('sample-id').value);
            if (isNaN(id) || id<0 || id>=totalSamples) return;
            document.getElementById('question-text').innerHTML = '<span class="loading">Loading...</span>';
            document.getElementById('answer-text').innerHTML = '<span class="loading">Loading...</span>';
            const r = await fetch('/api/sample?split='+split+'&id='+id);
            const d = await r.json();
            if (d.error) { document.getElementById('question-text').textContent = d.error; return; }
            for (let i=0;i<6;i++) document.getElementById('img-'+i).src = d.images[i];
            document.getElementById('question-text').innerHTML = d.question_html;
            document.getElementById('answer-text').innerHTML = d.answer_html;
            document.getElementById('bbox-counts').textContent = 'Q:'+d.q_bbox_count+' | A:'+d.a_bbox_count;
            if (d.bev) document.getElementById('bev-img').src = d.bev;
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
        loadDataset();
    </script>
</body>
</html>
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="QA Dataset Visualizer")
    parser.add_argument('--port', type=int, default=6060)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR)
    parser.add_argument('--pkl_path', type=str, default=PKL_PATH)
    args = parser.parse_args()
    DATA_DIR = args.data_dir
    PKL_PATH = args.pkl_path

    print(f"QA Dataset Visualizer")
    print(f"  Data dir: {DATA_DIR}")
    print(f"  Loading datasets...")
    for split in ['train', 'val']:
        d = get_dataset(split)
        if d: print(f"  {split}: {len(d):,} samples")
        else: print(f"  {split}: not found")

    print(f"  Loading nuScenes for BEV...")
    get_loader()
    print(f"  Ready! Open http://<server-ip>:{args.port}")
    print(f"  Keys: Left/Right arrows, 'r' for random")
    app.run(host=args.host, port=args.port, debug=False)
