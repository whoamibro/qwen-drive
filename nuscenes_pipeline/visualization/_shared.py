"""
Shared visualization utilities used by qa_visualizer and verifier_visualizer.

Covers:
- BEV (Bird's Eye View) generator
- PIL helpers: image_to_base64, draw_bboxes_on_image
- Two bbox text parsers:
    * parse_bboxes_qa_format     — QA-style "category (Image N (CamName) bbox[x1,y1,x2,y2])"
    * parse_bboxes_seed_format   — Seed-style "Image N (CamName) 2D bbox [x1=..,y1=..,x2=..,y2=..]"
"""

import io
import re
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OBJ_COLORS = {
    'car': '#ff6b6b', 'truck': '#ffa502', 'bus': '#ff7f50',
    'trailer': '#cd853f', 'construction_vehicle': '#daa520',
    'pedestrian': '#ffff00', 'motorcycle': '#ff69b4',
    'bicycle': '#ee82ee', 'traffic_cone': '#ff4500', 'barrier': '#808080',
}


# ---------------------------------------------------------------------------
# Bbox text parsers
# ---------------------------------------------------------------------------

def parse_bboxes_qa_format(text):
    """
    Parse bboxes from QA-style text:
        category (Image N (CamName) bbox[x1,y1,x2,y2])

    Returns: list of (img_num, x1, y1, x2, y2, label)
    """
    pattern = re.compile(
        r'(\w[\w_]*)\s*\(Image\s+(\d+)\s*\([^)]*\)\s*bbox\[(\d+),(\d+),(\d+),(\d+)\]\)'
    )
    results = []
    found_spans = set()
    for m in pattern.finditer(text):
        results.append((int(m.group(2)), int(m.group(3)), int(m.group(4)),
                         int(m.group(5)), int(m.group(6)), m.group(1)))
        found_spans.add((m.start(), m.end()))

    pattern2 = re.compile(r'Image\s+(\d+)\s*\([^)]*\)\s*bbox\[(\d+),(\d+),(\d+),(\d+)\]')
    for m in pattern2.finditer(text):
        if not any(m.start() >= s and m.end() <= e for s, e in found_spans):
            results.append((int(m.group(1)), int(m.group(2)), int(m.group(3)),
                             int(m.group(4)), int(m.group(5)), "object"))
    return results


def parse_bboxes_seed_format(text):
    """
    Parse bboxes from seed-data-style text (risk_assessment / traffic prompts):
        OBJ 1: car [Image 4 (Rear-left) 2D bbox [x1=0, y1=110, x2=54, y2=173]]

    Associates the OBJ-level category with each per-image bbox.
    Returns: list of (img_num, x1, y1, x2, y2, label) where label includes "OBJ N: category"
    """
    results = []
    # Match "OBJ N: category" headers line-by-line to keep per-obj association
    obj_pattern = re.compile(r'OBJ\s+(\d+):\s*(\w[\w_]*)')
    bbox_pattern = re.compile(
        r'Image\s+(\d+)\s*\(([^)]*)\)\s*2D\s+bbox\s*\[\s*x1=(\d+),\s*y1=(\d+),\s*x2=(\d+),\s*y2=(\d+)\s*\]'
    )

    for line in text.split('\n'):
        obj_match = obj_pattern.search(line)
        if not obj_match:
            continue
        obj_id = obj_match.group(1)
        category = obj_match.group(2)
        for m in bbox_pattern.finditer(line):
            img_num = int(m.group(1))
            x1, y1, x2, y2 = int(m.group(3)), int(m.group(4)), int(m.group(5)), int(m.group(6))
            label = f"OBJ {obj_id} {category}"
            results.append((img_num, x1, y1, x2, y2, label))
    return results


# ---------------------------------------------------------------------------
# PIL helpers
# ---------------------------------------------------------------------------

def draw_bboxes_on_image(img, bboxes, color):
    """Draw a list of (x1,y1,x2,y2,label) bboxes on a PIL image (before any flip)."""
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for x1, y1, x2, y2, label in bboxes:
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        tb = draw.textbbox((x1, y1 - 16), label, font=font)
        draw.rectangle([tb[0]-1, tb[1]-1, tb[2]+1, tb[3]+1], fill=color)
        draw.text((x1, y1 - 16), label, fill=(255, 255, 255), font=font)
    return img


def image_to_base64(img, fmt="JPEG", quality=85):
    """Encode a PIL image to a base64 data URI string."""
    buf = io.BytesIO()
    if fmt.upper() == "JPEG":
        img.save(buf, format=fmt, quality=quality)
        mime = "image/jpeg"
    else:
        img.save(buf, format=fmt)
        mime = f"image/{fmt.lower()}"
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode()}"


# ---------------------------------------------------------------------------
# BEV generator
# ---------------------------------------------------------------------------

def generate_bev(sample, loader, bev_range=50.0):
    """Generate Bird's Eye View as a base64 PNG data URI.

    Draws ego + all GT objects within range, color-coded by category,
    with heading arrows and velocity vectors. Scene rotated so ego faces up.
    """
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
    except Exception:
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


# ---------------------------------------------------------------------------
# Panoramic builder (6-view with optional bbox overlays, rear-camera flip)
# ---------------------------------------------------------------------------

CAMERA_ORDER = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT',
]
REAR_INDICES = {3, 4, 5}


def detect_bbox_coord_width(bboxes_iter):
    """Infer bbox coordinate-space image width from a sequence of bboxes.

    Args:
        bboxes_iter: iterable of (img_num, x1, y1, x2, y2, label) tuples

    Returns:
        One of {400, 800, 1600} based on the max x value, or None if empty.
    """
    max_x = 0
    for tup in bboxes_iter:
        # Accept (x1,y1,x2,y2,label) or (img_num,x1,y1,x2,y2,label)
        if len(tup) == 5:
            x1, _, x2, _, _ = tup
        else:
            _, x1, _, x2, _, _ = tup
        if x2 > max_x:
            max_x = x2
    if max_x == 0:
        return None
    if max_x > 800:
        return 1600
    if max_x > 400:
        return 800
    return 400


def build_panoramic(sample, loader, bbox_overlays=None, resize_factor=2,
                    bbox_coord_width=None):
    """
    Build 6-view panoramic images (as a list of base64 data URIs in egocentric order).

    Args:
        sample: NuScenesSample object
        loader: NuScenesDataLoader
        bbox_overlays: dict of {img_num_1based: [(bboxes, color), ...]}
                       bboxes is list of (x1,y1,x2,y2,label).
                       Allows layering multiple bbox sets (e.g., green + pink).
        resize_factor: Display resize factor. Controls the output image size
                       (original_width // resize_factor per view).
        bbox_coord_width: The image width in which the bbox coordinates are
                          expressed. If different from the displayed image width,
                          bboxes are scaled proportionally. If None, assumes
                          bboxes are already in the displayed image space.

    Returns:
        List of 6 base64 data URI strings in egocentric order.
    """
    bbox_overlays = bbox_overlays or {}

    panoramic_b64 = []
    for view_idx, cam_name in enumerate(CAMERA_ORDER):
        img_num = view_idx + 1
        cam = sample.cameras[cam_name]
        img_path = cam.image_path

        # Load image
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            # Try relative to project root
            alt_path = img_path.lstrip('./')
            img = Image.open(alt_path).convert('RGB')

        # Resize for display (does NOT have to match bbox coord space)
        if resize_factor > 1:
            w, h = img.size
            img = img.resize((w // resize_factor, h // resize_factor), Image.LANCZOS)

        # Flip rear cameras FIRST for egocentric consistency, so we draw bboxes
        # with readable (non-mirrored) text in the final orientation.
        is_rear = view_idx in REAR_INDICES
        if is_rear:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        display_w = img.size[0]
        # Scale bboxes from their native coord space to the displayed image space
        if bbox_coord_width and bbox_coord_width != display_w:
            bbox_scale = display_w / bbox_coord_width
        else:
            bbox_scale = 1.0

        def _scale(bs):
            if bbox_scale == 1.0:
                return bs
            return [(int(x1 * bbox_scale), int(y1 * bbox_scale),
                     int(x2 * bbox_scale), int(y2 * bbox_scale), label)
                    for x1, y1, x2, y2, label in bs]

        def _flip(bs):
            return [(display_w - x2, y1, display_w - x1, y2, label)
                    for x1, y1, x2, y2, label in bs]

        # Draw bbox overlays. Scale first (to display space), then flip x for rear.
        for bboxes, color in bbox_overlays.get(img_num, []):
            scaled = _scale(bboxes)
            to_draw = _flip(scaled) if is_rear else scaled
            draw_bboxes_on_image(img, to_draw, color)

        panoramic_b64.append(image_to_base64(img))

    return panoramic_b64
