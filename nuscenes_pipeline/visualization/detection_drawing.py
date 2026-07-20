"""
Shared drawing helpers for traffic light & pole detection results.

Used by:
  - nuscenes_pipeline.visualization.detection_visualizer (web dashboard)
  - nuscenes_pipeline.modules.traffic_light_pole_detection (--visualize batch output)

All 2D coordinates in detection JSONs live in the UNFLIPPED full-resolution image
space; rear-camera images are flipped for display, so x coordinates are mirrored
via `mirror_w` before drawing.
"""

import io
import base64
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CAMERA_ORDER = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT',
]
REAR_CAMERAS = {'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'}
CAM_SHORT_LABELS = {
    'CAM_FRONT_LEFT': '1: Front-left', 'CAM_FRONT': '2: Front',
    'CAM_FRONT_RIGHT': '3: Front-right', 'CAM_BACK_LEFT': '4: Rear-left (flip)',
    'CAM_BACK': '5: Rear (flip)', 'CAM_BACK_RIGHT': '6: Rear-right (flip)',
}

# Traffic light bbox color by detected state (RGB)
STATE_COLORS = {
    'red': (231, 76, 60), 'red_arrow': (192, 57, 43),
    'yellow': (241, 196, 15),
    'green': (46, 204, 113), 'green_arrow': (26, 188, 156),
    'off': (149, 165, 166), 'unknown': (52, 152, 219),
}
# Pole segment color by type (RGB)
POLE_COLORS = {
    'vertical_pole': (255, 255, 255), 'mast_arm': (255, 0, 255),
    'gantry': (255, 165, 0), 'span_wire': (0, 255, 255),
}


def _get_font(size=13):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _draw_label(draw, xy, text, color, font):
    tb = draw.textbbox(xy, text, font=font)
    draw.rectangle([tb[0] - 1, tb[1] - 1, tb[2] + 1, tb[3] + 1], fill=color)
    draw.text(xy, text, fill=(0, 0, 0), font=font)


def draw_traffic_lights(img, lights, scale, mirror_w=None):
    """Draw detected traffic light bboxes on a (display-scaled) PIL image.

    scale: display_width / full_res_width. mirror_w: full-res image width when
    the image is displayed flipped (rear cameras) — x coords are mirrored.
    """
    draw = ImageDraw.Draw(img)
    font = _get_font(13)
    for tl in lights:
        x1, y1, x2, y2 = tl['bbox_2d']
        if mirror_w is not None:
            x1, x2 = mirror_w - x2, mirror_w - x1
        x1, y1, x2, y2 = x1 * scale, y1 * scale, x2 * scale, y2 * scale
        state = tl.get('state', 'unknown')
        color = STATE_COLORS.get(state, STATE_COLORS['unknown'])
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        b3d = tl.get('bbox_3d')
        depth = f" {b3d['depth_m']:.0f}m" if b3d else ""
        _draw_label(draw, (x1, max(0, y1 - 16)), f"{tl.get('id','TL')} {state}{depth}", color, font)
    return img


def draw_poles(img, poles, scale, mirror_w=None):
    """Draw detected pole line segments on a (display-scaled) PIL image."""
    draw = ImageDraw.Draw(img)
    font = _get_font(12)
    for pole in poles:
        (x1, y1), (x2, y2) = pole['line_2d']
        if mirror_w is not None:
            x1, x2 = mirror_w - x1, mirror_w - x2
        x1, y1, x2, y2 = x1 * scale, y1 * scale, x2 * scale, y2 * scale
        seg_type = pole.get('segment_type', 'vertical_pole')
        color = POLE_COLORS.get(seg_type, (255, 255, 255))
        draw.line([x1, y1, x2, y2], fill=color, width=4)
        # Endpoint markers: filled circle at the base/junction, ring at the far end
        draw.ellipse([x1 - 5, y1 - 5, x1 + 5, y1 + 5], fill=color)
        draw.ellipse([x2 - 5, y2 - 5, x2 + 5, y2 + 5], outline=color, width=2)
        l3d = pole.get('line_3d')
        depth = f" {l3d['depth_m']:.0f}m" if l3d else ""
        _draw_label(draw, ((x1 + x2) / 2 + 6, (y1 + y2) / 2), f"{pole.get('id','P')}{depth}", color, font)
    return img


def make_detection_overlay_fn(lights, poles, show_lights=True, show_poles=True):
    """Build a qa_visualizer.generate_bev overlay callback drawing lifted 3D detections."""
    def overlay(ax, to_plot):
        from matplotlib.patches import Polygon as MplPolygon
        if show_lights:
            for tl in lights:
                b3d = tl.get('bbox_3d')
                if not b3d:
                    continue
                cx, cy = b3d['center'][0], b3d['center'][1]
                _, w, _ = b3d['size']
                yaw = b3d['yaw']
                p = to_plot([cx, cy])
                state = tl.get('state', 'unknown')
                c = '#%02x%02x%02x' % STATE_COLORS.get(state, STATE_COLORS['unknown'])
                # Housing footprint (thin box perpendicular to its facing yaw)
                half_w, half_t = max(w, 0.6) / 2, 0.3
                corners = np.array([[-half_t, -half_w], [half_t, -half_w],
                                    [half_t, half_w], [-half_t, half_w]])
                cos_y, sin_y = np.cos(yaw), np.sin(yaw)
                R = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
                corners_ego = (R @ corners.T).T + np.array([cx, cy])
                corners_plot = np.array([to_plot(c2) for c2 in corners_ego])
                ax.add_patch(MplPolygon(corners_plot, closed=True, linewidth=1.5,
                                        edgecolor=c, facecolor=c + '66', zorder=12))
                ax.text(p[0], p[1] + 1.2, f"{tl.get('id','TL')} z={b3d['center'][2]:.1f}",
                        fontsize=6, color=c, ha='center', zorder=13)
        if show_poles:
            for pole in poles:
                l3d = pole.get('line_3d')
                if not l3d:
                    continue
                b = to_plot(l3d['bottom'][:2])
                t = to_plot(l3d['top'][:2])
                seg_type = pole.get('segment_type', 'vertical_pole')
                c = '#%02x%02x%02x' % POLE_COLORS.get(seg_type, (255, 255, 255))
                ax.plot([b[0], t[0]], [b[1], t[1]], color=c, linewidth=2, zorder=12)
                ax.plot(b[0], b[1], marker='o', markersize=5, color=c, zorder=13)
                ax.text(b[0], b[1] - 1.5, pole.get('id', 'P'), fontsize=6,
                        color=c, ha='center', va='top', zorder=13)
    return overlay


def render_annotated_view(cam_name, image_path, lights, poles, display_scale=0.5,
                          show_lights=True, show_poles=True, cam_label=True):
    """Load one camera image and draw its detections; returns a PIL image.

    lights/poles: detections belonging to THIS camera only. Rear cameras are
    flipped for display (coords mirrored automatically).
    """
    img = Image.open(image_path).convert('RGB')
    full_w, full_h = img.size
    img = img.resize((int(full_w * display_scale), int(full_h * display_scale)), Image.LANCZOS)
    is_rear = cam_name in REAR_CAMERAS
    if is_rear:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    scale = img.size[0] / full_w
    mirror_w = full_w if is_rear else None
    if show_lights and lights:
        img = draw_traffic_lights(img, lights, scale, mirror_w)
    if show_poles and poles:
        img = draw_poles(img, poles, scale, mirror_w)
    if cam_label:
        _draw_label(ImageDraw.Draw(img), (4, 4), CAM_SHORT_LABELS[cam_name], (15, 52, 96), _get_font(14))
    return img


def render_detection_composite(sample, loader, lights, poles, display_scale=0.5,
                               bev_range=50.0, with_bev=True):
    """Render one composite PIL image per sample: annotated 3x2 6-view grid,
    with the BEV (GT objects + lifted 3D detections) pasted on the right.

    lights/poles: full per-sample detection lists (each entry carries a
    'camera' field); they are grouped per view internally.
    """
    lights_by_cam, poles_by_cam = {}, {}
    for tl in lights:
        lights_by_cam.setdefault(tl.get('camera'), []).append(tl)
    for p in poles:
        poles_by_cam.setdefault(p.get('camera'), []).append(p)

    views = []
    for cam_name in CAMERA_ORDER:
        views.append(render_annotated_view(
            cam_name, sample.cameras[cam_name].image_path,
            lights_by_cam.get(cam_name, []), poles_by_cam.get(cam_name, []),
            display_scale=display_scale,
        ))

    cell_w, cell_h = views[0].size
    grid_w, grid_h = cell_w * 3, cell_h * 2

    bev_img = None
    if with_bev:
        # Lazy import: qa_visualizer pulls in matplotlib/flask, which the
        # detection module should not require unless --visualize is used.
        from nuscenes_pipeline.visualization.qa_visualizer import generate_bev
        bev_b64 = generate_bev(sample, loader, bev_range=bev_range,
                               overlay_fn=make_detection_overlay_fn(lights, poles))
        bev_img = Image.open(io.BytesIO(base64.b64decode(bev_b64.split(',', 1)[1]))).convert('RGB')
        bev_img = bev_img.resize((int(bev_img.width * grid_h / bev_img.height), grid_h), Image.LANCZOS)

    canvas = Image.new('RGB', (grid_w + (bev_img.width if bev_img else 0), grid_h), (17, 17, 17))
    for i, view in enumerate(views):
        canvas.paste(view, ((i % 3) * cell_w, (i // 3) * cell_h))
    if bev_img is not None:
        canvas.paste(bev_img, (grid_w, 0))

    footer = (f"sample {sample.sample_idx} | {sample.token} | "
              f"{len(lights)} lights, {len(poles)} pole segments")
    _draw_label(ImageDraw.Draw(canvas), (4, grid_h - 20), footer, (15, 52, 96), _get_font(14))
    return canvas
