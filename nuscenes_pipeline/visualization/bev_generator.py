"""
Bird's Eye View (BEV) Visualization Generator for nuScenes Samples

Generates ego-centric BEV visualizations with:
  - Ego vehicle rectangle and heading arrow
  - Ego velocity vector (computed from consecutive frames)
  - 3D ground truth objects with oriented bounding boxes
  - Object heading arrows and velocity vectors
  - Range circles and grid overlay
  - Dark theme with color-coded object classes

The scene is rotated so the ego vehicle always faces upward (+Y direction).

Usage:
    # Generate BEV for a single sample
    python -m nuscenes_pipeline.visualization.bev_generator --sample_idx 42

    # Generate BEV for a range of samples
    python -m nuscenes_pipeline.visualization.bev_generator --start_idx 0 --end_idx 100

    # Custom output directory and range
    python -m nuscenes_pipeline.visualization.bev_generator --start_idx 0 --end_idx 50 \\
        --output_dir bev_vis_results --bev_range 60
"""

import os
import io
import base64
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import matplotlib.patches as mpatches

from nuscenes_pipeline.core.nuscenes_data_loader import NuScenesDataLoader


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EGO_LENGTH = 4.7   # meters
EGO_WIDTH = 1.9    # meters
EGO_REAR_TO_CENTER = 2.35  # rear-axle to geometric center (meters)

OBJ_COLORS = {
    'car': '#ff6b6b',
    'truck': '#ffa502',
    'bus': '#ff7f50',
    'trailer': '#cd853f',
    'construction_vehicle': '#daa520',
    'pedestrian': '#ffff00',
    'motorcycle': '#ff69b4',
    'bicycle': '#ee82ee',
    'traffic_cone': '#ff4500',
    'barrier': '#808080',
}
DEFAULT_COLOR = '#ffffff'

VEL_SCALE = 2.0       # velocity arrow scale factor
VEL_THRESHOLD = 0.5   # minimum velocity (m/s) to draw arrow


# ---------------------------------------------------------------------------
# Core BEV generation
# ---------------------------------------------------------------------------

def generate_bev(
    sample,
    loader: NuScenesDataLoader,
    bev_range: float = 50.0,
    figsize: tuple = (12, 12),
    dpi: int = 150,
) -> plt.Figure:
    """
    Create a Bird's Eye View (BEV) figure in ego-centric coordinates.

    Coordinate transform pipeline:
        1. Object positions: ego FLU -> global ENU -> centered at ego -> rotated so ego faces up
        2. Object velocities: ego FLU -> global ENU -> rotated so ego faces up
        3. Ego velocity: computed from consecutive global positions -> rotated

    Args:
        sample: NuScenesSample with gt_boxes, gt_velocity, gt_names, cameras
        loader: NuScenesDataLoader (for CAMERA_NAMES and get_sample)
        bev_range: Range in meters for each direction (default 50 = 100m total)
        figsize: Figure size in inches
        dpi: Resolution

    Returns:
        matplotlib Figure object (caller is responsible for saving/closing)
    """
    # --- Ego pose ---
    first_cam = sample.cameras[loader.CAMERA_NAMES[0]]
    ego_pos_global = np.array(first_cam.ego2global_translation[:2])
    qw, qx, qy, qz = first_cam.ego2global_rotation
    ego_yaw_global = np.arctan2(2 * (qw * qz + qx * qy),
                                1 - 2 * (qy * qy + qz * qz))

    # Rotation: ego frame -> global frame
    cos_yaw = np.cos(ego_yaw_global)
    sin_yaw = np.sin(ego_yaw_global)
    R_ego2global = np.array([[cos_yaw, -sin_yaw],
                              [sin_yaw,  cos_yaw]])

    # Rotation: make ego face up (+Y in plot)
    rot_angle = np.pi / 2 - ego_yaw_global
    cos_rot = np.cos(rot_angle)
    sin_rot = np.sin(rot_angle)
    R_scene = np.array([[cos_rot, -sin_rot],
                         [sin_rot,  cos_rot]])

    # --- Figure setup ---
    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='#1a1a2e')
    ax.set_facecolor('#16213e')
    ax.set_xlim(-bev_range, bev_range)
    ax.set_ylim(-bev_range, bev_range)
    ax.set_aspect('equal')

    # Grid
    grid_spacing = 10
    for i in range(-int(bev_range), int(bev_range) + 1, grid_spacing):
        ax.axhline(y=i, color='#2a3f5f', linewidth=0.5, alpha=0.5)
        ax.axvline(x=i, color='#2a3f5f', linewidth=0.5, alpha=0.5)

    # Range circles
    for r in [10, 20, 30, 40, 50]:
        if r <= bev_range:
            circle = plt.Circle((0, 0), r, fill=False, color='#3a5a8f',
                                linewidth=0.8, linestyle='--', alpha=0.6)
            ax.add_patch(circle)
            ax.text(r * 0.707, r * 0.707, f'{r}m',
                    fontsize=8, color='#6a8abf', alpha=0.8)

    # --- Ego vehicle ---
    ego_rect = Rectangle(
        (-EGO_WIDTH / 2, 0),
        EGO_WIDTH, EGO_LENGTH,
        angle=0,
        linewidth=2, edgecolor='#00ff88', facecolor='#00ff8844',
        zorder=10,
    )
    ax.add_patch(ego_rect)

    # Heading arrow
    ax.annotate('', xy=(0, EGO_LENGTH + 2), xytext=(0, EGO_REAR_TO_CENTER),
                arrowprops=dict(arrowstyle='->', color='#00ff88', lw=2),
                zorder=11)
    ax.text(0, -2, 'EGO', fontsize=10, color='#00ff88',
            ha='center', va='top', fontweight='bold')

    # --- Ego velocity (from next frame) ---
    ego_vel_global = np.array([0.0, 0.0])
    try:
        next_idx = sample.sample_idx + 1
        next_sample = loader.get_sample(next_idx)
        if next_sample.scene_token == sample.scene_token:
            next_cam = next_sample.cameras[loader.CAMERA_NAMES[0]]
            next_pos = np.array(next_cam.ego2global_translation[:2])
            dt = (next_sample.timestamp - sample.timestamp) / 1e6
            if dt > 0:
                ego_vel_global = (next_pos - ego_pos_global) / dt
    except Exception:
        pass

    if np.linalg.norm(ego_vel_global) > VEL_THRESHOLD:
        ego_vel_rotated = R_scene @ ego_vel_global
        vel_end = np.array([0, EGO_REAR_TO_CENTER]) + ego_vel_rotated * VEL_SCALE
        ax.annotate('', xy=(vel_end[0], vel_end[1]),
                    xytext=(0, EGO_REAR_TO_CENTER),
                    arrowprops=dict(arrowstyle='->', color='#00ffff',
                                    lw=2, linestyle='--'),
                    zorder=9)

    # --- Draw objects ---
    objects_drawn = []

    if len(sample.gt_boxes) > 0:
        for obj_idx in range(len(sample.gt_boxes)):
            bbox = sample.gt_boxes[obj_idx]
            velocity = sample.gt_velocity[obj_idx]
            obj_name = (sample.gt_names[obj_idx]
                        if sample.gt_names is not None else "object")

            obj_pos_ego = np.array([bbox[0], bbox[1]])
            obj_l, obj_w = bbox[3], bbox[4]
            obj_yaw_ego = bbox[6]
            distance = np.linalg.norm(obj_pos_ego)

            if abs(obj_pos_ego[0]) > bev_range or abs(obj_pos_ego[1]) > bev_range:
                continue

            # ego -> global -> centered -> scene-rotated
            obj_pos_centered = R_ego2global @ obj_pos_ego
            obj_pos_plot = R_scene @ obj_pos_centered

            obj_yaw_global = obj_yaw_ego + ego_yaw_global
            obj_yaw_plot = obj_yaw_global + rot_angle

            color = OBJ_COLORS.get(obj_name.lower(), DEFAULT_COLOR)

            # Oriented bounding box
            corners_local = np.array([
                [-obj_l / 2, -obj_w / 2],
                [ obj_l / 2, -obj_w / 2],
                [ obj_l / 2,  obj_w / 2],
                [-obj_l / 2,  obj_w / 2],
            ])
            cos_obj = np.cos(obj_yaw_plot)
            sin_obj = np.sin(obj_yaw_plot)
            R_obj = np.array([[cos_obj, -sin_obj],
                               [sin_obj,  cos_obj]])
            corners_plot = (R_obj @ corners_local.T).T + obj_pos_plot

            obj_polygon = Polygon(corners_plot, closed=True,
                                  linewidth=1.5, edgecolor=color,
                                  facecolor=color + '44', zorder=5)
            ax.add_patch(obj_polygon)

            # Heading arrow
            heading_len = min(obj_l, 3.0)
            heading_dir = (np.array([np.cos(obj_yaw_plot),
                                     np.sin(obj_yaw_plot)]) * heading_len)
            ax.annotate('', xy=obj_pos_plot + heading_dir, xytext=obj_pos_plot,
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.5),
                        zorder=6)

            # Velocity vector (ego FLU -> global -> scene-rotated)
            vel_ego = np.array([velocity[0], velocity[1]])
            vel_magnitude = np.linalg.norm(vel_ego)

            if vel_magnitude > VEL_THRESHOLD:
                vel_global = R_ego2global @ vel_ego
                vel_rotated = R_scene @ vel_global
                vel_end = obj_pos_plot + vel_rotated * VEL_SCALE
                ax.annotate('', xy=vel_end, xytext=obj_pos_plot,
                            arrowprops=dict(arrowstyle='->', color='#00ffff',
                                            lw=1, linestyle='--'),
                            zorder=4)

            # Label
            ax.text(obj_pos_plot[0], obj_pos_plot[1] + obj_w / 2 + 1.5,
                    f"{obj_name}\n{distance:.1f}m",
                    fontsize=7, color=color, ha='center', va='bottom', alpha=0.9)

            objects_drawn.append({
                'name': obj_name,
                'distance': distance,
                'position': tuple(obj_pos_plot),
                'velocity': vel_magnitude,
            })

    # --- Coordinate axes ---
    ax.annotate('', xy=(0, 10), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#55ff55', lw=2), zorder=2)
    ax.text(1, 10, 'Forward', fontsize=9, color='#55ff55', va='center')

    ax.annotate('', xy=(10, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#ff5555', lw=2), zorder=2)
    ax.text(10, -1.5, 'Right', fontsize=9, color='#ff5555', ha='center')

    # --- Legend ---
    legend_elements = [
        mpatches.Patch(facecolor='#00ff8844', edgecolor='#00ff88',
                       linewidth=2, label='Ego Vehicle'),
        plt.Line2D([0], [0], color='#00ffff', linewidth=2,
                   linestyle='--', label='Velocity Vector'),
    ]
    drawn_types = set(obj['name'].lower() for obj in objects_drawn)
    for obj_type in sorted(drawn_types):
        c = OBJ_COLORS.get(obj_type, DEFAULT_COLOR)
        legend_elements.append(
            mpatches.Patch(facecolor=c + '44', edgecolor=c,
                           linewidth=1.5, label=obj_type.capitalize())
        )
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8,
              facecolor='#1a1a2e', edgecolor='#3a5a8f', labelcolor='white')

    # --- Title and info ---
    scene_token = sample.scene_token[:16] if sample.scene_token else 'N/A'
    ego_yaw_deg = np.degrees(ego_yaw_global)
    ego_speed = np.linalg.norm(ego_vel_global)

    ax.set_title(
        f'BEV Visualization (Ego-Centric, Ego Faces Up)\n'
        f'Sample {sample.sample_idx} | Scene: {scene_token}... '
        f'| Ego Yaw: {ego_yaw_deg:.1f}\u00b0',
        fontsize=12, color='white', pad=10,
    )
    ax.text(0.02, 0.02,
            f"Objects: {len(objects_drawn)} | Ego Speed: {ego_speed:.1f} m/s "
            f"| Range: \u00b1{bev_range}m",
            transform=ax.transAxes, fontsize=9, color='#aaaaaa', va='bottom')
    ax.set_xlabel('\u2190 Left    |    Right \u2192 (meters)',
                  fontsize=10, color='#aaaaaa')
    ax.set_ylabel('\u2190 Rear    |    Forward \u2192 (meters)',
                  fontsize=10, color='#aaaaaa')
    ax.tick_params(colors='#666666', labelsize=8)

    plt.tight_layout()
    return fig


def save_bev(
    sample,
    loader: NuScenesDataLoader,
    output_path: str,
    bev_range: float = 50.0,
    dpi: int = 150,
):
    """Generate and save a BEV visualization to disk."""
    fig = generate_bev(sample, loader, bev_range=bev_range, dpi=dpi)
    fig.savefig(output_path, dpi=dpi, facecolor='#1a1a2e',
                bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


def bev_to_base64(
    sample,
    loader: NuScenesDataLoader,
    bev_range: float = 50.0,
    dpi: int = 120,
) -> str:
    """Generate a BEV visualization and return as a base64 data URI string."""
    fig = generate_bev(sample, loader, bev_range=bev_range,
                       figsize=(8, 8), dpi=dpi)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, facecolor='#1a1a2e',
                bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate BEV visualizations for nuScenes samples",
    )
    parser.add_argument("--pkl_path", type=str,
                        default=os.environ.get("NUSCENES_PKL_PATH",
                                               "nuscenes2d_ego_temporal_infos_val.pkl"),
                        help="Path to nuScenes pickle file")
    parser.add_argument("--sample_idx", type=int, default=None,
                        help="Single sample index to visualize")
    parser.add_argument("--start_idx", type=int, default=None,
                        help="Start index for range processing")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="End index for range processing")
    parser.add_argument("--output_dir", type=str, default="bev_vis_results",
                        help="Output directory for BEV images")
    parser.add_argument("--bev_range", type=float, default=50.0,
                        help="Range in meters for each direction (default 50)")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Image resolution (default 150)")
    args = parser.parse_args()

    loader = NuScenesDataLoader(args.pkl_path)
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine sample indices
    if args.sample_idx is not None:
        indices = [args.sample_idx]
    elif args.start_idx is not None:
        end = args.end_idx if args.end_idx is not None else len(loader) - 1
        indices = list(range(args.start_idx, end + 1))
    else:
        indices = [0]

    print(f"Generating BEV for {len(indices)} sample(s) -> {args.output_dir}/")

    for idx in indices:
        sample = loader.get_sample(idx)
        scene_tok = sample.scene_token[:16] if sample.scene_token else "unknown"
        sample_tok = sample.token[:16] if sample.token else "unknown"
        filename = f"{idx:04d}_{scene_tok}_{sample_tok}_bev.png"
        output_path = os.path.join(args.output_dir, filename)

        save_bev(sample, loader, output_path,
                 bev_range=args.bev_range, dpi=args.dpi)
        print(f"  [{idx:04d}] saved: {filename}")

    print(f"Done. {len(indices)} BEV image(s) saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
