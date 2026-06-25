"""
Panoramic 6-view Visualization Generator for nuScenes Samples

Tiles the 6 surround cameras into a 2x3 grid in ego-centric viewing order:

    Row 1: | CAM_FRONT_LEFT | CAM_FRONT | CAM_FRONT_RIGHT |
    Row 2: | CAM_BACK_LEFT* | CAM_BACK* | CAM_BACK_RIGHT* |   (* horizontally flipped)

Rear cameras are mirrored so left/right in the image stays consistent with the
ego's perspective (something on the ego's left appears on the left of the tile).

Usage:
    # Single sample
    python -m nuscenes_pipeline.visualization.pan_generator --sample_idx 42

    # Range
    python -m nuscenes_pipeline.visualization.pan_generator \\
        --pkl_path ./data/nuscenes/nuscenes2d_ego_temporal_infos_train.pkl \\
        --start_idx 0 \\
        --output_dir pan_vis_results_train
"""

import os
import argparse
from PIL import Image

from nuscenes_pipeline.core.nuscenes_data_loader import NuScenesDataLoader


CAMERA_ORDER = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT',  'CAM_BACK',  'CAM_BACK_RIGHT',
]
REAR_INDICES = {3, 4, 5}
GRID_COLS = 3
GRID_ROWS = 2


def build_panoramic_grid(sample, resize_factor: int = 2) -> Image.Image:
    """Load 6 cameras, flip rear ones, tile into a 2x3 PIL image."""
    tiles = []
    for view_idx, cam_name in enumerate(CAMERA_ORDER):
        img = Image.open(sample.cameras[cam_name].image_path).convert('RGB')
        if resize_factor > 1:
            w, h = img.size
            img = img.resize((w // resize_factor, h // resize_factor), Image.LANCZOS)
        if view_idx in REAR_INDICES:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        tiles.append(img)

    # All nuScenes cams share 1600x900; after a uniform resize they tile cleanly.
    # We normalize to the first tile's size in case any camera differs.
    tile_w, tile_h = tiles[0].size
    canvas = Image.new('RGB', (tile_w * GRID_COLS, tile_h * GRID_ROWS))
    for idx, tile in enumerate(tiles):
        if tile.size != (tile_w, tile_h):
            tile = tile.resize((tile_w, tile_h), Image.LANCZOS)
        row, col = divmod(idx, GRID_COLS)
        canvas.paste(tile, (col * tile_w, row * tile_h))
    return canvas


def main():
    parser = argparse.ArgumentParser(
        description="Generate 2x3 panoramic camera grids for nuScenes samples",
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
                        help="End index for range processing (inclusive)")
    parser.add_argument("--output_dir", type=str, default="pan_vis_results",
                        help="Output directory for panoramic images")
    parser.add_argument("--resize_factor", type=int, default=2,
                        help="Per-tile downscale factor (default 2 -> 800x450 tiles)")
    args = parser.parse_args()

    loader = NuScenesDataLoader(args.pkl_path)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.sample_idx is not None:
        indices = [args.sample_idx]
    elif args.start_idx is not None:
        end = args.end_idx if args.end_idx is not None else len(loader) - 1
        indices = list(range(args.start_idx, end + 1))
    else:
        indices = [0]

    print(f"Generating panoramic for {len(indices)} sample(s) -> {args.output_dir}/")

    for idx in indices:
        sample = loader.get_sample(idx)
        scene_tok = sample.scene_token[:16] if sample.scene_token else "unknown"
        sample_tok = sample.token[:16] if sample.token else "unknown"
        filename = f"{idx:04d}_{scene_tok}_{sample_tok}_pan.png"
        output_path = os.path.join(args.output_dir, filename)

        grid = build_panoramic_grid(sample, resize_factor=args.resize_factor)
        grid.save(output_path)
        print(f"  [{idx:04d}] saved: {filename}")

    print(f"Done. {len(indices)} panoramic image(s) saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
