import os
import cv2
import numpy as np
from glob import glob
import subprocess
import argparse

def get_id_from_visualization(filename):
    """Extract ID from visualization filename (index 0 after split by '_')"""
    basename = os.path.basename(filename)
    parts = basename.replace('.png', '').split('_')
    return parts[0]

def get_id_from_bev(filename):
    """Extract ID from BEV filename (index 0 after split by '_')"""
    basename = os.path.basename(filename)
    parts = basename.replace('.png', '').split('_')
    return parts[0]

def main():
    parser = argparse.ArgumentParser(description='Create visualization video')
    parser.add_argument('--output', '-o', type=str, default='combined_visualization.mp4',
                        help='Output video filename')
    parser.add_argument('--start', '-s', type=int, default=0,
                        help='Start index')
    parser.add_argument('--end', '-e', type=int, default=100,
                        help='End index')
    parser.add_argument('--sv_dir', type=str, default='.',
                        help='Directory to save the output video')
    parser.add_argument('--vis_dir', type=str, default='pan_vis_results',
                        help='Directory containing panoramic visualization images')
    parser.add_argument('--bev_dir', type=str, default='bev_vis_results',
                        help='Directory containing BEV visualization images')
    args = parser.parse_args()

    vis_dir = args.vis_dir
    bev_dir = args.bev_dir
    temp_dir = os.path.join(args.sv_dir, 'temp_frames')

    # Create save directory if it doesn't exist
    os.makedirs(args.sv_dir, exist_ok=True)
    output_video = os.path.join(args.sv_dir, args.output)

    # Create temp directory for combined frames
    os.makedirs(temp_dir, exist_ok=True)

    # Target IDs based on start and end arguments
    target_ids = [f'{i:04d}' for i in range(args.start, args.end)]

    # Build mapping from ID to file path
    vis_files = glob(os.path.join(vis_dir, '*.png'))
    bev_files = glob(os.path.join(bev_dir, '*.png'))

    vis_id_map = {get_id_from_visualization(f): f for f in vis_files}
    bev_id_map = {get_id_from_bev(f): f for f in bev_files}

    # Collect paired images for target IDs
    paired_images = []
    for target_id in target_ids:
        if target_id in vis_id_map and target_id in bev_id_map:
            paired_images.append({
                'id': target_id,
                'vis': vis_id_map[target_id],
                'bev': bev_id_map[target_id]
            })
        else:
            print(f"Warning: ID {target_id} not found in both directories")

    # Sort by ID
    paired_images.sort(key=lambda x: x['id'])

    if not paired_images:
        print("No paired images found!")
        return

    print(f"Found {len(paired_images)} paired images")

    # Read first pair to determine dimensions
    first_vis = cv2.imread(paired_images[0]['vis'])
    first_bev = cv2.imread(paired_images[0]['bev'])

    vis_h, vis_w = first_vis.shape[:2]
    bev_h, bev_w = first_bev.shape[:2]

    print(f"Visualization image size: {vis_w}x{vis_h}")
    print(f"BEV image size: {bev_w}x{bev_h}")

    # Match heights for side-by-side display
    target_height = max(vis_h, bev_h)
    # Make height divisible by 2 for H.264 encoding
    target_height = target_height if target_height % 2 == 0 else target_height + 1

    # Calculate new widths maintaining aspect ratio
    new_vis_w = int(vis_w * target_height / vis_h)
    new_bev_w = int(bev_w * target_height / bev_h)

    frame_width = new_vis_w + new_bev_w
    # Make width divisible by 2 for H.264 encoding
    frame_width = frame_width if frame_width % 2 == 0 else frame_width + 1
    frame_height = target_height

    print(f"Output frame size: {frame_width}x{frame_height}")

    # Process each pair and save as temp frames
    for i, pair in enumerate(paired_images):
        print(f"Processing ID: {pair['id']}")

        vis_img = cv2.imread(pair['vis'])
        bev_img = cv2.imread(pair['bev'])

        # Resize to match target height
        vis_resized = cv2.resize(vis_img, (new_vis_w, target_height))
        bev_resized = cv2.resize(bev_img, (new_bev_w, target_height))

        # Concatenate side by side (visualization on left, BEV on right)
        combined = np.hstack([vis_resized, bev_resized])

        # Ensure frame dimensions match expected size (pad if needed)
        if combined.shape[1] != frame_width:
            padded = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            padded[:, :combined.shape[1]] = combined
            combined = padded

        # Save as temp frame
        frame_path = os.path.join(temp_dir, f'frame_{i:04d}.png')
        cv2.imwrite(frame_path, combined)

    # Use ffmpeg to create video with H.264 codec
    print("\nCreating video with ffmpeg...")
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-framerate', '2',
        '-i', os.path.join(temp_dir, 'frame_%04d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '18',
        output_video
    ]
    subprocess.run(ffmpeg_cmd, check=True)

    # Clean up temp frames
    for f in glob(os.path.join(temp_dir, '*.png')):
        os.remove(f)
    os.rmdir(temp_dir)

    print(f"\nVideo saved to: {output_video}")

if __name__ == '__main__':
    main()
