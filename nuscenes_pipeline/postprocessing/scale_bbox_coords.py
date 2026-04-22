"""
Scale bbox coordinates in SFT dataset JSON files.

Bbox coordinates were computed on resized images (resize_factor=2 -> 800x450).
This script multiplies the x1,y1,x2,y2 values of every bbox[...] pattern by a
scale factor so the coordinates match the ORIGINAL image resolution (1600x900).

Pattern matched:
    bbox[x1,y1,x2,y2]   (4 comma-separated non-negative integers, no spaces)

Example:
    Before: "car (Image 4 (Rear-left) bbox[691,219,712,247])"
    After:  "car (Image 4 (Rear-left) bbox[1382,438,1424,494])"   (scale=2)

Usage:
    # Default: scale by 2, process train + val in-place in sft_dataset/
    python -m nuscenes_pipeline.postprocessing.scale_bbox_coords

    # Dry run (report counts without modifying files)
    python -m nuscenes_pipeline.postprocessing.scale_bbox_coords --dry_run

    # Custom scale factor (e.g., for resize_factor=4 -> scale=4)
    python -m nuscenes_pipeline.postprocessing.scale_bbox_coords --scale 4

    # Custom data dir or file suffix
    python -m nuscenes_pipeline.postprocessing.scale_bbox_coords --data_dir sft_dataset_v4

IMPORTANT:
    This operation is NOT idempotent. Running it twice would multiply coords
    by the scale factor twice. A marker file
    `.bbox_scaled_<scale>x` is written alongside processed JSONs to detect
    re-runs; use --force to override.
"""

import os
import re
import json
import argparse


# Pattern: bbox[x1,y1,x2,y2] with 4 non-negative integers, no spaces
# We anchor on 'bbox[' to avoid collisions with other bracketed lists.
_BBOX_RE = re.compile(r'bbox\[(\d+),(\d+),(\d+),(\d+)\]')


def scale_bbox_in_text(text: str, scale: int) -> tuple[str, int]:
    """Return (scaled_text, num_replacements) for all bbox[...] patterns."""
    count = 0

    def _sub(m):
        nonlocal count
        count += 1
        x1, y1, x2, y2 = (int(m.group(i)) for i in range(1, 5))
        return f"bbox[{x1*scale},{y1*scale},{x2*scale},{y2*scale}]"

    return _BBOX_RE.sub(_sub, text), count


def scale_conversation(conv: dict, scale: int) -> int:
    """Scale bbox coords in a single SFT entry. Returns total replacements."""
    total = 0
    if 'conversations' not in conv:
        return 0
    for turn in conv['conversations']:
        if turn.get('from') in ('human', 'gpt', 'gt'):
            text = turn.get('value', '')
            if not isinstance(text, str):
                continue
            scaled, n = scale_bbox_in_text(text, scale)
            if n > 0:
                turn['value'] = scaled
                total += n
    return total


def scale_dataset(data: list, scale: int) -> dict:
    """Scale all bbox coords across the dataset. Returns stats."""
    totals = {'bboxes_scaled': 0, 'samples_modified': 0}
    for entry in data:
        n = scale_conversation(entry, scale)
        if n > 0:
            totals['samples_modified'] += 1
            totals['bboxes_scaled'] += n
    return totals


def main():
    parser = argparse.ArgumentParser(
        description="Scale bbox[x1,y1,x2,y2] coordinates in SFT dataset JSON files",
    )
    parser.add_argument('--data_dir', type=str, default='sft_dataset',
                        help='Directory containing SFT train/val JSON files')
    parser.add_argument('--scale', type=int, default=2,
                        help='Multiplier applied to every bbox coordinate (default: 2)')
    parser.add_argument('--suffix', type=str, default='',
                        help='Suffix appended to base filenames, e.g. "_no_contrast"')
    parser.add_argument('--dry_run', action='store_true',
                        help='Report counts without modifying files')
    parser.add_argument('--force', action='store_true',
                        help='Skip the idempotency check and scale again even if a marker file exists')
    args = parser.parse_args()

    filenames = [
        f'sft_train_no_objlist{args.suffix}.json',
        f'sft_val_no_objlist{args.suffix}.json',
    ]

    marker_path = os.path.join(args.data_dir, f'.bbox_scaled_{args.scale}x')
    if os.path.exists(marker_path) and not args.force and not args.dry_run:
        print(f"ERROR: marker file exists at {marker_path}")
        print(f"  A previous run of this script already scaled bboxes by {args.scale}x.")
        print(f"  Re-running would multiply coordinates a second time.")
        print(f"  Pass --force to override, or delete the marker file manually.")
        return

    print(f"Scaling bbox coordinates by {args.scale}x in {args.data_dir}/")
    if args.dry_run:
        print("[DRY RUN] No files will be modified.")
    print()

    grand_totals = {'bboxes_scaled': 0, 'samples_modified': 0}

    for fname in filenames:
        fpath = os.path.join(args.data_dir, fname)
        if not os.path.exists(fpath):
            print(f"  {fname}: not found, skipping")
            continue

        print(f"Processing {fname}...")
        with open(fpath, 'r') as f:
            data = json.load(f)
        print(f"  Loaded {len(data):,} samples")

        totals = scale_dataset(data, args.scale)
        print(f"  Bboxes scaled:     {totals['bboxes_scaled']:,}")
        print(f"  Samples modified:  {totals['samples_modified']:,} / {len(data):,}")

        grand_totals['bboxes_scaled'] += totals['bboxes_scaled']
        grand_totals['samples_modified'] += totals['samples_modified']

        if not args.dry_run:
            with open(fpath, 'w') as f:
                json.dump(data, f, ensure_ascii=False)
            print(f"  Saved: {fpath}")
        print()

    print("=" * 60)
    print(f"Total bboxes scaled across all files: {grand_totals['bboxes_scaled']:,}")
    print(f"Total samples modified:               {grand_totals['samples_modified']:,}")
    print("=" * 60)

    if not args.dry_run and grand_totals['bboxes_scaled'] > 0:
        with open(marker_path, 'w') as f:
            f.write(f"Bbox coordinates scaled by {args.scale}x.\n")
        print(f"Wrote marker file: {marker_path}")


if __name__ == "__main__":
    main()
