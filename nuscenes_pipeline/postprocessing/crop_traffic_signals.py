"""
Crop every detected traffic SIGNAL (not pole) out of the nuScenes camera images
using the `tl_bboxes2d` boxes stored in the infos pkl by
`add_traffic_lights_to_infos.py`, and save them for downstream status
classification (light observable / unobservable; red / yellow / green).

Output layout (default --output_dir ./cropped_ts_p):

    cropped_ts_p/
        train/{split}_{sample_idx:05d}_{token}_{CAM_NAME}_s{sig_idx:02d}.jpg
        val/  ...
        train_manifest.jsonl     one row per crop (see below)
        val_manifest.jsonl

File name = the full address of the box inside the pkl, so a classifier's
output can be written back without any lookup:
    split       -> which pkl
    sample_idx  -> data['infos'][sample_idx]
    token       -> sanity check against info['token']
    CAM_NAME    -> cam_idx = list(info['cams']).index(CAM_NAME)
    sig_idx     -> info['tl_bboxes2d'][cam_idx][sig_idx]  (and tl_scores2d / tl_pole_idx2d)

Manifest row (jsonl):
    {"file", "split", "sample_idx", "token", "camera", "cam_idx", "sig_idx",
     "det_id", "bbox_xyxy", "score", "pole_idx", "crop_w", "crop_h",
     "pad_px", "image_path"}

Crops are taken from the full-resolution 1600x900 image (same space as the
boxes). `--pad` adds context pixels around the box (clipped to the image);
default 0 = exact bbox. Existing crops are skipped unless --overwrite.

Usage:
    python -m nuscenes_pipeline.postprocessing.crop_traffic_signals \\
        --pkl_path data/nuscenes/nuscenes2d_ego_temporal_infos_train.pkl --split train
    python -m nuscenes_pipeline.postprocessing.crop_traffic_signals \\
        --pkl_path data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl --split val
"""

import argparse
import json
import os
import pickle
from multiprocessing import Pool

from PIL import Image


def resolve_image_path(data_path, data_root):
    if os.path.exists(data_path):
        return data_path
    rel = data_path
    for prefix in ('./data/nuscenes/', 'data/nuscenes/'):
        if rel.startswith(prefix):
            rel = rel[len(prefix):]
            break
    return os.path.join(data_root, rel)


def crop_name(split, sample_idx, token, cam, sig_idx):
    return f'{split}_{sample_idx:05d}_{token}_{cam}_s{sig_idx:02d}.jpg'


def build_jobs(infos, split, out_img_dir, data_root):
    """One job per camera image that contains at least one signal."""
    jobs = []
    for sample_idx, info in enumerate(infos):
        cam_names = list(info['cams'].keys())
        for cam_idx, cam in enumerate(cam_names):
            boxes = info['tl_bboxes2d'][cam_idx]
            if len(boxes) == 0:
                continue
            scores = info['tl_scores2d'][cam_idx]
            ids = info['tl_ids2d'][cam_idx]
            pole_idx = info['tl_pole_idx2d'][cam_idx]
            crops = []
            for sig_idx, box in enumerate(boxes):
                crops.append({
                    'file': os.path.join(split, crop_name(split, sample_idx, info['token'], cam, sig_idx)),
                    'split': split,
                    'sample_idx': sample_idx,
                    'token': info['token'],
                    'camera': cam,
                    'cam_idx': cam_idx,
                    'sig_idx': sig_idx,
                    'det_id': ids[sig_idx] if sig_idx < len(ids) else None,
                    'bbox_xyxy': [round(float(v), 1) for v in box],
                    'score': round(float(scores[sig_idx]), 4),
                    'pole_idx': int(pole_idx[sig_idx]),
                })
            jobs.append({
                'image_path': resolve_image_path(info['cams'][cam]['data_path'], data_root),
                'crops': crops,
            })
    return jobs


_CFG = {}


def _init(out_dir, pad, overwrite, quality):
    _CFG.update(out_dir=out_dir, pad=pad, overwrite=overwrite, quality=quality)


def _process(job):
    out_dir, pad = _CFG['out_dir'], _CFG['pad']
    rows, n_skipped = [], 0
    todo = [c for c in job['crops']
            if _CFG['overwrite'] or not os.path.exists(os.path.join(out_dir, c['file']))]
    img = None
    if todo:
        try:
            img = Image.open(job['image_path']).convert('RGB')
        except (OSError, FileNotFoundError) as e:
            return [], 0, [f"{job['image_path']}: {e}"]
    for c in job['crops']:
        row = dict(c)
        row['pad_px'] = pad
        row['image_path'] = job['image_path']
        out_path = os.path.join(out_dir, c['file'])
        if c not in todo:
            n_skipped += 1
            try:
                with Image.open(out_path) as ex:
                    row['crop_w'], row['crop_h'] = ex.size
            except OSError:
                row['crop_w'] = row['crop_h'] = None
            rows.append(row)
            continue
        w, h = img.size
        x1, y1, x2, y2 = c['bbox_xyxy']
        x1 = max(0, int(x1 - pad)); y1 = max(0, int(y1 - pad))
        x2 = min(w, int(round(x2 + pad))); y2 = min(h, int(round(y2 + pad)))
        if x2 - x1 < 1: x2 = min(w, x1 + 1)
        if y2 - y1 < 1: y2 = min(h, y1 + 1)
        crop = img.crop((x1, y1, x2, y2))
        crop.save(out_path, quality=_CFG['quality'])
        row['crop_w'], row['crop_h'] = crop.size
        rows.append(row)
    return rows, n_skipped, []


def main():
    parser = argparse.ArgumentParser(description='Crop detected traffic signals from nuScenes images')
    parser.add_argument('--pkl_path', type=str, required=True)
    parser.add_argument('--split', type=str, required=True, choices=['train', 'val'])
    parser.add_argument('--output_dir', type=str, default='./cropped_ts_p')
    parser.add_argument('--data_root', type=str, default='./data/nuscenes',
                        help='Used when the pkl data_path does not exist as-is')
    parser.add_argument('--pad', type=int, default=0, help='Context pixels added around each box')
    parser.add_argument('--quality', type=int, default=95)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--limit', type=int, default=None, help='Only process the first N infos (debug)')
    args = parser.parse_args()

    with open(args.pkl_path, 'rb') as f:
        infos = pickle.load(f)['infos']
    if 'tl_bboxes2d' not in infos[0]:
        raise SystemExit('pkl has no tl_bboxes2d — run add_traffic_lights_to_infos.py first')
    if args.limit:
        infos = infos[:args.limit]

    img_dir = os.path.join(args.output_dir, args.split)
    os.makedirs(img_dir, exist_ok=True)
    jobs = build_jobs(infos, args.split, img_dir, args.data_root)
    n_crops = sum(len(j['crops']) for j in jobs)
    print(f'{args.split}: {len(infos)} infos -> {len(jobs)} images with signals, {n_crops} crops')

    rows, n_skipped, errors = [], 0, []
    with Pool(args.workers, initializer=_init,
              initargs=(args.output_dir, args.pad, args.overwrite, args.quality)) as pool:
        for i, (r, s, e) in enumerate(pool.imap_unordered(_process, jobs, chunksize=8), 1):
            rows.extend(r); n_skipped += s; errors.extend(e)
            if i % 2000 == 0 or i == len(jobs):
                print(f'  {i}/{len(jobs)} images, {len(rows)} crops', flush=True)

    rows.sort(key=lambda r: (r['sample_idx'], r['cam_idx'], r['sig_idx']))
    manifest = os.path.join(args.output_dir, f'{args.split}_manifest.jsonl')
    with open(manifest, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')

    sizes = [(r['crop_w'], r['crop_h']) for r in rows if r['crop_w']]
    tiny = sum(1 for w, h in sizes if w < 16 or h < 16)
    print(f'Wrote {len(rows) - n_skipped} new crops ({n_skipped} already existed) to {img_dir}')
    print(f'Manifest: {manifest} ({len(rows)} rows); crops with a side < 16 px: {tiny}')
    if errors:
        print(f'{len(errors)} image errors, first: {errors[0]}')


if __name__ == '__main__':
    main()
