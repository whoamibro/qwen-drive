"""
Add SAM3 traffic-signal / traffic-light-pole 2D boxes into a nuScenes
temporal-infos pkl.

Reads the per-sample JSONs produced by sam3/scripts/nuscenes/sam3_nuscenes_detection.py
(optionally post-processed by refine_sam3_results.py), i.e. `{idx:04d}_{token}.json`
with `per_view[CAM_NAME].detections[*] = {prompt, bbox_xyxy, score, id, ...}` in
original 1600x900 pixel coordinates, and writes them into every info using the
same per-camera parallel-list convention as the existing `bboxes2d` field
(one entry per camera, in `info['cams']` order):

    info['tl_bboxes2d']       list[6] of float32 (S_i, 4)  signal xyxy, clipped to image
    info['tl_scores2d']       list[6] of float32 (S_i,)
    info['tl_ids2d']          list[6] of list[str]         source detection ids ("D_k")
    info['tl_pole_bboxes2d']  list[6] of float32 (P_i, 4)  pole xyxy, clipped to image
    info['tl_pole_scores2d']  list[6] of float32 (P_i,)
    info['tl_pole_ids2d']     list[6] of list[str]
    info['tl_pole_idx2d']     list[6] of int64   (S_i,)    index into tl_pole_bboxes2d[i]
                                                          of the pole carrying the signal,
                                                          -1 if no pole was associated
    data['metadata']['traffic_light_source']              provenance (dir, rule, stats)

Signal -> pole association (the SAM3 JSONs carry no such link): a pole is a
candidate for a signal when the signal box intersects the pole box expanded by
`--assoc_margin` pixels (same camera). Among candidates the one with the largest
intersection area wins; ties go to the nearest box-center distance.

Samples matched by token. Samples without a result JSON get empty arrays so
every key exists uniformly. Existing `tl_*` keys are overwritten (idempotent).
The pkl is written atomically (tmp file + rename), in place by default.

Usage:
    python -m nuscenes_pipeline.postprocessing.add_traffic_lights_to_infos \\
        --pkl_path data/nuscenes/nuscenes2d_ego_temporal_infos_train.pkl \\
        --results_dir /yjj_rnd_home/research/vla/sam3/out_refined
    python -m nuscenes_pipeline.postprocessing.add_traffic_lights_to_infos \\
        --pkl_path data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl \\
        --results_dir /yjj_rnd_home/research/vla/sam3/out_val_refined
"""

import argparse
import glob
import json
import os
import pickle
from collections import Counter
from datetime import datetime

import numpy as np

SIGNAL_PROMPT = 'Traffic_Signal'
POLE_PROMPT = 'Traffic_Light_Pole'
IMG_W, IMG_H = 1600, 900

TL_KEYS = ('tl_bboxes2d', 'tl_scores2d', 'tl_ids2d',
           'tl_pole_bboxes2d', 'tl_pole_scores2d', 'tl_pole_ids2d',
           'tl_pole_idx2d')


def index_results(results_dir):
    """{token: json_path} from all `{idx}_{token}.json` files."""
    paths = {}
    for path in glob.glob(os.path.join(results_dir, '*.json')):
        base = os.path.basename(path)[:-5]
        if '_' not in base:
            continue
        token = base.split('_', 1)[1]
        paths[token] = path
    return paths


def clip_box(box, w, h):
    x1, y1, x2, y2 = box
    return [min(max(x1, 0.0), w), min(max(y1, 0.0), h),
            min(max(x2, 0.0), w), min(max(y2, 0.0), h)]


def intersection_area(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    return iw * ih if iw > 0 and ih > 0 else 0.0


def associate_signals_to_poles(sig_boxes, pole_boxes, margin):
    """Return int array (S,) with pole index per signal, -1 if none."""
    out = np.full(len(sig_boxes), -1, dtype=np.int64)
    if not len(sig_boxes) or not len(pole_boxes):
        return out
    expanded = [[p[0] - margin, p[1] - margin, p[2] + margin, p[3] + margin]
                for p in pole_boxes]
    for si, s in enumerate(sig_boxes):
        sc = ((s[0] + s[2]) / 2, (s[1] + s[3]) / 2)
        best = None  # (-area, dist, idx)
        for pi, (p, pe) in enumerate(zip(pole_boxes, expanded)):
            area = intersection_area(s, pe)
            if area <= 0:
                continue
            pc = ((p[0] + p[2]) / 2, (p[1] + p[3]) / 2)
            dist = (sc[0] - pc[0]) ** 2 + (sc[1] - pc[1]) ** 2
            key = (-area, dist, pi)
            if best is None or key < best:
                best = key
        if best is not None:
            out[si] = best[2]
    return out


def empty_view():
    return {
        'tl_bboxes2d': np.zeros((0, 4), dtype=np.float32),
        'tl_scores2d': np.zeros((0,), dtype=np.float32),
        'tl_ids2d': [],
        'tl_pole_bboxes2d': np.zeros((0, 4), dtype=np.float32),
        'tl_pole_scores2d': np.zeros((0,), dtype=np.float32),
        'tl_pole_ids2d': [],
        'tl_pole_idx2d': np.zeros((0,), dtype=np.int64),
    }


def build_view(dets, img_w, img_h, min_score, clip, margin, stats):
    sig, pole = [], []
    for det in dets:
        box = det.get('bbox_xyxy')
        if box is None or len(box) != 4:
            stats['bad_box'] += 1
            continue
        score = float(det.get('score', 0.0))
        if min_score is not None and score < min_score:
            stats['dropped_score'] += 1
            continue
        box = [float(v) for v in box]
        if (box[0] < 0 or box[1] < 0 or box[2] > img_w or box[3] > img_h):
            stats['out_of_image'] += 1
            if clip:
                box = clip_box(box, img_w, img_h)
        if box[2] - box[0] <= 0 or box[3] - box[1] <= 0:
            stats['degenerate'] += 1
            continue
        entry = (box, score, str(det.get('id', '')))
        prompt = det.get('prompt')
        if prompt == SIGNAL_PROMPT:
            sig.append(entry)
        elif prompt == POLE_PROMPT:
            pole.append(entry)
        else:
            stats['unknown_prompt'] += 1

    sig_boxes = [e[0] for e in sig]
    pole_boxes = [e[0] for e in pole]
    pole_idx = associate_signals_to_poles(sig_boxes, pole_boxes, margin)
    stats['signals'] += len(sig)
    stats['poles'] += len(pole)
    stats['signals_with_pole'] += int((pole_idx >= 0).sum())
    return {
        'tl_bboxes2d': np.array(sig_boxes, dtype=np.float32).reshape(-1, 4),
        'tl_scores2d': np.array([e[1] for e in sig], dtype=np.float32),
        'tl_ids2d': [e[2] for e in sig],
        'tl_pole_bboxes2d': np.array(pole_boxes, dtype=np.float32).reshape(-1, 4),
        'tl_pole_scores2d': np.array([e[1] for e in pole], dtype=np.float32),
        'tl_pole_ids2d': [e[2] for e in pole],
        'tl_pole_idx2d': pole_idx,
    }


def apply_results(infos, result_paths, min_score, clip, margin):
    stats = Counter()
    postprocess = None
    for info in infos:
        cam_names = list(info['cams'].keys())
        views = []
        path = result_paths.get(info['token'])
        if path is None:
            stats['samples_missing'] += 1
            views = [empty_view() for _ in cam_names]
        else:
            with open(path) as f:
                res = json.load(f)
            if res.get('token') != info['token']:
                raise ValueError(f'token mismatch in {path}: {res.get("token")} vs {info["token"]}')
            if postprocess is None:
                postprocess = res.get('postprocess')
            stats['samples_matched'] += 1
            per_view = res.get('per_view', {})
            for cam in cam_names:
                v = per_view.get(cam)
                if v is None:
                    stats['views_missing'] += 1
                    views.append(empty_view())
                    continue
                views.append(build_view(v.get('detections', []),
                                        v.get('width', IMG_W), v.get('height', IMG_H),
                                        min_score, clip, margin, stats))
        for key in TL_KEYS:
            info[key] = [v[key] for v in views]
        if sum(len(v['tl_bboxes2d']) + len(v['tl_pole_bboxes2d']) for v in views) == 0:
            stats['samples_empty'] += 1
    return stats, postprocess


def main():
    parser = argparse.ArgumentParser(
        description='Write SAM3 traffic signal / pole 2D boxes into a nuScenes infos pkl')
    parser.add_argument('--pkl_path', type=str, required=True)
    parser.add_argument('--results_dir', type=str, required=True,
                        help='SAM3 result JSON dir (e.g. sam3/out_refined, sam3/out_val_refined)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output pkl path (default: overwrite --pkl_path in place)')
    parser.add_argument('--min_score', type=float, default=None,
                        help='Drop detections below this score (default: keep all)')
    parser.add_argument('--no_clip', action='store_true',
                        help='Keep boxes that extend outside the image unclipped '
                             '(default: clip to image bounds like bboxes2d)')
    parser.add_argument('--assoc_margin', type=float, default=20.0,
                        help='Pixels the pole box is expanded by when associating signals (default 20)')
    args = parser.parse_args()

    result_paths = index_results(args.results_dir)
    if not result_paths:
        raise SystemExit(f'No result JSONs found in {args.results_dir}')
    print(f'Indexed {len(result_paths)} result JSONs from {args.results_dir}')

    with open(args.pkl_path, 'rb') as f:
        data = pickle.load(f)
    infos = data['infos']
    print(f'Loaded {len(infos)} infos from {args.pkl_path}')

    stats, postprocess = apply_results(infos, result_paths, args.min_score,
                                       not args.no_clip, args.assoc_margin)

    n_sig, n_pole = stats['signals'], stats['poles']
    with_pole = stats['signals_with_pole']
    print(f"Samples: matched {stats['samples_matched']}, missing {stats['samples_missing']}, "
          f"no detections {stats['samples_empty']}")
    print(f"Signals: {n_sig}  (with pole {with_pole}, "
          f"{100.0 * with_pole / n_sig if n_sig else 0:.1f}%)   Poles: {n_pole}")
    print(f"Boxes out of image: {stats['out_of_image']} "
          f"({'clipped' if not args.no_clip else 'kept'}), degenerate dropped: {stats['degenerate']}, "
          f"dropped by score: {stats['dropped_score']}, unknown prompt: {stats['unknown_prompt']}, "
          f"bad box: {stats['bad_box']}, views missing: {stats['views_missing']}")

    data.setdefault('metadata', {})['traffic_light_source'] = {
        'results_dir': os.path.abspath(args.results_dir),
        'sam3_postprocess': postprocess,
        'min_score': args.min_score,
        'clipped_to_image': not args.no_clip,
        'assoc_rule': (f'signal box intersects pole box expanded by {args.assoc_margin}px '
                       f'(same camera); max intersection area, then nearest center; -1 if none'),
        'coordinate_space': 'original_image_pixels (1600x900)',
        'stats': dict(stats),
        'written_at': datetime.now().isoformat(),
    }

    # Resolve symlinks so an in-place write updates the real file instead of
    # replacing the link with a copy (data/nuscenes/*.pkl are symlinks).
    out_path = os.path.realpath(args.output or args.pkl_path)
    tmp_path = out_path + '.tmp'
    with open(tmp_path, 'wb') as f:
        pickle.dump(data, f)
    os.replace(tmp_path, out_path)
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
