"""
Apply hand-labeled driving commands (Phase 4) into the demo pkls.

Reads the per-scene JSONs written by the driving-command labeler
(<out_root>/command_labels/scene_*.json), maps sample_token → command, and
replaces the DUMMY (-1) `gt_navigation_command` / `gt_planning_command` in
every subclip pkl (and the merged labeling pkl, when present).

Samples the labeler skipped are filled from the nearest labeled sample in
the SAME scene (temporal continuity) and reported explicitly; if a whole
scene is unlabeled its samples stay at -1 (also reported).

Usage:
    python -m custom_dataset.apply_command_labels \\
        --out_root /yjj_rnd_home/data/vlm_dataset/demo_dataset_nusc
"""

import argparse
import glob
import json
import os
import pickle
import re

SUBCLIP_RE = re.compile(r'^demo_.+_\d{2}\.pkl$')


def load_labels(labels_dir):
    """{sample_token: command} from all scene_*.json files."""
    labels = {}
    for path in sorted(glob.glob(os.path.join(labels_dir, 'scene_*.json'))):
        with open(path) as f:
            data = json.load(f)
        for token, idx_map in data.get('sample_labels', {}).items():
            for _gidx, label in idx_map.items():
                labels[token] = int(label)
    return labels


def apply_to_pkl(path, labels):
    """Write commands into one pkl. Returns (n_direct, filled, unlabeled)."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    infos = data['infos']

    n_direct = 0
    filled = []      # (token, borrowed_from_index_distance, value)
    unlabeled = []
    for i, info in enumerate(infos):
        cmd = labels.get(info['token'])
        if cmd is None:
            # Nearest labeled neighbor within the same scene.
            best = None
            for j, other in enumerate(infos):
                if (other['scene_token'] == info['scene_token']
                        and other['token'] in labels):
                    d = abs(j - i)
                    if best is None or d < best[0]:
                        best = (d, labels[other['token']])
            if best is None:
                unlabeled.append(info['token'])
                continue
            filled.append((info['token'], best[0], best[1]))
            cmd = best[1]
        else:
            n_direct += 1
        info['gt_navigation_command'] = cmd
        info['gt_planning_command'] = cmd

    with open(path, 'wb') as f:
        pickle.dump(data, f)
    return n_direct, filled, unlabeled


def main():
    parser = argparse.ArgumentParser(
        description='Write labeled driving commands into the demo pkls')
    parser.add_argument('--out_root', type=str,
                        default='/yjj_rnd_home/data/vlm_dataset/demo_dataset_nusc')
    parser.add_argument('--labels_dir', type=str, default=None,
                        help='Default: <out_root>/command_labels')
    args = parser.parse_args()

    labels_dir = args.labels_dir or os.path.join(args.out_root,
                                                 'command_labels')
    labels = load_labels(labels_dir)
    print(f'{len(labels)} labeled samples loaded from {labels_dir}')

    targets = sorted(f for f in os.listdir(args.out_root)
                     if SUBCLIP_RE.match(f))
    merged = os.path.join(args.out_root, 'demo_all_subclips.pkl')
    paths = [os.path.join(args.out_root, f) for f in targets]
    if os.path.isfile(merged):
        paths.append(merged)

    total_direct = 0
    for path in paths:
        n_direct, filled, unlabeled = apply_to_pkl(path, labels)
        total_direct += n_direct
        tag = os.path.basename(path)
        extra = ''
        if filled:
            extra += ''.join(
                f'\n    FILLED {t[:16]}... from neighbor {d} step(s) away '
                f'-> {v}' for t, d, v in filled)
        if unlabeled:
            extra += f'\n    UNLABELED (left -1): ' + ', '.join(
                t[:16] for t in unlabeled)
        print(f'  {tag}: {n_direct} direct{extra}')
    print('Done.')


if __name__ == '__main__':
    main()
