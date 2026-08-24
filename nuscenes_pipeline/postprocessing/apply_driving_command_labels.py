"""
Apply hand-labeled driving commands into a nuScenes temporal-infos pkl.

Reads the per-scene JSONs written by the driving-command labeler
(nuscenes_pipeline.visualization.driving_command_labeler), maps
sample_token -> command, and writes each label into the matching info's
`gt_navigation_command` field. `gt_planning_command` is left untouched
(it keeps the original nuScenes planning command) — this matches how
nuscenes2d_ego_temporal_infos_val.pkl was built.

Samples the labeler skipped inside a partially labeled scene are filled
from the nearest labeled sample of the SAME scene (temporal continuity)
and reported explicitly. Scenes with no labels at all get
`gt_navigation_command = None` so the key exists uniformly.

The pkl is written atomically (tmp file + rename), in place by default.

Usage:
    python -m nuscenes_pipeline.postprocessing.apply_driving_command_labels \\
        --pkl_path /yjj_rnd_home/data/vlm_dataset/nuscenes/nuscenes2d_ego_temporal_infos_train.pkl \\
        --labels_dir /yjj_rnd_home/data/vlm_dataset/qwen-drive-outputs/driving_command_labels_train
"""

import argparse
import glob
import json
import os
import pickle
from collections import Counter, defaultdict


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


def apply_labels(infos, labels):
    """Set gt_navigation_command on every info. Returns stats."""
    scene_positions = defaultdict(list)  # scene_token -> [(pkl_idx, token)]
    for i, info in enumerate(infos):
        scene_positions[info['scene_token']].append((i, info['token']))

    n_direct = 0
    filled = []            # (token, index_distance, value)
    unlabeled_scenes = set()
    for i, info in enumerate(infos):
        cmd = labels.get(info['token'])
        if cmd is None:
            # Nearest labeled neighbor within the same scene.
            best = None
            for j, tok in scene_positions[info['scene_token']]:
                if tok in labels:
                    d = abs(j - i)
                    if best is None or d < best[0]:
                        best = (d, labels[tok])
            if best is None:
                unlabeled_scenes.add(info['scene_token'])
                info['gt_navigation_command'] = None
                continue
            filled.append((info['token'], best[0], best[1]))
            cmd = best[1]
        else:
            n_direct += 1
        info['gt_navigation_command'] = cmd
    return n_direct, filled, unlabeled_scenes


def main():
    parser = argparse.ArgumentParser(
        description='Write labeled driving commands into a nuScenes infos pkl')
    parser.add_argument('--pkl_path', type=str, required=True)
    parser.add_argument('--labels_dir', type=str, required=True)
    parser.add_argument('--output', type=str, default=None,
                        help='Output pkl path (default: overwrite --pkl_path)')
    args = parser.parse_args()

    labels = load_labels(args.labels_dir)
    print(f'{len(labels)} labeled samples loaded from {args.labels_dir}')

    with open(args.pkl_path, 'rb') as f:
        data = pickle.load(f)
    infos = data['infos']
    print(f'{len(infos)} infos loaded from {args.pkl_path}')

    n_direct, filled, unlabeled_scenes = apply_labels(infos, labels)

    print(f'  direct: {n_direct}')
    for tok, d, v in filled:
        print(f'  FILLED {tok[:16]}... from neighbor {d} step(s) away -> {v}')
    n_none = sum(1 for i in infos if i['gt_navigation_command'] is None)
    if unlabeled_scenes:
        print(f'  UNLABELED scenes (gt_navigation_command=None): '
              f'{len(unlabeled_scenes)} scenes, {n_none} samples')
    dist = Counter(i['gt_navigation_command'] for i in infos
                   if i['gt_navigation_command'] is not None)
    print(f'  command distribution: {dict(sorted(dist.items()))}')

    out_path = args.output or args.pkl_path
    tmp_path = out_path + '.tmp'
    with open(tmp_path, 'wb') as f:
        pickle.dump(data, f)
    os.replace(tmp_path, out_path)
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
