"""
Merge the per-subclip demo pkls into ONE pkl for tools that take a single
--pkl_path (e.g. the driving-command labeler). Subclip scene boundaries are
preserved — every subclip keeps its own scene_token, and the labeler groups
contiguous samples by scene_token, so the merged file labels exactly like
24 short scenes.

The merged file lives next to the subclip pkls (same data_root), so all
relative sample paths keep resolving.

Usage:
    python -m custom_dataset.merge_subclip_pkls \\
        --out_root /yjj_rnd_home/data/vlm_dataset/demo_dataset_nusc
"""

import argparse
import os
import pickle
import re

SUBCLIP_RE = re.compile(r'^demo_.+_\d{2}\.pkl$')


def main():
    parser = argparse.ArgumentParser(
        description='Merge subclip pkls into one labeling pkl')
    parser.add_argument('--out_root', type=str,
                        default='/yjj_rnd_home/data/vlm_dataset/demo_dataset_nusc')
    parser.add_argument('--output_name', type=str,
                        default='demo_all_subclips.pkl')
    parser.add_argument('--per_clip', action='store_true',
                        help='Instead of one merged file, write one pkl per '
                             'CLIP (grouping its subclips, e.g. 8 x 15 '
                             'samples -> <clip>.pkl with 8 scenes).')
    args = parser.parse_args()

    names = sorted(f for f in os.listdir(args.out_root)
                   if SUBCLIP_RE.match(f))
    if not names:
        raise SystemExit(f'no subclip pkls in {args.out_root}')

    def write_merged(out_path, subclip_names):
        infos = []
        for name in subclip_names:
            with open(os.path.join(args.out_root, name), 'rb') as f:
                infos.extend(pickle.load(f)['infos'])
        with open(out_path, 'wb') as f:
            pickle.dump({'metadata': {'version': 'custom-demo-v1-merged'},
                         'infos': infos}, f)
        n_scenes = len({i['scene_token'] for i in infos})
        print(f'{len(subclip_names)} pkls -> {out_path} '
              f'({len(infos)} samples, {n_scenes} scenes)')

    if args.per_clip:
        groups = {}
        for name in names:
            clip = re.sub(r'_\d{2}\.pkl$', '', name)
            groups.setdefault(clip, []).append(name)
        for clip, group in sorted(groups.items()):
            write_merged(os.path.join(args.out_root, f'{clip}.pkl'),
                         sorted(group))
    else:
        write_merged(os.path.join(args.out_root, args.output_name), names)


if __name__ == '__main__':
    main()
