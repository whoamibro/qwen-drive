"""
Write VLM-classified traffic-signal status into a nuScenes infos pkl.

Reads the jsonl produced by
`nuscenes_pipeline.modules.traffic_signal_status_classifier` and, for every
signal box in `tl_bboxes2d`, adds parallel per-camera arrays (same layout as
`tl_bboxes2d`: list of 6, one entry per camera in info['cams'] order):

    info['tl_signal_type2d']      list[6] of np.str_ (S_i,)   vehicle | pedestrian | other | not_a_signal | unknown
    info['tl_light_observable2d'] list[6] of bool    (S_i,)   True if the lamp state was readable
    info['tl_light_color2d']      list[6] of np.str_ (S_i,)   red | yellow | green | off | unknown
    info['tl_lit_shape2d']        list[6] of np.str_ (S_i,)   circle | arrow_* | pedestrian | other | unknown
    info['tl_status_conf2d']      list[6] of float32 (S_i,)   model confidence, -1 if no result
    data['metadata']['traffic_light_status_source']           provenance + stats

Signals with no classifier row (or an unparsable response) get
signal_type/color/shape = "unknown", observable = False, conf = -1, so every
array is always the same length as `tl_bboxes2d[i]`. Rows are matched by
(sample_idx, token, cam_idx, sig_idx) — the crop file name encodes the same
address. Idempotent, atomic write, in place by default.

Usage:
    python -m nuscenes_pipeline.postprocessing.apply_traffic_signal_status \\
        --pkl_path data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl \\
        --status traffic_signal_status_results/val_status.jsonl
"""

import argparse
import json
import os
import pickle
from collections import Counter
from datetime import datetime

import numpy as np

STR_DTYPE = "<U16"


def load_status(path):
    """{(sample_idx, cam_idx, sig_idx): row} — later rows win (re-runs)."""
    rows = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated trailing line from an interrupted run
            rows[(r["sample_idx"], r["cam_idx"], r["sig_idx"])] = r
    return rows


def apply_status(infos, status):
    stats = Counter()
    color_counts, type_counts = Counter(), Counter()
    for sample_idx, info in enumerate(infos):
        n_cams = len(info["cams"])
        # drop keys from withdrawn experiments (aspect/orientation labeling)
        info.pop("tl_apparent_aspect2d", None)
        info.pop("tl_orientation2d", None)
        types, obs, colors, shapes, confs = [], [], [], [], []
        for cam_idx in range(n_cams):
            n = len(info["tl_bboxes2d"][cam_idx])
            t = np.full(n, "unknown", dtype=STR_DTYPE)
            o = np.zeros(n, dtype=bool)
            c = np.full(n, "unknown", dtype=STR_DTYPE)
            s = np.full(n, "unknown", dtype=STR_DTYPE)
            k = np.full(n, -1.0, dtype=np.float32)
            for sig_idx in range(n):
                stats["signals"] += 1
                r = status.get((sample_idx, cam_idx, sig_idx))
                if r is None:
                    stats["missing"] += 1
                    continue
                if r["token"] != info["token"]:
                    raise ValueError(f"token mismatch at sample {sample_idx}: {r['token']} vs {info['token']}")
                lab = r.get("label")
                if not lab:
                    stats["unparsed"] += 1
                    continue
                stats["labeled"] += 1
                t[sig_idx] = lab["signal_type"]
                o[sig_idx] = bool(lab["light_observable"])
                c[sig_idx] = lab["light_color"]
                s[sig_idx] = lab["lit_shape"]
                k[sig_idx] = lab["confidence"]
                type_counts[lab["signal_type"]] += 1
                if lab["light_observable"]:
                    color_counts[lab["light_color"]] += 1
                else:
                    stats["unobservable"] += 1
            types.append(t); obs.append(o); colors.append(c); shapes.append(s); confs.append(k)
        info["tl_signal_type2d"] = types
        info["tl_light_observable2d"] = obs
        info["tl_light_color2d"] = colors
        info["tl_lit_shape2d"] = shapes
        info["tl_status_conf2d"] = confs
    return stats, type_counts, color_counts


def main():
    parser = argparse.ArgumentParser(description="Write traffic-signal status labels into a nuScenes infos pkl")
    parser.add_argument("--pkl_path", required=True)
    parser.add_argument("--status", required=True, help="classifier output jsonl")
    parser.add_argument("--output", default=None, help="Output pkl (default: in place)")
    args = parser.parse_args()

    status = load_status(args.status)
    print(f"Loaded {len(status)} status rows from {args.status}")
    with open(args.pkl_path, "rb") as f:
        data = pickle.load(f)
    infos = data["infos"]
    if "tl_bboxes2d" not in infos[0]:
        raise SystemExit("pkl has no tl_bboxes2d — run add_traffic_lights_to_infos.py first")

    stats, type_counts, color_counts = apply_status(infos, status)
    print(f"Signals: {stats['signals']}  labeled {stats['labeled']}  unparsed {stats['unparsed']}  "
          f"missing {stats['missing']}")
    print(f"signal_type: {dict(type_counts)}")
    print(f"light_color (observable only): {dict(color_counts)}  unobservable: {stats['unobservable']}")

    first = next(iter(status.values()), {})
    data.setdefault("metadata", {})["traffic_light_status_source"] = {
        "status_file": os.path.abspath(args.status),
        "pad_px": first.get("pad_px"),
        "stats": dict(stats),
        "signal_type_counts": dict(type_counts),
        "light_color_counts": dict(color_counts),
        "written_at": datetime.now().isoformat(),
    }

    out_path = os.path.realpath(args.output or args.pkl_path)
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(data, f)
    os.replace(tmp_path, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
