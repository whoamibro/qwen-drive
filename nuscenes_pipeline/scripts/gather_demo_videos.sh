#!/bin/bash
################################################################################
# Gather rendered demo scene MP4s into one flat directory, numbered temporally
# (by scene order in the val pkl) for easy browsing.
#
# Usage:
#   bash nuscenes_pipeline/scripts/gather_demo_videos.sh [SRC_ROOT] [DEST]
#
#   SRC_ROOT  where to search for <scene>/videos/scene_*.mp4 (searched
#             recursively; default: demo_test_results — i.e. all batches)
#   DEST      flat output dir (default: <this scripts dir>/demo_vid)
#
# Environment overrides:
#   PKL_PATH  (default: <project_root>/data/nuscenes/..._val.pkl)
#
# When the same scene+category appears in more than one batch under SRC_ROOT,
# the batch directory name is added to the filename to avoid collisions.
# Existing up-to-date files in DEST are left untouched (safe to re-run).
################################################################################

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

SRC_ROOT="${1:-demo_test_results}"
DEST="${2:-$SCRIPT_DIR/demo_vid}"
PKL_PATH="${PKL_PATH:-$PROJECT_ROOT/data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl}"

[ -d "$SRC_ROOT" ] || { echo "ERROR: SRC_ROOT=$SRC_ROOT not found" >&2; exit 1; }

python3 - "$PKL_PATH" "$SRC_ROOT" "$DEST" <<'EOF'
import glob, os, pickle, shutil, sys
pkl, src_root, dest = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(dest, exist_ok=True)

d = pickle.load(open(pkl, 'rb'))
scene_order = {}
for i in d['infos']:
    scene_order.setdefault(i['scene_token'], len(scene_order))

vids = sorted(glob.glob(os.path.join(src_root, '**', 'videos', 'scene_*.mp4'),
                        recursive=True))
if not vids:
    print(f"[gather] no scene_*.mp4 found under {src_root}", file=sys.stderr)
    sys.exit(1)

entries = []
for v in vids:
    suffix = os.path.basename(v).split('_', 1)[1]        # <tok16>_<CAT>.mp4
    tok16 = suffix.split('_')[0]
    full = next((t for t in scene_order if t.startswith(tok16)), None)
    if full is None:
        print(f"[gather] WARNING: {v} matches no scene in pkl, skipping", file=sys.stderr)
        continue
    batch = os.path.relpath(v, src_root).split(os.sep)[0]
    entries.append((scene_order[full], batch, suffix, v))

# Disambiguate with the batch name only when a scene+category repeats across batches.
from collections import Counter
dup = Counter(s for _, _, s, _ in entries)

n_copied = 0
width = max(2, len(str(len(entries))))
for n, (_, batch, suffix, v) in enumerate(sorted(entries), 1):
    name = f"{n:0{width}d}_" + (f"{batch}_" if dup[suffix] > 1 else "") + f"scene_{suffix}"
    dst = os.path.join(dest, name)
    if not os.path.exists(dst) or os.path.getmtime(v) > os.path.getmtime(dst):
        shutil.copy2(v, dst)
        n_copied += 1
print(f"[gather] {len(entries)} videos ({n_copied} copied) -> {dest}")
EOF
