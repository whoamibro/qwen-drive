#!/bin/bash
################################################################################
# Demo batch runner — N random val scenes through run_demo_tester_8gpu.sh
#
# Samples N scenes from the val pkl (seeded, full 32-char tokens), runs the
# 8-GPU demo tester on each with a single custom question (Mode B), and
# optionally renders one MP4 per scene via demo_scene_video.
#
# Prerequisites:
#   - GPUs free (the tester loads a full bf16 model copy per worker)
#   - ckpts/qwen3_vl_8b_instruct present (or BASE_MODEL override)
#
# Usage:
#   LORA_PATH=output/curriculum_v2_c10_0621/C10__seed0/ckpt_100 \
#   bash nuscenes_pipeline/scripts/run_demo_batch.sh [N_SCENES] [SEED]
#
#   N_SCENES  number of random scenes (default: 30)
#   SEED      sampling seed (default: 42)
#
# Environment overrides:
#   LORA_PATH    (REQUIRED unless VLLM_URL is set) LoRA adapter dir
#   VLLM_URL     OpenAI-compatible server URL (e.g. http://localhost:8000/v1);
#                switches inference to the vLLM backend — no in-process model
#   MODEL_NAME   served model to request with VLLM_URL (default: qwen3vl-8b);
#                use a --lora-modules adapter name for the fine-tuned mode
#   QUESTION     (default: the safety-awareness demo question)
#   BATCH_NAME   output subdir under demo_test_results/ (default: batch_<lora leafdir>)
#   PKL_PATH     (default: <project_root>/data/nuscenes/..._val.pkl, absolute)
#   SCENES_FILE  file with one scene token per line — skips sampling entirely
#   RENDER_VIDEO=1  render scene MP4s after inference (needs ffmpeg on PATH;
#                   e.g. PATH="/opt/conda/envs/cosmos-transfer1/bin:$PATH"),
#                   then gather them into DEMO_VID_DIR, numbered temporally
#   DEMO_VID_DIR (default: <this scripts dir>/demo_vid) flat gather target
#
# Success per scene is judged by the merged predictions.json, NOT the tester's
# exit code (run_demo_tester_8gpu.sh exits 1 on success when RENDER_VIDEO is
# unset — its last line is a failing `[ ... ] && echo`). Scenes that already
# have a predictions.json are skipped, so the batch is resumable.
################################################################################

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

N_SCENES=${1:-30}
SEED=${2:-42}

# vLLM backend: VLLM_URL (+ MODEL_NAME) replaces in-process model loading;
# LORA_PATH is then unnecessary — MODEL_NAME picks base vs LoRA on the server.
if [ -n "${VLLM_URL:-}" ]; then
    MODEL_NAME="${MODEL_NAME:-qwen3vl-8b}"
    LORA_PATH="${LORA_PATH:-none}"
    BATCH_NAME="${BATCH_NAME:-batch_${MODEL_NAME}}"
elif [ -z "${LORA_PATH:-}" ]; then
    echo "ERROR: LORA_PATH is required (env var) unless VLLM_URL is set." >&2
    exit 1
fi

QUERY="To drive safely, is there any object that we have to be aware of?"
#QUERY="Is there any possibility of collision, if ego vehicle rise up the speed?"
QUESTION="${QUESTION:-$QUERY}"

PKL_PATH="${PKL_PATH:-$PROJECT_ROOT/data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl}"
BATCH_NAME="${BATCH_NAME:-batch_$(basename "$LORA_PATH")}"
OUTPUT_ROOT="demo_test_results/$BATCH_NAME"
mkdir -p "$OUTPUT_ROOT"

# Scene list: explicit file wins; otherwise sample N_SCENES tokens with SEED.
if [ -n "${SCENES_FILE:-}" ]; then
    [ -f "$SCENES_FILE" ] || { echo "ERROR: SCENES_FILE=$SCENES_FILE not found" >&2; exit 1; }
else
    SCENES_FILE="$OUTPUT_ROOT/scenes_n${N_SCENES}_seed${SEED}.txt"
    python3 - "$PKL_PATH" "$N_SCENES" "$SEED" "$SCENES_FILE" <<'EOF'
import pickle, random, sys
pkl, n, seed, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
d = pickle.load(open(pkl, 'rb'))
tokens = list(dict.fromkeys(i['scene_token'] for i in d['infos']))  # pkl (temporal) order
random.seed(seed)
picked = random.sample(tokens, min(n, len(tokens)))
open(out, 'w').write('\n'.join(picked) + '\n')
print(f"[demo_batch] sampled {len(picked)}/{len(tokens)} scenes (seed={seed}) -> {out}")
EOF
fi

TOTAL=$(grep -c . "$SCENES_FILE")
echo "============================================================"
echo "Demo batch"
echo "  lora_path : $LORA_PATH"
echo "  question  : $QUESTION"
echo "  scenes    : $TOTAL (list: $SCENES_FILE)"
echo "  outputs   : $OUTPUT_ROOT/<scene16>/"
echo "============================================================"

FAILED=()
n=0
while read -r TOKEN; do
    [ -z "$TOKEN" ] && continue
    n=$((n+1))
    SCENE_DIR="$OUTPUT_ROOT/${TOKEN:0:16}"
    if [ -f "$SCENE_DIR/predictions.json" ]; then
        echo "[$n/$TOTAL] $TOKEN — already done, skipping"
        continue
    fi
    echo ""
    echo "######## [$n/$TOTAL] scene $TOKEN ########"
    LORA_PATH="$LORA_PATH" PKL_PATH="$PKL_PATH" OUTPUT_DIR="$SCENE_DIR" \
        bash nuscenes_pipeline/scripts/run_demo_tester_8gpu.sh "$TOKEN" "$QUESTION"
    if [ ! -f "$SCENE_DIR/predictions.json" ]; then
        echo "FAILED: $TOKEN (no predictions.json; see $SCENE_DIR/_worker_logs/)" >&2
        FAILED+=("$TOKEN")
    fi
done < "$SCENES_FILE"

if [ "${RENDER_VIDEO:-0}" = "1" ]; then
    echo ""
    echo "######## Rendering scene videos ########"
    for p in "$OUTPUT_ROOT"/*/predictions.json; do
        [ -f "$p" ] || continue
        VID_DIR="$(dirname "$p")/videos"
        if compgen -G "$VID_DIR/*.mp4" > /dev/null; then
            echo "[render] $(dirname "$p") — videos exist, skipping"
            continue
        fi
        python3 -m nuscenes_pipeline.visualization.demo_scene_video --predictions "$p" \
            || echo "RENDER_FAILED: $p" >&2
    done

    # Gather all rendered MP4s into one flat directory, numbered temporally
    # (by scene order in the pkl) for easy browsing.
    DEMO_VID_DIR="${DEMO_VID_DIR:-$SCRIPT_DIR/demo_vid}"
    python3 - "$PKL_PATH" "$OUTPUT_ROOT" "$DEMO_VID_DIR" <<'EOF'
import glob, os, pickle, shutil, sys
pkl, out_root, dest = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(dest, exist_ok=True)
d = pickle.load(open(pkl, 'rb'))
scene_order = {}
for i in d['infos']:
    scene_order.setdefault(i['scene_token'], len(scene_order))
vids = sorted(glob.glob(os.path.join(out_root, '*', 'videos', 'scene_*.mp4')))
pairs = []
for v in vids:
    tok16 = os.path.basename(v).split('_')[1]
    full = next((t for t in scene_order if t.startswith(tok16)), None)
    if full is None:
        print(f"[gather] WARNING: {v} matches no scene in pkl, skipping", file=sys.stderr)
        continue
    pairs.append((scene_order[full], v))
n_copied = 0
for n, (_, v) in enumerate(sorted(pairs), 1):
    suffix = os.path.basename(v).split('_', 1)[1]        # <tok16>_<CAT>.mp4
    dst = os.path.join(dest, f"{n:02d}_scene_{suffix}")
    if not os.path.exists(dst) or os.path.getmtime(v) > os.path.getmtime(dst):
        shutil.copy2(v, dst)
        n_copied += 1
print(f"[gather] {len(pairs)} videos ({n_copied} copied) -> {dest}")
EOF
fi

echo ""
echo "============================================================"
echo "Batch complete: $((n - ${#FAILED[@]}))/$TOTAL scenes OK"
if [ "${#FAILED[@]}" -gt 0 ]; then
    printf 'Failed scenes:\n'; printf '  %s\n' "${FAILED[@]}"
    exit 1
fi
echo "Results: $OUTPUT_ROOT/"
