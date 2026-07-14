#!/bin/bash
# Evaluate ONE stage (final adapter or a specific checkpoint subdir) using all
# 8 GPUs in parallel via sample-level data parallelism.
#
# How it parallelises:
#   - Each of 8 workers loads its own copy of the LoRA-merged model on its
#     assigned GPU and iterates the categories in CURRICULUM ORDER
#     (OBS -> IDN -> AAS -> SRO -> TSS -> RML -> DRA -> RWP -> ESC -> CHR).
#   - Within each category, worker i processes samples at indices
#     {i, i+8, i+16, ...} (sample_stride=8, sample_offset=i).
#   - When all 8 workers finish, the merger combines their 8 partial
#     eval_report*.json + eval_predictions*.json files into a canonical
#     eval_report.json + eval_predictions.json. Categories in the merged
#     report appear in curriculum order.
#
# Workers iterate categories in lockstep order (OBS first, then IDN, ...)
# but are NOT strictly synchronised across category boundaries. The merged
# output is curriculum-ordered regardless. If you need strict serialisation
# (no worker starts IDN until all finish OBS), use the per-category fan-out
# approach instead.
#
# Usage:
#   bash qwen-vl-finetune/scripts/eval_single_stage_8gpu.sh \\
#       <stage_name> [<checkpoint_subdir>]
#
# Examples:
#   # Final adapter of stage_04_TSS
#   bash qwen-vl-finetune/scripts/eval_single_stage_8gpu.sh stage_04_TSS
#
#   # Intermediate checkpoint of stage_04_TSS
#   bash qwen-vl-finetune/scripts/eval_single_stage_8gpu.sh stage_04_TSS checkpoint-151
#
# Environment overrides:
#   OUTPUT_ROOT  (default: output/curriculum_v2_0609)
#   EVAL_DIR     (default: sft_dataset/eval_subset_200)
#   BASE_MODEL   (default: ckpts/qwen3_vl_8b_instruct)
#   NPROC        (default: 8) — number of GPU workers
#   GPU_IDS      (default: 0,1,...,NPROC-1)
#   IOU_THRESH   (default: 0.8)
#   SANITY_N     (default: 2) — sanity samples per category PER WORKER
#   N_PER_CAT_LIMIT  (default: unset) — cap samples/category for debug
#   FORCE_REEVAL=1  — overwrite existing eval_report.json

set -e
set -o pipefail  # propagate non-zero status through `python | sed | tee`

STAGE_NAME="${1:?usage: $0 <stage_name> [<checkpoint_subdir>]}"
CHECKPOINT_SUBDIR="${2:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT/qwen-vl-finetune:${PYTHONPATH:-}"

OUTPUT_ROOT=${OUTPUT_ROOT:-output/curriculum_v2_0609}
EVAL_DIR=${EVAL_DIR:-sft_dataset/eval_subset_200}
BASE_MODEL=${BASE_MODEL:-ckpts/qwen3_vl_8b_instruct}
IOU_THRESH=${IOU_THRESH:-0.8}
SANITY_N=${SANITY_N:-2}

STAGE_DIR="$OUTPUT_ROOT/$STAGE_NAME"
if [ ! -d "$STAGE_DIR" ]; then
    echo "ERROR: stage dir does not exist: $STAGE_DIR" >&2
    exit 1
fi

EFFECTIVE_DIR="$STAGE_DIR"
CKPT_FLAG=()
if [ -n "$CHECKPOINT_SUBDIR" ]; then
    if [ ! -d "$STAGE_DIR/$CHECKPOINT_SUBDIR" ]; then
        echo "ERROR: $STAGE_DIR/$CHECKPOINT_SUBDIR not found" >&2
        echo "Available: " >&2
        ls -d "$STAGE_DIR"/checkpoint-* 2>/dev/null | xargs -n1 basename >&2
        exit 1
    fi
    EFFECTIVE_DIR="$STAGE_DIR/$CHECKPOINT_SUBDIR"
    CKPT_FLAG=(--lora_checkpoint_subdir "$CHECKPOINT_SUBDIR")
fi

# GPU pool
DETECTED_NPROC=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 0)
[ "$DETECTED_NPROC" -le 0 ] && DETECTED_NPROC=8
NPROC=${NPROC:-$DETECTED_NPROC}

if [ -n "${GPU_IDS:-}" ]; then
    IFS=',' read -ra GPU_ARR <<<"$GPU_IDS"
    if [ "${#GPU_ARR[@]}" != "$NPROC" ]; then
        echo "ERROR: GPU_IDS has ${#GPU_ARR[@]} entries but NPROC=$NPROC" >&2
        exit 1
    fi
else
    GPU_ARR=()
    for ((i=0; i<NPROC; i++)); do GPU_ARR+=("$i"); done
fi

LOG_DIR="$EFFECTIVE_DIR/eval_logs"
mkdir -p "$LOG_DIR"

# Skip-if-done
FINAL_REPORT="$EFFECTIVE_DIR/eval_report.json"
if [ -f "$FINAL_REPORT" ] && [ "${FORCE_REEVAL:-0}" != "1" ]; then
    echo "[skip] $FINAL_REPORT already exists. Set FORCE_REEVAL=1 to overwrite."
    exit 0
fi

EXTRA_FLAGS=()
if [ -n "${N_PER_CAT_LIMIT:-}" ]; then
    EXTRA_FLAGS+=(--n_per_cat_limit "$N_PER_CAT_LIMIT")
fi

echo "============================================"
echo "  Single-stage 8-GPU eval (sample-stride)"
echo "  stage_dir:     $STAGE_DIR"
echo "  effective:     $EFFECTIVE_DIR"
echo "  eval_dir:      $EVAL_DIR"
echo "  GPUs:          ${GPU_ARR[*]} (NPROC=$NPROC)"
echo "  log_dir:       $LOG_DIR"
echo "  category order: OBS -> IDN -> AAS -> SRO -> TSS -> RML -> DRA -> RWP -> ESC -> CHR"
echo "============================================"
echo "Streaming all $NPROC worker outputs to THIS terminal. Each line is prefixed"
echo "  with the worker tag, e.g. [w3g3] for worker 3 on GPU 3."
echo "  Full per-worker logs are also written to $LOG_DIR/."
echo "============================================"

# ----------------------------------------------------------------------
# Fan out: each worker handles samples at index k where k % NPROC == i.
# ----------------------------------------------------------------------
PIDS=()
for ((i=0; i<NPROC; i++)); do
    GPU="${GPU_ARR[$i]}"
    SUFFIX="_worker${i}"
    LOG_FILE="$LOG_DIR/eval_worker${i}_gpu${GPU}.log"
    PREFIX="w${i}g${GPU}"
    echo "[launch] GPU $GPU  worker $i (stride=$NPROC offset=$i)  (prefix [$PREFIX])  -> $LOG_FILE"
    {
        CUDA_VISIBLE_DEVICES="$GPU" python -u -m nuscenes_pipeline.modules.sft_model_tester \
            --base_model "$BASE_MODEL" \
            --lora_path "$STAGE_DIR" \
            "${CKPT_FLAG[@]}" \
            --per_category_eval_dir "$EVAL_DIR" \
            --eval_v2 \
            --iou_threshold "$IOU_THRESH" \
            --sanity_print_n "$SANITY_N" \
            --save_predictions \
            --resize_factor 1 \
            --max_new_tokens 1024 \
            --sample_stride "$NPROC" \
            --sample_offset "$i" \
            --report_suffix "$SUFFIX" \
            "${EXTRA_FLAGS[@]}" \
            2>&1
    } | sed -u "s/^/[$PREFIX] /" | tee "$LOG_FILE" &
    PIDS+=($!)
done

echo
echo "Waiting for ${#PIDS[@]} workers ..."
FAIL=0
for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        echo "  pid $PID exited non-zero — check logs in $LOG_DIR"
        FAIL=$((FAIL + 1))
    fi
done

if [ "$FAIL" -gt 0 ]; then
    echo
    echo "WARNING: $FAIL worker(s) failed. Inspecting logs may be required before merge."
fi

# ----------------------------------------------------------------------
# Merge the 8 partial reports into canonical eval_report.json +
# eval_predictions.json. Categories are written in curriculum order.
# ----------------------------------------------------------------------
echo
echo "============================================"
echo "  Merging $NPROC partial reports into canonical eval_report.json"
echo "============================================"

python - <<PY
import glob, json, os, sys

EFFECTIVE_DIR = os.path.abspath("$EFFECTIVE_DIR")
NPROC = $NPROC

CURRICULUM_ORDER = ["OBS","IDN","AAS","SRO","TSS","RML","DRA","RWP","ESC","CHR"]

report_files = sorted(glob.glob(os.path.join(EFFECTIVE_DIR, "eval_report_worker*.json")))
pred_files   = sorted(glob.glob(os.path.join(EFFECTIVE_DIR, "eval_predictions_worker*.json")))
if not report_files:
    print("ERROR: no eval_report_worker*.json files found in", EFFECTIVE_DIR, file=sys.stderr)
    sys.exit(2)

worker_reports = []
for p in report_files:
    with open(p) as f: worker_reports.append(json.load(f))
worker_preds = []
for p in pred_files:
    with open(p) as f: worker_preds.append(json.load(f))

# Collect all categories across workers (any partial worker may have skipped
# a cat if it had zero samples in its slice — unlikely with 200 samples / 8
# workers but handle defensively).
all_cats = set()
for r in worker_reports:
    all_cats.update(r.get("metrics", {}).keys())
ordered_cats = [c for c in CURRICULUM_ORDER if c in all_cats]
ordered_cats += sorted(all_cats - set(CURRICULUM_ORDER))

# Merge counters across workers, recompute aggregate metrics.
def _key_iou(metric_entry):
    for k in metric_entry:
        if k.startswith("grounding_acc@"):
            return k
    return "grounding_acc@0.8"

merged_metrics = {}
for cat in ordered_cats:
    n = 0; n_grounded = 0; n_gt_boxes = 0
    n_ans_correct = 0; n_parsed_answer = 0
    gt_box_view_correct = 0; gt_box_iou_correct = 0
    n_grounded_with_valid_pred = 0
    spurious_pred_boxes = 0
    iou_key = None
    # T1 — completeness raw accumulators (gated: only fold in if at least one
    # worker emitted them, so backward-compat with pre-T1 reports holds).
    has_t1 = False
    per_view_gt_count      = [0] * 6
    per_view_matched_count = [0] * 6
    view_confusion         = [[0] * 6 for _ in range(6)]
    n_missing_boxes_sum   = 0
    ref_completeness_sum  = 0.0
    for r in worker_reports:
        m = r.get("metrics", {}).get(cat)
        if m is None: continue
        if iou_key is None: iou_key = _key_iou(m)
        n += m["n"]
        n_grounded += m["n_grounded"]
        n_gt_boxes += m["n_gt_boxes"]
        # Recover counts from rates + n
        n_ans_correct += round(m["answer_acc"] * m["n"])
        n_parsed_answer += round(m["parse_rate"] * m["n"])
        if m.get("view_acc") is not None and m["n_gt_boxes"] > 0:
            gt_box_view_correct += round(m["view_acc"] * m["n_gt_boxes"])
        if m.get(iou_key) is not None and m["n_gt_boxes"] > 0:
            gt_box_iou_correct  += round(m[iou_key] * m["n_gt_boxes"])
        gfv = m.get("grounding_format_valid")
        if gfv is not None and m["n_grounded"] > 0:
            n_grounded_with_valid_pred += round(gfv * m["n_grounded"])
        spurious_pred_boxes += m.get("spurious_pred_boxes", 0)
        # T1 raw counts (append-only schema; missing keys mean pre-T1 worker)
        if "per_view_gt_count" in m and "_raw_per_view_matched_count" in m:
            has_t1 = True
            for v in range(6):
                per_view_gt_count[v]      += m["per_view_gt_count"][v]
                per_view_matched_count[v] += m["_raw_per_view_matched_count"][v]
            for i in range(6):
                for j in range(6):
                    view_confusion[i][j] += m.get("view_confusion", [[0]*6]*6)[i][j]
            n_missing_boxes_sum  += m.get("_raw_n_missing_boxes_sum", 0)
            ref_completeness_sum += m.get("_raw_ref_completeness_sum", 0.0)

    if iou_key is None: iou_key = "grounding_acc@0.8"
    merged_metrics[cat] = {
        "n": n,
        "n_grounded": n_grounded,
        "n_gt_boxes": n_gt_boxes,
        "answer_acc": n_ans_correct / max(n, 1),
        "view_acc": (gt_box_view_correct / n_gt_boxes) if n_gt_boxes else None,
        iou_key:    (gt_box_iou_correct  / n_gt_boxes) if n_gt_boxes else None,
        "grounding_format_valid": (n_grounded_with_valid_pred / n_grounded)
                                  if n_grounded > 0 else None,
        "parse_rate": n_parsed_answer / max(n, 1),
        "spurious_pred_boxes": spurious_pred_boxes,
    }
    if has_t1:
        merged_metrics[cat].update({
            "referring_completeness": (ref_completeness_sum / n_grounded) if n_grounded > 0 else None,
            "n_missing_boxes":  n_missing_boxes_sum  / max(n, 1),
            "n_spurious_boxes": spurious_pred_boxes  / max(n, 1),
            "per_view_recall":  [(per_view_matched_count[v] / per_view_gt_count[v])
                                 if per_view_gt_count[v] else None
                                 for v in range(6)],
            "per_view_gt_count": per_view_gt_count,
            "view_confusion":   view_confusion,
            "_raw_per_view_matched_count":   per_view_matched_count,
            "_raw_n_missing_boxes_sum":      n_missing_boxes_sum,
            "_raw_ref_completeness_sum":     ref_completeness_sum,
        })

macro = (sum(merged_metrics[c]["answer_acc"] for c in ordered_cats) / len(ordered_cats)
         if ordered_cats else 0.0)

# Stitch the schema from the first worker (keeps timestamp/lora_path/etc).
canonical = dict(worker_reports[0])
canonical.pop("sample_stride", None)
canonical.pop("sample_offset", None)
canonical["metrics"] = {c: merged_metrics[c] for c in ordered_cats}
canonical["macro_answer_acc"] = macro
canonical["merged_from_workers"] = len(worker_reports)

out_path = os.path.join(EFFECTIVE_DIR, "eval_report.json")
with open(out_path, "w") as f:
    json.dump(canonical, f, indent=2)

# Merge predictions — concat per-category sample lists, sort by 'i' so the
# final list matches val-subset order.
canonical_preds = {}
if worker_preds:
    seen_cats = set()
    for wp in worker_preds:
        seen_cats.update(wp.keys())
    for cat in ordered_cats:
        merged_list = []
        for wp in worker_preds:
            merged_list.extend(wp.get(cat, []))
        merged_list.sort(key=lambda r: r.get("i", -1))
        canonical_preds[cat] = merged_list

preds_path = os.path.join(EFFECTIVE_DIR, "eval_predictions.json")
with open(preds_path, "w") as f:
    json.dump(canonical_preds, f, indent=2)

# Pretty-print the summary in curriculum order.
print(f"  merged from {len(worker_reports)} worker reports")
print(f"  -> {out_path}")
print(f"  -> {preds_path}")
print(f"  macro_answer_acc = {macro:.3f}")
print(f"  {'CAT':>4} {'n':>5} {'gnd':>5} {'answer':>8} {'view':>7} {'g@iou':>7} {'fmt_val':>8}")
for cat in ordered_cats:
    m = merged_metrics[cat]
    iou_key = _key_iou(m)
    va = m.get("view_acc"); ga = m.get(iou_key); fv = m.get("grounding_format_valid")
    print(f"  {cat:>4} {m['n']:>5} {m['n_grounded']:>5} "
          f"{m['answer_acc']:>8.3f} "
          f"{(va if va is not None else float('nan')):>7.3f} "
          f"{(ga if ga is not None else float('nan')):>7.3f} "
          f"{(fv if fv is not None else float('nan')):>8.3f}")
PY

MERGE_EXIT=$?
if [ "$MERGE_EXIT" -ne 0 ]; then
    echo "ERROR: merge step failed (exit $MERGE_EXIT). Per-worker reports remain in $EFFECTIVE_DIR/eval_report_worker*.json"
    exit "$MERGE_EXIT"
fi

# Optional cleanup of per-worker files (keep by default for debugging).
if [ "${KEEP_WORKER_FILES:-1}" = "0" ]; then
    rm -f "$EFFECTIVE_DIR"/eval_report_worker*.json "$EFFECTIVE_DIR"/eval_predictions_worker*.json
    echo "  (removed per-worker files; KEEP_WORKER_FILES=1 to retain)"
fi
echo "Done."
