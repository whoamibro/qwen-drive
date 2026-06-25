#!/bin/bash
# Standalone eval phase for an experiment trained with --skip_eval.
#
# Forwards all CLI args to `qwenvl.experiments.eval_run`. Sets PYTHONPATH so
# you don't have to.
#
# Usage:
#   bash qwen-vl-finetune/scripts/eval_experiment.sh \
#       --run_dir output/curriculum_v2_exp/B0__seed0
#
#   # Force re-eval of all checkpoints:
#   bash qwen-vl-finetune/scripts/eval_experiment.sh \
#       --run_dir output/curriculum_v2_exp/B0__seed0 --force
#
#   # Only re-eval specific stages / checkpoints:
#   bash qwen-vl-finetune/scripts/eval_experiment.sh \
#       --run_dir output/curriculum_v2_exp/B0__seed0 \
#       --only stage_04_TSS,stage_05_RML --force

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT/qwen-vl-finetune:${PYTHONPATH:-}"

exec python -m qwenvl.experiments.eval_run "$@"
