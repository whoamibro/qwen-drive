#!/bin/bash
# Thin wrapper for qwenvl.experiments.run_experiment.
#
# The `qwenvl` package lives under qwen-vl-finetune/, not the project root,
# so `python -m qwenvl.experiments.run_experiment` from the project root
# fails with ModuleNotFoundError. This wrapper exports the right PYTHONPATH
# and forwards all CLI args verbatim, so the invocation matches the
# documented examples without the user having to type the env var.
#
# Usage:
#   bash qwen-vl-finetune/scripts/run_experiment.sh \
#       --exp_id B0 --mode sequential --lr_schedule per_stage_wsd \
#       --grounding_floor off --replay off \
#       --output_root output/curriculum_v2_baseline_0615 --seed 0

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT/qwen-vl-finetune:${PYTHONPATH:-}"

exec python -m qwenvl.experiments.run_experiment "$@"
