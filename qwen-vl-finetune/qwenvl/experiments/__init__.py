"""Single-experiment ablation training framework for curriculum-v2.

Each invocation runs exactly one experiment, fully configured by CLI knobs.
No preset table — every run is defined by the combination of `--mode`,
`--lr_schedule`, `--grounding_floor`, `--replay`, `--sampler`. The
`--exp_id` arg is a free-form label used only for output directory naming
and reproducibility.

See `curriculum_v2_ablation_experiments.md` at the project root for the
knob semantics, composition rules, and a HUMAN-REFERENCE table of the 16
canonical flag combinations (B0..C12). That table is documentation only —
it has no code counterpart, so the CLI is the single source of truth.

GPU-limited workflow: run one experiment at a time, then run the standalone
aggregator (`qwenvl.experiments.aggregate`) to refresh the master
comparison table.
"""
