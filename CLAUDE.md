# CLAUDE.md — Project-Specific Context for Claude Code

This file documents conventions, architecture decisions, and non-obvious details
for the qwen-drive nuScenes autolabeling + SFT pipeline. Auto-loaded in every
Claude Code session on this repo.

---

## 1. Pipeline Architecture

Five stages, with dependencies:

```
Stage 1A  risk_assessment            ─┐
Stage 1B  traffic_signal_analysis    ─┼─→  Stage 2  question_selector  →  Stage 3  answer_generator  →  postprocessing  →  SFT training  →  SFT testing
Stage 1C  traffic_sign_extraction    ─┘
```

- Stages 1A/1B/1C are independent and can run in parallel.
- Stage 2 consumes 1A + 1B outputs as prior context for the `Dynamic_Agents_and_Risk_Assessment` and `Traffic_Signs_and_Signals` categories respectively.
- Stage 3 consumes 1A + 1B + 1C outputs — 1C (sign) is prepended to **all 10 categories**, 1A only to `Dynamic_Agents_and_Risk_Assessment`, 1B only to `Traffic_Signs_and_Signals`.

### Module paths

| Stage | Module | Shell script |
|---|---|---|
| 1A | `nuscenes_pipeline.modules.risk_assessment` | `scripts/run_risk_assessment.sh` |
| 1B | `nuscenes_pipeline.modules.traffic_signal_analysis` | `scripts/run_traffic_signal_analysis.sh` |
| 1C | `nuscenes_pipeline.modules.traffic_sign_extraction` | `scripts/run_traffic_sign_extraction.sh` |
| 2  | `nuscenes_pipeline.modules.question_selector` | `scripts/run_question_selector.sh` |
| 3  | `nuscenes_pipeline.modules.answer_generator` | `scripts/run_answer_generator.sh` |
| SFT test | `nuscenes_pipeline.modules.sft_model_tester` | `scripts/run_sft_model_tester.sh` |

---

## 2. File Naming Conventions

| Output dir | File suffix |
|---|---|
| `risk_assessment_results/` | `*_single_frame.json` |
| `traffic_signal_analysis_results/` | `*_traffic_signal.json` |
| `traffic_sign_results/` | `*_sign.json` |
| `qa_outputs/<category>/` or `qa_outputs/all/` | `sample_{idx}_applicable_questions.json`, `sample_{idx}_qa_summary.json`, `sample_{idx}_inference_detailed.json` |
| `qa_results/` | `sample_{idx}_qa_results.json`, `sample_{idx}_qa_detailed.json` |
| `sft_dataset/` | `sft_train_no_objlist.json`, `sft_val_no_objlist.json` |

**Loader pattern for prior-analysis files**: `nuscenes_pipeline.modules.question_selector._load_prior_analysis_response(dir, idx, suffix)` looks for `{idx:04d}_*_{suffix}.json`. Adding a new prior type means adding a tiny helper that just passes the suffix.

---

## 3. Key CLI Flags (non-obvious ones)

### Stage 2 — question_selector

- `--sampling_ratio FLOAT` — randomly samples this percentage from `[start_idx, end_idx]` range. E.g., `5` → 300 samples. Seeded for reproducibility.
- `--seed INT` (default 42) — the sampling RNG seed.

### Stage 3 — answer_generator

- `--from_stage1` — auto-discover sample indices from Stage 2 output files instead of using `start_idx`/`end_idx`. Essential when sampled subset was processed in Stage 2.
- `--answer_mode {a_r, r_a}` — `a_r` (default) = answer-first-then-reasoning (chain-of-thought); `r_a` = reasoning-first-then-answer (autolabel style, VLM derives answer from reasoning rather than justifying a pre-chosen one).
- `--sign_results_dir DIR` — Stage 1C output dir. Prepended to ALL 10 categories.
- `--risk_results_dir DIR` — Stage 1A. Prepended only to `Dynamic_Agents_and_Risk_Assessment`.
- `--traffic_results_dir DIR` — Stage 1B. Prepended only to `Traffic_Signs_and_Signals`.

### Stage 3 prompt-prepend ordering (when multiple priors apply)

Category-specific priors (risk/signal) are prepended AFTER sign, so they sit closer to the question body (higher recency = higher attention).

| Category | Nearest → furthest from question body |
|---|---|
| `Dynamic_Agents_and_Risk_Assessment` | base ← risk ← sign |
| `Traffic_Signs_and_Signals` | base ← signal ← sign |
| Other 8 categories | base ← sign |

---

## 4. MCQ Answer Format (v4 question bank)

- Exactly 5 options labeled (A)–(E).
- Number of correct options **varies per pair (1–5)**, sampled randomly in `pre_instantiate_pairs()` and stored as `num_correct_positive` / `num_correct_contrastive` in each pair dict. The prompt surfaces these as `[MCQ] num_correct_positive: N` lines.
- Answer format: single letter (`"B"`) or comma-separated (`"A,C,D"`). Count must match `num_correct_*`.
- Distractors drawn from `expected_answers` pool; generated if pool too small.

---

## 5. Prompt Conventions

- **Ego-centric perspective**: both `question_selector` and `answer_generator` system prompts contain an "IMPORTANT — Ego-Centric Perspective" block. Spatial references are ego-relative, "relevance" is defined relative to the ego's driving context, and templates' `ego_centric_note` field is surfaced per-template as "Ego-Centric Guidance".
- **Short-answer + bullet reasoning**: Answer field is minimal (single letter for MCQ, yes/no for y_or_n, etc.). Reasoning is bullet-point format (3–6 bullets), NOT prose. Exception: `open_ended` answer type allows 2–4 sentence descriptive answers.

---

## 6. Image Processing — Critical Details

### Qwen3-VL uses Qwen2-VL's processor under the hood
- No dedicated `Qwen3VLImageProcessor` class exists. `AutoProcessor.from_pretrained(...)` returns `Qwen3VLProcessor`, whose `.image_processor` is `Qwen2VLImageProcessorFast`.
- The `max_pixels` and `min_pixels` attributes exist but are just aliases. The actual limits live in `self.size["longest_edge"]` and `self.size["shortest_edge"]`. Setting one without the other is ineffective — always set both.

### Double-resize happens in training
In `train_nuscenes_qwen3vl.py`, `process_image()` does:
1. Manual `img.resize(1/resize_factor)` via PIL LANCZOS
2. Then the processor's `smart_resize` rounds to patch multiples and caps by `max_pixels`

With `resize_factor=2` and `max_pixels=50176`, an 1600×900 image becomes 800×450 (LANCZOS) then ~140×280 (bilinear inside smart_resize). Final ViT input is only ~140×280 per view → **small objects like traffic lights are unreadable**. The manual resize is wasted work when `max_pixels` is the binding constraint.

### Coordinate space of bbox labels
- `transform_obj_to_bbox.py --resize_factor N` produces bboxes in the image space AFTER dividing original dimensions by N.
- **Default N=1** → bboxes in full 1600×900 space (matches the original nuScenes image resolution). ViT input is then downscaled by the processor's `smart_resize` to whatever `max_pixels` allows; training is consistent as long as the same pipeline is used at inference.
- Older runs used N=2 (800×450). If you have legacy data in 800×450 space, use `scale_bbox_coords.py --scale 2` to scale up to 1600×900, OR re-run `transform_obj_to_bbox` with the new default.
- Online scaling at training load time was prototyped then reverted — we keep labels in a fixed coord space instead.

---

## 7. SFT Training — Gotchas

### NCCL timeout
Default is 10 minutes. If eval takes longer (and the full val set is 35K samples → ~10 hours), the next training step's ALLREDUCE hangs waiting for ranks still finalizing eval/save, and NCCL times out.

**Fix**: `export NCCL_TIMEOUT=3600` + `export TORCH_NCCL_TIMEOUT_SEC=3600` at the top of `run_nuscenes_lora.sh`.

### Eval set size
The full `sft_val_no_objlist.json` (35K samples) is too big for in-training eval. Create a 300-sample subset:
```python
import json, random
data = json.load(open('sft_dataset/sft_val_no_objlist.json'))
random.seed(42)
json.dump(random.sample(data, 300), open('sft_dataset/sft_val_small.json', 'w'))
```
Point `VAL_DATA` in the training shell script at this file. Reserve the full val set for final evaluation only.

### Token budget (per sample, with `max_pixels=50176`)
- System prompt: ~418 tokens (constant)
- Vision (6 views): ~300 tokens
- Human query: ~299 tokens
- GPT response (loss applied): mean 139, p99 490, max 695
- Total: ~1,200 mean, ~1,800 p99

`--model_max_length 8192` is generous. Could drop to 2048 to save memory.

### Raising `max_pixels`
- With `resize_factor=2` (800×450 input), raising `max_pixels` past 360,000 is a no-op because the input is the binding constraint.
- With `resize_factor=1` (full 1600×900 input), `max_pixels=802816` gives ~644×1176 per view (966 tokens/view × 6 = 5,796 vision tokens per sample). Still within `model_max_length=8192` but ~5× step time.
- Bbox labels are now in 1600×900 space by default (after `transform_obj_to_bbox.py --resize_factor 1`, which is the default), so no label-space change is needed when switching between `resize_factor` values at training time.

---

## 8. Post-processing Pipeline (in order)

```bash
# Step 1: OBJ IDs → 2D bbox descriptions
python -m nuscenes_pipeline.postprocessing.transform_obj_to_bbox \
    --input_dir qa_results --output_dir sft_dataset --data_root ./data/nuscenes

# Step 2: Build SFT train/val JSONs
python -m nuscenes_pipeline.postprocessing.prepare_sft_dataset \
    --qa_dir sft_dataset --output_dir sft_dataset --data_root ./data/nuscenes

# Step 3: Cleanse OBJ refs (Rounds 1-3)
python -m nuscenes_pipeline.postprocessing.cleanse_obj_references \
    --input sft_dataset/sft_train_no_objlist.json
python -m nuscenes_pipeline.postprocessing.cleanse_obj_references \
    --input sft_dataset/sft_val_no_objlist.json

# Step 4: Fix motion state claims against GT velocity
python -m nuscenes_pipeline.postprocessing.fix_motion_states \
    --data_dir sft_dataset --data_root ./data/nuscenes

# Step 5: Convert to Qwen3-VL unified JSON format (reasoning + grounding + answer)
python -m nuscenes_pipeline.postprocessing.convert_to_qwen3vl_format \
    --input sft_dataset/sft_train_no_objlist.json \
    --output sft_dataset/sft_train_qwen3vl.json
python -m nuscenes_pipeline.postprocessing.convert_to_qwen3vl_format \
    --input sft_dataset/sft_val_no_objlist.json \
    --output sft_dataset/sft_val_qwen3vl.json
```

`--data_root ./data/nuscenes` is required because pkl image paths are `data/nuscenes/samples/...` — the scripts strip `data/nuscenes/` from the relative path, so `data_root` must point at `./data/nuscenes` (not the project root).

`cleanse_obj_references.py` now handles both singular and plural forms: `OBJ ID`, `OBJ IDs`, `object list`, `object lists`.

**Step 5 (Qwen3-VL format)**:
- Legacy output (`sft_{train,val}_no_objlist.json`) embeds bboxes inline in prose — deprecated Qwen-VL 1 style.
- New output (`sft_{train,val}_qwen3vl.json`) has gpt value as a unified JSON string with keys: `reasoning` (bullet-point string with bboxes removed), `grounding` (list of `{image_idx, camera, bbox_2d, label}` objects), `answer` (short factual response).
- **Preserves `from: "system"` as its own turn** (does not merge into human). Our training pipeline `train_nuscenes_qwen3vl.py` supports system turns natively via `preprocess_with_system_prompt()` — system content is rendered as `<|im_start|>system\n...<|im_end|>\n`, masked with `IGNORE_INDEX` so it forms prefix context but not loss targets. Note: this differs from the official `qwen-vl-finetune` and `Qwen-VL-Series-Finetune` repos, which drop or mishandle per-sample system turns; we use our custom training script.
- Idempotency: writes `.format_qwen3vl` marker in the output directory.
- Train with `--data_path sft_dataset/sft_train_qwen3vl.json` + leave `--enable_reasoning` as default (False). The model learns to emit the unified JSON directly.

---

## 9. Commit Conventions

- **No `Co-Authored-By: Claude` lines** in commits — user's explicit preference (saved in memory).
- When making git commits, always use HEREDOC format:
  ```bash
  git commit -m "$(cat <<'EOF'
  Subject line

  - Bullet 1
  - Bullet 2
  EOF
  )"
  ```

---

## 10. Common Operations

### Run subset autolabel pipeline (5% of dataset)
```bash
# Stage 1A/1B/1C on full dataset (reusable)
bash nuscenes_pipeline/scripts/run_risk_assessment.sh 0 6018 8
bash nuscenes_pipeline/scripts/run_traffic_signal_analysis.sh 0 6018 8
bash nuscenes_pipeline/scripts/run_traffic_sign_extraction.sh 0 6018 8

# Stage 2: select templates for 5% sampled subset (~300 samples)
bash nuscenes_pipeline/scripts/run_question_selector.sh 0 6018 all 8 5

# Stage 3: auto-discover the sampled indices from Stage 2 output
bash nuscenes_pipeline/scripts/run_answer_generator.sh 0 6018 all 8 from_stage1

# For reasoning-first autolabel style
bash nuscenes_pipeline/scripts/run_answer_generator.sh 0 6018 all 8 from_stage1 r_a
```

### Preview prompts without inference (debugging)
```bash
python -m nuscenes_pipeline.scripts.show_prompts \
    --category Traffic_Signs_and_Signals --sample_idx 23
```

### Launch vLLM server (prerequisite for all stages)
```bash
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct --tensor-parallel-size 8
```

### Test SFT-trained model on val set
See `## Evaluation & Benchmarks` in README for the `sft_model_tester` workflow.

---

## 11. Environment Variables

| Variable | Purpose | Used by |
|---|---|---|
| `NUSCENES_PKL_PATH` | Path to the pkl file | all stages |
| `QUESTION_BANK_PATH` | Path to question bank JSON | Stages 2, 3 |
| `NCCL_TIMEOUT`, `TORCH_NCCL_TIMEOUT_SEC` | NCCL collective timeout in seconds | SFT training |

The active question bank is `./data/question_bank_v4_static.json` (251 templates, 44 sub-categories, variable-correct MCQ support).
