# Drive Agent Pipeline

A multi-stage VQA pipeline for autonomous driving scene understanding on the nuScenes dataset, powered by Qwen3-VL served via vLLM.

## Table of Contents

### Autolabel pipeline (raw nuScenes → QA pairs)
1. [Pipeline Overview](#pipeline-overview)
2. [Prerequisites](#prerequisites)
3. [Stage 1A: Risk Assessment](#stage-1a-risk-assessment)
4. [Stage 1B: Traffic Analysis](#stage-1b-traffic-analysis)
5. [Stage 2: Question Selector](#stage-2-question-selector)
6. [Stage 3: Answer Generator](#stage-3-answer-generator)
7. [Package Structure](#package-structure)
8. [Visualization Tools](#visualization-tools)
9. [Post-Processing Pipeline](#post-processing-pipeline)
10. [Full Pipeline Example (End-to-End)](#full-pipeline-example-end-to-end)

### SFT training
11. [SFT Training](#sft-training)
12. [Curriculum Learning (Sequential Multi-Stage SFT)](#curriculum-learning-sequential-multi-stage-sft)
13. [Curriculum-v2 Training and Evaluation](#curriculum-v2-training-and-evaluation)
14. [Ablation Experiments Framework](#ablation-experiments-framework)

### Evaluation and inspection
15. [SFT Model Testing (Qualitative Inference)](#sft-model-testing-qualitative-inference)
16. [Demo Tester (Per-Scene Frame-by-Frame Inference)](#demo-tester-per-scene-frame-by-frame-inference)
17. [Evaluation & Benchmarks](#evaluation--benchmarks)
18. [Eval Sample Visualizer (Browser-Based)](#eval-sample-visualizer-browser-based)
19. [Resize Factor Guide](#resize-factor-guide)
20. [Supporting Documentation](#supporting-documentation)

---

## Pipeline Overview

```
Stage 1A: Risk Assessment      Analyze driving risks, hazards, and TTC per sample
Stage 1B: Traffic Analysis      Identify traffic signal states per sample
                                        |
                          (Stage 1A + 1B results feed into Stage 2)
                                        |
Stage 2:  Question Selector     Select applicable question templates from the question bank
                                        |
                          (Stage 2 outputs feed into Stage 3)
                                        |
Stage 3:  Answer Generator      Generate grounded QA pairs with contrastive answers
```

Stages 1A and 1B are independent and can run in parallel.
Stage 2 requires both Stage 1A and 1B results.
Stage 3 requires Stage 2 results.

---

## Prerequisites

### vLLM Server

All modules require a running vLLM server with the Qwen3-VL model:

```bash
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct --tensor-parallel-size 8
```

### Environment Variables (Optional)

Set these to avoid passing paths on every invocation:

| Variable | Description | Example |
|----------|-------------|---------|
| `NUSCENES_PKL_PATH` | Path to nuScenes pickle file | `/data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl` |
| `QUESTION_BANK_PATH` | Path to question bank JSON | `/data/qa_dataset/question_bank.json` |

If not set, the modules fall back to looking for the files in the current working directory.

### Working Directory

All commands should be run from the `qwen-drive/` root directory:

```bash
cd /path/to/qwen-drive
```

---

## Stage 1A: Risk Assessment

Analyzes driving risks, hazards, and time-to-collision (TTC) for each nuScenes sample using 6-view egocentric camera images and 3D object annotations.

### How to Run

```bash
# Via shell script (recommended defaults)
bash nuscenes_pipeline/scripts/run_risk_assessment.sh [START_IDX] [END_IDX] [NUM_WORKERS]

# Via Python module
python -m nuscenes_pipeline.modules.risk_assessment \
    --start_idx 0 --end_idx 6018 --num_workers 8 \
    --resize_factor 2 --to_global --3dod --proj2img \
    --filter_length 50 --rear_filter 20 \
    --max_new_tokens 4096 \
    --results_dir risk_assessment_results \
    --question "Analyze risks and hazards following these categories: Risk Categories: 1. Static Hazards: Parked vehicles, road infrastructure, visibility obstructions 2. Dynamic Risks: Potential pedestrian/vehicle emergence zones, blind spots 3. Environmental Factors: Weather, lighting, road geometry 4. Situational Awareness: Areas requiring increased vigilance Output Format: 1. Immediate Risks (requires immediate attention) 2. Potential Risks (monitor closely) 3. Recommended Actions (specific driving advice) 4. Overall Risk Level(Low/Moderate/High with brief justification) Instructions: - Analyze the recommended actions in the overall context of ego-centric surrounding scenes. - Double-check all directions of images before finalizing. - Prioritize by severity and likelihood. - Focus on actionable insight. - For the assessment of the Overall Risk Level, compute the collision risk using the provided collision-risk formula by substituting the 3D information and velocity of all objects and the ego vehicle, and present the evaluated result accordingly."
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_name` | str | `Qwen/Qwen3-VL-235B-A22B-Instruct` | Model name served by vLLM |
| `--api_base` | str | `http://localhost:8000/v1` | vLLM API endpoint URL |
| `--api_key` | str | `EMPTY` | API key for the vLLM server |
| `--pkl_path` | str | env `NUSCENES_PKL_PATH` | Path to nuScenes pickle file |
| `--start_idx` | int | `0` | Starting sample index (inclusive) |
| `--end_idx` | int | `6018` | Ending sample index (inclusive) |
| `--question` | str | **required** | Risk assessment question prompt |
| `--resize_factor` | int | `4` | Image downscale factor. Use `2` for production runs (1/2 of original 1600x900) |
| `--max_new_tokens` | int | `4096` | Maximum tokens the model generates per sample |
| `--num_workers` | int | `8` | Number of parallel multiprocessing workers |
| `--to_global` | flag | `False` | Use global ENU coordinates instead of ego-relative FLU coordinates |
| `--3dod` | flag | `False` | Include 3D object detection ground truth (bounding boxes, velocities, TTC) in the prompt |
| `--filter_length` | float | `20.0` | Maximum distance (meters) from ego vehicle to include 3D objects |
| `--rear_filter` | float | `None` | Maximum distance (meters) for non-vehicle objects behind the ego vehicle. Vehicles behind ego are always kept regardless of this filter |
| `--proj2img` | flag | `False` | Project 3D object positions onto image pixel coordinates in the prompt |
| `--results_dir` | str | `risk_assessment_results` | Directory to save per-sample result JSONs |
| `--log_dir` | str | `risk_assessment_logs` | Directory for execution logs and failed-index tracking |

### Input

| Input | Description |
|-------|-------------|
| nuScenes pickle file (`--pkl_path`) | Pre-processed nuScenes validation set containing 6,019 samples with camera images, 3D annotations, ego poses, and velocities |
| 6-view camera images | Loaded from paths stored in the pickle file. Camera order: Front-Left, Front, Front-Right, Rear-Left, Rear, Rear-Right. Rear cameras are horizontally flipped for egocentric consistency |

### Output

Per-sample JSON files saved to `{results_dir}/`:

**Filename pattern:** `{idx:04d}_{scene_token}_{sample_token}_single_frame.json`

**Contents:**
- `system_prompt`: The system prompt sent to the model
- `user_question`: The user question with scene context (ego state, 3D objects, TTC)
- `response`: Model's risk assessment response
- `inference_time`: Time taken for the API call
- `sample_metadata`: Sample index, scene token, timestamp

---

## Stage 1B: Traffic Analysis

Identifies all traffic signals visible in the 6-view camera images and determines their states (red, yellow, green), orientation, and relevance to the ego vehicle's lane.

### How to Run

```bash
# Via shell script (recommended defaults)
bash nuscenes_pipeline/scripts/run_traffic_analysis.sh [START_IDX] [END_IDX] [NUM_WORKERS]

# Via Python module
python -m nuscenes_pipeline.modules.traffic_analysis \
    --start_idx 0 --end_idx 6018 --num_workers 8 \
    --resize_factor 1 --to_global \
    --max_new_tokens 4096 \
    --results_dir traffic_analysis_results \
    --question "Analyze the traffic signals visible in the images. Identify which signal governs the ego vehicle's driving status and lane, then report its current state."
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_name` | str | `Qwen/Qwen3-VL-235B-A22B-Instruct` | Model name served by vLLM |
| `--api_base` | str | `http://localhost:8000/v1` | vLLM API endpoint URL |
| `--api_key` | str | `EMPTY` | API key for the vLLM server |
| `--pkl_path` | str | env `NUSCENES_PKL_PATH` | Path to nuScenes pickle file |
| `--sample_indices` | int+ | `None` | Specific sample indices to analyze (e.g., `--sample_indices 0 10 20`) |
| `--start_idx` | int | `None` | Start index for range-based processing |
| `--end_idx` | int | `None` | End index for range-based processing |
| `--question` | str | (built-in default) | Custom traffic analysis question |
| `--resize_factor` | int | `4` | Image downscale factor. Use `1` for production runs (full 1600x900 resolution for small signal detection) |
| `--max_new_tokens` | int | `4096` | Maximum tokens the model generates per sample |
| `--num_workers` | int | `8` | Number of parallel multiprocessing workers |
| `--to_global` | flag | `False` | Use global ENU coordinates in the prompt |
| `--visualize` | flag | `False` | Generate visualization images alongside results |
| `--viz_output_dir` | str | `traffic_visualizations` | Directory for visualization output |
| `--results_dir` | str | `traffic_analysis_results` | Directory to save per-sample result JSONs |

**Note on sample selection:** Use either `--sample_indices` for specific samples OR `--start_idx`/`--end_idx` for a range. If only `--start_idx` is given, all samples from that index to the end of the dataset are processed.

### Input

| Input | Description |
|-------|-------------|
| nuScenes pickle file (`--pkl_path`) | Same pickle file as Stage 1A |
| 6-view camera images | Same camera images. Traffic analysis uses **full resolution** (`--resize_factor 1`) for better small-signal detection |

### Output

Per-sample JSON files saved to `{results_dir}/`:

**Filename pattern:** `{idx:04d}_{scene_token}_{sample_token}_traffic.json`

**Contents:**
- `system_prompt`: The system prompt with camera heading information
- `user_question`: The traffic analysis question
- `response`: Model's traffic signal analysis (signal states, orientations, lane governance)
- `inference_time`: Time taken for the API call
- `sample_metadata`: Sample index, scene token, camera heading table

---

## Stage 2: Question Selector

Selects applicable question templates from a question bank for each nuScenes sample. For each template, the module validates whether the template's placeholders can be grounded in the current scene using the 6-view images, 3D object data, and prior analysis results from Stages 1A and 1B.

### How to Run

```bash
# Via shell script (recommended defaults)
bash nuscenes_pipeline/scripts/run_question_selector.sh [START_IDX] [END_IDX] [CATEGORY] [NUM_WORKERS] [SAMPLING_RATIO]

# Full dataset
bash nuscenes_pipeline/scripts/run_question_selector.sh 0 6018 all 8

# 5% random subset (recommended for testing)
bash nuscenes_pipeline/scripts/run_question_selector.sh 0 6018 all 8 5

# Via Python module
python -m nuscenes_pipeline.modules.question_selector \
    --start_idx 0 --end_idx 6018 --category all --num_workers 8 \
    --sampling_ratio 5 --seed 42 \
    --resize_factor 2 --filter_distance 50 --rear_filter 20 \
    --batch_size 5 --max_new_tokens 1024 \
    --risk_results_dir risk_assessment_results \
    --traffic_results_dir traffic_analysis_results \
    --output_dir qa_outputs
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_name` | str | `Qwen/Qwen3-VL-235B-A22B-Instruct` | Model name served by vLLM |
| `--api_base` | str | `http://localhost:8000/v1` | vLLM API endpoint URL |
| `--api_key` | str | `EMPTY` | API key for the vLLM server |
| `--pkl_path` | str | env `NUSCENES_PKL_PATH` | Path to nuScenes pickle file |
| `--question_bank` | str | env `QUESTION_BANK_PATH` | Path to question bank JSON containing template definitions across 10 categories |
| `--vqa_results_dir` | str | `vqa_results` | Directory containing prior VQA result files (used for context building) |
| `--output_dir` | str | `qa_outputs` | Output directory for generated results |
| `--risk_results_dir` | str | `risk_assessment_results` | Directory containing Stage 1A risk assessment JSONs. Used as prior context for the `Dynamic_Agents_and_Risk_Assessment` category |
| `--traffic_results_dir` | str | `traffic_analysis_results` | Directory containing Stage 1B traffic analysis JSONs. Used as prior context for the `Traffic_Signs_and_Signals` category |
| `--start_idx` | int | `0` | Starting sample index (inclusive) |
| `--end_idx` | int | `6018` | Ending sample index (inclusive) |
| `--sampling_ratio` | float | `None` | Percentage of samples to randomly select (e.g., `5` for 5%). When set, randomly samples this ratio from the `[start_idx, end_idx]` range |
| `--seed` | int | `42` | Random seed for reproducible sampling |
| `--category` | str | **required** | Question category to process. One of the 10 categories listed below, or `all` to process all categories |
| `--filter_distance` | float | `20.0` | Maximum distance (meters) from ego to include 3D objects in scene analysis |
| `--rear_filter` | float | `None` | Maximum distance (meters) for non-vehicle objects behind ego |
| `--batch_size` | int | `5` | Number of templates per VLM validation batch. Smaller values reduce GPU OOM risk |
| `--resize_factor` | int | `2` | Image downscale factor (1/n of original 1600x900) |
| `--max_new_tokens` | int | `1024` | Maximum tokens the model generates per validation batch |
| `--num_workers` | int | `8` | Number of parallel multiprocessing workers |
| `--log_dir` | str | `continuous_qa_logs` | Directory for execution logs |

**Valid categories for `--category`:**

| Category | Description |
|----------|-------------|
| `Observation` | Object presence and counting |
| `Identification` | Object type classification |
| `Attributes_and_States` | Object attributes (color, state, damage) |
| `Spatial_Relationships_and_Occlusion` | Relative positions and occlusion |
| `Traffic_Signs_and_Signals` | Traffic signal/sign identification |
| `Road_Markings_and_Lane_Configuration` | Lane markings and road layout |
| `Dynamic_Agents_and_Risk_Assessment` | Moving agents and collision risk |
| `Right_of_Way_and_Planning` | Driving decisions and right-of-way |
| `Environmental_and_Sensor_Conditions` | Weather, lighting, sensor quality |
| `Causal_and_Hypothetical_Reasoning` | "What if" and cause-effect reasoning |
| `all` | Process all 10 categories sequentially |

### Input

| Input | Description |
|-------|-------------|
| nuScenes pickle file (`--pkl_path`) | Same pickle file as previous stages |
| Question bank (`--question_bank`) | JSON file containing question templates organized by 10 categories. Each template has placeholder tags with candidate values and expected answer types |
| Stage 1A results (`--risk_results_dir`) | Risk assessment JSONs from Stage 1A. Provides prior context for risk-related categories |
| Stage 1B results (`--traffic_results_dir`) | Traffic analysis JSONs from Stage 1B. Provides prior context for traffic signal categories |

**Question bank structure (v3/v4 — ego-centric with sub-categories):**
```json
{
  "version": "v3_egocentric_static",
  "categories": [
    {
      "category": "Observation",
      "description": "...",
      "sub_categories": [
        {
          "sub_category": "Ego-Path Presence Detection",
          "templates": [
            {
              "template_id": "OBS-EP-001",
              "template": "Are there any <object> in the ego-vehicle's <spatial_relation>?",
              "placeholders": {
                "object": ["pedestrians", "vehicles", ...],
                "spatial_relation": ["current lane", "intended path", ...]
              },
              "answer_type": "y_or_n",
              "question_type": "observation",
              "ego_centric_note": "Spatial relation is ego-anchored; model must determine which camera views correspond to the specified zone."
            }
          ]
        }
      ]
    }
  ]
}
```

**Answer types:** `y_or_n` (151), `mcq` (68, 5-option A-E), `open_ended` (21), `num_count` (8), `distance` (3)

### Output

Three JSON files per sample, saved to `{output_dir}/`:

| File | Description |
|------|-------------|
| `sample_{idx}_qa_summary.json` | Category statistics: how many templates were tested vs. applicable per category |
| `sample_{idx}_applicable_questions.json` | List of applicable templates with verified placeholder values grounded in the scene |
| `sample_{idx}_inference_detailed.json` | Full debug info including raw VLM responses for each validation batch |

Log files saved to `{log_dir}/`:
- `continuous_qa_vllm_{timestamp}.log` — execution summary
- `failed_qa_indices_{timestamp}.txt` — indices that failed (for retry)

---

## Stage 3: Answer Generator

Takes question selector outputs (Stage 2), classifies each template's placeholders as ENTITY (grounded to specific objects) or LEXICAL (vocabulary choices), pre-instantiates concrete question-answer pairs, and uses the VLM to generate positive answers with contrastive variations for training data diversity.

### How to Run

```bash
# Via shell script (recommended defaults)
bash nuscenes_pipeline/scripts/run_answer_generator.sh [START_IDX] [END_IDX] [CATEGORY] [NUM_WORKERS] [MODE]

# Full range mode (default)
bash nuscenes_pipeline/scripts/run_answer_generator.sh 0 6018 all 8

# Auto-discover mode (processes only samples that have Stage 2 outputs)
bash nuscenes_pipeline/scripts/run_answer_generator.sh 0 6018 all 8 from_stage1

# Via Python module
python -m nuscenes_pipeline.modules.answer_generator \
    --from_stage1 --category all --num_workers 8 \
    --resize_factor 2 --filter_distance 50 --rear_filter 20 \
    --max_new_tokens 16384 --max_pairs 3 \
    --stage1_dir qa_outputs \
    --output_dir qa_results \
    --risk_results_dir risk_assessment_results \
    --traffic_results_dir traffic_analysis_results
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_name` | str | `Qwen/Qwen3-VL-235B-A22B-Instruct` | Model name served by vLLM |
| `--api_base` | str | `http://localhost:8000/v1` | vLLM API endpoint URL |
| `--api_key` | str | `EMPTY` | API key for the vLLM server |
| `--pkl_path` | str | env `NUSCENES_PKL_PATH` | Path to nuScenes pickle file |
| `--question_bank` | str | env `QUESTION_BANK_PATH` | Path to question bank JSON (needed for original placeholder definitions and expected answer types) |
| `--stage1_dir` | str | `qa_outputs` | Stage 2 (question selector) output directory. Supports both per-category subdirs (`qa_outputs/Observation/`) and all-category subdir (`qa_outputs/all/`) |
| `--output_dir` | str | `qa_results` | Output directory for generated QA pairs |
| `--risk_results_dir` | str | `risk_assessment_results` | Directory containing Stage 1A risk assessment JSONs. Prepended to prompt for `Dynamic_Agents_and_Risk_Assessment` templates |
| `--traffic_results_dir` | str | `traffic_analysis_results` | Directory containing Stage 1B traffic analysis JSONs. Prepended to prompt for `Traffic_Signs_and_Signals` templates |
| `--start_idx` | int | `0` | Starting sample index (inclusive). Ignored when `--from_stage1` is set |
| `--end_idx` | int | `6018` | Ending sample index (inclusive). Ignored when `--from_stage1` is set |
| `--from_stage1` | flag | `False` | Auto-discover sample indices from Stage 2 output files instead of using `start_idx`/`end_idx` range. Processes only samples that have applicable question results |
| `--category` | str | `all` | Question category to process, or `all`. Same category choices as Stage 2 |
| `--filter_distance` | float | `50.0` | Maximum distance (meters) from ego to include objects in scene context |
| `--rear_filter` | float | `20.0` | Maximum distance (meters) for non-vehicle objects behind ego |
| `--resize_factor` | int | `2` | Image downscale factor (1/n of original 1600x900) |
| `--max_new_tokens` | int | `16384` | Maximum tokens the model generates per template. Higher than other stages because answers include detailed reasoning |
| `--max_pairs` | int | `3` | Maximum pre-instantiated QA pairs per template. Each pair uses a different combination of grounded placeholder values |
| `--num_workers` | int | `8` | Number of parallel multiprocessing workers |
| `--log_dir` | str | `qa_gen_logs` | Directory for execution logs |

### Input

| Input | Description |
|-------|-------------|
| nuScenes pickle file (`--pkl_path`) | Same pickle file as previous stages |
| Question bank (`--question_bank`) | Same question bank as Stage 2. Used to look up original placeholder definitions and expected answer types |
| Stage 2 results (`--stage1_dir`) | Question selector output directory containing `sample_{idx}_applicable_questions.json` files with verified templates and grounded placeholders |

### Output

Two JSON files per sample, saved to `{output_dir}/`:

| File | Description |
|------|-------------|
| `sample_{idx}_qa_results.json` | Generated QA pairs per template. Each entry includes: the instantiated question, positive answer, contrastive QA variations (altered placeholders), entity grounding details, and placeholder classification (ENTITY vs LEXICAL) |
| `sample_{idx}_qa_detailed.json` | Full debug info with raw VLM responses, pre-instantiation details, and per-template reasoning traces |

Log files saved to `{log_dir}/`:
- `qa_gen_stage2_{timestamp}.log` — execution summary
- `failed_qa_gen_indices_{timestamp}.txt` — indices that failed (for retry)

### Thinking-model variant

For runs that should use `Qwen/Qwen3-VL-235B-A22B-Thinking` instead of the default Instruct model, use the dedicated sibling script `run_answer_generator_thinking.sh`. It mirrors the Instruct script's positional arguments but switches the model name, raises `--max_new_tokens` to `32768` to accommodate the `<think>...</think>` prefix, and writes to a separate output namespace so it never clobbers the Instruct baseline.

vLLM must be launched with the matching reasoning parser so the JSON-only payload arrives in `message.content` (the chain-of-thought is routed to `message.reasoning_content`):

```bash
vllm serve Qwen/Qwen3-VL-235B-A22B-Thinking \
    --tensor-parallel-size 8 \
    --reasoning-parser qwen3 \
    --max-model-len 65536
```

Invocation matches the Instruct script's signature:

```bash
bash nuscenes_pipeline/scripts/run_answer_generator_thinking.sh 0 6018 all 8 from_stage1
```

Output directories used by the Thinking variant:

| Path | Description |
|------|-------------|
| `qa_results_thinking/` | QA pair result files (same schema as `qa_results/`) |
| `answer_generator_thinking_logs/` | Execution logs |
| `prior_disagreements_thinking/` | Prior-disagreement traces |

Post-processing (`transform_obj_to_bbox`, `prepare_sft_dataset`, etc.) needs `--input_dir qa_results_thinking` (or a rename) to consume these outputs.

### RL (GRPO) variant

`nuscenes_pipeline/modules/answer_generator_rl.py` (+ `run_answer_generator_rl.sh`) is a separate fork producing RL-training data (spec: `thinking_answer_generator_4_rl_impl.md`). Structural differences from the SFT modules:

- **Tier-banded `think` array** per pair side — the student model's `<think>` learning target. Step-count bands per category (2-4 / 3-5 / 4-6 / 5-8 by tier); exactly 2 steps when the questioned entity is absent.
- **Fixed key order** `think → reasoning → answer` (no `a_r`/`r_a` switch).
- **`contrast_status` self-report** (`achieved` / `same_answer` / `skipped`) replaces the must-differ constraint; `same_answer` pairs are emitted honestly and harvested downstream as unpaired QA. Contrast is optional for DRA/RML.
- **Validation gate + one retry** per template (malformed JSON, think-band violations, missing contrast_status); failed retries are kept with `rl_flags` — no silent drops. `status_mismatch` cross-checks the self-report against the actual answers.
- **Provenance**: every pair side carries `qa_id` (`s0710_t002_p1_pos` format).
- **Run report**: `qa_results_rl_thinking/rl_run_report.json` — parse success, per-category band compliance, contrast_status distribution, flag counts.

The grounding contract (`tag_mappings` etc.) is frozen and identical to the SFT modules. Outputs are isolated: `qa_results_rl_thinking/`, `answer_generator_rl_logs/`, `prior_disagreements_rl/`.

```bash
# Smoke test first (T10): 5 samples x 2 templates per category
bash nuscenes_pipeline/scripts/run_answer_generator_rl.sh smoke "100,228,304,371,759"

# Full run (same positional signature as the Thinking script, minus ANSWER_MODE;
# 6th arg "resume" skips samples with existing results)
bash nuscenes_pipeline/scripts/run_answer_generator_rl.sh 0 6018 all 8 from_stage1 resume
```

---

## Package Structure

```
qwen-drive/
  nuscenes_pipeline/
    __init__.py
    core/
      __init__.py
      nuscenes_data_loader.py         Core data loading from nuScenes pickle
      nuscenes_prompt_generator.py    Prompt construction, TTC computation, OBB collision detection
      qa_utils.py                     SceneAnalyzer, question bank loader, template validation utilities
    modules/
      __init__.py
      risk_assessment.py              Stage 1A - Risk/hazard analysis
      traffic_analysis.py             Stage 1B - Traffic signal state analysis
      question_selector.py            Stage 2  - Template selection from question bank
      answer_generator.py             Stage 3  - Contrastive QA pair generation
      answer_generator_rl.py          Stage 3 fork - RL (GRPO) training data with think traces
      sft_prompt_builder.py           SFT training prompt construction
      sft_model_tester.py             Qualitative inference test for SFT-trained LoRA model
      demo_tester.py                  Per-frame scene inference with the SFT model (demo bank / custom question)
      merge_demo_predictions.py       Merge per-worker demo_tester outputs into predictions.json
    visualization/
      __init__.py
      bev_generator.py                BEV visualization with ego, objects, velocities
      make_video.py                   Combine panoramic + BEV images into video
      make_scene_videos.py            Per-scene video reels (multi-sample stitching)
      pan_generator.py                6-view panoramic image builder
      pretty_formatting.py            Clean and format JSON result fields
      qa_visualizer.py                Flask web dashboard for QA dataset verification
      eval_sample_visualizer.py       Flask dashboard for v2 eval outputs — GT vs PRED side-by-side (§18)
      demo_scene_video.py             Per-category demo MP4s from demo_tester predictions (§16)
      driving_command_labeler.py      Flask tool to hand-label GT driving commands (7 classes)
    postprocessing/
      __init__.py
      transform_obj_to_bbox.py        Replace OBJ IDs with 2D bbox descriptions; rear cameras x-flipped
      cleanse_obj_references.py       Remove leaked OBJ refs and meta-references (gpt turns only)
      fix_motion_states.py            Correct motion claims against GT velocity
      prepare_sft_dataset.py          Build SFT train/val JSON (system + MCQ options + 6-image prompt)
      convert_to_qwen3vl_format.py    Pack gpt response into reasoning/grounding/answer JSON with native grounding tokens
      build_global_grounded_pool.py   Pre-materialize the global grounded pool for GF (§14)
      count_qa_stats.py               QA pair statistics utility
    scripts/
      run_risk_assessment.sh          Shell script for Stage 1A
      run_traffic_analysis.sh         Shell script for Stage 1B
      run_question_selector.sh        Shell script for Stage 2
      run_answer_generator.sh         Shell script for Stage 3
      run_postprocessing.sh           Wrapper for the full 5-step post-processing pipeline
      run_sft_model_tester.sh         Shell script for SFT model inference test
      run_answer_generator_rl.sh      Shell script for the RL (GRPO) Stage 3 fork
      run_demo_tester_8gpu.sh         8-GPU frame-stride fan-out for demo_tester + auto-merge (§16)
      run_command_labeler.sh          Launch the driving-command labeling web tool
      show_prompts.py                 Prompt preview (prints system+user prompts without inference)
  qwen-vl-finetune/
    configs/
      curriculum_v1.yaml              Per-stage WSD baseline (sequential 10-stage curriculum)
      curriculum_v2.yaml              v2 schedule + composite-loss coefficients (§13)
    qwenvl/
      curriculum/
        config.py                     Stage dataclass loader; emit_shell helper for shell-script orchestration
      train/
        wsd_scheduler.py              Warmup-Stable-Decay LR schedule (per-stage cycle)
        composite_loss.py             Six-term composite SFT loss (§13)
        token_role_masks.py           Per-token role masks (answer / gate / coord / image_idx)
      experiments/                    Ablation framework (§14)
        config.py                     CLI parser + ExpConfig dataclass + manifest writer
        samplers.py                   CompositeSampler — two-stage per-category uniform; §5.2 composition
        lr_schedules.py               global_wsd + per_stage_wsd_relaxed LR variants
        train.py                      Ablation training entry point (subclasses v2 trainer)
        run_experiment.py             Orchestrator: one experiment per invocation
        eval_run.py                   Standalone eval for --skip_eval trained experiments
        aggregate.py                  _master_comparison.{csv,md} builder
        verify_sampler.py             Offline acceptance gate for sampler exposure ratios
    scripts/
      run_curriculum_v2.sh            10-stage v2 training orchestrator (§13)
      eval_curriculum_v2.sh           Standalone per-stage eval pass (§13)
      eval_curriculum_v2_parallel.sh  8-stage fan-out across 8 GPUs (§13)
      eval_intermediate_8stages.sh    Hardcoded mapping for known intermediate checkpoints (§13)
      eval_single_stage_8gpu.sh       Sample-stride parallelism within one stage (§13)
      run_experiment.sh               Wrapper for the ablation launcher (§14)
      eval_experiment.sh              Wrapper for the ablation eval_run (§14)
```

```
  train_nuscenes_qwen3vl.py           SFT training script (LoRA / full)
  train_nuscenes_qwen3vl_v2.py        v2 training script with composite loss + per-cat eval (§13)
```

### Core Modules

- **nuscenes_data_loader.py**: Loads the pre-processed nuScenes pickle file, provides `NuScenesDataLoader` class for accessing samples with 6-view camera images, 3D bounding boxes, velocities, ego poses, and planning annotations. Handles egocentric camera flipping (rear cameras are horizontally mirrored).

- **nuscenes_prompt_generator.py**: Constructs VQA prompts with ego vehicle state, 3D object lists with TTC computation. Implements OBB collision detection using the Separating Axis Theorem (SAT) in global coordinates, with acceleration-aware quadratic motion model for TTC estimation.

- **qa_utils.py**: Provides `SceneAnalyzer` for extracting structured scene context (objects, ego state, camera visibility), `load_question_bank()` for loading template definitions, and category-specific prior builders that incorporate risk/traffic analysis results.

---

## Visualization Tools

### bev_generator.py

Generates Bird's Eye View (BEV) visualizations for nuScenes samples. Draws the ego vehicle, all ground truth objects with oriented bounding boxes, heading arrows, velocity vectors, and range circles on a dark-themed plot. The scene is rotated so the ego vehicle always faces upward.

```bash
# Single sample
python -m nuscenes_pipeline.visualization.bev_generator --sample_idx 42

# Range of samples
python -m nuscenes_pipeline.visualization.bev_generator \
    --start_idx 0 --end_idx 100 \
    --output_dir bev_vis_results --bev_range 50
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--pkl_path` | str | env `NUSCENES_PKL_PATH` | Path to nuScenes pickle file |
| `--sample_idx` | int | `None` | Single sample index to visualize |
| `--start_idx` | int | `None` | Start index for range processing |
| `--end_idx` | int | `None` | End index for range processing |
| `--output_dir` | str | `bev_vis_results` | Output directory for BEV PNG images |
| `--bev_range` | float | `50.0` | Range in meters for each direction (50 = 100m total field of view) |
| `--dpi` | int | `150` | Image resolution |

**Output:** `{idx:04d}_{scene_token}_{sample_token}_bev.png` per sample

**Library usage:**
```python
from nuscenes_pipeline.visualization.bev_generator import generate_bev, save_bev, bev_to_base64

# Save to file
save_bev(sample, loader, "output.png", bev_range=50.0)

# Get as base64 for web embedding
uri = bev_to_base64(sample, loader, bev_range=50.0)

# Get raw matplotlib Figure for custom processing
fig = generate_bev(sample, loader, bev_range=50.0)
```

### make_video.py

Combines panoramic (6-view) and BEV visualization images side-by-side into an MP4 video using ffmpeg.

```bash
python -m nuscenes_pipeline.visualization.make_video \
    --vis_dir pan_vis_results \
    --bev_dir bev_vis_results \
    --start 0 --end 100 \
    --sv_dir videos \
    --output combined_visualization.mp4
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output`, `-o` | str | `combined_visualization.mp4` | Output video filename |
| `--start`, `-s` | int | `0` | Start index |
| `--end`, `-e` | int | `100` | End index |
| `--sv_dir` | str | `.` | Directory to save the output video |
| `--vis_dir` | str | `pan_vis_results` | Directory containing panoramic visualization images |
| `--bev_dir` | str | `bev_vis_results` | Directory containing BEV visualization images |

**Requires:** `ffmpeg`, `opencv-python`

### pretty_formatting.py

Extracts and cleans text fields (system_prompt, prompt, response) from pipeline result JSON files. Handles corrupted text artifacts (mojibake, broken tokens) and outputs clean Python multiline string literals.

```bash
python -m nuscenes_pipeline.visualization.pretty_formatting \
    risk_assessment_results/0000_sample.json \
    --fields system_prompt prompt response \
    --out cleaned_output.py
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `json_path` | str | **required** | Path to the result JSON file to clean |
| `--fields` | str+ | `system_prompt prompt response` | Fields to extract and clean from the JSON |
| `--out` | str | `None` (stdout) | Output file path. If omitted, prints to stdout |

### qa_visualizer.py

Interactive Flask web dashboard for verifying the SFT QA dataset. Displays 6-view panoramic images with bounding boxes extracted from questions (green) and answers (pink), alongside a BEV visualization with ground truth objects and velocity vectors.

```bash
python -m nuscenes_pipeline.visualization.qa_visualizer \
    --data_dir /path/to/qa_dataset \
    --pkl_path /path/to/nuscenes.pkl \
    --port 6060
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--port` | int | `6060` | Web server port |
| `--host` | str | `0.0.0.0` | Web server host |
| `--data_dir` | str | env `QA_DATASET_DIR` | Directory containing SFT dataset JSONs (`sft_train_no_objlist.json`, `sft_val_no_objlist.json`) |
| `--pkl_path` | str | env `NUSCENES_PKL_PATH` | Path to nuScenes pickle file (for BEV generation) |

**Keyboard shortcuts** (in browser):
- Left/Right arrows: navigate between samples
- `r`: jump to a random sample

**Requires:** `flask`, `matplotlib`, `Pillow`

### pan_generator.py

Builds a single 6-view panoramic PNG (FL / F / FR on top, BL / B / BR on bottom; rear views x-flipped) from a nuScenes sample. Pairs with `bev_generator.py` outputs to feed `make_video.py`.

```bash
python -m nuscenes_pipeline.visualization.pan_generator \
    --start_idx 0 --end_idx 100 --output_dir pan_vis_results
```

### make_scene_videos.py

Stitches the per-sample outputs of `pan_generator.py` + `bev_generator.py` into one MP4 video **per nuScenes scene** (vs `make_video.py`'s single combined video). Useful for inspecting model behavior temporally within each scene. Output files are numbered temporally — scenes are ordered by their earliest sample index and written as `scene_0001.mp4`, `scene_0002.mp4`, … (not named by scene token).

```bash
python -m nuscenes_pipeline.visualization.make_scene_videos \
    --vis_dir pan_vis_results --bev_dir bev_vis_results \
    --output_dir scene_videos
```

### eval_sample_visualizer.py

Flask dashboard for the v2 evaluation outputs (§18). Renders GT vs PRED side-by-side: green / pink box overlays on 6-view images, parsed reasoning and answer fields, per-box IoU + view-OK match table. See [§18](#eval-sample-visualizer-browser-based) for launch commands.

### driving_command_labeler.py

Flask browser tool to hand-label the **ground-truth driving command** per sample. Shows the 3 forward camera views (FL / F / FR) plus a BEV pane, with 7 command classes: `0` Turn left, `1` Turn right, `2` Go straight, `3` Follow lane, `4` Change lane to left, `5` Change lane to right, `6` U-Turn. Click a button or press keys `0`–`6`; each label autosaves and the view auto-advances (re-labeling with a different command overwrites and stays on the sample; Backspace clears). Output is one JSON per scene (`{scene_id, scene_index, sample_labels}`); existing label files in `--output_dir` are reloaded on startup, so labeling resumes across server restarts.

```bash
# PORT (default 6062) and SPLIT (train | val, default train).
# SPLIT selects the pkl and writes labels to driving_command_labels_<SPLIT>/
bash nuscenes_pipeline/scripts/run_command_labeler.sh 6062 train
```

---

## Post-Processing Pipeline

After Stage 3 finishes, the QA results go through five post-processing steps that produce the final Qwen3-VL SFT training dataset. The single wrapper script runs the whole sequence and supports two output modes:

```bash
# Mixed flat files (default): sft_{train,val}_qwen3vl.json
bash nuscenes_pipeline/scripts/run_postprocessing.sh

# Per-category curriculum-learning files: sft_{train,val}_qwen3vl_<ABBR>.json
MODE=curriculum bash nuscenes_pipeline/scripts/run_postprocessing.sh
```

Override defaults via env vars: `MODE` (`full` | `curriculum`), `PKL_PATH`, `DATA_ROOT`, `QA_INPUT_DIR`, `SFT_DIR`. End-to-end runtime is ~2-3 minutes on the 5%-subset (300 samples) for either mode.

### Mode summary

| | `MODE=full` (default) | `MODE=curriculum` |
|---|---|---|
| Step 2 output | 2 mixed files | 20 per-category files (10 cats × {train, val}) |
| Steps 3 & 5 | run once each | loop over the 20 files |
| Step 4 | 1 invocation | 1 invocation with `--curriculum` — loader + image-index + velocity cache **shared** across all 20 files so pkl-load startup is paid once |
| Final outputs | `sft_dataset/sft_{train,val}_qwen3vl.json` | `sft_dataset/sft_{train,val}_qwen3vl_{OBS,IDN,AAS,SRO,TSS,RML,DRA,RWP,ESC,CHR}.json` |
| Use case | single LoRA run, baseline experiments | curriculum-learning chained stages (easy → hard) |

The 3-letter category abbreviations used throughout:

| Source category | Abbr |
|---|---|
| `Observation` | `OBS` |
| `Identification` | `IDN` |
| `Attributes_and_States` | `AAS` |
| `Spatial_Relationships_and_Occlusion` | `SRO` |
| `Traffic_Signs_and_Signals` | `TSS` |
| `Road_Markings_and_Lane_Configuration` | `RML` |
| `Dynamic_Agents_and_Risk_Assessment` | `DRA` |
| `Right_of_Way_and_Planning` | `RWP` |
| `Environmental_and_Sensor_Conditions` | `ESC` |
| `Causal_and_Hypothetical_Reasoning` | `CHR` |

Canonical curriculum order (easy perceptual → hard reasoning):
`OBS → IDN → AAS → SRO → TSS → RML → DRA → RWP → ESC → CHR`

### Pipeline Data Flow

```
answer_generator output (qa_results/sample_*_qa_results.json — OBJ IDs)
        |
        v
[Step 1] transform_obj_to_bbox.py    OBJ IDs → 2D bbox descriptions in 1600x900
                                      pixel space; rear cameras x-mirrored.
        |
        v
sft_dataset/sample_*_qa_results.json  (bbox-transformed QA results)
        |
        v
[Step 2] prepare_sft_dataset.py       Build 3-turn SFT conversations. MCQ options
                                      appended under TASK. Each sample tagged with
                                      its source `category`.
                                      --full       -> 1 train + 1 val file
                                      --curriculum -> 10 train + 10 val files
        |
        v
sft_dataset/sft_{train,val}_no_objlist[_ABBR].json
        |
        v
[Step 3] cleanse_obj_references.py    Remove leaked OBJ refs (gpt turns only).
        |
        v
[Step 4] fix_motion_states.py         Correct motion state claims vs GT velocity.
                                      --curriculum amortizes pkl/index/cache across
                                      all per-category files in one invocation.
        |
        v
[Step 5] convert_to_qwen3vl_format.py System turn preserved; gpt value packed
                                      into unified JSON (reasoning/grounding/answer);
                                      bboxes rendered with native Qwen3-VL grounding
                                      tokens + 0-1000 normalized coords.
        |
        v
sft_dataset/sft_{train,val}_qwen3vl[_ABBR].json   (FINAL — ready for SFT training)
```

### Step 1: Transform OBJ to Bbox

Replaces OBJ ID references in QA answers with visual 2D bounding box descriptions, projecting 3D ground-truth boxes through each camera intrinsic. **Rear-camera bboxes are x-mirrored** (`x_new = W - x_old`) so coordinates match the horizontally flipped image that the model is shown both at annotation time and during training.

```bash
python -m nuscenes_pipeline.postprocessing.transform_obj_to_bbox \
    --input_dir qa_results \
    --output_dir sft_dataset \
    --data_root ./data/nuscenes \
    --resize_factor 1
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input_dir` | str | `qa_results/Observation` | Directory with answer generator output (contains OBJ IDs) |
| `--output_dir` | str | `sft_dataset/Observation` | Output directory for bbox-transformed results |
| `--pkl_path` | str | env `NUSCENES_PKL_PATH` | Path to nuScenes pickle file |
| `--data_root` | str | env `NUSCENES_DATA_ROOT` | nuScenes data directory (where `samples/CAM_*/` images live) |
| `--resize_factor` | int | `1` | Image resize factor for bbox coord space. `1` = full 1600x900 (default); `2` = 800x450 |

**Transformation examples:**
- `"OBJ 30"` → `"pedestrian (Image 2 (Front) bbox[788,234,799,314])"`
- `"OBJ 5 (pedestrian, 46.6m ahead)"` → `"pedestrian (46.6m ahead, Image 1 bbox[691,219,712,247])"`

### Step 2: Prepare SFT Dataset

Converts bbox-transformed QA results into Qwen3-VL SFT training format (3-turn conversations: system, user, assistant) **without 3D object list** in the prompt. For MCQ questions, the labeled options `(A)..(E)` are appended under the `TASK:` block so the model sees what each letter refers to. Every sample carries a top-level `category` field copied from its source `qa_result` entry — downstream steps preserve this tag.

Two output modes via mutually-exclusive flags:

```bash
# Mixed flat files (default)
python -m nuscenes_pipeline.postprocessing.prepare_sft_dataset \
    --qa_dir sft_dataset \
    --output_dir sft_dataset \
    --data_root ./data/nuscenes \
    --val_size 40000 \
    --full

# 20 per-category files (10 cats × train+val)
python -m nuscenes_pipeline.postprocessing.prepare_sft_dataset \
    --qa_dir sft_dataset \
    --output_dir sft_dataset \
    --data_root ./data/nuscenes \
    --val_size 40000 \
    --curriculum
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--qa_dir` | str | `sft_dataset/Observation` | Directory with bbox-transformed QA results |
| `--pkl_path` | str | env `NUSCENES_PKL_PATH` | Path to nuScenes pickle file |
| `--data_root` | str | env `NUSCENES_DATA_ROOT` | nuScenes data directory (where `samples/CAM_*/` images live) |
| `--output_dir` | str | `sft_dataset` | Output directory for SFT JSON files |
| `--resize_factor` | int | `2` | Image resize factor |
| `--val_size` | int | `40000` | Validation sample cap. In `--curriculum` mode this cap is applied per category. |
| `--seed` | int | `42` | Random seed for train/val split |
| `--full` | flag | (default) | Emit 2 mixed files: `sft_{train,val}_no_objlist.json` |
| `--curriculum` | flag | — | Bucket samples by source category and emit 20 files: `sft_{train,val}_no_objlist_{OBS,IDN,AAS,SRO,TSS,RML,DRA,RWP,ESC,CHR}.json` |

**Outputs:**
- `--full`: `sft_train_no_objlist.json`, `sft_val_no_objlist.json`
- `--curriculum`: 20 files matching `sft_{train,val}_no_objlist_<ABBR>.json`

### Step 3: Cleanse OBJ References (Rounds 1-3)

Removes any remaining leaked OBJ references that survived the bbox transformation. **Only `gpt` turns are cleansed** — the human/system prompts are built from templates and never carry OBJ refs, so cleansing them would destructively collapse intentional `\n\n` paragraph breaks (e.g., the blank line between `TASK:` and the MCQ `Options:` block).

- **Round 1**: Direct OBJ N patterns (e.g., `OBJ 36`, `(OBJ 36)`)
- **Round 2**: Sentences containing meta-references (`"spatial data"`, `"OBJ ID"`, `"object list"`, `"prior knowledge"`, etc.)
- **Round 3**: Edge-case phrase replacements (`"spatial database"` → `"scene"`, `"object list"` → `"scene"`)

```bash
# Cleanse train set
python -m nuscenes_pipeline.postprocessing.cleanse_obj_references \
    --input sft_dataset/sft_train_no_objlist.json

# Cleanse val set
python -m nuscenes_pipeline.postprocessing.cleanse_obj_references \
    --input sft_dataset/sft_val_no_objlist.json

# Dry run (report only, no modifications)
python -m nuscenes_pipeline.postprocessing.cleanse_obj_references \
    --input sft_dataset/sft_train_no_objlist.json --dry_run
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input` | str | **required** | SFT JSON file to cleanse |
| `--output` | str | same as input | Output path (default: overwrite in place) |
| `--dry_run` | flag | `False` | Report counts without modifying the file |

### Step 4: Fix Motion States (Round 5)

Cross-references motion state claims in answers against ground truth velocity data. Corrects contradictions where text says "moving" but velocity < 0.1 m/s, or "parked" but velocity > 0.1 m/s.

The loader, scene analyzer, image index, and per-sample velocity cache are built **once** and reused across every input file in a single invocation — so the pkl-load startup cost is paid only once regardless of how many files are processed.

```bash
# Full mode (mixed files)
python -m nuscenes_pipeline.postprocessing.fix_motion_states \
    --data_dir sft_dataset \
    --data_root ./data/nuscenes

# Curriculum mode (also picks up sft_{train,val}_no_objlist_<ABBR>.json)
python -m nuscenes_pipeline.postprocessing.fix_motion_states \
    --data_dir sft_dataset \
    --data_root ./data/nuscenes \
    --curriculum
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data_dir` | str | `sft_dataset` | Directory containing the SFT JSON files |
| `--pkl_path` | str | env `NUSCENES_PKL_PATH` | Path to nuScenes pickle file |
| `--data_root` | str | env `NUSCENES_DATA_ROOT` | nuScenes data directory (where `samples/CAM_*/` images live) |
| `--curriculum` | flag | `False` | Widen the file list to include all `sft_{train,val}_no_objlist_*.json` matches. Shared loader/index/cache amortize pkl-load across all 20 files. |

**Phrase corrections** (22 patterns):
- Motion → Stationary (velocity < 0.1 m/s): `"actively riding"` → `"stationary"`, `"in motion"` → `"stationary"`, etc.
- Stationary → Motion (velocity ≥ 0.1 m/s): `"is parked"` → `"is moving"`, `"pulled over"` → `"in motion"`, etc.

### Step 5: Convert to Qwen3-VL Unified JSON

Repacks each sample into the format the SFT trainer consumes:
- **System turn preserved** at index 0 (`from: "system"`). Our `train_nuscenes_qwen3vl.py` renders it as `<|im_start|>system\n...<|im_end|>\n` and masks with `IGNORE_INDEX` so it serves as prefix context but not loss target.
- **gpt.value packed into a single JSON string** with three keys:
  - `reasoning` — bullet-point string with bbox parentheticals stripped.
  - `grounding` — list of `{image_idx, camera, ref}` objects. The `ref` field uses Qwen3-VL's **native grounding special tokens**: `<|object_ref_start|>label<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>`. Coordinates are normalized to **[0, 1000]** using each image's actual PIL-read dimensions, matching the convention Qwen-VL was pretrained on.
  - `answer` — short factual response (single letter for MCQ, `Yes`/`No`, short phrase, or 2-4 sentences for `open_ended`).

```bash
python -m nuscenes_pipeline.postprocessing.convert_to_qwen3vl_format \
    --input  sft_dataset/sft_train_no_objlist.json \
    --output sft_dataset/sft_train_qwen3vl.json
python -m nuscenes_pipeline.postprocessing.convert_to_qwen3vl_format \
    --input  sft_dataset/sft_val_no_objlist.json \
    --output sft_dataset/sft_val_qwen3vl.json
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input` | str | **required** | Pre-conversion SFT JSON (from Steps 1-4) |
| `--output` | str | **required** | Output Qwen3-VL JSON |
| `--dry_run` | flag | `False` | Report stats and sample preview without writing |
| `--force` | flag | `False` | Overwrite the `.format_qwen3vl` idempotency marker |

**Output schema** (one sample):
```json
{
  "image": ["...CAM_FRONT_LEFT.jpg", "...CAM_FRONT.jpg", ...],
  "conversations": [
    {"from": "system", "value": "You are an autonomous driving..."},
    {"from": "human",  "value": "...TASK: ...\n\nOptions:\n(A) ...\n(B) ..."},
    {"from": "gpt",    "value": "{\n  \"reasoning\": \"- ...\",\n  \"grounding\": [{\"image_idx\": 1, \"camera\": \"Front-left\", \"ref\": \"<|object_ref_start|>pedestrian<|object_ref_end|><|box_start|>(614,473),(705,760)<|box_end|>\"}],\n  \"answer\": \"No\"\n}"}
  ]
}
```

### QA Statistics Utility

```bash
# Analyze all QA results in a directory
python -m nuscenes_pipeline.postprocessing.count_qa_stats qa_results/

# Analyze a single file
python -m nuscenes_pipeline.postprocessing.count_qa_stats qa_results/sample_0_qa_results.json
```

---

## Full Pipeline Example (End-to-End)

```bash
cd /path/to/qwen-drive

export NUSCENES_PKL_PATH="./data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl"
export QUESTION_BANK_PATH="./data/question_bank_v4_static.json"

# === Stage 1A & 1B (can run in parallel) ===
bash nuscenes_pipeline/scripts/run_risk_assessment.sh 0 6018 8
bash nuscenes_pipeline/scripts/run_traffic_analysis.sh 0 6018 8

# === Stage 2: Question Selection ===
# Full dataset
bash nuscenes_pipeline/scripts/run_question_selector.sh 0 6018 all 8
# Or 5% subset for testing
bash nuscenes_pipeline/scripts/run_question_selector.sh 0 6018 all 8 5

# === Stage 3: Answer Generation ===
# Auto-discover mode (recommended when using sampled subset)
bash nuscenes_pipeline/scripts/run_answer_generator.sh 0 6018 all 8 from_stage1
# Or full range mode
bash nuscenes_pipeline/scripts/run_answer_generator.sh 0 6018 all 8

# === Post-Processing (5 steps in one wrapper) ===
# Default: mixed flat files for single-LoRA training
bash nuscenes_pipeline/scripts/run_postprocessing.sh
# Produces sft_dataset/sft_{train,val}_qwen3vl.json — final SFT inputs.

# Or: per-category files for curriculum-learning chained stages
MODE=curriculum bash nuscenes_pipeline/scripts/run_postprocessing.sh
# Produces sft_dataset/sft_{train,val}_qwen3vl_{OBS,IDN,...,CHR}.json
# Train sequentially: OBS -> IDN -> AAS -> SRO -> TSS -> RML -> DRA -> RWP -> ESC -> CHR
```

---

## SFT Training

Fine-tunes Qwen3-VL on the generated QA dataset using LoRA or full fine-tuning with DeepSpeed.

### LoRA Fine-Tuning (Recommended)

```bash
# Via shell script
NPROC_PER_NODE=8 bash qwen-vl-finetune/scripts/run_nuscenes_lora.sh

# Direct command
torchrun --nproc_per_node=8 train_nuscenes_qwen3vl.py \
    --deepspeed qwen-vl-finetune/scripts/zero2.json \
    --mode lora \
    --model_name_or_path ckpts/qwen3_vl_8b_instruct \
    --train_data_path sft_dataset/sft_train_qwen3vl.json \
    --val_data_path sft_dataset/sft_val_qwen3vl.json \
    --resize_factor 2 \
    --lora_r 64 --lora_alpha 128 --lora_dropout 0.05 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --bf16 --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-4 \
    --output_dir output_nuscenes_lora_qwen3vl \
    --gradient_checkpointing True \
    --report_to tensorboard
```

`train_nuscenes_qwen3vl.py` is our custom training script (separate from the vendored `qwen-vl-finetune` / `Qwen-VL-Series-Finetune` repos). It supports `from: "system"` turns natively via `preprocess_with_system_prompt()` — system content is rendered with the standard `<|im_start|>system\n...<|im_end|>\n` chat template and masked with `IGNORE_INDEX` so it forms prefix context but not loss targets.

### Full Fine-Tuning

```bash
NPROC_PER_NODE=8 bash qwen-vl-finetune/scripts/run_nuscenes_full.sh
```

### Key Training Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mode` | str | **required** | `lora` or `full` |
| `--model_name_or_path` | str | — | Path to Qwen3-VL model or HuggingFace ID |
| `--train_data_path` | str | `sft_dataset/sft_train_qwen3vl.json` | Training data from post-processing pipeline (Step 5 output) |
| `--val_data_path` | str | `sft_dataset/sft_val_qwen3vl.json` | Validation data (Step 5 output) |
| `--resize_factor` | int | `2` | Image downscale factor (must match pipeline) |
| `--lora_r` | int | `64` | LoRA rank |
| `--lora_alpha` | int | `128` | LoRA alpha |
| `--lora_dropout` | float | `0.05` | LoRA dropout |
| `--deepspeed` | str | — | DeepSpeed config path (`zero2.json` for LoRA, `zero3.json` for full) |
| `--bf16` | flag | — | Use bfloat16 training |
| `--data_flatten` | bool | `False` | Pack sequences for efficiency (full fine-tune only) |
| `--max_pixels` | int | `50176` | Max image pixels |
| `--min_pixels` | int | `784` | Min image pixels |
| `--lora_pretrained` | str | `None` | Warm-start LoRA from an existing adapter dir (used for curriculum-stage handoff). When set, the adapter's own LoraConfig is loaded; the `--lora_r/_alpha/_dropout` flags are ignored. |
| `--eval_dataset_paths_json` | str | `None` | Path to a JSON `{category: val_path}` mapping. When set, HF Trainer runs **per-category eval** at every `eval_steps` and logs `eval_<CAT>_loss` for each entry. Overrides `--val_data_path`. |
| `--use_wsd_scheduler` | bool | `False` | Activate the warmup-stable-decay LR scheduler (see Curriculum Learning section). |
| `--wsd_warmup_ratio` | float | `0.10` | Fraction of stage steps used for linear warmup (only if `--use_wsd_scheduler True`). |
| `--wsd_decay_ratio` | float | `0.20` | Fraction of stage steps used for linear decay to 0 (only if `--use_wsd_scheduler True`). |

> **Note (transformers ≥5):** `tokenizer.apply_chat_template(...)` now returns a `BatchEncoding` (dict-like) instead of a `list[int]`. `preprocess_with_system_prompt` unwraps the `"input_ids"` field on the fly, so the same training script works on both transformers 4.x and 5.x.

---

## Curriculum Learning (Sequential Multi-Stage SFT)

> **Note**: this section documents the **v1 curriculum** (sequential per-category SFT with standard causal CE).
> The **v2 successor** ([§13](#curriculum-v2-training-and-evaluation)) adds a six-term composite SFT loss
> (answer / gate / view / IoU weighting) plus per-category evaluation metrics (`answer_acc`, `view_acc`,
> `grounding_acc@<iou>`, `grounding_format_valid`). Both pipelines still ship and stay in sync; pick v1 for
> the simpler baseline, v2 for the ablation framework ([§14](#ablation-experiments-framework)) and the
> grounding-aware loss.

Trains the LoRA adapter across the 10 question categories **in sequence**, easy-perceptual → hard-reasoning, with each stage warm-starting from the previous stage's final adapter. Per-category eval losses are logged on every validation step across **all 10 categories** (not just the current stage's), and a per-category generation-accuracy eval runs at the end of every stage so you can spot catastrophic forgetting in the aggregation report.

### Curriculum order

```
OBS → IDN → AAS → SRO → TSS → RML → DRA → RWP → ESC → CHR
```

| code | full name |
|---|---|
| OBS | Observation |
| IDN | Identification |
| AAS | Attributes & States |
| SRO | Spatial Relationships & Occlusion |
| TSS | Traffic Signs & Signals |
| RML | Road Markings & Lane Configuration |
| DRA | Dynamic Agents & Risk Assessment |
| RWP | Right-of-Way & Planning |
| ESC | Environmental & Sensor Conditions |
| CHR | Causal & Hypothetical Reasoning |

### Per-stage LR shape (W-S-D)

Each stage has its **own** warmup → stable → decay cycle (default ratios `0.10 / 0.70 / 0.20`):

```
lr_mult
   1.0 |         ___________________
       |       /                     \
       |     /                         \
   0.0 |___/                             \___
            |--warmup--|------stable------|--decay--|
            0                                       stage_steps
```

### Components

| Path | Purpose |
|---|---|
| `qwen-vl-finetune/configs/curriculum_v1.yaml` | Real curriculum config (10 stages, full epochs). |
| `qwen-vl-finetune/configs/curriculum_smoke.yaml` | Smoke-test config (20 steps/stage, 20 samples/cat eval) for verifying the pipeline end-to-end in minutes. |
| `qwen-vl-finetune/qwenvl/curriculum/config.py` | YAML loader and shell-vars emitter (`--emit_shell <i>`). |
| `qwen-vl-finetune/qwenvl/train/wsd_scheduler.py` | `LambdaLR` factory implementing W-S-D. |
| `qwen-vl-finetune/scripts/run_curriculum.sh` | Orchestrator. Loops 10 stages: emit env, run `torchrun`, run stage-end eval. |
| `nuscenes_pipeline/postprocessing/build_eval_subset.py` | Stratified per-category val subset builder (default 200/cat). |
| `nuscenes_pipeline/scripts/run_stage_end_eval.sh` | Per-stage generation eval wrapper around `sft_model_tester`. |
| `hf_dataset_train/aggregate_curriculum_reports.py` | Builds the stage × category accuracy matrix + forgetting column. |

### Prerequisites

1. **Per-category SFT files** — produced by post-processing in curriculum mode:
   ```bash
   MODE=curriculum bash nuscenes_pipeline/scripts/run_postprocessing.sh
   # writes sft_dataset/sft_{train,val}_qwen3vl_{OBS,IDN,…,CHR}.json
   ```
2. **Stratified eval subset** — sampled once, reused for every stage's in-training eval **and** end-of-stage generation eval:
   ```bash
   python -m nuscenes_pipeline.postprocessing.build_eval_subset \
       --input_dir sft_dataset --n_per_cat 200
   # writes sft_dataset/eval_subset_200/sft_val_qwen3vl_<CAT>_subset.json + manifest.json
   ```

### Running the curriculum

#### Smoke test first (always)

20 optimizer steps/stage + 10 samples/cat for the stage-end eval. Confirms warm-start, dict-form eval, WSD shape, generation eval, and aggregation all work end-to-end before you commit GPU-hours.

```bash
python -m nuscenes_pipeline.postprocessing.build_eval_subset \
    --n_per_cat 20 --output_dir sft_dataset/eval_subset_smoke

END_STAGE=1 STAGE_END_EVAL_LIMIT=10 \
    bash qwen-vl-finetune/scripts/run_curriculum.sh \
    qwen-vl-finetune/configs/curriculum_smoke.yaml
```

What to look for in the logs:
- `[WSDTrainer] WSD scheduler: total_steps=<N> warmup_ratio=0.1 decay_ratio=0.2`
- Stage 1 only: `Warm-starting LoRA from: output/curriculum_smoke/stage_00_OBS`
- Eval lines containing all of `eval_OBS_loss … eval_CHR_loss`
- `Eval report saved to: …/eval_report.json` after each stage

#### Single-GPU full curriculum

```bash
bash qwen-vl-finetune/scripts/run_curriculum.sh \
    qwen-vl-finetune/configs/curriculum_v1.yaml
```

#### Multi-GPU (single node)

`NPROC_PER_NODE` is the only mandatory env var. ZeRO-2 + torchrun handle distribution automatically.

```bash
NPROC_PER_NODE=8 \
    bash qwen-vl-finetune/scripts/run_curriculum.sh \
    qwen-vl-finetune/configs/curriculum_v1.yaml
```

**Effective batch size scales with NPROC_PER_NODE.** With the v1 defaults (`per_device_train_batch_size=1`, `gradient_accumulation_steps=4`), 8 GPUs give effective batch 32 (vs 4 on 1 GPU). Consider scaling peak LR by ~√N when increasing GPU count (e.g. `peak_lr: 5.6e-4` for 8 GPUs ≈ `2e-4 × √8`). Override per-stage in the YAML rather than via env vars.

The WSD step counts adapt automatically — the scheduler is anchored to HF Trainer's computed `num_training_steps`, which already accounts for the distributed batch.

#### Multi-node

```bash
NPROC_PER_NODE=8 NNODES=2 NODE_RANK=$NODE_RANK \
MASTER_ADDR=node0.cluster MASTER_PORT=29500 \
    bash qwen-vl-finetune/scripts/run_curriculum.sh \
    qwen-vl-finetune/configs/curriculum_v1.yaml
```

(The orchestrator reads `NNODES` and `MASTER_ADDR`; you'd add `NODE_RANK` handling to `torchrun` if needed.)

### Orchestrator env-var overrides

| Variable | Purpose |
|---|---|
| `NPROC_PER_NODE` | GPUs per node (default 1). |
| `START_STAGE` / `END_STAGE` | Resume a partial curriculum (default 0 / last). |
| `SKIP_STAGE_END_EVAL=1` | Skip the per-stage generation eval (train only). |
| `STAGE_END_EVAL_LIMIT=N` | Cap generation-eval samples per category (smoke). |
| `DEEPSPEED_CONFIG=""` | Disable DeepSpeed entirely (single-GPU dev only). |
| `DEEPSPEED_CONFIG=path/to/zero3.json` | Use a different DeepSpeed config (e.g. ZeRO-3 if OOM). |

### Output structure

```
output/curriculum_v1/
├── stage_00_OBS/
│   ├── checkpoint-*/                      # HF Trainer checkpoints
│   ├── adapter_config.json
│   ├── adapter_model.safetensors          # final LoRA adapter (warm-start for stage 01)
│   ├── eval_dataset_paths.json            # written by the curriculum config emitter
│   ├── eval_report.json                   # per-category generation accuracy
│   └── eval_predictions.json              # per-sample predictions (if --save_predictions)
├── stage_01_IDN/
│   └── ...
├── ...
├── stage_09_CHR/
├── curriculum_report.csv                  # 10×10 stage × category accuracy matrix
└── curriculum_report.md                   # same matrix + forgetting column
```

### Aggregated report

Runs automatically at the end of `run_curriculum.sh`; can also be invoked manually:

```bash
python hf_dataset_train/aggregate_curriculum_reports.py \
    --output_root output/curriculum_v1
```

The Markdown report includes a **Forgetting** column:

```
forgetting(CAT) = max(acc_CAT across stages 0..N-1) − acc_CAT at final stage
```

Positive values indicate the model lost ground on `CAT` by the end of the curriculum — the main risk of sequential SFT.

### Curriculum config schema

Minimal example (defaults shown):

```yaml
output_root: output/curriculum_v1
base_model: ckpts/qwen3_vl_8b_instruct
train_data_dir: sft_dataset
eval_subset_dir: sft_dataset/eval_subset_200
eval_categories: [OBS, IDN, AAS, SRO, TSS, RML, DRA, RWP, ESC, CHR]

defaults:
  epochs: 1
  max_steps: -1          # -1 = ignore; >0 caps optimizer steps (smoke runs)
  peak_lr: 2.0e-4
  wsd: [0.10, 0.70, 0.20]
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4
  eval_steps: 500
  save_steps: 500
  save_total_limit: 2
  max_pixels: 1440208
  min_pixels: 784
  lora_r: 64
  lora_alpha: 128
  full_val_for_stage_end_eval: false   # set true to use the full per-cat val (slow)

stages:
  - { name: OBS }
  - { name: IDN, epochs: 2, peak_lr: 1.5e-4 }    # per-stage override example
  - { name: AAS }
  # ...
```

Any field under `defaults` can be overridden per-stage. The loader rejects unknown keys to fail loud rather than silently miss a typo.

---

## Curriculum-v2 Training and Evaluation

Curriculum-v2 builds on the per-category curriculum (§12) with two additions:

1. **Composite SFT loss** — six terms beyond standard causal CE: extra weights
   on the answer field and the empty-vs-non-empty grounding gate, a 6-way
   restricted softmax on the `image_idx` digit, and an IoU-aware
   re-weighting of coordinate tokens. See `loss_system.md` for the
   per-term spec and `loss_improvement_design_v3.md` for the design
   rationale.
2. **Per-category evaluation** — `sft_model_tester --eval_v2` reports
   `answer_acc`, `view_acc`, `grounding_acc@<iou>`, and
   `grounding_format_valid` per category. Prediction box extraction
   operates on raw token IDs so box delimiter special tokens aren't
   stripped at decode time.

   Since the T1 completeness work (spec: `v3_loss_completeness_task.md`),
   each category additionally reports grounding-completeness columns:
   `referring_completeness`, `n_missing_boxes`, `n_spurious_boxes`,
   `per_view_recall` (per 6-view slot), `per_view_gt_count`, and a 6×6
   `view_confusion` matrix. The metrics dict also carries `_raw_*` count
   fields (matched-box counts, missing-box and completeness sums) — not
   for human consumption: `eval_single_stage_8gpu.sh` sums them across
   its 8 workers and re-derives the ratios at merge time. The merge is
   backward-compatible — worker reports from pre-T1 checkpoints simply
   omit the completeness columns and the merger skips them.

### Run the v2 curriculum (10 stages)

```bash
# Training only (skip in-training + per-stage gen eval)
SKIP_INTRAINING_EVAL=1 SKIP_STAGE_END_EVAL=1 NPROC_PER_NODE=8 \
    bash qwen-vl-finetune/scripts/run_curriculum_v2.sh \
        qwen-vl-finetune/configs/curriculum_v2.yaml

# Standalone evaluation across all 10 stages × all 10 categories
bash qwen-vl-finetune/scripts/eval_curriculum_v2.sh \
    qwen-vl-finetune/configs/curriculum_v2.yaml

# Parallel evaluation: 8 stages × 8 GPUs (~30 min instead of ~4 hours)
bash qwen-vl-finetune/scripts/eval_curriculum_v2_parallel.sh \
    qwen-vl-finetune/configs/curriculum_v2.yaml

# Single-stage eval across 8 GPUs (sample-stride parallelism, ~4 min)
GPU_IDS="0,1,2,3,4,5,6,7" \
    bash qwen-vl-finetune/scripts/eval_single_stage_8gpu.sh \
        stage_04_TSS [checkpoint-XXX]
```

### Output structure

```
output/curriculum_v2_<run>/
├── stage_00_OBS/ … stage_09_CHR/
│   ├── adapter_*.safetensors
│   ├── eval_dataset_paths.json
│   ├── eval_report.json           # per-stage, all 10 cats
│   └── eval_predictions.json
├── curriculum_report.csv          # stage × category matrix
└── curriculum_report.md
```

Per-step training logs surface the composite-loss breakdown (`loss_base_ce`,
`loss_answer`, `loss_answer_w`, … plus `grounded_frac`); the invariant
`loss == loss_base_ce + Σ loss_*_w` holds by construction.

### Reference docs

- `loss_system.md` — as-built reference for the six loss terms
- `curriculum_v2_loss.md` — narrative explanation with code snippets
- `loss_improvement_design_v3.md` — design rationale (frozen invariants,
  per-term motivation, R-VLM / KLAL references)
- `curriculum_pipeline_analysis.md` — v1 pipeline architecture analysis

---

## Ablation Experiments Framework

Argument-driven launcher for the 16-run factorial ablation matrix
(`B0, F1, F2, F3, L1, L2, S7, C01–C08, C12`). Each run is fully
specified by four CLI knobs and produces a self-contained
`run_manifest.json` for reproducibility. No static preset table — the
16 canonical flag combinations live as a reference table in the spec.

### Knobs

| Knob | Values | Spec |
|---|---|---|
| `--mode` | `sequential` \| `mixed` | §2.1 |
| `--lr_schedule` | `per_stage_wsd` \| `global_wsd` \| `per_stage_wsd_relaxed` | §2.4 |
| `--grounding_floor` | `off` \| float in (0, 1) | §2.2 |
| `--replay` | `off` \| float in (0, 1) | §2.3 |

### Run a single experiment

```bash
# Examples
bash qwen-vl-finetune/scripts/run_experiment.sh \
    --exp_id B0 --mode sequential --lr_schedule per_stage_wsd \
    --grounding_floor off --replay off \
    --output_root output/curriculum_v2_exp --seed 0

bash qwen-vl-finetune/scripts/run_experiment.sh \
    --exp_id F3 --mode mixed --lr_schedule global_wsd \
    --grounding_floor off --replay off \
    --output_root output/curriculum_v2_exp --seed 0

# Dry-run (no GPU, prints resolved config, exits)
bash qwen-vl-finetune/scripts/run_experiment.sh \
    --exp_id C12 --mode mixed --lr_schedule global_wsd \
    --grounding_floor 0.25 --replay 0.10 \
    --output_root /tmp --dry_run
```

### Split training and evaluation

For long runs, `--skip_eval` trains only (skips per-stage gen-eval and
`summary.json`). Run evaluation later via `eval_experiment.sh`:

```bash
# Train only
bash qwen-vl-finetune/scripts/run_experiment.sh \
    --exp_id F1 --mode sequential --lr_schedule per_stage_wsd \
    --grounding_floor 0.25 --replay off \
    --output_root output/curriculum_v2_exp --seed 0 \
    --skip_eval

# Evaluate later (idempotent — re-running skips completed checkpoints)
bash qwen-vl-finetune/scripts/eval_experiment.sh \
    --run_dir output/curriculum_v2_exp/F1__seed0
```

### Sampler acceptance gate

Before launching any GF / ER / mixed experiment, verify the realized
per-category exposure matches the spec's expected weights:

```bash
PYTHONPATH=qwen-vl-finetune python -m qwenvl.experiments.verify_sampler \
    --mode mixed --lr_schedule global_wsd \
    --grounding_floor 0.25 --replay off \
    --num_draws 100000
```

Exits non-zero if any (pool, category) cell deviates more than
`--tolerance` (default 2%) from the expected `1/N`. Catches sampler /
spec mismatches before GPU time is spent.

### Master comparison

After eval finishes for each experiment:

```bash
PYTHONPATH=qwen-vl-finetune python -m qwenvl.experiments.aggregate \
    --output_root output/curriculum_v2_exp
```

Writes `_master_comparison.{csv,md}` sorted by `collapse_indicator`
(worst format-validity over checkpoints) so collapses surface at the
top of the table. Re-runnable safely after every new experiment.

### Optional: pre-materialize the global grounded pool

Saves ~5 min per stage startup on GF runs (avoids re-filtering the 10
per-category JSONs at every torchrun init):

```bash
python -m nuscenes_pipeline.postprocessing.build_global_grounded_pool
```

Orchestrator auto-detects the resulting
`sft_dataset/sft_train_qwen3vl_GLOBAL_GROUNDED.json` and prefers it
over the per-category list.

### Reference docs

- `curriculum_v2_ablation_experiments.md` — authoritative spec (knob
  semantics, composition rules, 16-run matrix, headline metrics,
  output layout, frozen-loss invariant, deferred difficulty-weighted
  mixture in Appendix A)
- `Mixed-train-implementation-issue.md` — postmortem for the
  per-category-uniform sampler fix (size-proportional bug, root cause,
  fix, validation across 12 sampler configurations)

---

## SFT Model Testing (Qualitative Inference)

Runs inference on a LoRA-fine-tuned Qwen3-VL-8B model using the **same prompt structure as training** (no-objlist variant: 6 surround-view images + ego status + task question). Saves results in the same format as `sft_train_no_objlist.json` so they can be loaded into the `qa_visualizer` dashboard for visual side-by-side comparison with ground truth answers.

### How to Run

```bash
# Via shell script (recommended)
bash nuscenes_pipeline/scripts/run_sft_model_tester.sh [MODE] [START] [END]

# Examples
bash nuscenes_pipeline/scripts/run_sft_model_tester.sh val 0 9       # Compare against val GT (indices 0..9)
bash nuscenes_pipeline/scripts/run_sft_model_tester.sh range 0 20    # Raw sample indices 0..20

# Via Python module
python -m nuscenes_pipeline.modules.sft_model_tester \
    --from_val --val_indices 0 1 2 3 4 5 \
    --base_model ckpts/qwen3_vl_8b_instruct \
    --lora_path output_nuscenes_lora_no_objlist_cleansing_qads/checkpoint-5500 \
    --output_dir /path/to/sft_dataset

# Custom question on arbitrary samples
python -m nuscenes_pipeline.modules.sft_model_tester \
    --sample_indices 0 10 50 100 \
    --question "Are there any pedestrians on the sidewalk?"

# Override paths via env vars
LORA_PATH=output_nuscenes_lora_no_objlist_cleansing_qads/checkpoint-11000 \
    bash nuscenes_pipeline/scripts/run_sft_model_tester.sh val 0 20

# Per-category accuracy mode (curriculum stage-end eval) — v1 schema:
# answer-only exact-match. Iterates sft_val_qwen3vl_<CAT>(_subset)?.json
# files in the eval dir, runs generation, parses the answer field, writes
# eval_report.json into --lora_path.
python -m nuscenes_pipeline.modules.sft_model_tester \
    --base_model ckpts/qwen3_vl_8b_instruct \
    --lora_path output/curriculum_v1/stage_03_SRO \
    --per_category_eval_dir sft_dataset/eval_subset_200 \
    --save_predictions

# Or via the curriculum wrapper
bash nuscenes_pipeline/scripts/run_stage_end_eval.sh \
    output/curriculum_v1/stage_03_SRO \
    sft_dataset/eval_subset_200 \
    ckpts/qwen3_vl_8b_instruct

# v2 per-category eval — adds --eval_v2 for the four-metric schema:
# answer_acc, view_acc, grounding_acc@<iou>, grounding_format_valid.
# Predicted bbox extraction operates on raw token IDs (handles special-token
# stripping correctly). See §13 for full v2 workflow.
python -m nuscenes_pipeline.modules.sft_model_tester \
    --base_model ckpts/qwen3_vl_8b_instruct \
    --lora_path output/curriculum_v2_<run>/stage_03_SRO \
    --per_category_eval_dir sft_dataset/eval_subset_200 \
    --eval_v2 --iou_threshold 0.8 \
    --save_predictions
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--base_model` | str | `ckpts/qwen3_vl_8b_instruct` | Path to base Qwen3-VL-8B model |
| `--lora_path` | str | `output_nuscenes_lora_no_objlist_cleansing_qads/checkpoint-5500` | Path to LoRA adapter checkpoint (merged at load time) |
| `--pkl_path` | str | `NUSCENES_PKL_PATH` | Path to nuScenes pickle file |
| `--sample_indices` | int+ | `None` | Specific sample indices to test (for ad-hoc inference) |
| `--start_idx` | int | `None` | Start index for range-based processing |
| `--end_idx` | int | `None` | End index for range-based processing |
| `--from_val` | flag | `False` | Load questions from the val set so each result can be compared against ground truth |
| `--val_path` | str | `sft_dataset/sft_val_no_objlist.json` | Path to val set JSON (used with `--from_val`) |
| `--val_indices` | int+ | `None` | Specific val set indices to test (used with `--from_val`) |
| `--question` | str | `None` | Custom question. When set, overrides val questions for the specified samples |
| `--resize_factor` | int | `2` | Image downscale factor (must match training) |
| `--max_new_tokens` | int | `2048` | Maximum tokens the model generates per sample |
| `--output_dir` | str | `sft_dataset` | Directory for output JSON |
| `--output_name` | str | `sft_test_<timestamp>.json` | Output filename |
| `--per_category_eval_dir` | str | `None` | Directory of `sft_val_qwen3vl_<CAT>(_subset)?.json` files. When set, runs the **per-category accuracy** code path: iterates each category, runs generation, parses the `{reasoning, grounding, answer}` JSON envelope, exact-matches on the `answer` field, and writes `eval_report.json` into `--lora_path`. The legacy qualitative-test code path (`--from_val`, `--sample_indices`, etc.) is skipped. |
| `--n_per_cat_limit` | int | `None` | Cap samples per category for smoke runs of the per-category eval. |
| `--save_predictions` | flag | `False` | Also dump per-sample predictions to `eval_predictions.json` alongside the report. |

### Input

| Input | Description |
|-------|-------------|
| Base model | Qwen3-VL-8B checkpoint (e.g., from `ckpts/qwen3_vl_8b_instruct`) |
| LoRA adapter | Output of the SFT training run (e.g., `output_nuscenes_lora_no_objlist_cleansing_qads/checkpoint-5500`). Merged into the base model at load time via `PeftModel.merge_and_unload()` |
| nuScenes pickle file | Same pickle used throughout the pipeline |
| Val JSON (optional) | `sft_val_no_objlist.json` when `--from_val` is set, providing GT answers for comparison |

### Output

One JSON file per run saved to `{output_dir}/{output_name}` in the **same format as `sft_train_no_objlist.json`**:

```json
[
  {
    "image": ["...CAM_FRONT_LEFT.jpg", "...CAM_FRONT.jpg", ...],
    "conversations": [
      {"from": "system", "value": "You are an autonomous driving analysis agent..."},
      {"from": "human",  "value": "=== Image 1: ... === <image> ... TASK: ..."},
      {"from": "gpt",    "value": "<model prediction>"},
      {"from": "gt",     "value": "<ground truth>"}   // only when --from_val
    ]
  },
  ...
]
```

The `"gt"` turn is only added in `--from_val` mode. Because the format matches the training data, the output file can be loaded directly into `nuscenes_pipeline.visualization.qa_visualizer` for visual inspection alongside the 6-view images and BEV overlay.

---

## Demo Tester (Per-Scene Frame-by-Frame Inference)

`nuscenes_pipeline/modules/demo_tester.py` runs the SFT-trained model on **every frame of one nuScenes scene** — for demo videos and qualitative temporal inspection. `--scene_token` accepts the full 32-char token or the 16-char prefix used in filenames. It reuses the exact prompt/inference path of `sft_model_tester` (system/user prompt builders, 6-view interleave with rear flip, token-ID grounding parse), so predictions are byte-comparable to the training-time distribution.

Two mutually exclusive modes:

- **Mode A** — `--demo_questions_path data/demo_questions.json`: runs the canonical 40-question demo bank (4 questions × 10 categories) on every frame; output groups the 40 answers per frame.
- **Mode B** — `--q "..."`: runs a single custom question on every frame.

```bash
# 8-GPU frame-stride fan-out (worker i owns frames with frame_pos % NPROC == i),
# then auto-merges the per-worker JSONs into a canonical predictions.json
LORA_PATH=output/curriculum_v2_f3_0618/F3__seed0/ckpt_100 \
    bash nuscenes_pipeline/scripts/run_demo_tester_8gpu.sh \
        ff6af17f52c34e9c data/demo_questions.json                              # Mode A

LORA_PATH=... bash nuscenes_pipeline/scripts/run_demo_tester_8gpu.sh \
        b526c20f7eed49f0 'Is the ego-vehicle safe to change lanes right?'      # Mode B
```

Single-GPU runs invoke the module directly (`python -m nuscenes_pipeline.modules.demo_tester --scene_token ... --lora_path ...` plus one of the two mode flags). The sharding contract is `--sample_stride` / `--sample_offset`; `merge_demo_predictions.py --worker_dir <dir>` merges the per-worker files — frame_pos values are disjoint across workers so concat + sort is lossless, and header fields (model paths, question set, etc.) are sanity-checked for cross-worker consistency. Output defaults to `demo_test_results/<scene_token[:16]>/`.

### demo_scene_video.py

Assembles a `predictions.json` into **one MP4 per category** (10 for the canonical bank). Each video cycles through the category's questions sequentially, playing each question across every frame of the scene: 6-view panoramic tiles (rear views flipped, same as training) plus a header block with the question, predicted answer, and the first lines of reasoning. Predicted grounding is drawn as pink boxes — coordinates are in Qwen-VL's [0, 1000] normalized space and map straight onto the tiles, no unflip needed.

```bash
python -m nuscenes_pipeline.visualization.demo_scene_video \
    --predictions demo_test_results/ff6af17f52c34e9c/predictions.json \
    --output_dir  demo_test_results/ff6af17f52c34e9c/videos \
    --framerate 2                    # --only_category <CAT> renders a single video
```

---

## Evaluation & Benchmarks

> **Note**: this section documents the original **qualitative inspection workflow** (predict on val
> samples + load into `qa_visualizer` for side-by-side review). For **quantitative per-category
> evaluation** (answer accuracy + view accuracy + grounding IoU + grounding format-validity), see the
> v2 eval flow in [§13](#curriculum-v2-training-and-evaluation) and the ablation-framework eval in
> [§14](#ablation-experiments-framework). The v2 / ablation paths use `sft_model_tester --eval_v2` and
> produce `eval_report.json` per checkpoint plus a `summary.json` per run; this section's workflow is
> still useful for one-off ad-hoc inspection.

Evaluation of the SFT-tuned Qwen3-VL model is performed **on the in-house nuScenes VLM dataset built by this pipeline** (not on general-purpose VLM benchmarks). The goal is to verify that the fine-tuned model answers autonomous-driving questions grounded in the 6-view surround images, ego state, and scene context produced by Stages 1–3.

### Evaluation Workflow

```
        SFT-tuned Qwen3-VL + LoRA checkpoint
                        |
                        v
[Step 1] sft_model_tester.py        Run inference on val set using the
                                     SAME prompt format as training
                        |
                        v
sft_test_<timestamp>.json            Predictions + GT answers in the
                                     sft_train_no_objlist.json schema
                        |
                        v
[Step 2] qa_visualizer.py            Flask dashboard: side-by-side
                                     6-view images + BEV + prediction vs GT
```

Step 1 produces model predictions aligned with the val-set GT answers; Step 2 loads that JSON into the QA visualizer so predictions can be inspected visually against the 6-view panorama and BEV overlay.

### Step 1: Run Inference on the Val Set

Use the SFT model tester described in [SFT Model Testing (Qualitative Inference)](#sft-model-testing-qualitative-inference) above. The `--from_val` flag loads questions from `sft_val_no_objlist.json` and appends a `"gt"` turn to each output entry so predictions can be compared against ground truth.

```bash
# Compare predictions vs val GT for indices 0..49
bash nuscenes_pipeline/scripts/run_sft_model_tester.sh val 0 49

# Or directly
python -m nuscenes_pipeline.modules.sft_model_tester \
    --from_val --val_indices 0 1 2 3 4 5 \
    --base_model ckpts/qwen3_vl_8b_instruct \
    --lora_path output_nuscenes_lora_no_objlist_cleansing_qads/checkpoint-5500 \
    --val_path sft_dataset/sft_val_no_objlist.json \
    --output_dir sft_dataset \
    --output_name sft_test_ckpt5500.json
```

The output JSON matches the training schema:
```json
[
  {
    "image": ["...CAM_FRONT_LEFT.jpg", "...CAM_FRONT.jpg", ...],
    "conversations": [
      {"from": "system", "value": "You are an autonomous driving analysis agent..."},
      {"from": "human",  "value": "... TASK: ..."},
      {"from": "gpt",    "value": "<model prediction>"},
      {"from": "gt",     "value": "<ground truth answer from val set>"}
    ]
  }
]
```

### Step 2: Qualitative Inspection via QA Visualizer

Load the prediction JSON into the Flask web dashboard to compare predictions vs GT alongside the surround-view images and BEV overlay:

```bash
python -m nuscenes_pipeline.visualization.qa_visualizer \
    --data_dir sft_dataset \
    --pkl_path ./data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl \
    --port 6060
```

The dashboard renders the 6-view panorama with question/answer bounding boxes (green = question-referenced objects, pink = answer-referenced objects), the BEV with ground-truth velocity vectors, and both the model prediction and the GT answer side-by-side. See the [qa_visualizer](#qa_visualizerpy) section for keyboard shortcuts and arguments.

### What to Look For

| Signal | Where to check |
|--------|----------------|
| Hallucinated objects (e.g., pedestrians that aren't in any camera view) | Compare GPT answer vs green/pink bboxes in the panorama |
| Incorrect motion-state claims | Compare "moving/stationary" language vs velocity arrows in the BEV |
| Wrong spatial zone ("front-right" vs "rear-left") | Check which camera view the referenced object appears in |
| MCQ option letter vs rationale mismatch | Inspect the bullet reasoning in the answer against the chosen letter |
| Missed traffic signals | Cross-reference with the Stage 1B traffic analysis for the same sample |

---

## Eval Sample Visualizer (Browser-Based)

Flask app for qualitative inspection of the v2 evaluation outputs.
Mirrors `qa_visualizer.py`'s 6-view image layout but specialized for
GT-vs-PRED comparison: overlays GT boxes in green, matched predicted
boxes in pink, and spurious predictions in dashed pink, alongside a
side-by-side reasoning / answer / per-box-match panel.

### Launch

```bash
# Single experiment
python -m nuscenes_pipeline.visualization.eval_sample_visualizer \
    --eval_predictions output/curriculum_v2_exp/B0__seed0/stage_00_OBS/eval_predictions.json \
    --val_subset_dir   sft_dataset/eval_subset_200 \
    --port 6061

# All stages of a curriculum (stage dropdown lets you switch)
python -m nuscenes_pipeline.visualization.eval_sample_visualizer \
    --curriculum_root output/curriculum_v2_exp/B0__seed0 \
    --val_subset_dir  sft_dataset/eval_subset_200 \
    --port 6061
```

Open `http://<server-ip>:6061` in a browser. Sample selection mirrors
the eval CLI's `--sanity_print_n N` (first 2 grounded samples per
category by default → 20 samples per stage). Use `--samples_per_cat N`
to surface more.

---

## Resize Factor Guide

| Module | Recommended | Reason |
|--------|-------------|--------|
| Risk Assessment (1A) | `2` | Sufficient resolution for scene-level risk analysis |
| Traffic Analysis (1B) | `1` | Full resolution needed for detecting small traffic signals |
| Question Selector (2) | `2` | Template validation does not require fine-grained detail |
| Answer Generator (3) | `2` | Answer quality is driven by reasoning, not pixel-level detail |

---

## Supporting Documentation

Project-root markdown documents covering the curriculum-v2 and
autolabel-variant work in depth (cross-linked from §6, §13 and §14):

| Document | Purpose |
|---|---|
| `v3_loss_completeness_task.md` | Spec for the grounding-completeness work (T1–T3): per-view completeness measurement columns in the v2 eval, per-view presence supervision, view-stratified grounding-floor sampling. Motivating failure case and scope guardrails (referring-grounding only, no detection/set loss). |
| `thinking_answer_generator_4_rl_impl.md` | Spec for the RL (GRPO) Stage 3 fork (`answer_generator_rl.py`): tier-banded think arrays, `contrast_status` self-report, validation gate + retry, provenance `qa_id`s, run report. |
| `loss_system.md` | As-built reference for the 6-term composite loss currently in `composite_loss.py`. Includes per-term spec with code snippets, mask construction, aggregation flow, per-component logging table, numerical-correctness checklist. |
| `loss_improvement_design_v3.md` | Design rationale for the composite loss. Frozen-loss invariant, per-term motivation, R-VLM / KLAL references, phased rollout. |
| `curriculum_v2_loss.md` | Narrative explanation of the loss with stepped code excerpts (superseded by `loss_system.md` for as-built reference; kept for explanatory context). |
| `curriculum_v2_ablation_experiments.md` | Authoritative spec for the 16-run ablation matrix. Knob semantics, composition rules, eval protocol, headline metrics, output layout, deferred Appendix A (difficulty-weighted mixture). |
| `Mixed-train-implementation-issue.md` | Postmortem for the per-category-uniform sampler fix. Spec/implementation mismatch root cause, 12-config validation results, impact on pre-fix mixed-mode adapters. |
| `curriculum_pipeline_analysis.md` | Architecture analysis of the v1 sequential curriculum. Useful background context for understanding what v2 changes. |
