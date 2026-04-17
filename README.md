# Drive Agent Pipeline

A multi-stage VQA pipeline for autonomous driving scene understanding on the nuScenes dataset, powered by Qwen3-VL served via vLLM.

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
      sft_prompt_builder.py           SFT training prompt construction
      sft_model_tester.py             Qualitative inference test for SFT-trained LoRA model
    visualization/
      __init__.py
      bev_generator.py                BEV visualization with ego, objects, velocities
      make_video.py                   Combine panoramic + BEV images into video
      pretty_formatting.py            Clean and format JSON result fields
      qa_visualizer.py                Flask web dashboard for QA dataset verification
    postprocessing/
      __init__.py
      transform_obj_to_bbox.py        Replace OBJ IDs with 2D bbox descriptions
      cleanse_obj_references.py       Remove leaked OBJ refs and meta-references
      fix_motion_states.py            Correct motion claims against GT velocity
      prepare_sft_dataset.py          Build SFT train/val JSON (no object list)
      count_qa_stats.py               QA pair statistics utility
    scripts/
      run_risk_assessment.sh          Shell script for Stage 1A
      run_traffic_analysis.sh         Shell script for Stage 1B
      run_question_selector.sh        Shell script for Stage 2
      run_answer_generator.sh         Shell script for Stage 3
      run_sft_model_tester.sh         Shell script for SFT model inference test
      show_prompts.py                 Prompt preview (prints system+user prompts without inference)
```

```
  evaluation/
    RealWorldQA/
      run_realworldqa.py              Inference + evaluation
      infer_instruct.sh               Inference shell script
      eval_instruct.sh                Evaluation shell script
    mmmu/
      run_mmmu.py                     MMMU benchmark runner
      infer_instruct.sh / eval_instruct.sh
    MathVision/
      run_mathv.py                    MathVision benchmark runner
      infer_instruct.sh / eval_instruct.sh
    ODinW-13/
      run_odinw.py                    Object detection benchmark (COCO AP)
      infer_instruct.sh / eval_instruct.sh
    VideoMME/
      run_videomme.py                 Video understanding benchmark
      infer_instruct.sh / eval_instruct.sh
  web_demo_mm.py                      Interactive Gradio web demo
  train_nuscenes_qwen3vl.py           SFT training script (LoRA / full)
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

---

## Post-Processing Pipeline

After the 4 pipeline stages complete, the QA results go through post-processing to produce a clean SFT training dataset. The post-processing removes leaked 3D object list artifacts so the trained VLM relies on visual perception.

### Pipeline Data Flow

```
answer_generator output (qa_results/Observation/)
        |
        v
[Step 1] transform_obj_to_bbox.py    OBJ IDs → 2D bbox descriptions
        |
        v
sft_dataset/Observation/              (bbox-transformed QA results)
        |
        v
[Step 2] prepare_sft_dataset.py       Build SFT conversation format
        |
        v
sft_dataset/sft_train_no_objlist.json + sft_val_no_objlist.json
        |
        v
[Step 3] cleanse_obj_references.py    Remove leaked OBJ refs (Rounds 1-3)
        |
        v
[Step 4] fix_motion_states.py         Correct motion state claims
        |
        v
sft_dataset/sft_{train,val}_no_objlist.json  (FINAL — ready for SFT training)
```

### Step 1: Transform OBJ to Bbox

Replaces OBJ ID references in QA answers with visual 2D bounding box descriptions.

```bash
python -m nuscenes_pipeline.postprocessing.transform_obj_to_bbox \
    --input_dir qa_results \
    --output_dir sft_dataset \
    --data_root ./data/nuscenes \
    --resize_factor 2
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input_dir` | str | `qa_results/Observation` | Directory with answer generator output (contains OBJ IDs) |
| `--output_dir` | str | `sft_dataset/Observation` | Output directory for bbox-transformed results |
| `--pkl_path` | str | env `NUSCENES_PKL_PATH` | Path to nuScenes pickle file |
| `--data_root` | str | env `NUSCENES_DATA_ROOT` | nuScenes data directory (where `samples/CAM_*/` images live) |
| `--resize_factor` | int | `2` | Image resize factor (must match pipeline resize_factor) |

**Transformation examples:**
- `"OBJ 30"` → `"pedestrian (Image 2 (Front) bbox[788,234,799,314])"`
- `"OBJ 5 (pedestrian, 46.6m ahead)"` → `"pedestrian (46.6m ahead, Image 1 bbox[691,219,712,247])"`

### Step 2: Prepare SFT Dataset

Converts bbox-transformed QA results into Qwen3-VL SFT training format (3-turn conversations: system, user, assistant) **without 3D object list** in the prompt.

```bash
python -m nuscenes_pipeline.postprocessing.prepare_sft_dataset \
    --qa_dir sft_dataset \
    --output_dir sft_dataset \
    --data_root ./data/nuscenes \
    --val_size 40000
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--qa_dir` | str | `sft_dataset/Observation` | Directory with bbox-transformed QA results |
| `--pkl_path` | str | env `NUSCENES_PKL_PATH` | Path to nuScenes pickle file |
| `--data_root` | str | env `NUSCENES_DATA_ROOT` | nuScenes data directory (where `samples/CAM_*/` images live) |
| `--output_dir` | str | `sft_dataset` | Output directory for SFT JSON files |
| `--resize_factor` | int | `2` | Image resize factor |
| `--val_size` | int | `40000` | Fixed number of validation samples (rest goes to train) |
| `--seed` | int | `42` | Random seed for train/val split |

**Output:** `sft_train_no_objlist.json` and `sft_val_no_objlist.json`

### Step 3: Cleanse OBJ References (Rounds 1-3)

Removes any remaining leaked OBJ references that survived the bbox transformation:
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

```bash
python -m nuscenes_pipeline.postprocessing.fix_motion_states \
    --data_dir sft_dataset \
    --data_root ./data/nuscenes
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data_dir` | str | `sft_dataset` | Directory containing `sft_train_no_objlist.json` and `sft_val_no_objlist.json` |
| `--pkl_path` | str | env `NUSCENES_PKL_PATH` | Path to nuScenes pickle file |
| `--data_root` | str | env `NUSCENES_DATA_ROOT` | nuScenes data directory (where `samples/CAM_*/` images live) |

**Phrase corrections** (22 patterns):
- Motion → Stationary (velocity < 0.1 m/s): `"actively riding"` → `"stationary"`, `"in motion"` → `"stationary"`, etc.
- Stationary → Motion (velocity ≥ 0.1 m/s): `"is parked"` → `"is moving"`, `"pulled over"` → `"in motion"`, etc.

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

# === Post-Processing ===
# Step 1: OBJ → bbox transformation
python -m nuscenes_pipeline.postprocessing.transform_obj_to_bbox \
    --input_dir qa_results --output_dir sft_dataset --data_root ./data/nuscenes

# Step 2: Build SFT dataset
python -m nuscenes_pipeline.postprocessing.prepare_sft_dataset \
    --qa_dir sft_dataset --output_dir sft_dataset --data_root ./data/nuscenes

# Step 3: Cleanse leaked OBJ references
python -m nuscenes_pipeline.postprocessing.cleanse_obj_references \
    --input sft_dataset/sft_train_no_objlist.json
python -m nuscenes_pipeline.postprocessing.cleanse_obj_references \
    --input sft_dataset/sft_val_no_objlist.json

# Step 4: Fix motion state contradictions
python -m nuscenes_pipeline.postprocessing.fix_motion_states \
    --data_dir sft_dataset --data_root ./data/nuscenes
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
    --train_data_path sft_dataset/sft_train_no_objlist.json \
    --val_data_path sft_dataset/sft_val_no_objlist.json \
    --resize_factor 2 \
    --lora_r 64 --lora_alpha 128 --lora_dropout 0.05 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --bf16 --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-4 \
    --output_dir output_nuscenes_lora_no_objlist \
    --gradient_checkpointing True \
    --report_to tensorboard
```

### Full Fine-Tuning

```bash
NPROC_PER_NODE=8 bash qwen-vl-finetune/scripts/run_nuscenes_full.sh
```

### Key Training Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mode` | str | **required** | `lora` or `full` |
| `--model_name_or_path` | str | — | Path to Qwen3-VL model or HuggingFace ID |
| `--train_data_path` | str | `sft_dataset/sft_train_no_objlist.json` | Training data from post-processing pipeline |
| `--val_data_path` | str | `sft_dataset/sft_val_no_objlist.json` | Validation data |
| `--resize_factor` | int | `2` | Image downscale factor (must match pipeline) |
| `--lora_r` | int | `64` | LoRA rank |
| `--lora_alpha` | int | `128` | LoRA alpha |
| `--lora_dropout` | float | `0.05` | LoRA dropout |
| `--deepspeed` | str | — | DeepSpeed config path (`zero2.json` for LoRA, `zero3.json` for full) |
| `--bf16` | flag | — | Use bfloat16 training |
| `--data_flatten` | bool | `False` | Pack sequences for efficiency (full fine-tune only) |
| `--max_pixels` | int | `50176` | Max image pixels |
| `--min_pixels` | int | `784` | Min image pixels |

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

## Evaluation & Benchmarks

After SFT training, evaluate the fine-tuned model on multiple benchmarks. Each benchmark has an `infer` step (run vLLM inference) and an `eval` step (compute metrics).

### Available Benchmarks

| Benchmark | Type | Metric | Description |
|-----------|------|--------|-------------|
| **RealWorldQA** | Image QA | Accuracy | Real-world visual question answering |
| **MMMU** | Image QA | Accuracy (A-D MCQ) | Multi-discipline multimodal understanding |
| **MathVision** | Image QA | Accuracy | Mathematical visual reasoning |
| **ODinW-13** | Object Detection | COCO AP | Object detection in the wild (13 domains) |
| **VideoMME** | Video QA | Accuracy | Video understanding (short/long duration) |

### How to Run

Each benchmark lives in `evaluation/<benchmark>/` with standardized scripts:

```bash
cd evaluation/<benchmark>

# Step 1: Inference — runs vLLM on the benchmark dataset
bash infer_instruct.sh

# Step 2: Evaluation — computes metrics
bash eval_instruct.sh
```

**Before running**, edit the shell scripts to set your paths:
- `--model-path`: path to your SFT fine-tuned model checkpoint
- `--data-dir`: path to the benchmark dataset

### Example: RealWorldQA

```bash
cd evaluation/RealWorldQA

# Inference
python run_realworldqa.py infer \
    --model-path /path/to/your/sft-checkpoint \
    --dataset RealWorldQA \
    --data-dir /path/to/realworldqa_data \
    --output-file results/RealWorldQA_results.jsonl \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max-new-tokens 32768 \
    --temperature 0.7 --top-p 0.8 --top-k 20

# Evaluation
python run_realworldqa.py eval \
    --data-dir /path/to/realworldqa_data \
    --input-file results/RealWorldQA_results.jsonl \
    --output-file results/RealWorldQA_evaluation.csv \
    --dataset RealWorldQA \
    --eval-model gpt-3.5-turbo-0125 --api-type dash --nproc 4
```

### Example: MMMU

```bash
cd evaluation/mmmu

python run_mmmu.py infer \
    --model-path /path/to/your/sft-checkpoint \
    --data-dir /path/to/mmmu_data \
    --dataset MMMU_DEV_VAL \
    --output-file results/mmmu_dev_val_predictions.jsonl \
    --max-new-tokens 32768 \
    --temperature 0.7 --top-p 0.8 --top-k 20

python run_mmmu.py eval \
    --data-dir /path/to/mmmu_data \
    --input-file results/mmmu_dev_val_predictions.jsonl \
    --output-file results/mmmu_dev_val_eval_results.csv \
    --dataset MMMU_DEV_VAL \
    --eval-model gpt-3.5-turbo-0125 --api-type dash --nproc 16
```

### Common Inference Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model-path` | str | — | Path to fine-tuned model checkpoint or HuggingFace model ID |
| `--data-dir` | str | — | Path to benchmark dataset |
| `--output-file` | str | — | Output JSONL file for predictions |
| `--tensor-parallel-size` | int | `1` | Number of GPUs for tensor parallelism |
| `--gpu-memory-utilization` | float | `0.9` | GPU memory fraction for vLLM |
| `--max-model-len` | int | `128000` | Maximum model context length |
| `--max-new-tokens` | int | `32768` | Maximum tokens to generate |
| `--temperature` | float | `0.7` | Sampling temperature |
| `--top-p` | float | `0.8` | Top-p (nucleus) sampling |
| `--top-k` | int | `20` | Top-k sampling |

### Web Demo

Interactive Gradio web interface for testing the fine-tuned model with custom images and questions:

```bash
# Using HuggingFace backend
python web_demo_mm.py --backend hf \
    --checkpoint-path /path/to/your/sft-checkpoint \
    --server-port 7860

# Using vLLM backend (faster)
python web_demo_mm.py --backend vllm \
    --checkpoint-path /path/to/your/sft-checkpoint \
    --server-port 7860 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.7
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--checkpoint-path`, `-c` | str | `Qwen/Qwen3-VL-235B-A22B-Instruct` | Model checkpoint path |
| `--backend` | str | `hf` | Backend: `hf` (HuggingFace) or `vllm` |
| `--server-port` | int | `7860` | Web server port |
| `--server-name` | str | `127.0.0.1` | Web server host |
| `--flash-attn2` | flag | `False` | Enable Flash Attention 2 (HF backend) |
| `--tensor-parallel-size` | int | `1` | GPU count for tensor parallelism (vLLM backend) |
| `--gpu-memory-utilization` | float | `0.7` | GPU memory fraction (vLLM backend) |

---

## Resize Factor Guide

| Module | Recommended | Reason |
|--------|-------------|--------|
| Risk Assessment (1A) | `2` | Sufficient resolution for scene-level risk analysis |
| Traffic Analysis (1B) | `1` | Full resolution needed for detecting small traffic signals |
| Question Selector (2) | `2` | Template validation does not require fine-grained detail |
| Answer Generator (3) | `2` | Answer quality is driven by reasoning, not pixel-level detail |
