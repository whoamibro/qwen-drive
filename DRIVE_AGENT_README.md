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

**Filename pattern:** `{idx:04d}_{scene_token}_{sample_token}_traffic_v8.json`

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
bash nuscenes_pipeline/scripts/run_question_selector.sh [START_IDX] [END_IDX] [CATEGORY] [NUM_WORKERS]

# Via Python module
python -m nuscenes_pipeline.modules.question_selector \
    --start_idx 0 --end_idx 6018 --category all --num_workers 8 \
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

**Question bank structure:**
```json
{
  "categories": [
    {
      "category": "Observation",
      "description": "...",
      "templates": [
        {
          "template": "Are there any <object> <preposition> the <place>?",
          "placeholders": {
            "object": ["pedestrians", "vehicles", ...],
            "place": ["lane", "roadway", ...],
            "preposition": ["in", "on"]
          },
          "answer_type": "y_or_n",
          "question_type": "observation"
        }
      ]
    }
  ]
}
```

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
bash nuscenes_pipeline/scripts/run_answer_generator.sh [START_IDX] [END_IDX] [CATEGORY] [NUM_WORKERS]

# Via Python module
python -m nuscenes_pipeline.modules.answer_generator \
    --start_idx 0 --end_idx 6018 --category all --num_workers 8 \
    --resize_factor 2 --filter_distance 50 --rear_filter 20 \
    --max_new_tokens 16384 --max_pairs 3 \
    --stage1_dir qa_outputs_stage1 \
    --output_dir qa_outputs_stage2
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_name` | str | `Qwen/Qwen3-VL-235B-A22B-Instruct` | Model name served by vLLM |
| `--api_base` | str | `http://localhost:8000/v1` | vLLM API endpoint URL |
| `--api_key` | str | `EMPTY` | API key for the vLLM server |
| `--pkl_path` | str | env `NUSCENES_PKL_PATH` | Path to nuScenes pickle file |
| `--question_bank` | str | env `QUESTION_BANK_PATH` | Path to question bank JSON (needed for original placeholder definitions and expected answer types) |
| `--stage1_dir` | str | `qa_outputs_stage1` | Stage 2 (question selector) output directory. Contains the applicable question files per sample |
| `--output_dir` | str | `qa_outputs_stage2` | Output directory for generated QA pairs |
| `--start_idx` | int | `0` | Starting sample index (inclusive) |
| `--end_idx` | int | `6018` | Ending sample index (inclusive) |
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

## Full Pipeline Example

```bash
cd /path/to/qwen-drive

# Set environment variables
export NUSCENES_PKL_PATH="/data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl"
export QUESTION_BANK_PATH="/data/qa_dataset/question_bank.json"

# Stage 1A & 1B can run in parallel (on separate GPU servers or sequentially)
bash nuscenes_pipeline/scripts/run_risk_assessment.sh 0 6018 8
bash nuscenes_pipeline/scripts/run_traffic_analysis.sh 0 6018 8

# Stage 2: requires Stage 1A + 1B results
bash nuscenes_pipeline/scripts/run_question_selector.sh 0 6018 all 8

# Stage 3: requires Stage 2 results
bash nuscenes_pipeline/scripts/run_answer_generator.sh 0 6018 all 8
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
      sft_prompt_builder.py           SFT training prompt construction
    visualization/
      __init__.py
      bev_generator.py                BEV visualization with ego, objects, velocities
      make_video.py                   Combine panoramic + BEV images into video
      pretty_formatting.py            Clean and format JSON result fields
      qa_visualizer.py                Flask web dashboard for QA dataset verification
    scripts/
      run_risk_assessment.sh          Shell script for Stage 1A
      run_traffic_analysis.sh         Shell script for Stage 1B
      run_question_selector.sh        Shell script for Stage 2
      run_answer_generator.sh         Shell script for Stage 3
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

## Resize Factor Guide

| Module | Recommended | Reason |
|--------|-------------|--------|
| Risk Assessment (1A) | `2` | Sufficient resolution for scene-level risk analysis |
| Traffic Analysis (1B) | `1` | Full resolution needed for detecting small traffic signals |
| Question Selector (2) | `2` | Template validation does not require fine-grained detail |
| Answer Generator (3) | `2` | Answer quality is driven by reasoning, not pixel-level detail |
