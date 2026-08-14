# Report — Custom Dataset → nuScenes-Format Demo Adaptation Plan

- **Date**: 2026-08-13
- **Author**: Claude Code session (with Yongjin Jeon)
- **Context**: Adapting the qwen-drive demo pipeline (`run_demo_batch.sh` → `run_demo_tester_8gpu.sh` → `demo_tester.py`, vLLM or HF backend) from nuScenes to a custom 3-clip dataset at `/yjj_rnd_home/data/vlm_dataset/demo_dataset/`, for demoing the LoRA-SFT-trained Qwen3-VL-8B model.

---

## 1. Original Request (verbatim)

> Good. Now I want to change the input dataset(for now we are using nuScenes) into custom dataset(stored in the '/yjj_rnd_home/data/vlm_dataset/demo_dataset/'). In the root of the custom dataset directory "/yjj_rnd_home/data/vlm_dataset/demo_dataset/' there are 3 sub directories exist and each one represents the different clip. Each clip has same leaf directory structure. images(contain 7view images), maps/calib_tuned(contains the intrinsic and extrinsic parameters of each view's camera), GER_MACHET18_20260414_192304_ccan_3_0.csv(contains the can data extracted while ego vehicle's driving), maps/map(contains the traectory and pose of the ego vehicle at each keyframe). I think these infos are enough to run the demo using our trained model. Before start implementing the adaptation process for converting the data of custom dataset into the nuScenes format, we need to cut and manipulate the image due to the different camera spec. Our trained model only requires the 6view image so, we are goint to use only "h120_front_cam", "h100_side_frontleft_cam", "h100_side_frontright_cam", "h100_side_rearcenter_cam", "h100_side_rearleft_cam", "h100_side_rearright_cam". Unlikely to the nuScenes dataset, the intrinsics between front cam and the others are different so we have to cut out the patch(to have same ratio to the other view images) from the front cam image and resize them until the scale of the objects in the images are almost close to same. If we can do this by computing the numbers based on the mathematics then suggest me the methodology to do that, otherwise we have to build a webserver based visualization tool that can manually do such process and findout the proper numbers(crop size and pixel coords). After that, we have to extract the ego vehicle's pose and velocity at the image keyframe. Then we have to run the driving command labeling server and label the driving command of the demo samples. Could you build a solid and deeply systematic plan to execute all of these tasks?

---

## 2. Key Findings from the Data Audit

Two headline conclusions:

1. **The front-cam crop can be computed purely from calibration math** — no manual-tuning webserver is needed. All cameras are already rectified to pinhole models with zero distortion coefficients (verified: `2·atan(cx/fx)` reproduces each camera's nominal HFOV).
2. **The repo already has a browser-based command labeler** (`nuscenes_pipeline/visualization/driving_command_labeler.py`, Flask, works off any nuScenes-format pkl). Command labeling only needs a CAN-based auto-proposal step and a label→pkl merge script.

### Dataset facts

- **Clips**: 3 clips (`demo_1_20260414_{105436,112636,192304}`), each 599 frames at 10 Hz (~60 s), 7 synchronized views sharing identical µs timestamps in filenames.
- **Cameras** (from `maps/calib_tuned/*.yaml`):

| Channel | Resolution | fx | fy | cx | cy | actual HFOV / VFOV |
|---|---|---|---|---|---|---|
| `2160p_h120_front` | 3840×2160 | 1127.88 | 1634.36 | 1926.05 | 1081.85 | 119.5° / 66.9° |
| `1536p_h100_side_*` (×5) | 1920×1536 | ~770–778 | ~901–908 | ~954–962 | ~768–770 | ~101.5° / ~80.6° |
| `2160p_h30_front_zoom` | 3840×2160 | — | — | — | — | excluded |

- `distortion_model: kbm` but **all k1–k4 ≈ 0** → images behave as **rectified pinhole**. This is what makes a purely mathematical crop/scale solution valid.
- **fx ≠ fy on every camera, and the anisotropy differs** (front fx/fy = 0.69, sides ≈ 0.86). A single uniform crop+resize cannot equalize object scale on both axes — x and y must be scaled independently.
- The side yamls' `hfov: 120` field is wrong (copy-paste artifact); intrinsics say ~100°, matching the filenames. Trust intrinsics only.
- **Ego motion**: `maps/map/traj_lcs.txt` is a TUM-style dense trajectory (`t x y z qx qy qz qw`) of the **LiDAR** in a clip-local SLAM frame; timestamps match image timestamps 1:1 (match by timestamp, not row index — small end-of-clip off-by-1/2 counts). `lcs_extrinsic` (cam→lidar) + `vcs_extrinsic` (cam→vehicle/VCS) chain to the ego (VCS) pose.
- **CAN CSV** (~240 Hz): `Cluster_Display_Speed`, wheel speeds, `Yaw_Rate`, `Steering_Angle`, `Turn_Indicator_Left/Right`, `Gear_Position` — for velocity cross-validation and command auto-proposals.

### Camera → nuScenes mapping (rear flip then applies automatically by name in `prepare_messages`)

| Custom channel | nuScenes name |
|---|---|
| `2160p_h120_front` | CAM_FRONT |
| `1536p_h100_side_frontleft` / `frontright` | CAM_FRONT_LEFT / CAM_FRONT_RIGHT |
| `1536p_h100_side_rearleft` / `rearright` | CAM_BACK_LEFT / CAM_BACK_RIGHT |
| `1536p_h100_side_rearcenter` | CAM_BACK |

---

## 3. Phase 1 — Image Unification via a Common Virtual Camera (pure math)

### Methodology

Because every image is pinhole-rectified, "equal object scale" has an exact definition: for an object at distance Z, its pixel extent is `f·size/Z` per axis. So all six views must end with the **same effective focal length per axis**. Define one **virtual target camera** — output size `W_t×H_t`, isotropic focal `f_t`, principal point at center — and warp each source into it. Since source and target share the optical axis, the warp degenerates to an axis-aligned **subpixel crop centered on the principal point + per-axis resize** (one `PIL.Image.resize(..., box=(x1,y1,x2,y2))` call — no remap needed):

```
crop_w = W_t · fx_src / f_t          crop_h = H_t · fy_src / f_t
x1 = cx − crop_w/2,  x2 = cx + crop_w/2      (same for y around cy)
```

This simultaneously fixes (a) front-vs-side scale mismatch, (b) the fx≠fy anisotropy (output has square pixels), and (c) per-camera intrinsic differences among the five side cams (each gets its own crop from its own yaml).

### Target choice: `W_t×H_t = 1600×900`, `f_t = 684`

Feasibility (crop must fit inside the source): `f_t ≥ max over cams of (W_t·fx/W_src, H_t·fy/H_src)`. Binding constraint = front cam's vertical axis: `900·1634.36/2160 = 681.0`. Take **f_t = 684** for margin. Concrete numbers:

| View | crop (w×h) | crop x-range | crop y-range | resize to |
|---|---|---|---|---|
| front | 2638×2151 | [607, 3245] | [7, 2157] | 1600×900 |
| side (each from its own yaml; FL example) | ~1820×1194 | [44, 1864] | [173, 1367] | 1600×900 |

Result: all six views are 1600×900 (the exact nuScenes resolution the model trained on), pinhole, f = 684 both axes → **HFOV ≈ 98.9°, VFOV ≈ 66.7° per view, identical angular resolution everywhere**. The front is cropped from 120° to ~99°; the sides keep nearly their full 100° coverage with no upscaling.

**Conscious trade-off**: this matches views *to each other* at ~54% of nuScenes front-cam absolute scale (nuScenes front f≈1266 at 70°; f=684 is close to nuScenes' 110° CAM_BACK at f≈809). Matching nuScenes front scale would force cropping the sides to ~64° HFOV — unacceptable coverage loss. Keep `f_t` a config knob; if demo answers look scale-confused, A/B a higher `f_t`.

### Verification instead of manual tuning

Build a **verification visualizer** (static HTML composite per keyframe): 6-view grid at output spec with rear views flipped, plus two objective checks:
1. An object visible near the front/front-left view boundary should have equal pixel height in both views.
2. Project boxes from `maps/lidar_3d_od_results` through the virtual intrinsics `K_t = [[684,0,800],[0,684,450],[0,0,1]]` and confirm they land on the objects in all views (also validates the pinhole assumption end-to-end).

**Deliverables**: `custom_dataset/virtual_camera.py` (calib parsing + crop math), `custom_dataset/convert_images.py` (batch, multiprocessing, writes `<out_root>/samples/CAM_X/<clip>_<timestamp>.jpg`, idempotent via marker), `custom_dataset/verify_warp.py`.

---

## 4. Phase 2 — Ego Pose & Velocity per Keyframe

1. **Keyframe selection**: subsample 10 Hz → every 5th frame (2 Hz, matching nuScenes cadence and the demo video's `--framerate 2`) → ~120 samples/clip, ~360 total.
2. **Pose chain**: `T_map←vcs(t) = T_map←lidar(t) · T_lidar←cam · inv(T_vcs←cam)` using `lcs_extrinsic` and `vcs_extrinsic` from any one camera yaml (compute once; verify consistency across all six). The clip-local SLAM frame serves as "global" — the pipeline only uses relative displacement and yaw, so no geo-referencing is needed. Convert TUM `qx qy qz qw` → nuScenes `[w,x,y,z]`.
3. **Convention sanity checks** (must-do, cheap): VCS should be FLU — front cam at `t≈(1.97, −0.27, 1.42)` and quat `(0.5,−0.5,0.5,−0.5)` (canonical optical→FLU) look right; confirm front-right cam has ty<0, front-left ty>0. The yaml comments and `frame_from/frame_to` labels contradict each other, so validate transform direction empirically: reconstructed forward speed must be positive and match CAN.
4. **Velocity**: don't store it — `compute_ego_velocity/acceleration` derive it from adjacent `ego2global_translation` + timestamps at prompt-build time, exactly as for nuScenes. Our job is only accurate per-keyframe pose + µs timestamps.
5. **QA**: plot `|Δpos/Δt|` vs CAN `Cluster_Display_Speed` (resampled to keyframes) per clip; flag >10% divergence. Plot BEV trajectory against `maps/trajectory_grpah_xy.png`.

---

## 5. Phase 3 — nuScenes-Format pkl Assembly

Build `custom_dataset/build_infos_pkl.py` → `<out_root>/custom_demo_infos.pkl` with `{'metadata': {'version': 'custom-demo-v1'}, 'infos': [...]}`. Required fields (exactly what `NuScenesDataLoader.get_sample` reads, no more):

- Per info: `token` (md5-hex of clip+timestamp, 32 chars), `scene_token` (md5-hex of clip name — the 16-char-prefix scheme then works everywhere), `timestamp` (µs int), `gt_planning` = zeros(1,6,3), `gt_planning_command` (filled in Phase 4, default 3 "Follow lane"), `gt_navigation_command` (Phase 4), `gt_boxes` = zeros(0,7), `gt_velocity` = zeros(0,2), `gt_names` = empty array, `can_bus` = None, `description`/`location` = clip metadata.
- Per cam in `cams`: `data_path` = `'samples/CAM_X/....jpg'` **relative to pkl dir** (loader auto-derives `data_root` from pkl parent), `cam_intrinsic` = K_t (virtual intrinsics), `sensor2ego_rotation/translation` = from `vcs_extrinsic` (reordered quats), `ego2global_rotation/translation` = Phase 2 pose (same value on all six cams; the demo reads it from CAM_FRONT_LEFT), `timestamp`, `type`.

**Smoke test**: `NuScenesDataLoader(pkl).get_sample(0)`; then a `prepare_messages()` dry-run printing the ego-status block for a few frames — command/velocity/FLU numbers must be sane. Frames must be stored in temporal order per scene, scenes contiguous (the pipeline's velocity finite-difference and the labeler's scene grouping assume it).

---

## 6. Phase 4 — Driving Command Labeling

1. **Auto-proposal from CAN** (`custom_dataset/propose_commands.py`): per keyframe, windowed heuristics — turn indicators + integrated `Yaw_Rate` over ±3 s + `Steering_Angle` → propose Turn left/right, lane changes (indicator without large heading change), U-turn (|Δheading| > 150°), else Go straight/Follow lane. Write in the labeler's per-scene JSON format so proposals appear as pre-filled labels.
2. **Human review** with the existing labeler:
   ```bash
   python -m nuscenes_pipeline.visualization.driving_command_labeler \
       --pkl_path <custom pkl> --output_dir custom_command_labels --port 6062
   ```
   ~360 samples is a short session. Two small adaptations to verify/patch: `generate_bev` must tolerate empty `gt_boxes`; front-view rendering at 1600×900 is already the expected shape.
3. **Merge** (`custom_dataset/apply_command_labels.py`, new ~40-line script): read the per-scene label JSONs, write `gt_navigation_command` (and mirror into `gt_planning_command`) into the pkl by sample token.

---

## 7. Phase 5 — Run the Demo & Validate

```bash
# 1. Server (unchanged)
LORA_PATH=output/curriculum_v2_c10_0621/C10__seed0/ckpt_100 LORA_NAME=c10_ckpt100 \
bash nuscenes_pipeline/scripts/run_vllm_demo_server.sh 8

# 2. Scenes file with the 3 custom scene tokens, then:
VLLM_URL=http://localhost:8000/v1 MODEL_NAME=c10_ckpt100 BATCH_NAME=c10_custom_demo \
PKL_PATH=<out_root>/custom_demo_infos.pkl SCENES_FILE=<out_root>/scenes.txt \
bash nuscenes_pipeline/scripts/run_demo_batch.sh
```

`PKL_PATH` and `SCENES_FILE` overrides already exist, so **no demo-pipeline code changes should be needed**. Validate: grounding boxes land on real objects in all 6 views (especially rear/flipped ones), image_idx↔direction consistency, base vs LoRA comparison, then `render_demo_videos.sh` + `gather_demo_videos.sh` as before.

---

## 8. Risks / Open Items

- **Pinhole assumption**: if Phase 1 verification shows residual distortion (kbm with hidden coefficients elsewhere), fall back to `cv2.remap` through the KB model — the virtual-camera math and all downstream phases are unchanged.
- **Transform direction ambiguity** in the yamls (comments vs `frame_from/to`) — resolved empirically in Phase 2's sanity checks, not by trusting documentation.
- **Domain gap**: 8 MP German-road imagery vs nuScenes, and ~0.54× absolute object scale vs the nuScenes front cam. The `f_t` knob and a base-model A/B run are the mitigation. Note the demo feeds the processor at native resolution (16.7 MP budget), so 1600×900 outputs cost the same ~1,400 tokens/view as nuScenes demos.
- Rear-center camera FOV (100°) is narrower than nuScenes CAM_BACK (110°) — acceptable, no action.

## 9. Finalized Virtual-Camera Configuration (decided 2026-08-13, via crop_verifier)

Phase 1 has been implemented (`custom_dataset/virtual_camera.py` + `crop_verifier.py`) and the
image-unification design evolved beyond the original crop-only plan. Frozen settings for the
batch converter:

- **Warp**: TRUE pitch rotation per camera via rectification homography `H = K_src·R_x(α)·K_t⁻¹`
  (PIL PERSPECTIVE, bicubic); degenerates to subpixel crop+resize (LANCZOS) at zero rotation.
  NOT a shifted crop — measured elevations come from `camera_orientation.json`
  (trajectory-anchored tuned lcs extrinsics; `measure_camera_orientation.py`).
- **Target cameras**: 1600×900. Base f_t = **1100** (72.1°×44.5°) for 5 views;
  **CAM_BACK f_t = 702.9** (97.4°×65.3°) via the nuScenes back/front focal ratio 809.22/1266.42.
- **Pitch**: auto-level every camera (clamped to data). Measured axis elevations:
  FL +9.5°, F −0.03°, FR +9.8°, BL −10.3°, B −27.4°, BR −9.4°.
  Extra offsets (FINAL): **CAM_BACK_LEFT/RIGHT −5°** (axis −5°, footprints seated lower while
  keeping the top-wide trapezoid form). Black padding was explored (+5°/+10° on CAM_BACK) and
  rejected; side-alignment to the rear axis was explored and rejected.
- **CAM_BACK keystone** (FINAL): `pin_top` (footprint top edge = the source's full top border,
  (0,0)–(1920,0)) + `bottom_scale 0.85` (bottom edge 1640→1394 px). Zero black. This leaves the
  pure-pinhole family: the effective camera (by exact homography decomposition) is anamorphic
  fx=755.4 / fy=678.1 with principal point (801, 337) and axis elevation −13.4°.
- Rear views (BL/B/BR) horizontally flipped at composition/inference time, per training convention.
- **Frozen rig file**: `custom_dataset/virtual_rig.json` (written by
  `python -m custom_dataset.export_virtual_rig`) — per-camera source intrinsics/extrinsics,
  target cameras, warp quads + homographies, EFFECTIVE intrinsics (RQ decomposition,
  residual 0), and virtual axis orientations in the road frame. The batch converter and pkl
  builder must read THIS file; the crop verifier auto-loads it at startup.
- Caveat: the rear source has no data above +13° elevation — full leveling is physically
  impossible without black padding (rejected; the model never saw black bands in training).

## 10. Progress Log (2026-08-13)

- **Phase 1 COMPLETE**: `custom_dataset/convert_images.py` converted all 10,782 images
  (3 clips × 599 frames × 6 cams, stride 1) through the frozen rig →
  `/yjj_rnd_home/data/vlm_dataset/demo_dataset_nusc/samples/CAM_*/<clip>__<cam>__<ts>.jpg`
  (1600×900, q95, rear flip NOT baked in — applied at inference). 443 img/s × 64 workers.
- **Phases 2+3 COMPLETE** (`custom_dataset/build_infos_pkl.py`): per-keyframe ego poses from
  traj_lcs.txt chained via tuned front-cam lcs + nominal front-cam vcs extrinsics
  (T_map←vcs = T_map←lidar·T_lidar←front·T_front←vcs). **QA: reconstructed speed vs CAN
  Cluster_Display_Speed mean |Δv| = 0.06/0.08/0.11 m/s per clip.** 2 Hz keyframes (120/clip)
  cut into 15-sample subclips → **24 pkls** (`<clip>_01..08.pkl`, 8 per clip — the requested
  "15 subclips" is impossible with 15-sample chunks at 120 keyframes; chunk size is a flag).
  Schema mirrors nuscenes2d_ego_temporal_infos_val.pkl; cams carry the rig's effective
  intrinsics + derived virtual extrinsics; `gt_navigation_command = gt_planning_command = -1`
  (DUMMY — Phase 4 fills them); object/label fields empty.
- **Smoke test PASSED**: NuScenesDataLoader + extract_ego_state + prepare_messages on the new
  pkls — all image paths resolve, 6×1600×900 views, FLU velocity forward-positive, ego block
  renders; command shows "Unknown (-1)" until labeled.
- **Phase 4 COMPLETE (2026-08-14)**: all 360 keyframes hand-labeled via the command labeler
  (merged view `demo_all_subclips.pkl`, port 6006); 359 direct labels + 1 skipped sample
  auto-filled from its in-scene neighbor (`apply_command_labels.py`). Dummy −1s replaced in
  all 24 subclip pkls + merged pkl. Distribution: 105436 {GoStraight 116, LaneRight 4},
  112636 {TurnRight 29, GoStraight 88, LaneRight 3}, 192304 {GoStraight 120}.
- **Per-clip pkls** (`merge_subclip_pkls.py --per_clip`): the 24 subclips regrouped 0–7/8–15/16–23
  into 3 clip-level pkls — `<clip>.pkl`, 120 samples / 8 scenes each (scene = subclip).
- **Phase 5 COMPLETE (2026-08-14)**: demo ran on all 3 per-clip pkls (8 scenes × 15 frames each)
  through the vLLM server for BOTH models — base `qwen3vl-8b` and LoRA `c10_ckpt100`
  (24/24 scenes OK each). Videos rendered + gathered: `demo_vid/base_custom/<ts>/` and
  `demo_vid/c10_custom/<ts>/` (8 MP4s per clip, 7.5 s @ 2 fps). Base emits long free-form
  markdown (no JSON/grounding, as expected); the LoRA transfers its SFT format to the custom
  domain: short answer + bullet reasoning with per-view references + valid grounding boxes
  (e.g., correctly boxing the front-left white van). Gather caveat: `gather_demo_videos.sh`
  needs `PKL_PATH=<clip pkl>` for custom batches (defaults to the nuScenes val pkl).
  **All plan phases complete.**

## 11. Execution Order

Strictly Phase 1 → 2 → 3 → 4 → 5; each phase has a self-contained QA gate before the next. Starting point: Phase 1 (`virtual_camera.py` + converter + verification composites for a handful of frames from clip `192304`).
