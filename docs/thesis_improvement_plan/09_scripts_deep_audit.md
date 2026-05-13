# 09 -- Scripts & Tests Deep Audit

**Date:** 2026-03-31
**Scope:** Every `.py` file under `simulation_camera_scooter/scripts/` and `simulation_camera_scooter/tests/`

---

## Part A: Scripts Inventory (23 files)

### Legend

| Rating | Meaning |
|--------|---------|
| **USEFUL** | Works or would work with data present; keep |
| **NEEDS-FIX** | Broken but valuable; fix before relying on it |
| **REDUNDANT** | Duplicates another script's function |
| **DEAD** | No longer needed given current project state |

---

### A1. eval_hand_annotated_pipeline.py

**Path:** `scripts/eval_hand_annotated_pipeline.py` (619 lines)

**Purpose:** THE gold-standard evaluation script. Runs SegFormer inference (baseline + candidate models) on hand-annotated frames, computes per-frame segmentation metrics (IoU, precision, recall, F1) against GT masks, then runs 5 planner variants (bev_dt_full, bev_dt_nearfield, bev_graph, img_dt, img_midpoint) on both predicted and oracle-GT masks. Computes path metrics (inside_gt_ratio, mean_clearance_px, center_error_px, heading_deg). Generates contact-sheet comparison images.

**Inputs:**
- `--images-root` (default: `outputs/hand_annotations/v1/images/`) -- hand-annotated frame JPGs
- `--masks-root` (default: `outputs/hand_annotations/v1/masks/`) -- hand-annotated GT mask PNGs
- `--output-root` (default: `research/artifacts/`)
- `--device` (cuda/cpu)
- `--max-frames`, `--per-video-limit`, `--videos` (optional filters)

**Outputs:**
- `tables/segmentation_hand_annotations_per_frame.csv` -- per-frame seg metrics (**this is the per-frame data file for statistical analysis**)
- `tables/segmentation_hand_annotations_summary.csv` -- aggregated summary
- `tables/segmentation_hand_annotations_by_video.csv` -- per-video breakdown
- `tables/planner_hand_annotations_per_frame.csv` -- per-frame planner metrics (**another key per-frame data file**)
- `tables/planner_hand_annotations_summary.csv`
- `tables/planner_hand_annotations_by_video.csv`
- `images/planner_bev_vs_img_dt_contact_sheet.png`
- JSON variants of summaries

**Issues found:**
1. Line 250: `args` is used as a module-global but defined at line 618 (`args = parse_args()`). This works because `evaluate_segmentation` is only called from `main()` after `args` is set, but it is a code smell. The `device` should be passed as a parameter.
2. Hardcoded model paths (lines 43-44): `BASELINE_MODEL` and `CANDIDATE_MODEL` point to specific checkpoint directories. If these directories do not exist on the current machine, the script will fail at model load time. The defaults are reasonable but should be overridable via CLI args (they are not).
3. The `ConfidenceHoldCleaner` (line 153) implements hysteresis thresholding but its `band` parameter is not exposed via CLI.

**Can generate per-frame data?** YES -- the primary source of per-frame CSV data for thesis statistical analysis.

**Rating:** **USEFUL** -- The single most important evaluation script.

**Exact command:**
```bash
cd simulation_camera_scooter
python scripts/eval_hand_annotated_pipeline.py \
  --images-root ../outputs/hand_annotations/v1/images \
  --masks-root ../outputs/hand_annotations/v1/masks \
  --output-root ../research/artifacts \
  --device cuda
```

To limit to specific videos or frame counts:
```bash
python scripts/eval_hand_annotated_pipeline.py \
  --videos IMG_1877 IMG_1922 \
  --per-video-limit 50 \
  --max-frames 200
```

---

### A2. eval_binary_seg_models.py

**Path:** `scripts/eval_binary_seg_models.py` (261 lines)

**Purpose:** Full-video replay comparison of baseline vs candidate binary SegFormer checkpoint. Runs `live_heading_demo.run_live()` on all test videos for each model, collects per-run CSV logs, and summarizes into JSON + Markdown.

**Inputs:**
- `--candidate-model` (required) -- path to candidate checkpoint
- `--videos-dir` (default: `test_videos/`)
- `--baseline-model` (default: `config.MODEL_DIR`)
- `--baseline-thresh` / `--candidate-thresh` -- segmentation confidence thresholds
- `--max-frames`, `--save-video`, `--enable-detection`, `--planner-mode`

**Outputs:**
- Per-case directories with CSV logs and optional overlay videos
- `summary.json` and `summary.md` at output root

**Issues found:**
1. Requires `live_heading_demo.run_live()` which needs a GPU and the full pipeline stack.
2. No per-frame CSV export -- it summarizes from the CSV logs that `run_live` generates, so the per-frame data IS in the per-run CSVs, but this script only writes summaries.

**Can generate per-frame data?** INDIRECTLY -- the per-run CSVs contain per-frame data but this script only outputs summaries. The raw CSVs are saved to the log directories.

**Rating:** **USEFUL**

**Exact command:**
```bash
cd simulation_camera_scooter
python scripts/eval_binary_seg_models.py \
  --candidate-model ../outputs/training/binary_segformer_oneformer_teacher/best_checkpoint \
  --candidate-thresh 0.60 \
  --output-root ../outputs/evaluation/binary_model_replay \
  --save-video \
  --max-frames 500
```

---

### A3. eval_template_planner.py

**Path:** `scripts/eval_template_planner.py` (224 lines)

**Purpose:** Compare graph-first baseline planner vs Phase 11 template-approval planner on recorded videos. Runs `live_heading_demo.run_live()` in two modes (`graph_baseline` and `template_phase11`) for each video.

**Inputs:**
- `--videos` (required, one or more video paths)
- `--output-root` (default: `simulation_camera_scooter/eval_runs/template_planner`)
- `--max-frames`, `--stride`, `--planner-mode`, `--enable-detection`, `--save-video`

**Outputs:**
- Per-mode log CSVs + optional overlay videos
- `summary.json` and `summary.md`

**Issues found:**
1. The `_collect_rendered_video` function (line 40) uses `shutil.move` on the hardcoded filename `heading_demo_output.mp4`, relying on a side effect of `run_live`. Fragile.
2. Does not accept `--model-dir` or `--seg-conf-thresh` arguments, so always uses the default model.

**Can generate per-frame data?** INDIRECTLY -- via the CSV logs from run_live.

**Rating:** **USEFUL**

**Exact command:**
```bash
cd simulation_camera_scooter
python scripts/eval_template_planner.py \
  --videos test_videos/IMG_1877.mp4 test_videos/IMG_1922.mp4 \
  --output-root eval_runs/template_planner \
  --save-video \
  --max-frames 500
```

---

### A4. eval_waypoint_turn_planner.py

**Path:** `scripts/eval_waypoint_turn_planner.py` (721 lines)

**Purpose:** Phase 11.1 three-mode replay comparison: baseline, template, and waypoint_turn. The most comprehensive evaluator, with waypoint-turn-specific metrics (turn rate, hold rate, containment fail rate, boundary clearance), provenance logging of config thresholds, and both pandas and fallback CSV summarization.

**Inputs:**
- `--replay-set` OR `--videos` (mutually exclusive, one required)
- `--modes` (default: baseline, template, waypoint_turn)
- `--output-dir` (default: `outputs/evaluation/waypoint_turn_phase11_1`)
- `--max-frames`, `--stride`, `--model-dir`, `--seg-conf-thresh`, `--intent-schedule`

**Outputs:**
- Per-mode log CSVs
- `summary.json`, `comparison.md`, `thresholds.json`

**Issues found:**
1. Both `template` and `waypoint_turn` modes produce identical `run_live` kwargs (lines 149-159), making them functionally identical unless `--intent-schedule` is provided for waypoint_turn mode. The difference depends on the intent schedule or GPS data.
2. Imports `config as runtime_config` at module scope (line 61) which triggers config loading before argparse.

**Can generate per-frame data?** INDIRECTLY -- via the CSV logs.

**Rating:** **USEFUL**

**Exact command:**
```bash
cd simulation_camera_scooter
python scripts/eval_waypoint_turn_planner.py \
  --videos test_videos/VID_20260319_155939_00_017.mp4 \
  --modes baseline template waypoint_turn \
  --output-dir outputs/evaluation/waypoint_turn_phase11_1 \
  --max-frames 300

# Or with a replay manifest:
python scripts/eval_waypoint_turn_planner.py \
  --replay-set .planning/phases/11.1-gps-intent-corridor-waypoint-turn-planner/11.1-REPLAY_SET.txt \
  --modes baseline template waypoint_turn
```

---

### A5. train_binary_segformer.py

**Path:** `scripts/train_binary_segformer.py` (472 lines)

**Purpose:** Fine-tune a SegFormer checkpoint for binary drivable-mask segmentation. Implements full training loop with: data augmentation (flip, color jitter, Gaussian blur), CE + Dice loss, cosine LR schedule with linear warmup, mixed-precision (AMP), class weight balancing, best-checkpoint saving.

**Inputs:**
- `--images-root` (default: `annotation_frames/`)
- `--masks-root` (default: `outputs/pseudo_labels/.../masks`)
- `--init-model` (default: `models/my-segformer-road`)
- `--output-dir` (default: `outputs/training/binary_segformer_oneformer_teacher`)
- `--epochs` (10), `--batch-size` (4), `--lr` (5e-5), `--weight-decay` (1e-4)
- `--image-width` (640), `--image-height` (360), `--val-every` (5)
- `--seed` (1337), `--device`

**Outputs:**
- `best_checkpoint/` -- HuggingFace-compatible model directory
- `last_checkpoint/`
- `history.csv` -- epoch-level training metrics
- `summary.json` -- full training summary

**Issues found:**
1. Line 361: `torch.cuda.amp.GradScaler(enabled=...)` uses the deprecated API. Should be `torch.amp.GradScaler('cuda', enabled=...)` for PyTorch 2.0+. This produces a deprecation warning but still works.
2. Line 376: `torch.cuda.amp.autocast(enabled=...)` similarly deprecated.

**Can generate per-frame data?** No (this is a training script, not evaluation).

**Rating:** **USEFUL**

**Exact command (for the thesis models):**
```bash
cd simulation_camera_scooter
python scripts/train_binary_segformer.py \
  --images-root annotation_frames \
  --masks-root ../outputs/pseudo_labels/oneformer_cityscapes_swin_large_binary/masks \
  --init-model models/my-segformer-road \
  --output-dir ../outputs/training/binary_segformer_oneformer_teacher \
  --epochs 10 \
  --batch-size 4 \
  --lr 5e-5 \
  --val-every 5 \
  --seed 1337 \
  --device cuda
```

---

### A6. eval_research_improvements.py

**Path:** `scripts/eval_research_improvements.py` (512 lines)

**Purpose:** Evaluate the three research pipeline improvements (enhanced morphology, DT corridor, temporal smoothing) by comparing baseline vs enhanced on a test video. Uses a **stub BEV transform** (crop + threshold), not real segmentation.

**Inputs:**
- `--video` (auto-discovers if not specified)
- `--max-frames` (default: 200)

**Outputs:**
- Stdout comparison table
- `EVALUATION_REPORT.md` at project root

**Issues found:**
1. `clean_sidewalk_mask` and `compute_heading_smooth` imports verified to exist in `masks.py` and `heading.py` respectively -- no import issues.
2. Uses a stub BEV (`_passthrough_bev`) that crop+thresholds the frame, making absolute metric values meaningless. Only relative comparisons are valid.
3. Mutates the config module globals (lines 477-480) which is fragile and not thread-safe.
4. The report hardcodes date as `2026-03-17` (line 329).

**Can generate per-frame data?** No -- only generates aggregate metrics and a markdown report.

**Rating:** **USEFUL** -- but the stub BEV limits scientific validity; results are only meaningful for relative baseline-vs-enhanced comparison.

---

### A7. eval_simple_road.py

**Path:** `scripts/eval_simple_road.py` (751 lines)

**Purpose:** Evaluate the simplified simple-road pipeline vs the full baseline on test videos. Runs both pipelines side-by-side on each frame, computing mask coverage, temporal IoU, heading, lateral offset, and path jump metrics. Generates overlay videos and comparison strips.

**Inputs:**
- `--videos` (auto-discovers if not specified)
- `--max-frames` (300), `--frame-step` (3)
- `--output-dir`, `--device`

**Outputs:**
- Per-video: baseline/improved overlay videos, side-by-side comparison video, metrics JSON
- Updated `EVALUATION_REPORT.md`

**Issues found:**
1. Depends on `simple_road_pipeline.SimpleRoadProcessor` and `SimpleRoadConfig` which must exist.
2. Heavy -- requires GPU and full model stack.
3. The `EVALUATION_REPORT.md` append logic (line 580-590) tries to find and replace previous run sections, which is fragile string manipulation.

**Can generate per-frame data?** PARTIALLY -- the `MetricAccumulator` stores per-frame lists but they are summarized into aggregates in the JSON output. The per-frame arrays are not directly exported.

**Rating:** **USEFUL** -- Needed for simple-road vs baseline comparison.

---

### A8. benchmark_seg_stability.py

**Path:** `scripts/benchmark_seg_stability.py` (310 lines)

**Purpose:** Evaluate temporal segmentation stability across all SegFormer checkpoints. For each checkpoint, runs inference and computes consecutive-frame IoU.

**Inputs:**
- `--video` (default: `test_video_mar3_1_h264.mp4`)
- `--road-id` (1), `--conf-thresh` (0.6), `--max-frames` (500)

**Outputs:** Stdout table sorted by pct_stable. Identifies best checkpoint.

**Issues found:**
1. CHECKPOINTS list (line 44) is hardcoded with specific checkpoint names (`checkpoint-500` through `checkpoint-5000`). These may not exist.
2. PROCESSOR_DIR (line 60) is hardcoded to `models/my-segformer-road_new` which may not exist.
3. Line 156: `TemporalMaskSmoother._iou()` is called as a static/class method directly -- need to verify this private method exists.

**Can generate per-frame data?** No -- only aggregate stats.

**Rating:** **USEFUL** -- but only works if the specific checkpoints exist. Good for thesis Chapter 4 checkpoint selection narrative.

---

### A9. cityscapes_miou_segformer_b0.py

**Path:** `scripts/cityscapes_miou_segformer_b0.py` (357 lines)

**Purpose:** Standard Cityscapes 19-class mIoU evaluation for SegFormer-B0. Proper confusion-matrix-based IoU computation.

**Inputs:**
- `--cityscapes-root` (required) -- path to Cityscapes dataset
- `--model-id` (default: `nvidia/segformer-b0-finetuned-cityscapes-1024-1024`)
- `--split` (val), `--batch-size` (1), `--max-images`, `--device`, `--local-files-only`

**Outputs:**
- `metrics/cityscapes_miou_segformer_b0.json`
- `metrics/cityscapes_miou_segformer_b0_per_class.csv`

**Issues found:**
1. Requires the Cityscapes dataset to be downloaded locally. Not an issue per se, but the dataset must be obtained separately.
2. Uses `torchvision.datasets.Cityscapes` which requires specific directory structure.

**Can generate per-frame data?** No -- aggregate mIoU only.

**Rating:** **USEFUL** -- Essential for the thesis "Cityscapes mIoU" baseline number.

**Exact command:**
```bash
cd simulation_camera_scooter
python scripts/cityscapes_miou_segformer_b0.py \
  --cityscapes-root "D:/datasets/cityscapes" \
  --output-json metrics/cityscapes_miou_segformer_b0.json \
  --output-csv metrics/cityscapes_miou_segformer_b0_per_class.csv
```

---

### A10. tune_smoother.py

**Path:** `scripts/tune_smoother.py` (254 lines)

**Purpose:** Parameter sweep for EMA alpha x consistency_thresh on demo video. Single GPU pass to collect raw masks, then CPU-only replay through every (alpha, thresh) combination.

**Inputs:**
- `--video` (default: `test_video_mar3_1_h264.mp4`)
- `--max-frames` (500)

**Outputs:** Stdout table of sweep results, best params.

**Issues found:** None significant. Clean implementation.

**Rating:** **USEFUL** -- Good for thesis discussion of smoothing parameter selection.

---

### A11. eval_boundary_net.py

**Path:** `scripts/eval_boundary_net.py` (132 lines)

**Purpose:** Evaluate a trained TinyBoundaryNet checkpoint on a validation set.

**Inputs:**
- `--records` (required) -- JSONL with boundary records
- `--checkpoint` (required) -- `.pt` checkpoint
- `--output-json`, `--batch-size`, various threshold params

**Outputs:** JSON metrics to stdout and optional file.

**Issues found:** None significant.

**Rating:** **USEFUL** -- if boundary net experiments are discussed in thesis.

---

### A12. export_boundary_targets.py

**Path:** `scripts/export_boundary_targets.py` (81 lines)

**Purpose:** Convert mask records to row-wise boundary targets for TinyBoundaryNet training.

**Inputs:**
- `--records` (required) -- input JSONL
- `--output` (required) -- output JSONL
- `--limit`, `--row-step`, `--min-width-px`, `--skip-missing`

**Rating:** **USEFUL** -- data prep for boundary net.

---

### A13. train_boundary_net.py

**Path:** `scripts/train_boundary_net.py` (133 lines)

**Purpose:** Train TinyBoundaryNet from exported boundary JSONL records.

**Inputs:**
- `--train-records`, `--val-records` (required)
- `--output-dir`, `--epochs`, `--batch-size`, `--lr`, etc.

**Rating:** **USEFUL** -- training script for boundary net experiments.

---

### A14. measure_bev_survival.py

**Path:** `scripts/measure_bev_survival.py` (114 lines)

**Purpose:** Measure BEV calibration quality from a run log CSV. Computes pixel survival ratio (BEV mask pixels / front-view sidewalk pixels), heading jump statistics, and pass/fail gates.

**Inputs:** Optional CSV path (defaults to latest `logs/run_*.csv`).

**Outputs:** Stdout metrics with pass/fail gates.

**Rating:** **USEFUL** -- quick diagnostic for BEV calibration quality.

---

### A15. convert_videos.py

**Path:** `scripts/convert_videos.py` (98 lines)

**Purpose:** Convert MOV files to H264 MP4 using OpenCV.

**Rating:** **USEFUL** -- utility script.

---

### A16. extract_annotation_frames.py

**Path:** `scripts/extract_annotation_frames.py` (180 lines)

**Purpose:** Extract visually diverse frames from videos for hand annotation using scene-change detection and sharpness scoring.

**Rating:** **USEFUL** -- data prep for annotation pipeline.

---

### A17. generate_binary_pseudo_labels.py

**Path:** `scripts/generate_binary_pseudo_labels.py` (260 lines)

**Purpose:** Generate binary drivable pseudo-labels using OneFormer (Swin-L) as teacher. Collapses Cityscapes road+sidewalk classes to binary drivable mask.

**Issues found:**
1. Requires downloading the OneFormer Swin-L model (`shi-labs/oneformer_cityscapes_swin_large`), which is a large model (~1 GB).

**Rating:** **USEFUL** -- essential for the pseudo-label training pipeline described in thesis.

---

### A18. make_video_comparison_strips.py

**Path:** `scripts/make_video_comparison_strips.py` (102 lines)

**Purpose:** Build side-by-side comparison image strips from baseline/candidate replay videos.

**Rating:** **USEFUL** -- generates thesis figures.

---

### A19. prepare_hand_annotation_workspace.py

**Path:** `scripts/prepare_hand_annotation_workspace.py` (115 lines)

**Purpose:** Copy annotation images + starter masks into a workspace for manual correction.

**Rating:** **USEFUL** -- data prep utility.

---

### A20. replay_model_on_videos.py

**Path:** `scripts/replay_model_on_videos.py` (122 lines)

**Purpose:** Replay one segmentation model on all test videos, saving overlay videos and CSV logs.

**Inputs:**
- `--model-dir` (required), `--label` (required), `--output-root` (required)
- `--seg-conf-thresh` (0.6), `--videos-dir`, `--planner-mode`, `--max-frames`

**Rating:** **REDUNDANT** -- overlaps significantly with `eval_binary_seg_models.py` but for a single model. Could be useful as a simpler alternative.

---

### A21. tune_binary_threshold.py

**Path:** `scripts/tune_binary_threshold.py` (102 lines)

**Purpose:** Sweep segmentation confidence thresholds against pseudo-label validation split.

**Rating:** **USEFUL** -- generates the threshold-vs-IoU data for thesis.

---

### A22. learn_turn_schedule.py

**Path:** `scripts/learn_turn_schedule.py` (154 lines)

**Purpose:** Analyze an unguided replay CSV log to learn likely turn windows, then write a frame-range intent schedule JSON for use with `live_heading_demo.py`.

**Rating:** **USEFUL** -- utility for waypoint-turn evaluation.

---

### A23. calibrate_bev_examples.py

**Path:** `scripts/calibrate_bev_examples.py` (255 lines)

**Purpose:** Interactive GUI tool for BEV calibration. Browse exported frame examples, click 4 source points, see live warped segmentation preview.

**Issues found:**
1. Requires `cv2.namedWindow` and GUI display -- will not work headless.

**Rating:** **USEFUL** -- but only for interactive calibration sessions.

---

## Part B: Tests Inventory (13 files, ~110+ tests)

### B1. conftest.py

**Path:** `tests/conftest.py` (207 lines)

**Fixtures provided:**
- `_pin_ego_center` (autouse) -- pins `BEV_EGO_X_FRAC` to 0.5
- `straight_bev_mask` -- 220x220, 60px wide center corridor
- `curved_bev_mask` -- 220x220, left-curving corridor
- `wide_straight_bev_mask` -- 220x220, 100px wide center corridor
- `right_curved_bev_mask` -- 220x220, right-curving corridor
- `fragmented_near_field_bev_mask` -- broken near-field
- `false_pocket_bev_mask` -- side pocket near ego
- `straight_path_model` -- CubicPathModel for y=0
- `bev_h_matrix` -- simple 3x3 scale matrix (0.3x, 0.5y)
- `bev_obstacle_mask_500x600` -- all-white BEV mask
- `mock_detections` -- 2 YOLO detection dicts
- `commanded_left_bev_mask` -- left turn opening
- `commanded_right_bev_mask` -- right turn opening
- `unsupported_turn_bev_mask` -- fragmented side support
- `no_intent_straight_bev_mask` -- clean straight, no turns

### B2. test_realtime_nav_core.py (~280 lines, ~20 tests)

**Covers:** `BEVPathExtractor` (empty mask, straight corridor, wide corridor, return type, commit logic), `AdaptivePurePursuitController` (no path, straight, curved, rate limiting), waypoint-turn integration (commanded left/right engage, maneuver lock stability, no-intent preservation, unsupported turn hold/slowdown).

### B3. test_template_path_planner.py (~388 lines, ~17 tests)

**Covers:** `corridor_from_mask` (centered bounds, curving centerline, fragmented, false pocket), `generate_template_bank` (7 templates, families, curvature bounds), `approve_template_bank` (straight preference, curve preference, obstacle penalty, continuity bias, low confidence, margin gate, hold recommendation), GPS intent filtering (straight blocks turn, left excludes right, right can approve right), `eval_template_planner.summarize_log` smoke test, `_collect_rendered_video`.

### B4. test_waypoint_turn_planner.py (~730 lines, ~30+ tests)

**Covers:** Public contract stability (dataclass fields, function signatures), fixture support verification (left/right extent), unsupported turn handling, inactive states (no intent, straight intent, empty intent), target selection (left negative lateral, right positive lateral, cross-side rejection, decision band range), support scoring (threshold, false pocket rejection, weak support), containment gating (inside corridor, no centerline rejoin, confidence threshold, narrow corridor rejection, hold slowdown, path length), replay evaluator (manifest parsing, summary metrics -- waypoint_turn_rate, hold_rate, family switches, heading stats, low confidence, mode dispatch, comparison artifact).

### B5. test_heading.py (~116 lines, ~12 tests)

**Covers:** `compute_heading` (straight, left curve, right curve, single point), `heading_to_command` (straight, left, right, sharp left, sharp right, return type), `compute_speed` (near obstacle, close obstacle, max on straight, sharp turn, no path, interpolated turn).

### B6. test_bev_calibration.py (~79 lines, ~5 tests)

**Covers:** Calibration file is absolute path, `load_bev_params` returns 3-tuple, H is 3x3, H and H_inv are inverses, singular calibration falls back to default, missing file uses defaults.

### B7. test_bev_obstacle.py (~175 lines, ~8 tests)

**Covers:** OBS-01 through OBS-09: foot-point projection, metric conversion, EMA decay, EMA update, obstacle penalty (clear vs occupied), hard-block mask painting, out-of-bounds clamping, no-penalty with empty zones, full pipeline integration.

### B8. test_bev_predictor.py (~352 lines, ~18 tests)

**Covers:** `BEVPredictiveTracker` warp tests (forward, lateral, rotation, identity), path warp (forward, trim behind ego, turn, empty), should-skip logic (no state, straight allow, sharp turn block, low confidence block, max consecutive, gentle turn, low occupancy), compute/skip frame handlers (first frame, blend after skips, skip returns mask, skip increments count, empty mask drops path), ego displacement (with path, dead reckoning, zero speed), IoU helper.

### B9. test_temporal_smoother.py (~245 lines, ~6 tests)

**Covers:** `TemporalMaskSmoother` edge cases: empty mask first call, empty after non-empty history, alpha response time at min alpha (SEG-03), high consistency full alpha, low consistency reduced alpha, identical masks stabilization.

### B10. test_boundary_dataset_model.py (~88 lines, ~3 tests)

**Covers:** `BoundaryRecordDataset` loading, `boundary_collate` + `TinyBoundaryNet` forward pass, `boundary_loss` and `boundary_metrics` finiteness.

### B11. test_boundary_inference.py (~84 lines, ~3 tests)

**Covers:** `decode_boundary_prediction` centerline path, missing near-field flagging, previous path blending.

### B12. test_boundary_targets.py (~68 lines, ~4 tests)

**Covers:** `extract_boundary_targets` straight corridor, invalid rows, curving centerline tracking, `build_boundary_record` summary fields.

### B13. test_image_path_planner.py (~35 lines, ~2 tests)

**Covers:** `CameraMidpointPlanner` straight path, `CameraDtPlanner` curved corridor tracking.

---

## Part C: Coverage Gap Analysis

### What IS tested well:
- Template path planner: corridor extraction, template scoring, approval logic, GPS intent
- Waypoint turn planner: full algorithm + evaluator integration
- BEV obstacle projection: foot-point, metric conversion, EMA, penalty, hard-block
- BEV predictor: warp, skip logic, frame handling
- Temporal smoother: edge cases, parameter constraints
- Heading and speed computation
- BEV calibration loading and fallback
- Boundary model/dataset/inference chain

### What is NOT tested:
- **No integration tests** that run the full pipeline (`realtime_nav_core.py` main loop) end-to-end with a mock video
- **No tests for `fast_road_detector.py`** (SegFormer inference wrapper)
- **No tests for `skeleton.py`** (medial-axis path extraction -- legacy but still used as fallback)
- **No tests for `gps_navigator.py`** or `intent_picker.py`
- **No tests for `visualization.py`** or `data_logger.py`
- **No tests for `masks.py`** (clean_bev_mask_enhanced, though it is tested indirectly via corridor tests)
- **No tests for `safe_corridor.py`** (DtSafeCorridor -- only tested indirectly)
- **No tests for `path_smoother.py`** (PathTemporalSmoother, HeadingTemporalFilter)
- **No tests for `simple_road_pipeline.py`**
- **No tests for `image_path_planner.py` beyond basic smoke** (only 2 tests)

### What is NOT tested in scripts:
- **No unit tests for `eval_hand_annotated_pipeline.py`** (the most critical evaluation script)
- **No tests for `eval_simple_road.py`**
- **No tests for `eval_research_improvements.py`**
- Scripts are tested only where they export utility functions consumed by other test files (e.g., `eval_template_planner.summarize_log` is tested in `test_template_path_planner.py`, and `eval_waypoint_turn_planner` functions are tested in `test_waypoint_turn_planner.py`).

---

## Part D: Per-Frame Data Generation Capability

For thesis statistical analysis (confidence intervals, Wilcoxon tests, bootstrap), you need **per-frame CSV data**.

| Script | Per-Frame CSV? | Format |
|--------|---------------|--------|
| `eval_hand_annotated_pipeline.py` | **YES** | `segmentation_hand_annotations_per_frame.csv` (IoU, precision, recall, F1 per frame) and `planner_hand_annotations_per_frame.csv` (inside_gt_ratio, clearance, center_error, heading per frame) |
| `eval_binary_seg_models.py` | **INDIRECT** | Raw per-frame data is in the `run_*.csv` logs generated by `run_live`, saved under each case's log directory |
| `eval_template_planner.py` | **INDIRECT** | Same -- raw CSVs from `run_live` |
| `eval_waypoint_turn_planner.py` | **INDIRECT** | Same -- raw CSVs from `run_live` |
| `eval_simple_road.py` | **NO** | Per-frame arrays exist in memory but are only exported as aggregates in the JSON |
| `eval_research_improvements.py` | **NO** | Only aggregate metrics |
| `benchmark_seg_stability.py` | **NO** | Only aggregate pct_stable/pct_failure |
| `cityscapes_miou_segformer_b0.py` | **NO** | Aggregate confusion matrix only |

**Recommendation:** The `run_live()` CSVs generated by `eval_binary_seg_models.py`, `eval_template_planner.py`, and `eval_waypoint_turn_planner.py` contain per-frame columns including:
- `seg_iou`, `heading_smoothed_deg`, `has_path`, `path_source`, `selected_template_family`
- `bev_mask_occ_ratio`, `corridor_confidence`, `planner_low_confidence`, `planner_slowdown`
- `fps`, `sidewalk_mask_pixels`, `bev_mask_pixels`

These can be loaded with `pd.read_csv()` for any statistical test needed.

---

## Part E: Script Health Summary Table

| # | Script | Lines | Rating | Key Issue |
|---|--------|-------|--------|-----------|
| 1 | eval_hand_annotated_pipeline.py | 619 | **USEFUL** | Model paths not CLI-overridable; `args` used as global |
| 2 | eval_binary_seg_models.py | 261 | **USEFUL** | Needs GPU + full stack |
| 3 | eval_template_planner.py | 224 | **USEFUL** | No model-dir CLI arg; fragile video collection |
| 4 | eval_waypoint_turn_planner.py | 721 | **USEFUL** | template/waypoint_turn modes are identical without intent schedule |
| 5 | train_binary_segformer.py | 472 | **USEFUL** | Deprecated AMP API (warning only) |
| 6 | eval_research_improvements.py | 512 | **USEFUL** | Stub BEV limits scientific validity; hardcoded date |
| 7 | eval_simple_road.py | 751 | **USEFUL** | Heavy; no per-frame CSV export |
| 8 | benchmark_seg_stability.py | 310 | **USEFUL** | Hardcoded checkpoint list |
| 9 | cityscapes_miou_segformer_b0.py | 357 | **USEFUL** | Requires Cityscapes dataset |
| 10 | tune_smoother.py | 254 | **USEFUL** | Clean |
| 11 | eval_boundary_net.py | 132 | **USEFUL** | Clean |
| 12 | export_boundary_targets.py | 81 | **USEFUL** | Clean |
| 13 | train_boundary_net.py | 133 | **USEFUL** | Clean |
| 14 | measure_bev_survival.py | 114 | **USEFUL** | Clean |
| 15 | convert_videos.py | 98 | **USEFUL** | Utility |
| 16 | extract_annotation_frames.py | 180 | **USEFUL** | Utility |
| 17 | generate_binary_pseudo_labels.py | 260 | **USEFUL** | Requires OneFormer model download |
| 18 | make_video_comparison_strips.py | 102 | **USEFUL** | Clean |
| 19 | prepare_hand_annotation_workspace.py | 115 | **USEFUL** | Clean |
| 20 | replay_model_on_videos.py | 122 | **REDUNDANT** | Overlaps with eval_binary_seg_models.py |
| 21 | tune_binary_threshold.py | 102 | **USEFUL** | Clean |
| 22 | learn_turn_schedule.py | 154 | **USEFUL** | Clean |
| 23 | calibrate_bev_examples.py | 255 | **USEFUL** | GUI only; requires display |

**Summary:** 22 USEFUL, 0 NEEDS-FIX, 1 REDUNDANT, 0 DEAD.

---

## Part F: Recommended Commands for Fresh Thesis Data Generation

### F1. Cityscapes mIoU (Table 4.1 baseline)
```bash
cd simulation_camera_scooter
python scripts/cityscapes_miou_segformer_b0.py \
  --cityscapes-root "D:/datasets/cityscapes" \
  --model-id nvidia/segformer-b0-finetuned-cityscapes-1024-1024 \
  --output-json metrics/cityscapes_miou_segformer_b0.json \
  --output-csv metrics/cityscapes_miou_segformer_b0_per_class.csv
```

### F2. Hand-Annotated Evaluation (Tables 5.x segmentation + planner comparison)
```bash
cd simulation_camera_scooter
python scripts/eval_hand_annotated_pipeline.py \
  --images-root ../outputs/hand_annotations/v1/images \
  --masks-root ../outputs/hand_annotations/v1/masks \
  --output-root ../research/artifacts \
  --device cuda
```
Produces: per-frame CSVs for statistical tests, summary CSVs for tables, contact-sheet PNGs for figures.

### F3. Full-Video Replay (Table 5.x temporal stability)
```bash
cd simulation_camera_scooter
python scripts/eval_binary_seg_models.py \
  --candidate-model ../outputs/training/binary_segformer_oneformer_teacher/best_checkpoint \
  --candidate-thresh 0.60 \
  --output-root ../outputs/evaluation/binary_model_replay_full \
  --save-video
```

### F4. Template Planner vs Graph Baseline (Table 5.x planner comparison)
```bash
cd simulation_camera_scooter
python scripts/eval_template_planner.py \
  --videos test_videos/IMG_1877.mp4 test_videos/IMG_1922.mp4 test_videos/IMG_1924.mp4 \
  --output-root eval_runs/template_planner_thesis \
  --save-video
```

### F5. Waypoint Turn Three-Mode Comparison (Table 5.x turn evaluation)
```bash
cd simulation_camera_scooter
python scripts/eval_waypoint_turn_planner.py \
  --replay-set .planning/phases/11.1-gps-intent-corridor-waypoint-turn-planner/11.1-REPLAY_SET.txt \
  --modes baseline template waypoint_turn \
  --output-dir outputs/evaluation/waypoint_turn_thesis
```

### F6. Training Reproduction (for thesis methodology section)
```bash
cd simulation_camera_scooter
# Step 1: Generate pseudo-labels
python scripts/generate_binary_pseudo_labels.py \
  --input-root annotation_frames \
  --output-root ../outputs/pseudo_labels/oneformer_cityscapes_swin_large_binary \
  --save-previews \
  --device cuda

# Step 2: Fine-tune SegFormer
python scripts/train_binary_segformer.py \
  --images-root annotation_frames \
  --masks-root ../outputs/pseudo_labels/oneformer_cityscapes_swin_large_binary/masks \
  --init-model models/my-segformer-road \
  --output-dir ../outputs/training/binary_segformer_oneformer_teacher \
  --epochs 10 --batch-size 4 --lr 5e-5 --seed 1337

# Step 3: Tune threshold
python scripts/tune_binary_threshold.py \
  --images-root annotation_frames \
  --masks-root ../outputs/pseudo_labels/oneformer_cityscapes_swin_large_binary/masks \
  --model-dir ../outputs/training/binary_segformer_oneformer_teacher/best_checkpoint
```

---

## Part G: Test Coverage Summary

| Test File | Module(s) Tested | Tests | Status |
|-----------|-----------------|-------|--------|
| test_realtime_nav_core.py | BEVPathExtractor, PurePursuit, WaypointTurn integration | ~20 | Pass (expected) |
| test_template_path_planner.py | corridor, templates, approval, GPS intent | ~17 | Pass (expected) |
| test_waypoint_turn_planner.py | plan_waypoint_turn, evaluator functions | ~30+ | Pass (expected) |
| test_heading.py | compute_heading, heading_to_command, compute_speed | ~12 | Pass (expected) |
| test_bev_calibration.py | load_bev_params, fallbacks | ~5 | Pass (expected) |
| test_bev_obstacle.py | project_foot_to_bev, EMA, penalty | ~8 | Pass (expected) |
| test_bev_predictor.py | BEVPredictiveTracker | ~18 | Pass (expected) |
| test_temporal_smoother.py | TemporalMaskSmoother | ~6 | Pass (expected) |
| test_boundary_dataset_model.py | BoundaryRecordDataset, TinyBoundaryNet | ~3 | Pass (expected) |
| test_boundary_inference.py | decode_boundary_prediction | ~3 | Pass (expected) |
| test_boundary_targets.py | extract_boundary_targets | ~4 | Pass (expected) |
| test_image_path_planner.py | CameraDtPlanner, CameraMidpointPlanner | ~2 | Pass (expected) |

**Total: ~128 tests across 12 test files + conftest.py**

**Missing coverage (thesis-relevant):**
- masks.py (clean_bev_mask_enhanced) -- tested only indirectly
- safe_corridor.py (DtSafeCorridor) -- tested only indirectly
- path_smoother.py (temporal smoothing) -- not tested
- fast_road_detector.py -- not tested (requires GPU)
- skeleton.py -- not tested (legacy but still fallback)
- simple_road_pipeline.py -- not tested

**Verdict:** Test coverage is good for the core planner and obstacle modules. The main gap is around the segmentation and mask-processing pipeline, which is tested only through integration in the evaluation scripts.
