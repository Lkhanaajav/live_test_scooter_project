# Codebase Audit: simulation_camera_scooter/

**Date:** 2026-03-31
**Auditor:** Claude Code (automated deep audit)
**Scope:** Every Python file, script, test, model, data directory, and non-code asset

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Root-level Python modules | 31 files, 12,693 LOC |
| Scripts | 24 files, 6,307 LOC |
| Tests | 13 files (12 test + 1 conftest), 2,836 LOC |
| Total Python LOC | 21,836 |
| ACTIVE modules | 16 |
| LEGACY modules | 5 |
| RESEARCH modules | 5 |
| DEAD modules | 2 |
| Standalone scripts | 3 |
| Critical issue: `realtime_nav_core.py` | 2,826 LOC -- **far exceeds 800-line guideline** |
| Critical issue: `live_heading_demo.py` | 1,278 LOC -- **far exceeds 800-line guideline** |
| Missing `__init__.py` | Package is not a proper Python package |
| Missing `dt_path_planner.py` | Referenced by 2 files but does not exist |
| Empty `path_planners/` directory | Only contains `__pycache__/` |

---

## Root-Level Python Modules (31 files)

### 1. config.py
- **Lines:** 423
- **Status:** ACTIVE (core)
- **Purpose:** Single source of truth for ALL tunable constants (segmentation, BEV, path planning, obstacles, GPS, speeds, safety gates, research flags)
- **Dependencies:** `json`, `os`, `numpy`
- **Imported by:** Nearly every module (20+ importers)
- **Issues:**
  - Contains executable logic (`_resolve_model_dir()`, `_load_bev_calibration_meta()`) mixed with constants -- should be pure constants with lazy initialization
  - `_MODEL_DIR_CANDIDATES` list has 5 hardcoded filesystem paths
  - Stale comment line 252: "BEV_SIZE[1]=540 px covers NAV_BEV_FORWARD_M=9 m" but actual values are BEV_SIZE=(360,660) and NAV_BEV_FORWARD_M=11.0

### 2. realtime_nav_core.py
- **Lines:** 2,826
- **Status:** ACTIVE (core orchestrator)
- **Purpose:** BEV path extraction + adaptive pure pursuit controller; the main pipeline brain
- **Dependencies:** `config`, `template_path_planner`, `path_smoother` (lazy), `safe_corridor` (lazy), `dt_path_planner` (lazy, MISSING), `waypoint_turn_planner` (lazy)
- **Imported by:** `live_heading_demo`, `intent_picker`, tests (`conftest`, `test_realtime_nav_core`, `test_bev_obstacle`, `test_bev_predictor`, `test_template_path_planner`), scripts (`eval_research_improvements`, `eval_hand_annotated_pipeline`)
- **Issues:**
  - **2,826 lines -- 3.5x the 800-line guideline.** Contains 9 classes and dozens of methods
  - `BEVPathExtractor` class alone is ~2,360 lines -- a god object
  - Inline skeleton graph construction (lines 853+, 896+, 1827+) duplicates functionality from `skeleton.py`
  - `AdaptivePurePursuitController` (line 2735+) should be its own module
  - References `dt_path_planner` which does not exist (handled by try/except but misleading)
  - Imports `heapq` for Dijkstra -- this graph search is reimplemented inline rather than using `skeleton.py`

### 3. template_path_planner.py
- **Lines:** 873
- **Status:** ACTIVE (Phase 11 planner)
- **Purpose:** Corridor extraction and ego-anchored template arc approval for BEV masks
- **Dependencies:** `config`, `safe_corridor` (lazy)
- **Imported by:** `realtime_nav_core`, `waypoint_turn_planner`, `intent_picker`, tests, scripts
- **Issues:**
  - Slightly over 800-line limit (873)
  - Duplicate `_resample_polyline()` function also exists in `boundary_inference.py`
  - Duplicate `_clip()`, `_safe_norm()` utility functions exist in multiple modules

### 4. fast_road_detector.py
- **Lines:** 597
- **Status:** ACTIVE
- **Purpose:** SegFormer-based road/sidewalk segmentation inference with logging, ONNX acceleration
- **Dependencies:** `stabilization`, `config`
- **Imported by:** `live_heading_demo`, `intent_picker`, `camera_waypoint_pipeline`, scripts (`benchmark_seg_stability`, `tune_smoother`)
- **Issues:**
  - Contains a standalone `Config` dataclass (line 19) that shadows the module-level `config.py` pattern
  - Has its own `SystemInfo` dataclass (line 42) for hardware profiling -- could be extracted
  - `video_path` default "test_video_june_03_3.MOV" is hardcoded

### 5. live_heading_demo.py
- **Lines:** 1,278
- **Status:** ACTIVE (primary GUI entry point)
- **Purpose:** Full pipeline demo: camera -> SegFormer -> BEV -> path -> heading + speed command with live visualization
- **Dependencies:** `fast_road_detector`, `realtime_nav_core`, `path_planners` (MISSING), `config`, `data_logger`, `scooter_commander`, `gps_navigator`, `object_detector`, `bev_calibration`, `bev_obstacle`, `stabilization`, `masks`, `heading`, `visualization`, `bev_predictor`
- **Imported by:** scripts (`eval_binary_seg_models`, `eval_template_planner`, `eval_waypoint_turn_planner`, `replay_model_on_videos`)
- **Issues:**
  - **1,278 lines -- 1.6x the 800-line guideline**
  - Imports `path_planners` which is an empty directory (handled by try/except)
  - `run_live()` function is likely a monolithic entry point -- scripts import it to run replays
  - Contains both GUI logic and pipeline orchestration -- should be split

### 6. bev_calibration.py
- **Lines:** 254
- **Status:** ACTIVE
- **Purpose:** BEV calibration tool and parameter loader (homography H matrix)
- **Dependencies:** `config`
- **Imported by:** `live_heading_demo`, `intent_picker`, scripts (`calibrate_bev_examples`, `measure_bev_survival`, `eval_hand_annotated_pipeline`)
- **Issues:** None significant

### 7. bev_obstacle.py
- **Lines:** 170
- **Status:** ACTIVE
- **Purpose:** BEV obstacle projection from YOLO detections + temporal EMA smoothing grid
- **Dependencies:** `config`
- **Imported by:** `live_heading_demo`, tests (`test_bev_obstacle`)
- **Issues:** None significant. Well-documented, standalone.

### 8. bev_predictor.py
- **Lines:** 279
- **Status:** ACTIVE
- **Purpose:** Predictive BEV frame reuse -- skip expensive segmentation when scene is predictable
- **Dependencies:** `config`
- **Imported by:** `live_heading_demo`, tests (`test_bev_predictor`)
- **Issues:** None significant

### 9. masks.py
- **Lines:** 386
- **Status:** ACTIVE
- **Purpose:** BEV mask cleaning, splitting, grass suppression, component selection, enhanced morphology (Research Idea 1)
- **Dependencies:** `config`
- **Imported by:** `live_heading_demo`, `intent_picker`, scripts (`eval_simple_road`, `eval_research_improvements`, `eval_hand_annotated_pipeline`)
- **Issues:** None significant

### 10. heading.py
- **Lines:** 165
- **Status:** ACTIVE
- **Purpose:** Heading computation, command classification (straight/left/right/sharp), speed profiling
- **Dependencies:** `config`, `path_smoother` (lazy)
- **Imported by:** `live_heading_demo`, tests (`test_heading`), scripts (`eval_research_improvements`)
- **Issues:** Module-level mutable state (`_heading_filter = None`) -- violates immutability guideline

### 11. object_detector.py
- **Lines:** 103
- **Status:** ACTIVE
- **Purpose:** YOLOv8-nano obstacle detection wrapper
- **Dependencies:** `config`
- **Imported by:** `live_heading_demo`
- **Issues:** None significant. Clean and focused.

### 12. visualization.py
- **Lines:** 227
- **Status:** ACTIVE
- **Purpose:** HUD drawing functions for camera and BEV views
- **Dependencies:** `config`
- **Imported by:** `live_heading_demo`
- **Issues:** None significant

### 13. data_logger.py
- **Lines:** 92
- **Status:** ACTIVE
- **Purpose:** Per-frame CSV logger for thesis experiments
- **Dependencies:** None from package (stdlib only)
- **Imported by:** `live_heading_demo`
- **Issues:** None significant. Clean and focused.

### 14. stabilization.py
- **Lines:** 248
- **Status:** ACTIVE
- **Purpose:** Camera shake compensation (optical-flow stabilizer) + temporal mask smoother (EMA)
- **Dependencies:** `config`
- **Imported by:** `fast_road_detector`, `live_heading_demo`, `intent_picker`, tests (`test_temporal_smoother`), scripts (`benchmark_seg_stability`, `tune_smoother`, `eval_simple_road`)
- **Issues:** None significant

### 15. waypoint_turn_planner.py
- **Lines:** 538
- **Status:** ACTIVE (Phase 11.1)
- **Purpose:** GPS-intent corridor waypoint turn planner
- **Dependencies:** `template_path_planner`, `config`
- **Imported by:** `realtime_nav_core` (lazy), tests (`test_waypoint_turn_planner`)
- **Issues:** None significant. Well-documented.

### 16. gps_navigator.py
- **Lines:** 235
- **Status:** ACTIVE
- **Purpose:** Serial GPS NMEA reader + waypoint follower
- **Dependencies:** `config`
- **Imported by:** `live_heading_demo`
- **Issues:**
  - Uses threading for serial I/O -- no test coverage
  - `pyserial` is optional (graceful degradation)

### 17. path_smoother.py
- **Lines:** 245
- **Status:** RESEARCH (Idea 3)
- **Purpose:** Temporal EMA smoothing on cubic path coefficients + circular EMA on heading
- **Dependencies:** `config`
- **Imported by:** `realtime_nav_core` (lazy), `heading` (lazy), tests (`test_temporal_smoother` -- naming mismatch)
- **Issues:**
  - Test file is named `test_temporal_smoother.py` but tests `path_smoother.py` -- naming inconsistency

### 18. safe_corridor.py
- **Lines:** 381
- **Status:** RESEARCH (Idea 2)
- **Purpose:** Distance-transform Dijkstra corridor extraction
- **Dependencies:** `config`, `scipy` (optional)
- **Imported by:** `realtime_nav_core` (lazy), `template_path_planner` (lazy)
- **Issues:** None significant. Proper lazy import handling.

### 19. simple_road_pipeline.py
- **Lines:** 686
- **Status:** RESEARCH
- **Purpose:** Simplified pipeline tuned for clean simple-road navigation (fewer templates, tighter params)
- **Dependencies:** `config`
- **Imported by:** scripts (`eval_simple_road`)
- **Issues:** Not used by any active pipeline module. Only used by its own eval script.

### 20. image_path_planner.py
- **Lines:** 379
- **Status:** RESEARCH
- **Purpose:** Camera-space path planners (no BEV) -- DT planner and midpoint planner for comparison
- **Dependencies:** None from package (standalone)
- **Imported by:** tests (`test_image_path_planner`), scripts (`eval_hand_annotated_pipeline`)
- **Issues:** None significant. Standalone by design.

### 21. boundary_model.py
- **Lines:** 150
- **Status:** RESEARCH (experimental)
- **Purpose:** Tiny row-wise boundary network (CNN) for left/right sidewalk edge prediction
- **Dependencies:** `torch` only
- **Imported by:** tests (`test_boundary_dataset_model`), scripts (`eval_boundary_net`, `train_boundary_net`)
- **Issues:** Not used by any active pipeline. Pure research experiment.

### 22. boundary_inference.py
- **Lines:** 254
- **Status:** RESEARCH (experimental)
- **Purpose:** Decode row-wise boundary predictions into centerline path + confidence
- **Dependencies:** None from package
- **Imported by:** tests (`test_boundary_inference`), scripts (`eval_boundary_net`)
- **Issues:**
  - Duplicate `_resample_polyline()` function (also in `template_path_planner.py`)
  - Not used by any active pipeline

### 23. boundary_dataset.py
- **Lines:** 119
- **Status:** RESEARCH (experimental)
- **Purpose:** PyTorch Dataset for boundary-target JSONL records
- **Dependencies:** `torch` only
- **Imported by:** tests (`test_boundary_dataset_model`), scripts (`eval_boundary_net`, `train_boundary_net`)
- **Issues:** Not used by any active pipeline

### 24. boundary_targets.py
- **Lines:** 134
- **Status:** RESEARCH (experimental)
- **Purpose:** Boundary target extraction (row-based left/right edge positions) from segmentation masks
- **Dependencies:** None from package
- **Imported by:** tests (`test_boundary_targets`), scripts (`export_boundary_targets`)
- **Issues:** Not used by any active pipeline

### 25. skeleton.py
- **Lines:** 84
- **Status:** DEAD
- **Purpose:** Skeletonization and branch pruning utilities (Guo-Hall thinning)
- **Dependencies:** `config`
- **Imported by:** **NOTHING** -- no file imports this module
- **Issues:**
  - **Completely dead code.** `realtime_nav_core.py` has its own inline skeleton implementation
  - `camera_waypoint_pipeline.py` calls `skeletonize_guohall` but that file is also legacy/dead
  - Should be deleted or the functionality in `realtime_nav_core.py` should be extracted here

### 26. camera_waypoint_pipeline.py
- **Lines:** 392
- **Status:** LEGACY (superseded)
- **Purpose:** Original camera-space road detection + path planning pipeline
- **Dependencies:** `fast_road_detector`
- **Imported by:** **NOTHING**
- **Issues:**
  - **Completely dead code.** Not imported by any file
  - **Hardcodes all constants** (ROAD_ID, src_points, dst_points, bev_size) -- violates config.py rule
  - References old `models/my-segformer-road_new` directory
  - Uses `networkx` for graph algorithms -- the only file that imports `networkx`
  - Should be deleted

### 27. analyze_log.py
- **Lines:** 481
- **Status:** STANDALONE SCRIPT (misplaced)
- **Purpose:** Post-hoc analysis of CSV run logs: statistics, timing plots, heading/speed plots
- **Dependencies:** `pandas`, `matplotlib` (both optional-ish)
- **Imported by:** **NOTHING**
- **Issues:**
  - Should be in `scripts/` directory, not root
  - Self-contained -- not a library module

### 28. eval_cityscapes.py
- **Lines:** 217
- **Status:** STANDALONE SCRIPT (misplaced)
- **Purpose:** Cityscapes sidewalk-class IoU evaluation for thesis SegFormer
- **Dependencies:** `torch`, `transformers`, `PIL`
- **Imported by:** **NOTHING**
- **Issues:**
  - Should be in `scripts/` directory, not root
  - Hardcodes `DEFAULT_MODEL_DIR` to `"models/drivable-segformer-b0"` instead of using config
  - Self-contained -- not a library module

### 29. eval_rugd.py
- **Lines:** 237
- **Status:** STANDALONE SCRIPT (misplaced)
- **Purpose:** RUGD dataset drivable-surface IoU evaluation
- **Dependencies:** `torch`, `transformers`, `PIL`
- **Imported by:** **NOTHING**
- **Issues:**
  - Should be in `scripts/` directory, not root
  - Hardcodes `DEFAULT_MODEL_DIR` instead of using config
  - Self-contained

### 30. scooter_commander.py
- **Lines:** 47
- **Status:** ACTIVE (hardware interface)
- **Purpose:** Serial protocol for sending steering/speed commands to scooter hardware
- **Dependencies:** `pyserial` (optional)
- **Imported by:** `live_heading_demo`
- **Issues:**
  - Uses `print()` for logging instead of `logging` module
  - No test coverage

### 31. intent_picker.py
- **Lines:** 193
- **Status:** STANDALONE TOOL
- **Purpose:** Interactive frame-by-frame intent picker GUI (manual annotation tool)
- **Dependencies:** `fast_road_detector`, `bev_calibration`, `masks`, `stabilization`, `realtime_nav_core`, `template_path_planner`, `config`
- **Imported by:** **NOTHING** (standalone CLI tool)
- **Issues:**
  - Should arguably be in `scripts/`
  - Heavy dependency set for a standalone tool

---

## Scripts Directory (24 files, 6,307 LOC)

### Evaluation Scripts

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `eval_research_improvements.py` | 511 | Compare baseline vs enhanced pipeline (3 research ideas) | USEFUL |
| `eval_simple_road.py` | 750 | Evaluate simplified simple-road pipeline vs baseline | USEFUL |
| `eval_waypoint_turn_planner.py` | 720 | Phase 11.1 replay: baseline vs template vs waypoint turn | USEFUL |
| `eval_template_planner.py` | 223 | Phase 11 replay: graph-first vs template planner | USEFUL |
| `eval_binary_seg_models.py` | 260 | Compare binary SegFormer checkpoint vs baseline | USEFUL |
| `eval_hand_annotated_pipeline.py` | 619 | Research eval on hand-annotated GT frames | USEFUL |
| `eval_boundary_net.py` | 131 | Evaluate tiny boundary network | RESEARCH |

### Training Scripts

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `train_binary_segformer.py` | 471 | Fine-tune SegFormer for binary drivable mask | USEFUL |
| `train_boundary_net.py` | 132 | Train tiny boundary network | RESEARCH |

### Data/Annotation Scripts

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `extract_annotation_frames.py` | 179 | Extract diverse frames from videos for annotation | USEFUL |
| `generate_binary_pseudo_labels.py` | 259 | Generate pseudo-labels with OneFormer teacher | USEFUL |
| `export_boundary_targets.py` | 80 | Export boundary targets from mask records | RESEARCH |
| `prepare_hand_annotation_workspace.py` | 114 | Copy images + masks for manual correction | USEFUL |

### Calibration/Tuning Scripts

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `calibrate_bev_examples.py` | 254 | Interactive BEV calibration on frame examples | USEFUL |
| `benchmark_seg_stability.py` | 309 | Evaluate temporal seg stability across checkpoints | USEFUL |
| `tune_smoother.py` | 253 | Sweep EMA alpha x consistency_thresh grid | USEFUL |
| `tune_binary_threshold.py` | 101 | Sweep segmentation thresholds | USEFUL |
| `learn_turn_schedule.py` | 153 | Learn turn windows from replay CSV logs | NICHE |

### Utility Scripts

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `convert_videos.py` | 97 | Convert MOV to H264 MP4 | USEFUL |
| `make_video_comparison_strips.py` | 101 | Side-by-side video comparisons | USEFUL |
| `replay_model_on_videos.py` | 121 | Replay one model on all test videos | USEFUL |
| `measure_bev_survival.py` | 113 | Measure BEV calibration quality from run logs | NICHE |
| `cityscapes_miou_segformer_b0.py` | 356 | Full Cityscapes mIoU evaluation | USEFUL |

### Script Issues
- `eval_hand_annotated_pipeline.py` line 34: imports `dt_path_planner` which does not exist (will fail unless wrapped in try/except -- it is NOT wrapped)
- Several scripts do `os.sys.path.insert(0, ...)` which is non-standard (should use `sys.path`)
- No `__init__.py` in `scripts/` -- cannot be imported as a package

---

## Tests Directory (13 files, 2,836 LOC)

### Test Coverage Map

| Test File | Lines | Module(s) Tested | Coverage Quality |
|-----------|-------|-------------------|-----------------|
| `conftest.py` | 206 | Shared fixtures (masks, extractors) | Good |
| `test_realtime_nav_core.py` | 279 | `realtime_nav_core` | Moderate -- tests BEVPathExtractor basics |
| `test_template_path_planner.py` | 387 | `template_path_planner`, `heading` | Good |
| `test_waypoint_turn_planner.py` | 729 | `waypoint_turn_planner` | Excellent -- very thorough |
| `test_bev_predictor.py` | 352 | `bev_predictor` | Good |
| `test_bev_obstacle.py` | 174 | `bev_obstacle` | Good |
| `test_bev_calibration.py` | 78 | `bev_calibration` | Basic |
| `test_heading.py` | 115 | `heading` | Moderate |
| `test_temporal_smoother.py` | 244 | `path_smoother`, `stabilization` | Good (naming mismatch) |
| `test_boundary_dataset_model.py` | 87 | `boundary_dataset`, `boundary_model` | Basic |
| `test_boundary_inference.py` | 83 | `boundary_inference` | Basic |
| `test_boundary_targets.py` | 68 | `boundary_targets` | Basic |
| `test_image_path_planner.py` | 34 | `image_path_planner` | Minimal (1 test) |

### Modules with NO Test Coverage

| Module | Lines | Status | Risk |
|--------|-------|--------|------|
| `config.py` | 423 | ACTIVE | Low (mostly constants) |
| `fast_road_detector.py` | 597 | ACTIVE | **HIGH** -- core segmentation, no unit tests |
| `live_heading_demo.py` | 1,278 | ACTIVE | **HIGH** -- main entry point, no unit tests |
| `masks.py` | 386 | ACTIVE | **MEDIUM** -- used by many, tested indirectly only |
| `visualization.py` | 227 | ACTIVE | Low (display only) |
| `data_logger.py` | 92 | ACTIVE | Low (simple I/O) |
| `gps_navigator.py` | 235 | ACTIVE | **MEDIUM** -- threading + serial, no tests |
| `object_detector.py` | 103 | ACTIVE | **MEDIUM** -- YOLO wrapper, no unit tests |
| `scooter_commander.py` | 47 | ACTIVE | Low (simple serial) |
| `safe_corridor.py` | 381 | RESEARCH | **MEDIUM** -- complex Dijkstra, no direct tests |
| `simple_road_pipeline.py` | 686 | RESEARCH | Low (research variant) |
| `skeleton.py` | 84 | DEAD | N/A |
| `camera_waypoint_pipeline.py` | 392 | DEAD | N/A |

### Test Issues
- `test_temporal_smoother.py` tests `path_smoother.py` and `stabilization.py` -- **naming mismatch**
- `test_image_path_planner.py` has only 34 lines / 1 test -- minimal coverage
- No test for `masks.py` despite being imported by many modules
- No test for `fast_road_detector.py` despite being the core perception module
- No integration test that runs the full pipeline end-to-end

---

## Models Directory

| Directory | Size | Purpose | Status |
|-----------|------|---------|--------|
| `my-segformer-road/` | ~14 MB | Original SegFormer road model (3 sub-checkpoints) | ACTIVE |
| `my-segformer-road_new/` | ~14 MB | Newer SegFormer variant (7 sub-checkpoints) | UNCLEAR |
| `checkpoint-500/` through `checkpoint-5000/` | ~14 MB each | Training intermediate checkpoints | CLEANUP CANDIDATE |
| **Total models/** | **756 MB** | | |

### Issues
- **10 loose checkpoint directories** (500, 1000, ..., 5000) at the models root -- likely training artifacts that should be cleaned up
- `my-segformer-road_new/` -- unclear if this is used or superseded
- The `config.py` `_MODEL_DIR_CANDIDATES` does NOT reference any model under `models/` except `my-segformer-road` -- the new checkpoints are under `outputs/training/`

---

## Non-Python Files at Root

| File | Size | Purpose | Status |
|------|------|---------|--------|
| `bev_calibration.npy` | 160 B | Active homography matrix | ACTIVE |
| `bev_calibration_meta.json` | 349 B | Calibration metadata | ACTIVE |
| `bev_calibration_backup_20260312.npy` | 160 B | Backup calibration | CLEANUP |
| `bev_H.npy` | 200 B | **Legacy** homography (pre-calibration tool) | DEAD |
| `bev_Hinv.npy` | 200 B | **Legacy** inverse homography | DEAD |
| `cityscapes_iou_drivable-segformer-b0.json` | 10.6 KB | Evaluation results | SHOULD MOVE to metrics/ |
| `rugd_iou_drivable-segformer-b0.json` | 14.6 KB | Evaluation results | SHOULD MOVE to metrics/ |
| `sample_route.csv` | 352 B | Sample GPS waypoints | ACTIVE |
| `test_video_june_03_3.mp4` | **76 MB** | Test video | SHOULD MOVE to test_videos/ |
| `yolov8n.pt` | **6.3 MB** | YOLOv8 nano weights | ACTIVE but duplicates root-level copy |
| `turn_analysis.png` | 962 KB | Turn analysis visualization | SHOULD MOVE to eval_runs/ |
| `turn_detail.png` | 1.1 MB | Turn detail visualization | SHOULD MOVE to eval_runs/ |
| `turns_survey.png` | 1.2 MB | Turn survey visualization | SHOULD MOVE to eval_runs/ |
| `CALIBRATION_SOP.md` | 3.3 KB | Calibration procedure | REFERENCE |
| `MUST_READ_TURN_CONTAINMENT.md` | 5.3 KB | Turn containment documentation | REFERENCE |
| `PHASE_11_1_CHANGELOG.md` | 3.5 KB | Phase 11.1 changelog | REFERENCE |
| `RUNTIME_RUNBOOK.md` | 7.0 KB | Runtime operations guide | REFERENCE |
| `ReadME.tex` | 88 B | Stub LaTeX readme | DEAD |

---

## Data Directories

| Directory | Contents | Size | Status |
|-----------|----------|------|--------|
| `test_videos/` | 11 video files (MOV + MP4) | ~3.3 GB | ACTIVE -- test corpus |
| `result/` | 1 overlay video (seg_bev_overlay_1931.mp4) | **763 MB** | CLEANUP -- generated output |
| `logs/` | 40+ CSV run logs + meta JSON | ~10 MB | USEFUL for analysis |
| `eval_runs/` | 19 eval output directories | varies | USEFUL for thesis |
| `demo_outputs/` | Debug images + eval directories | ~5 MB | CLEANUP |
| `overnight_runs/` | 2 experiment directories | small | HISTORICAL |
| `metrics/` | 5 JSON performance summaries + boundary smoke dir | small | USEFUL |
| `annotation_frames/` | 4 video frame sets (~100 frames each) | ~40 MB | USEFUL for fine-tuning |
| `intent_schedules/` | 1 JSON turn schedule | tiny | NICHE |
| `path_planners/` | **EMPTY** (only `__pycache__/`) | 0 | DEAD -- should delete |

---

## Dependency Graph (Simplified)

```
config.py  <──────────────────────────────── (imported by everything)
    │
    ├── template_path_planner.py ──── safe_corridor.py [lazy]
    │       │
    │       ├── realtime_nav_core.py ──── path_smoother.py [lazy]
    │       │       │                     waypoint_turn_planner.py [lazy]
    │       │       │                     dt_path_planner.py [MISSING, lazy]
    │       │       │
    │       │       └── live_heading_demo.py
    │       │               ├── fast_road_detector.py ── stabilization.py
    │       │               ├── bev_calibration.py
    │       │               ├── bev_obstacle.py
    │       │               ├── bev_predictor.py
    │       │               ├── masks.py
    │       │               ├── heading.py ── path_smoother.py [lazy]
    │       │               ├── object_detector.py
    │       │               ├── visualization.py
    │       │               ├── data_logger.py
    │       │               ├── scooter_commander.py
    │       │               └── gps_navigator.py
    │       │
    │       └── waypoint_turn_planner.py
    │
    ├── boundary_* cluster (isolated research island):
    │       boundary_model.py
    │       boundary_dataset.py
    │       boundary_inference.py
    │       boundary_targets.py
    │
    └── DEAD: skeleton.py, camera_waypoint_pipeline.py
```

### Circular Dependencies
- **None found.** All lazy imports (try/except) prevent circular import issues.
- The dependency graph is a clean DAG (directed acyclic graph).

---

## Findings and Recommendations

### FILES TO DELETE

| File | Reason |
|------|--------|
| `skeleton.py` | Dead code. Not imported by anything. `realtime_nav_core.py` has its own inline implementation. |
| `camera_waypoint_pipeline.py` | Dead code. Not imported by anything. Superseded by `realtime_nav_core.py`. Hardcodes constants. |
| `bev_H.npy` | Legacy pre-calibration homography. Superseded by `bev_calibration.npy`. |
| `bev_Hinv.npy` | Legacy inverse homography. |
| `ReadME.tex` | 88-byte stub with no content. |
| `path_planners/` (directory) | Empty directory with only `__pycache__/`. |
| `bev_calibration_backup_20260312.npy` | One-off backup. Should be in version control, not as a file. |

### FILES TO MOVE

| File | From | To | Reason |
|------|------|----|--------|
| `analyze_log.py` | root | `scripts/` | Standalone analysis script, not a library module |
| `eval_cityscapes.py` | root | `scripts/` | Standalone evaluation script |
| `eval_rugd.py` | root | `scripts/` | Standalone evaluation script |
| `intent_picker.py` | root | `scripts/` | Standalone GUI tool |
| `test_video_june_03_3.mp4` | root | `test_videos/` | Test video belongs with other test videos |
| `cityscapes_iou_*.json` | root | `metrics/` | Evaluation results |
| `rugd_iou_*.json` | root | `metrics/` | Evaluation results |
| `turn_*.png` (3 files) | root | `eval_runs/` or `metrics/` | Generated analysis images |

### FILES TO SPLIT

| File | Lines | Recommendation |
|------|-------|----------------|
| `realtime_nav_core.py` | 2,826 | Extract `AdaptivePurePursuitController` (line 2735+) into `pure_pursuit.py`. Extract inline skeleton graph code into a proper `graph_path.py`. Consider extracting `BEVPathExtractor` helper methods into `bev_path_helpers.py`. Target: main file ~800 lines. |
| `live_heading_demo.py` | 1,278 | Extract GUI/visualization loop logic from pipeline logic. The `run_live()` function should be a thin wrapper around pipeline calls. |

### FILES TO RENAME

| Current Name | Suggested Name | Reason |
|-------------|----------------|--------|
| `test_temporal_smoother.py` | `test_path_smoother.py` | Tests `path_smoother.py`, not a module called "temporal_smoother" |

### FILES TO MERGE

| Files | Recommendation |
|-------|----------------|
| `boundary_model.py` + `boundary_inference.py` + `boundary_dataset.py` + `boundary_targets.py` | These 4 files form an isolated research cluster (657 LOC total). Consider merging into a single `boundary_network.py` or keeping as-is in a `boundary/` subdirectory. |

### MISSING FILES

| File | Referenced By | Impact |
|------|---------------|--------|
| `dt_path_planner.py` | `realtime_nav_core.py` (lazy), `eval_hand_annotated_pipeline.py` (NOT lazy) | `realtime_nav_core.py` handles gracefully via try/except. **`eval_hand_annotated_pipeline.py` will crash** on import. |
| `path_planners/__init__.py` (+ actual planners) | `live_heading_demo.py` (lazy) | Handled via try/except. Empty directory should be deleted. |
| `__init__.py` | Package structure | `simulation_camera_scooter/` is not a proper Python package. All imports rely on `sys.path` manipulation. |

### CODE QUALITY ISSUES

1. **Duplicate utility functions:**
   - `_resample_polyline()` exists in both `template_path_planner.py` and `boundary_inference.py`
   - `_clip()` exists in `template_path_planner.py` and `image_path_planner.py`
   - `_safe_norm()` exists in `template_path_planner.py`
   - Should be extracted to a shared `utils.py`

2. **Mutable module-level state:**
   - `heading.py` line 39: `_heading_filter = None` -- global mutable singleton
   - Violates immutability rule from coding style guide

3. **Hardcoded values violating config.py rule:**
   - `camera_waypoint_pipeline.py` -- hardcodes ROAD_ID, src_points, dst_points, bev_size
   - `fast_road_detector.py` -- hardcodes `video_path` default
   - `eval_cityscapes.py` -- hardcodes DEFAULT_MODEL_DIR
   - `eval_rugd.py` -- hardcodes DEFAULT_MODEL_DIR

4. **Stale comments:**
   - `config.py` line 252: comment says "BEV_SIZE[1]=540" but actual BEV_SIZE=(360,660), so BEV_SIZE[1]=660

5. **print() instead of logging:**
   - `scooter_commander.py` uses `print()` for all output
   - `camera_waypoint_pipeline.py` uses `print()` for status

6. **Large binary files in repo:**
   - `test_video_june_03_3.mp4` (76 MB) at root
   - `yolov8n.pt` (6.3 MB) at root AND likely duplicated at project root
   - `result/seg_bev_overlay_1931.mp4` (763 MB) -- generated output in repo
   - `models/` (756 MB) with 10 training checkpoint directories
   - `test_videos/` (3.3 GB) -- should be in `.gitignore` or LFS

### MISSING TEST COVERAGE (HIGH PRIORITY)

| Module | Lines | Risk | Recommendation |
|--------|-------|------|----------------|
| `fast_road_detector.py` | 597 | HIGH | Add unit tests for `FastRoadDetector` initialization, inference mocking |
| `masks.py` | 386 | MEDIUM | Add tests for `clean_bev_mask_enhanced`, `select_main_component`, `ego_connected_mask` |
| `gps_navigator.py` | 235 | MEDIUM | Add tests for waypoint distance calc, NMEA parsing (mock serial) |
| `object_detector.py` | 103 | MEDIUM | Add tests for detection filtering, distance estimation |
| `safe_corridor.py` | 381 | MEDIUM | Add tests for corridor extraction on synthetic masks |

---

## Size Summary

| Category | Files | LOC | % of Total |
|----------|-------|-----|-----------|
| Core pipeline (ACTIVE) | 16 | 9,387 | 43% |
| Demo/GUI (ACTIVE) | 1 | 1,278 | 6% |
| Research (experimental) | 5 | 1,845 | 8% |
| Legacy/Dead | 2 | 476 | 2% |
| Standalone scripts (misplaced) | 4 | 1,128 | 5% |
| Scripts directory | 24 | 6,307 | 29% |
| Tests | 13 | 2,836 | 13% |
| **Total** | **65** | **21,836** | **100%** |

### Disk Usage Summary

| Item | Size |
|------|------|
| Python source code | ~1.5 MB |
| Model checkpoints (`models/`) | 756 MB |
| Test videos (`test_videos/`) | 3.3 GB |
| Root test video | 76 MB |
| Generated outputs (`result/`) | 763 MB |
| YOLO weights | 6.3 MB |
| **Total directory** | **~5+ GB** |
