# Directory Structure — Autonomous Scooter Navigation

## Project Layout

```
live_test_scooter_project/
├── simulation_camera_scooter/           # Main Python package (67 .py modules)
│   ├── config.py                        # **Single source of truth for ALL constants
│   ├── realtime_nav_core.py             # **Pipeline orchestrator (entry point)
│   ├── live_heading_demo.py             # Standalone GUI demo
│   ├── simple_road_pipeline.py          # Research variant (simplified binary mask)
│   │
│   ├── [Segmentation & Model]
│   ├── fast_road_detector.py            # SegFormer inference + temporal smoothing
│   ├── stabilization.py                 # Camera shake compensation
│   ├── bev_calibration.py               # Homography matrix management
│   ├── bev_*.npy                        # Calibration matrices (binary numpy)
│   ├── bev_calibration_meta.json        # Calibration metadata
│   │
│   ├── [BEV & Path Extraction]
│   ├── bev_predictor.py                 # Predictive frame reuse via motion
│   ├── bev_obstacle.py                  # Obstacle projection to BEV space
│   ├── masks.py                         # Morphological mask cleaning
│   ├── skeleton.py                      # Medial-axis skeleton (legacy)
│   ├── safe_corridor.py                 # DT-based corridor (research)
│   │
│   ├── [Path Planning]
│   ├── template_path_planner.py         # **Corridor + template arc approval
│   ├── waypoint_turn_planner.py         # **GPS-intent turn planning (Phase 11.1)
│   ├── dt_path_planner.py               # DT Ridge planner (research fallback)
│   ├── image_path_planner.py            # Image-space planner (legacy)
│   │
│   ├── [Obstacle & Intent]
│   ├── object_detector.py               # YOLOv8n inference wrapper
│   ├── heading.py                       # Heading estimation from path
│   ├── path_smoother.py                 # Temporal EMA on path/heading
│   ├── gps_navigator.py                 # GPS waypoint following
│   ├── intent_picker.py                 # Intent type classification
│   ├── camera_waypoint_pipeline.py      # Image-space waypoint planner
│   │
│   ├── [Visualization & Logging]
│   ├── visualization.py                 # HUD overlays (camera + BEV)
│   ├── data_logger.py                   # Session telemetry logging
│   ├── analyze_log.py                   # Post-processing analysis
│   │
│   ├── [Boundary Detection (Experimental)]
│   ├── boundary_model.py                # Road boundary segmentation model
│   ├── boundary_inference.py            # Boundary inference
│   ├── boundary_dataset.py              # Boundary dataset loading
│   ├── boundary_targets.py              # Boundary target generation
│   │
│   ├── [Hardware Interface]
│   ├── scooter_commander.py             # Motor control (if present)
│   │
│   ├── tests/                           # Pytest suite (14 test files)
│   ├── conftest.py                      # **Shared pytest fixtures
│   ├── test_realtime_nav_core.py        # Integration tests (core pipeline)
│   ├── test_template_path_planner.py    # Template planner unit tests
│   ├── test_waypoint_turn_planner.py    # Waypoint turn planner tests
│   ├── test_bev_obstacle.py             # Obstacle projection tests
│   ├── test_bev_predictor.py            # BEV prediction tests
│   ├── test_heading.py                  # Heading estimation tests
│   ├── test_temporal_smoother.py        # Temporal filter tests
│   ├── test_bev_calibration.py          # BEV calibration tests
│   ├── test_image_path_planner.py       # Image space planner tests
│   ├── test_boundary_*.py               # Boundary network tests (3 files)
│   │
│   ├── scripts/                         # Evaluation, training, analysis scripts
│   ├── eval_waypoint_turn_planner.py    # **Evaluate Phase 11.1 turn planning
│   ├── eval_template_planner.py         # Evaluate Phase 11 template planner
│   ├── eval_research_improvements.py    # Compare research implementations
│   ├── eval_simple_road.py              # Evaluate simplified pipeline
│   ├── replay_model_on_videos.py        # Inference on test videos
│   ├── benchmark_seg_stability.py       # Segmentation stability benchmarks
│   ├── train_binary_segformer.py        # **Binary segmentation training
│   ├── tune_binary_threshold.py         # **Confidence threshold tuning
│   ├── train_boundary_net.py            # Boundary network training
│   ├── tune_smoother.py                 # Temporal smoother tuning
│   ├── generate_binary_pseudo_labels.py # OneFormer teacher → pseudo labels
│   ├── calibrate_bev_examples.py        # BEV calibration data collection
│   ├── extract_annotation_frames.py     # Extract fine-tuning samples
│   ├── eval_cityscapes.py               # Cityscapes evaluation
│   ├── eval_rugd.py                     # RUGD outdoor segmentation eval
│   ├── eval_boundary_net.py             # Boundary network evaluation
│   ├── convert_videos.py                # Video format conversion
│   ├── make_video_comparison_strips.py  # Side-by-side video export
│   ├── measure_bev_survival.py          # BEV mask quality metrics
│   ├── learn_turn_schedule.py           # Learn turn timing schedule
│   ├── export_boundary_targets.py       # Export boundary ground truth
│   ├── prepare_hand_annotation_workspace.py  # Annotation setup
│   │
│   ├── models/                          # Model checkpoints (directory)
│   ├── my-segformer-road/               # Default SegFormer checkpoint
│   ├── │
│   ├── annotation_frames/               # Fine-tuning frame extracts (by video)
│   ├── IMG_1878/, IMG_1921/, etc.       # Individual frame packages
│   │
│   ├── demo_outputs/                    # Output from live demos
│   ├── eval_runs/                       # Evaluation run results
│   ├── overnight_runs/                  # Long-running overnight evals
│   ├── path_planners/                   # Alternative planner implementations
│   ├── result/                          # Demo/test output directory
│   ├── logs/                            # Session logs (CSV + MP4)
│   ├── metrics/                         # Computed metric files
│   ├── test_videos/                     # Test video files
│   ├── intent_schedules/                # Learned turn schedules
│   │
│   ├── [Documentation]
│   ├── CALIBRATION_SOP.md               # BEV calibration standard operating procedure
│   ├── MUST_READ_TURN_CONTAINMENT.md    # Turn path containment algorithm
│   ├── RUNTIME_RUNBOOK.md               # Field deployment checklist
│   ├── PHASE_11_1_CHANGELOG.md          # Phase 11.1 changes (waypoint turns)
│   ├── ReadME.tex                       # LaTeX notes
│   │
│   └── .claude/                         # Claude Code workspace (project-local)
│       ├── agents/                      # Custom task agents
│       ├── commands/                    # Slash command definitions
│       ├── hooks/                       # Automation hooks
│       └── .claude/                     # Claude configuration
│
├── .planning/                           # Planning & analysis directory
│   └── codebase/                        # **Codebase documentation
│       ├── ARCHITECTURE.md              # **THIS FILE: system design overview
│       ├── STRUCTURE.md                 # **THIS FILE: directory layout
│       └── [Additional future docs]
│
├── thesis/                              # LaTeX paper (do not modify)
│   ├── main.tex
│   ├── figures/                         # Thesis figures & graphs
│   └── tools/                           # Build scripts
│
├── outputs/                             # Training outputs
│   └── training/                        # Checkpoints from training scripts
│       ├── binary_segformer_oneformer_teacher/
│       ├── binary_segformer_old400_img1931_vid017_020/
│       └── [other checkpoint directories]
│
├── research/                            # Research-specific outputs
│   ├── [research improvement experiments]
│   └── [evaluation results]
│
├── .claude/                             # Claude Code global config (shared)
│   ├── CLAUDE.md                        # **Project instructions
│   ├── settings.json                    # Configuration
│   ├── agents/                          # Custom agents
│   ├── commands/                        # Global commands
│   ├── rules/                           # Coding rules
│   │   ├── common/                      # Common rules (all languages)
│   │   │   ├── coding-style.md          # Immutability, file organization
│   │   │   ├── git-workflow.md          # Commit format, PR workflow
│   │   │   ├── security.md              # Security checklist
│   │   │   └── testing.md               # Test coverage requirements
│   │   └── python/                      # Python-specific rules
│   │       ├── coding-style.md          # PEP 8, type annotations, immutability
│   │       └── testing.md               # pytest patterns
│   ├── get-shit-done/                   # GSD framework (deprecated)
│   └── hooks/                           # Pre-commit hooks
│
├── .git/                                # Git repository
├── .gitignore                           # Git ignore patterns
│
├── [Documentation Files]
├── CLAUDE.md                            # **Project instructions (root copy)
├── RESEARCH_REVIEW.md                   # Research ideas review
├── IMPLEMENTATION_PLAN.md               # Implementation roadmap
├── EVALUATION_REPORT.md                 # Research evaluation results
├── FINAL_SUMMARY.md                     # Research summary
├── TRAINING_LOG.md                      # Binary segmentation training log
├── CHANGELOG_RESEARCH_IMPL.md           # Research changes
├── NEXT_STEPS.md                        # Future work
├── PHASE11_RESEARCH.md                  # Phase 11 research notes
├── PHASE11_RESULTS.md                   # Phase 11 evaluation
├── PHASE11_VIDEO_EVAL.md                # Phase 11 video testing
├── PHASE11_WORKLOG.md                   # Phase 11 work log
│
└── [Model Artifacts]
    ├── bev_calibration.npy              # Homography matrix at project root
    ├── yolov8n.pt                       # YOLOv8n pretrained weights (6.5 MB)
    └── [other binary files]
```

---

## Naming Conventions

### Python Modules

| Pattern | Purpose | Example |
|---------|---------|---------|
| `*_core.py` | Main orchestrator | `realtime_nav_core.py` |
| `*_planner.py` | Path planning | `template_path_planner.py`, `waypoint_turn_planner.py` |
| `*_detector.py` | Detection/inference | `object_detector.py`, `fast_road_detector.py` |
| `*_navigator.py` | GPS/navigation | `gps_navigator.py` |
| `*_picker.py` | Selection logic | `intent_picker.py` |
| `*_model.py` | Neural network wrappers | `boundary_model.py` |
| `*_smoother.py` | Temporal filtering | `path_smoother.py` |
| `test_*.py` | Pytest files | `test_realtime_nav_core.py` |
| `eval_*.py` | Evaluation scripts | `eval_waypoint_turn_planner.py` |
| `train_*.py` | Training scripts | `train_binary_segformer.py` |

### Data Classes

| Convention | Purpose | Example |
|------------|---------|---------|
| `*Result` | Immutable output | `TemplateApprovalResult`, `WaypointTurnResult` |
| `*Config` | Configuration (frozen dataclass) | `WaypointTurnPlannerConfig` |
| `*Target` | Planning candidate | `WaypointTurnTarget` |
| `*Corridor` | Path boundary | `Corridor`, `DtCorridorResult` |
| `*Grid` | Spatial accumulation | `ObstacleEMAGrid` |

### Configuration Constants

| Scope | Location | Example |
|-------|----------|---------|
| Segmentation | `config.py` lines 13–91 | `MODEL_DIR`, `SEG_INPUT_RES` |
| BEV | `config.py` lines 93–141 | `BEV_SIZE`, `BEV_EGO_X_FRAC` |
| Path tuning | `config.py` lines 143–149 | `DT_CORE_THRESH`, `PRUNE_BRANCH_LEN` |
| Heading thresholds | `config.py` lines 151–154 | `HEADING_STRAIGHT_THRESH` |
| Speed profile | `config.py` lines 156–163 | `SPEED_MAX`, `SPEED_TURN` |
| Obstacle | `config.py` lines 165–189 | `OBSTACLE_CLASSES`, `YOLO_CONF_THRESH` |
| GPS | `config.py` lines 191–196 | `EARTH_RADIUS_M`, `GPS_STEER_GAIN` |
| Colors | `config.py` lines 198–212 | `COLOR_STRAIGHT`, `PATH_COLORS` |
| Stabilization | `config.py` lines 214–220 | `STABILIZATION_ENABLED` |
| Masks & BEV | `config.py` lines 222–286 | `MASK_SMOOTH_ALPHA`, `MORPH_ENHANCED` |
| Safety gates | `config.py` lines 229–235 | `SEG_IOU_FAIL`, `SPEED_SEG_UNSTABLE` |
| Research flags | `config.py` lines 268–414 | `MORPH_ENHANCED`, `DT_CORRIDOR_ENABLED`, `PATH_SMOOTH_ENABLED` |
| Waypoint turn | `config.py` lines 339–423 | `WAYPOINT_DECISION_BAND_MIN_M`, `WAYPOINT_TURN_ENABLED` |

**Rule**: Never hardcode values in modules. Use `config.py` imports instead.

---

## Key File Purposes

### **Critical (Never Skip)**

| File | Purpose | Size |
|------|---------|------|
| `config.py` | Single source of truth for all constants | 424 lines |
| `realtime_nav_core.py` | Pipeline orchestrator (entry point) | 3,600+ lines |
| `template_path_planner.py` | Corridor + template arc approval | 400+ lines |
| `waypoint_turn_planner.py` | GPS-intent turn planner (Phase 11.1) | 600+ lines |

### **High-Priority (Often Modified)**

| File | Purpose | Size |
|------|---------|------|
| `fast_road_detector.py` | SegFormer inference + temporal smoothing | 750+ lines |
| `bev_obstacle.py` | Obstacle projection to BEV | 200+ lines |
| `masks.py` | BEV mask cleaning (research variants) | 350+ lines |
| `heading.py` | Heading from path curvature | 150+ lines |
| `path_smoother.py` | Temporal EMA on path/heading | 250+ lines |

### **Support (Imported as Needed)**

| File | Purpose | Size |
|------|---------|------|
| `bev_calibration.py` | Homography matrix I/O | 300+ lines |
| `bev_predictor.py` | Predictive frame reuse | 350+ lines |
| `object_detector.py` | YOLOv8 wrapper | 100+ lines |
| `gps_navigator.py` | GPS waypoint following | 250+ lines |
| `intent_picker.py` | Intent classification | 200+ lines |
| `skeleton.py` | Medial-axis extraction (legacy) | 400+ lines |
| `visualization.py` | HUD drawing | 300+ lines |
| `data_logger.py` | Telemetry logging | 150+ lines |
| `stabilization.py` | Camera shake compensation | 200+ lines |

### **Research (Feature Flags)**

| File | Purpose | Flag | Size |
|------|---------|------|------|
| `safe_corridor.py` | DT-based corridor | `DT_CORRIDOR_ENABLED` | 250+ lines |
| `dt_path_planner.py` | DT Ridge fallback planner | `DT_PLANNER_ENABLED` | 300+ lines |
| `simple_road_pipeline.py` | Simplified binary pipeline | — | 400+ lines |

### **Tests**

| File | Coverage | Lines |
|------|----------|-------|
| `conftest.py` | Shared pytest fixtures | 200+ |
| `test_realtime_nav_core.py` | Integration tests (core pipeline) | 400+ |
| `test_template_path_planner.py` | Template planner unit tests | 450+ |
| `test_waypoint_turn_planner.py` | Waypoint turn planner unit tests | 800+ |
| `test_bev_obstacle.py` | Obstacle projection unit tests | 300+ |
| `test_bev_predictor.py` | BEV prediction tests | 400+ |
| `test_heading.py` | Heading estimation tests | 150+ |
| `test_temporal_smoother.py` | Path/heading filter tests | 300+ |
| Others (5 files) | Boundary, calibration, image planner | ~1000 total |

---

## Directory Map: Where Things Live

### Models & Checkpoints

```
simulation_camera_scooter/models/
├── my-segformer-road/           # Default SegFormer checkpoint (from config.py)
│   ├── config.json              # SegFormer config
│   ├── pytorch_model.bin        # Model weights
│   ├── preprocessor_config.json
│   └── training_summary.json    # Validation IoU reference
│
outputs/training/                # (project root) Checkpoints from training
├── binary_segformer_oneformer_teacher/
│   ├── best_checkpoint/
│   │   ├── pytorch_model.bin
│   │   ├── config.json
│   │   └── training_summary.json
├── binary_segformer_old400_img1931_vid017_020/
└── [other checkpoints ranked by validation IoU]
```

**Lookup logic** (config.py line 75–85): Auto-selects highest validation IoU checkpoint.

### Calibration & Metadata

```
simulation_camera_scooter/
├── bev_calibration.npy              # 4-point homography matrix (binary)
├── bev_calibration_meta.json        # Metadata (ego_x_frac, source video)
├── bev_H.npy, bev_Hinv.npy         # Legacy format (still supported)
└── bev_calibration_backup_*.npy    # Snapshots from calibration runs

[Project root]/
└── bev_calibration.npy              # Shared copy
```

### Test & Demo Data

```
simulation_camera_scooter/
├── test_videos/                 # Test video files (.mp4, .MOV)
├── annotation_frames/           # Fine-tuning frame extracts
│   ├── IMG_1878/
│   │   ├── frame_*.jpg          # Extracted frames
│   │   └── metadata.json        # Frame info
│   └── [other video directories]
├── demo_outputs/                # Output from live demos
│   ├── baseline_june_intent_gui/
│   └── [other demo runs]
└── logs/                        # Session logs
    └── [video_name]/
        ├── telemetry.csv        # Frame-by-frame metrics
        ├── overlay.mp4          # Video with HUD
        └── manifest.json        # Session metadata
```

### Training & Evaluation

```
scripts/
├── train_binary_segformer.py    # **Binary segmentation training
├── tune_binary_threshold.py     # **Confidence threshold optimization
├── eval_waypoint_turn_planner.py # **Phase 11.1 evaluation
├── eval_template_planner.py     # Phase 11 evaluation
├── eval_research_improvements.py # A/B test research variants
└── [12+ other evaluation scripts]

outputs/                         # Training outputs
└── training/
    ├── binary_segformer_oneformer_teacher/
    │   ├── best_checkpoint/
    │   ├── training_summary.json # Best validation IoU
    │   └── logs/
    ├── binary_segformer_old400_*/
    └── [other training runs]
```

### Documentation

```
[Project root]/
├── CLAUDE.md                    # **Project instructions
├── RESEARCH_REVIEW.md           # Research ideas review
├── IMPLEMENTATION_PLAN.md       # Research implementation roadmap
├── EVALUATION_REPORT.md         # Results of research improvements
├── FINAL_SUMMARY.md             # Research summary
├── TRAINING_LOG.md              # Binary segmentation training log
├── CHANGELOG_RESEARCH_IMPL.md   # Changes from research work
├── NEXT_STEPS.md                # Future work recommendations
├── PHASE11_*.md                 # Phase 11 documentation (4 files)
└── [other research docs]

simulation_camera_scooter/
├── CALIBRATION_SOP.md           # BEV calibration procedure
├── MUST_READ_TURN_CONTAINMENT.md # Turn path containment algorithm
├── RUNTIME_RUNBOOK.md           # Field deployment checklist
├── PHASE_11_1_CHANGELOG.md      # Phase 11.1 (waypoint turns) changes
└── ReadME.tex                   # LaTeX notes

.planning/codebase/             # **THIS SECTION
├── ARCHITECTURE.md              # System design overview
└── STRUCTURE.md                 # Directory layout (THIS FILE)
```

---

## Dependency Graph (Import Structure)

### Core Pipeline

```
realtime_nav_core.py              # Entry point
├── config.py                      # Configuration (no dependencies)
├── fast_road_detector.py          # Segmentation
│   └── config.py
├── bev_calibration.py             # BEV homography
│   └── config.py
├── template_path_planner.py       # Template planner
│   ├── config.py
│   └── safe_corridor.py (optional)
├── waypoint_turn_planner.py       # Turn planner
│   ├── config.py
│   ├── template_path_planner.py
│   └── [indirect: config]
├── dt_path_planner.py (optional)  # DT ridge fallback
├── object_detector.py             # Obstacle detection
│   └── config.py
├── bev_obstacle.py                # Obstacle projection
│   └── config.py
├── heading.py                     # Heading estimation
│   └── config.py
├── path_smoother.py               # Temporal smoothing
│   └── config.py
├── masks.py                       # Mask cleaning
│   └── config.py
├── visualization.py               # HUD rendering
│   └── config.py
├── data_logger.py                 # Telemetry
│   └── config.py
└── [etc.]
```

### No Circular Dependencies
- **Rule**: Modules only import config.py + specific utilities, never reverse
- **Exception**: Optional research modules (guarded by try-except + flags)

---

## Convention Checklist

### Code Organization
- ✅ Many small files (200–400 lines typical, max 800)
- ✅ High cohesion (one responsibility per module)
- ✅ Low coupling (config.py as single import source)
- ✅ No hardcoded constants (all in config.py)

### Data Immutability
- ✅ Output classes use `@dataclass(frozen=True)`
- ✅ Numpy arrays returned as new objects (not modified)
- ✅ Configuration uses frozen dataclasses

### Type Safety
- ✅ All function signatures have type annotations
- ✅ Return types always specified
- ✅ Numpy arrays typed (`np.ndarray`, shape hints in docstrings)

### Error Handling
- ✅ Try-except at module boundaries (imports)
- ✅ Feature flags for optional modules (`_HAS_*` patterns)
- ✅ Fail-safe defaults (never None returns)
- ✅ Explicit logging of errors

### Testing
- ✅ Pytest framework (conftest.py fixtures)
- ✅ Unit tests per module (test_*.py)
- ✅ Integration tests in test_realtime_nav_core.py
- ✅ Target coverage ≥80%

---

## Quick Reference: Where to Find Things

| "I want to..." | File/Directory |
|---|---|
| Understand the whole system | `.planning/codebase/ARCHITECTURE.md` |
| Change speed thresholds | `config.py` lines 159–163 |
| Improve path smoothing | `path_smoother.py` + `config.py` lines 289–298 |
| Add a new planner | Create `*_planner.py` in simulation_camera_scooter/, import in realtime_nav_core.py |
| Train a new segmentation model | `scripts/train_binary_segformer.py` |
| Evaluate a planner | `scripts/eval_*_planner.py` |
| Calibrate BEV | `simulation_camera_scooter/bev_calibration.py` or `scripts/calibrate_bev_examples.py` |
| Run the demo | `python simulation_camera_scooter/live_heading_demo.py` |
| Review safety gates | `realtime_nav_core.py` + `config.py` lines 229–235, 380–391 |
| Check test coverage | `pytest simulation_camera_scooter/tests/ --cov=simulation_camera_scooter --cov-report=term-missing` |
| View field logs | `simulation_camera_scooter/logs/[video_name]/telemetry.csv` + `overlay.mp4` |
| Learn about Phase 11.1 | `simulation_camera_scooter/MUST_READ_TURN_CONTAINMENT.md`, `config.py` lines 339–423 |
| Understand research improvements | `.planning/codebase/ARCHITECTURE.md` section "Enhancement options" + root docs |
