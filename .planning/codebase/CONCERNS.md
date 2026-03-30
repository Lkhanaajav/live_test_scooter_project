# Codebase Concerns & Technical Debt

**Date:** 2026-03-30
**Repository:** live_test_scooter_project (autonomous scooter navigation)
**Main code:** `simulation_camera_scooter/` (21.8k lines, 31 Python modules)

---

## 1. Code Quality Issues

### 1.1 Oversized Modules

Three files exceed the 800-line limit and violate single-responsibility principle:

| File | Lines | Main Classes/Functions | Issue |
|------|-------|------------------------|-------|
| `realtime_nav_core.py` | 2,826 | 9 classes, 50+ methods | Core path extraction + pure pursuit controller combined; difficult to test independently; tight coupling between skeleton/template/medial-axis logic and control loop |
| `live_heading_demo.py` | 1,278 | 1 class, 30+ methods | Full pipeline orchestrator: camera I/O, segmentation, BEV transform, obstacle detection, GPS navigation, control output; impossible to unit test or reuse components |
| `template_path_planner.py` | 873 | 7 classes, 40+ methods | Template bank, approval logic, corridor extraction; mixes geometric computation with approval heuristics |

**Impact:**
- High cognitive load for new contributors
- Difficult to isolate bugs
- Cannot reuse individual path-planning strategies without pulling in control logic
- Testing requires full-stack setup

**Recommendation:**
- Extract `BEVPathExtractor` → `bev_path_extraction.py`
- Extract `AdaptivePurePursuitController` → `pure_pursuit_controller.py`
- Move corridor logic → `corridor_extraction.py`

---

### 1.2 Oversized Functions

Four functions exceed 100 lines, indicating complexity that should be decomposed:

| Module | Function | Lines | Concern |
|--------|----------|-------|---------|
| `fast_road_detector.py` | `process_video()` | 110+ | Video processing loop with inline model inference, frame processing, logging all interleaved |
| `bev_calibration.py` | `run_calibration()` | 169+ | Interactive calibration with UI rendering, event handling, matrix computation mixed together |
| `template_path_planner.py` | `approve_template_bank()` | 142+ | Complex approval logic with nested loops, multiple thresholds, unclear branching |
| `boundary_inference.py` | (multiple) | 131+ | Boundary model inference with shape validation, tensor operations, model loading |

**Impact:**
- Difficult to follow control flow
- Easy to miss edge cases during modification
- Hard to unit test individual steps
- Violates "max 50 lines per function" guidance

---

### 1.3 Missing Type Annotations

While most functions have type hints, several patterns lack them:

```python
# From realtime_nav_core.py, line 626 - broad exception handling
try:
    pts_m = path_model.sample_xy(ds_m=max(0.08, float(self.cfg.path_sample_ds_m)))
except Exception:  # ← TOO BROAD
    return np.zeros((0, 2), dtype=np.int32)
```

Found ~8 instances of bare `except Exception:` in:
- `config.py`: lines 30, 38, 64, 131
- `bev_calibration.py`: lines 30, 38, 227
- `object_detector.py`: line 20
- `realtime_nav_core.py`: lines 626, 1644, 2031
- `safe_corridor.py`: line 203

**Impact:**
- Silently swallows real errors (import failures, attribute errors, etc.)
- Masks debugging during development
- Makes code fragile when dependencies change

---

## 2. Technical Debt & Design Issues

### 2.1 Hardcoded Values in Non-Config Files

Seven files contain duplicated or hardcoded constants that belong in `config.py`:

| File | Line(s) | Hardcoded Value | Should Use |
|------|---------|-----------------|-----------|
| `camera_waypoint_pipeline.py` | 29–50 | `ROAD_ID=1, SIDEWALK_ID=2, CAMERA_WIDTH=1280, CAMERA_HEIGHT=720, CAMERA_FPS=30` | Import from `config.py` |
| `fast_road_detector.py` | 21–36 | `conf_thresh=0.6, frame_step=2, smoothing_weight=0.2` | Should reference config |
| `eval_cityscapes.py` | 97–104 | `CITYSCAPES_ROAD_ID=7, CITYSCAPES_SIDEWALK_ID=8` | Move to config.py |
| `eval_rugd.py` | 109–120 | `DRIVABLE_COLORS={...}, DEVICE="cuda"` | Move to config.py |

**Risk:**
- Duplicate constants = inconsistent behavior across pipelines
- Hard to adjust thresholds globally without hunting through files
- Violates CLAUDE.md design rule: "config.py is the single source of truth"

---

### 2.2 Missing Error Recovery & Fallback Patterns

Several critical paths lack graceful degradation:

| Module | Issue | Example |
|--------|-------|---------|
| `object_detector.py` | No fallback when YOLOv8 model download fails | `self.model = None` set, but calling `.detect()` returns `[]` without logging |
| `fast_road_detector.py` | No handling if segmentation inference fails | Model loading errors crash the pipeline |
| `config.py` | Model resolution logic is silent when no checkpoints found | Returns fallback dir without warning if all preferred models missing |
| `bev_calibration.py` | JSON corruption in metadata not explicitly handled | Bare `except Exception: return {}` masks real issues |

**Impact:**
- Silent failures in real-time operation (no log message = hard to debug on RPi)
- Scooter behavior becomes unpredictable when models fail
- No distinguish between "model not found" vs. "I/O error" vs. "corrupted weights"

---

### 2.3 Fragile Test Suite (3 Failing Tests)

**Current:** 154 passing, **3 failing**

#### Test Failure 1: `test_bev_predictor.py::TestOnComputeFrame::test_first_frame`
```
TypeError: PathPlanResult.__init__() missing 1 required positional argument: 'control_path_px'
```
- Root cause: Test fixture `_make_path_result()` does not match refactored `PathPlanResult` dataclass signature
- Impact: Path predictor tests cannot run; regression risk on frame-skipping logic

#### Test Failure 2: `test_bev_predictor.py::TestOnComputeFrame::test_blend_after_skips`
- Same root cause as above
- Both tests use stale fixture

#### Test Failure 3: `test_waypoint_turn_planner.py::TestReplayManifestParsing::test_replay_set_file_can_be_loaded`
```
FileNotFoundError: .planning/phases/11.1-gps-intent-corridor-waypoint-turn-planner/11.1-REPLAY_SET.txt
```
- Root cause: Test expects file in `simulation_camera_scooter/.planning/...` but actual file is at project root `.planning/...`
- Impact: Replay manifest loading untested; waypoint-turn evaluation may silently fail

**Recommendation:**
- Fix test fixtures to match current dataclass signatures
- Update file path resolution to use project root instead of module-relative paths

---

### 2.4 Import Cycles & Soft Failures

Optional imports with graceful fallback in `realtime_nav_core.py` (lines 42–79):

```python
try:
    from path_smoother import PathTemporalSmoother as _PathTemporalSmoother
    _HAS_PATH_SMOOTHER = True
except ImportError:
    _HAS_PATH_SMOOTHER = False
    _PathTemporalSmoother = None  # type: ignore[assignment,misc]
```

**Issues:**
1. `# type: ignore` comments bypass mypy (13 instances across codebase)
2. Feature flags like `_HAS_PATH_SMOOTHER` allow code to degrade silently
3. Hard to tell at runtime why a module didn't load (import error vs. module doesn't exist)

**Risk:** Research features (path smoothing, DT corridor) can silently disable without warning

---

## 3. Performance & Real-Time Concerns

### 3.1 Blocking Operations in Main Loop

`realtime_nav_core.py` and `live_heading_demo.py` perform I/O without async:

| Operation | Blocking | Impact |
|-----------|----------|--------|
| Model loading (SegFormer, YOLOv8) | Yes (startup only) | ~5–10 seconds on RPi 4; scooter unresponsive during boot |
| GPU memory allocation | Yes | ~2–3 seconds; can cause frame drops |
| JSON checkpoint loading in config resolution | Yes (startup) | Multiple file reads in `_resolve_model_dir()`; not parallelized |
| BEV calibration matrix computation | Yes | Perspective transform computed per frame, not cached |

**Recommendation:**
- Cache BEV matrices per frame size
- Pre-allocate GPU memory on startup
- Profile on actual RPi 4 hardware

### 3.2 Memory Bloat from Lazy Dependencies

Large libraries loaded but not always used:

| Library | Size (approx) | Usage Pattern | Concern |
|---------|---------------|---------------|---------|
| `torch` | 1.5 GB | Loaded by all detector modules | Even CPU-only path loads full GPU support |
| `transformers` | 200 MB | SegFormer model loading | Unused code if model fails to download |
| `ultralytics` | 50 MB | YOLOv8 wrapper | Optional; should be lazy-loaded |

**RPi 4 Impact:**
- Only 4 GB RAM available
- Swapping kills performance below 10 Hz target
- No lazy loading of heavy deps

---

## 4. Security & Input Validation Gaps

### 4.1 Unsafe File Operations

Most file opens lack proper encoding/error handling:

```python
# From eval_rugd.py, line 185
pil_img = Image.open(local_img).convert("RGB")  # No try/except; will crash if file corrupted

# From data_logger.py, line 58
self._file = open(self.csv_path, "w", newline="")  # No error if directory doesn't exist
```

**Risk:**
- Corrupted video files crash the pipeline
- Missing calibration files cause silent failures
- No audit trail of what failed

### 4.2 Insufficient Input Validation

Frame dimensions, mask shapes, and path coordinates not validated at system boundaries:

```python
# From realtime_nav_core.py, no validation that frame_hw is reasonable
def process_frame(self, frame_bgr, frame_hw):
    # No checks: frame could be (0, 0), (99999, 99999), etc.
    h_work, w_work = int(self.cfg.work_size[0]), int(self.cfg.work_size[1])
    ...
```

**Impact:**
- Out-of-bounds array access possible
- Silent numerical failures in distance transforms
- No graceful degradation

### 4.3 Floating-Point Clipping Without Bounds Checking

Repeated use of manual clipping instead of schema validation:

```python
# config.py, line 141
BEV_EGO_X_FRAC = float(np.clip(float(_BEV_META.get("ego_x_frac", 0.5)), 0.05, 0.95))

# Multiple instances of magic numbers instead of constants
x = int(np.clip(x, 0, max(0, w_bev - 1)))  # Clipping but no validation that w_bev > 0
```

**Recommendation:**
- Use pydantic or dataclasses with validators
- Validate at module entry points, not deep in computation

---

## 5. Fragile Areas (Prone to Breaking)

### 5.1 BEV Coordinate System Fragility

The BEV coordinate system is used in 15+ files with minimal documentation:

```
Assumption: (0,0) = bottom-left, forward = decreasing y
Violating this silently produces inverted paths
```

**Files affected:**
- `realtime_nav_core.py`, `template_path_planner.py`, `safe_corridor.py`, `masks.py`, `heading.py`, `visualization.py`

**Risk:** Any refactoring of skeleton/medial-axis extraction can invert paths without obvious error

**Recommendation:** Add invariant assertions at module boundaries:
```python
assert path_px[-1, 1] < path_px[0, 1], "Path should point upward (decreasing y)"
```

---

### 5.2 Temporal State Management (Frame Skipping, EMA)

Multiple research improvements use temporal state (EMA, path smoothing, DT corridor) that can be disrupted:

| Feature | State Type | Reset Logic | Fragility |
|---------|-----------|-------------|-----------|
| Path temporal smoothing | EMA coefficients | Topology change detection (line 150+) | What triggers "topology change"? Unclear thresholds |
| Heading filter | Circular EMA | >45° jump (config.py:298) | Single large heading error resets all smoothing |
| Predictive frame reuse | Previous mask | Confidence floor (0.50) | No bounds checking on confidence values |

**Risk:** Small config changes (HEADING_SMOOTH_RESET_DEG) drastically change behavior; no A/B testing harness

---

### 5.3 Coupled Model Loading Pipeline

`config.py` auto-detects best checkpoint by validation IoU:

```python
# Lines 23–85
_MODEL_DIR_CANDIDATES = [...]  # 4 hardcoded paths
ranked = [(iou, -priority, path) for path in _MODEL_DIR_CANDIDATES if isdir(path)]
ranked.sort(reverse=True)  # Silent fallback if all fail
```

**Risk:**
- New model trained → added to candidates list → **all existing code reweights silently**
- No version pinning; code that passes tests may fail with new model checkpoint
- Validation IoU from `training_summary.json` not guaranteed to match runtime performance

**Recommendation:**
- Explicit model versioning (e.g., `MODEL_VERSION = "2026-03-24"` in config.py)
- Only auto-detect during dev; pin model in production

---

## 6. Missing Tests & Coverage Gaps

### 6.1 No Unit Tests for Critical Modules

| Module | Tests | Issue |
|--------|-------|-------|
| `masks.py` | None | Road segmentation cleaning has no coverage; `clean_bev_mask_enhanced()` untested |
| `safe_corridor.py` | None | Distance-transform corridor extraction untested; `get_default_dt_corridor()` has no regression tests |
| `path_smoother.py` | Partial | Temporal EMA logic tested (`test_temporal_smoother.py`) but not integrated with realtime loop |
| `gps_navigator.py` | None | GPS intent conditioning untested; no validation of waypoint-turn decisions |
| `scooter_commander.py` | None | Serial output to hardware untested; could silently fail to send commands |
| `stabilization.py` | None | Camera shake compensation untested |

### 6.2 No Integration Tests for Real-Time Pipeline

Tests exist for individual modules but **zero integration tests** for full pipeline:
- Camera → Segmentation → BEV → Path Planning → Control output

**Risk:** Pipeline-level bugs (timing, state leaks, tensor shape mismatches) only found during live testing

### 6.3 No Regression Tests for Research Improvements

Three research features added (morphological cleaning, DT corridor, path smoothing) with **no tests to prevent reversion**:

- No test that `MORPH_ENHANCED=True` produces cleaner output than `False`
- No test that `DT_CORRIDOR_ENABLED` improves path containment
- No test that `PATH_SMOOTH_ENABLED` reduces jitter

---

## 7. Dependency Risks

### 7.1 Outdated or Unsupported Dependencies

No `requirements.txt` or `pyproject.toml` found. Hard to determine exact versions:

**Inferred dependencies:**
- `torch`, `transformers`, `ultralytics` — No version pins
- `opencv-python` — No version specified
- `numpy`, `scipy` — No version specified
- `psutil` — Used in `fast_road_detector.py`, version unknown

**Risk on RPi 4:**
- `transformers` may have updated to versions requiring newer CUDA
- `torch` nightly builds can introduce breaking changes
- No way to reproduce exact environment

### 7.2 Heavy GPU Dependencies for CPU-Fallback Code

All inference code assumes GPU optionality:

```python
# From object_detector.py, line 14–22
def _default_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
        # ... fallback to CPU
    except Exception:
        pass
    return "cpu"
```

But `torch` is loaded unconditionally. On systems without CUDA, this still pulls in 500 MB+ of unused GPU support.

**Recommendation:**
- Create `requirements-rpi.txt` with CPU-only torch
- Use environment variable to choose CPU/GPU at startup

---

## 8. Documentation Gaps

### 8.1 Missing Module Docstrings

14 of 31 modules lack comprehensive docstrings explaining:
- What the module does
- Key assumptions (BEV coords, real-time constraints)
- Dependencies and side effects

**Modules with minimal docs:**
- `bev_obstacle.py`
- `boundary_model.py`, `boundary_targets.py`
- `intent_picker.py`
- `scooter_commander.py`
- `skeleton.py`

### 8.2 Configuration Constants Lack Rationale

`config.py` has 95+ constants with values but **no comments on where they came from**:

```python
HEADING_SMOOTH_RESET_DEG = 45.0  # ← Why 45.0 and not 30.0 or 60.0?
WAYPOINT_ACQUIRE_SUPPORT_MIN = 0.40  # ← Empirically tuned where? When?
```

**Impact:** Maintainers cannot safely adjust thresholds; no scientific backing documented

---

## 9. Summary: Risk Ranking

| Category | Severity | Count | Immediate Action |
|----------|----------|-------|------------------|
| Failing tests | High | 3 | Fix `PathPlanResult` fixture, update file paths |
| Over-sized modules | High | 3 | Refactor into separate files |
| Bare exception handlers | Medium | 8 | Replace with specific exception types |
| Hardcoded values | Medium | 7 | Move to config.py |
| Missing unit tests | Medium | 6 | Add tests for `masks.py`, `safe_corridor.py` |
| No integration tests | Medium | 1 | Create end-to-end pipeline test |
| Model version drift | Low | 1 | Pin model version in config |
| Missing docs | Low | 14 | Add docstrings & rationale comments |

---

## 10. Recommended Fixes (Priority Order)

### Week 1: Stability
1. Fix 3 failing tests (PathPlanResult, replay manifest)
2. Replace bare `except Exception:` with specific exceptions
3. Add input validation at pipeline boundaries

### Week 2: Architecture
4. Extract `BEVPathExtractor` to separate module
5. Extract `AdaptivePurePursuitController` to separate module
6. Add unit tests for `masks.py` and `safe_corridor.py`

### Week 3: Documentation & DevOps
7. Create `requirements.txt` and `requirements-rpi.txt`
8. Document BEV coordinate system invariants
9. Add configuration comments with rationale

### Week 4: Research
10. Add regression tests for MORPH_ENHANCED, DT_CORRIDOR, PATH_SMOOTH
11. Create integration test for full pipeline
12. Pin model version; disable auto-detect in production

---

**End of CONCERNS.md**
