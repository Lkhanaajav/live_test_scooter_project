# Testing

This document describes the testing framework, patterns, and practices used in the scooter navigation project.

## Test Framework & Setup

### Pytest Configuration

**Framework**: pytest (Python's standard testing framework)

**Test discovery**:
- Test files: `tests/test_*.py` (12 test files)
- Test functions: `test_*` or methods in `Test*` classes
- Fixtures: defined in `conftest.py` with `@pytest.fixture` decorator

**Running tests**:

```bash
cd simulation_camera_scooter

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=term-missing

# Run single test file
pytest tests/test_heading.py -v

# Run specific test class or function
pytest tests/test_heading.py::TestComputeHeading::test_straight_path_gives_zero_heading -v

# Run tests matching a pattern
pytest tests/ -k "test_heading" -v
```

### Fixtures (conftest.py)

Shared pytest fixtures are centralized in `simulation_camera_scooter/tests/conftest.py`:

**Path setup fixture**:
```python
@pytest.fixture(autouse=True)
def _pin_ego_center():
    """Pin BEV_EGO_X_FRAC to 0.5 so tests don't depend on calibration files."""
    original = config.BEV_EGO_X_FRAC
    config.BEV_EGO_X_FRAC = 0.5
    yield
    config.BEV_EGO_X_FRAC = original
```

**Key fixtures** (19 total):
- **Mask fixtures**: `straight_bev_mask`, `curved_bev_mask`, `wide_straight_bev_mask`, `right_curved_bev_mask`, `fragmented_near_field_bev_mask`, `false_pocket_bev_mask`
- **Path model fixtures**: `straight_path_model`
- **Projection fixtures**: `bev_h_matrix`, `bev_obstacle_mask_500x600`
- **Detection fixtures**: `mock_detections`
- **Phase 11.1 fixtures**: `commanded_left_bev_mask`, `commanded_right_bev_mask`, `unsupported_turn_bev_mask`, `no_intent_straight_bev_mask`

---

## Test File Organization

### Test Files (12 total)

| File | Purpose | Pattern |
|------|---------|---------|
| `test_heading.py` | Heading computation, command classification, speed profiling | 3 test classes |
| `test_bev_obstacle.py` | YOLO BEV obstacle projection (Phase 03.1) | Mix of functions + fixtures |
| `test_bev_calibration.py` | Homography calibration and edge cases | Class-based tests |
| `test_bev_predictor.py` | Predictive BEV frame reuse | Fixture-driven |
| `test_boundary_dataset_model.py` | Road boundary annotation and inference | Dataset validation |
| `test_boundary_inference.py` | Boundary post-processing | Integration tests |
| `test_boundary_targets.py` | Boundary target generation | Numerical validation |
| `test_image_path_planner.py` | Image-space path planning (deprecated) | Legacy tests |
| `test_realtime_nav_core.py` | BEVPathExtractor, pure pursuit controller | 2 test classes, 15+ tests |
| `test_skeleton.py` | Skeletonization and branch pruning | Unit tests for algorithms |
| `test_stabilization.py` | Camera shake compensation | EMA filter validation |
| `test_template_path_planner.py` | Corridor extraction, template approval | Comprehensive test suite |

### Naming Convention

Test functions follow the pattern:
```
test_<noun>_<verb>_<outcome>
```

**Examples**:
- `test_straight_path_gives_zero_heading` — what input, what expected output
- `test_compute_speed_stops_near_obstacle` — condition and behavior
- `test_process_empty_mask_returns_no_path` — input state, return type
- `test_heading_to_command_sharp_left` — what angle maps to what command
- `test_ema_decay` — operation being tested

---

## Test Patterns

### Class-Based Test Organization

Tests grouped in `Test*` classes for logical organization:

```python
class TestComputeHeading:
    """Tests for the compute_heading() function."""

    def test_straight_path_gives_zero_heading(self):
        """Vertical path (straight ahead) should return ~0 degrees."""
        path = [(110, 220), (110, 100)]
        angle = compute_heading(path)
        assert abs(angle) < 5.0

    def test_left_curve_gives_negative_heading(self):
        """Path curving left (x decreasing) should return negative heading."""
        path = [(110, 220), (90, 180), (70, 140), (60, 100)]
        angle = compute_heading(path)
        assert angle < 0.0

    def test_right_curve_gives_positive_heading(self):
        """Path curving right (x increasing) should return positive heading."""
        path = [(110, 220), (130, 180), (150, 140), (160, 100)]
        angle = compute_heading(path)
        assert angle > 0.0

    def test_single_point_returns_zero(self):
        """Less than 2 points should return 0."""
        assert compute_heading([]) == 0.0
        assert compute_heading([(110, 110)]) == 0.0


class TestHeadingToCommand:
    """Tests for heading-to-command mapping."""

    def test_heading_to_command_straight(self):
        """Small angle should map to STRAIGHT."""
        cmd, color = heading_to_command(0.0)
        assert cmd == "STRAIGHT"

    def test_heading_to_command_sharp_left(self):
        """Large negative angle should map to SHARP LEFT."""
        cmd, color = heading_to_command(-(HEADING_TURN_THRESH + 10))
        assert cmd == "SHARP LEFT"
```

**Benefit**: logically grouped assertions, clear test structure, easier navigation.

### Fixture-Driven Testing

Fixtures provide test data (masks, homographies, models):

```python
def test_process_straight_corridor_returns_path(self, straight_bev_mask):
    """A clear straight corridor should yield a valid path."""
    extractor = BEVPathExtractor()
    result = extractor.process(straight_bev_mask)
    assert isinstance(result, PathPlanResult)
    assert result.best_path_m.ndim == 2
    assert result.best_path_m.shape[1] == 2

def test_metric_conversion(self, bev_h_matrix, bev_obstacle_mask_500x600, mock_detections):
    """OBS-02: metric coordinate of projected foot matches expected."""
    det = {"bbox": (270, 200, 330, 400), "class_name": "person", ...}
    forward_m, lateral_m = detection_to_metric(det, bev_h_matrix, ...)
    assert abs(forward_m - 6.59) < 0.1
    assert abs(lateral_m - (-2.10)) < 0.15
```

**Benefits**:
- Reusable test data (same mask used by 10+ tests)
- Declarative dependencies (function signature lists what's needed)
- Setup/teardown handled automatically by pytest

### Parametrized Testing

Test the same logic with multiple inputs:

```python
@pytest.mark.parametrize(
    "heading_deg,expected_cmd",
    [
        (0.0, "STRAIGHT"),
        (-10.0, "LEFT"),
        (10.0, "RIGHT"),
        (-25.0, "SHARP LEFT"),
        (25.0, "SHARP RIGHT"),
    ]
)
def test_heading_to_command(heading_deg, expected_cmd):
    """Test all heading ranges map to correct commands."""
    cmd, _ = heading_to_command(heading_deg)
    assert cmd == expected_cmd
```

**Usage in codebase**: Less common than class-based grouping; used for range validation.

### Assertion Patterns

#### Type Checking
```python
def test_returns_tuple_of_string_and_tuple(self):
    """heading_to_command should return (str, tuple)."""
    cmd, color = heading_to_command(0.0)
    assert isinstance(cmd, str)
    assert isinstance(color, tuple)
    assert len(color) == 3
```

#### Numerical Comparison (with tolerance)
```python
def test_straight_path_gives_zero_heading(self):
    """Vertical path should return ~0 degrees."""
    path = [(110, 220), (110, 100)]
    angle = compute_heading(path)
    assert abs(angle) < 5.0  # ±5 degree tolerance
```

#### Range Checks
```python
def test_process_returns_path_with_positive_progress(self, straight_bev_mask):
    """Path should have forward progress."""
    result = extractor.process(straight_bev_mask)
    if result.has_path:
        assert float(np.max(result.best_path_m[:, 0])) > 0.0
```

#### Equality and Properties
```python
def test_process_empty_mask_returns_no_path(self):
    """All-black mask should produce no path."""
    extractor = BEVPathExtractor()
    mask = np.zeros((220, 220), dtype=np.uint8)
    result = extractor.process(mask)
    assert result.has_path is False
```

#### Array Shape & Content
```python
def test_process_returns_valid_path_structure(self, straight_bev_mask):
    """Result should have consistent array shapes."""
    result = extractor.process(straight_bev_mask)
    assert isinstance(result, PathPlanResult)
    assert result.best_path_m.ndim == 2
    assert result.best_path_m.shape[1] == 2  # (N, 2) — N points × (x, y)
```

---

## Coverage & Test Scope

### Current Coverage

- **Tested modules** (~12): heading, bev_obstacle, bev_calibration, skeleton, realtime_nav_core, template_path_planner, etc.
- **Test count**: 104 tests across 12 files
- **Coverage target**: 80% (enforced by project rules)
- **Framework**: pytest + pytest-cov

### What's Tested

**Unit tests** (single function isolation):
- Pure functions: `compute_heading()`, `heading_to_command()`, `compute_speed()`
- Math utilities: `_polyline_curvature_mean()`, `_resample_polyline()`, `_clip()`
- Skeletonization: `skeletonize_guohall()`, `prune_small_branches()`

**Integration tests** (multiple modules together):
- `BEVPathExtractor.process()` — mask → skeleton → graph → path
- `detection_to_metric()` + `project_foot_to_bev()` — detection pipeline
- `AdaptivePurePursuitController.update()` — path → steering command

**Fixture-based tests** (behavior with various inputs):
- Straight corridors, curved corridors, fragmented masks
- Wide vs narrow passages
- Missing or occluded support

### What's NOT Tested (Known Gaps)

- **Real-time performance**: no throughput/latency benchmarks (checked manually)
- **Hardware integration**: no actual camera feed, motor commands
- **GPS navigation**: minimal tests (not a priority for pure-pursuit validation)
- **Visualization code**: not tested (rendering is visual verification only)
- **Training/model loading**: dataset classes tested, but model training not exercised

---

## Running Tests Locally

### Standard Test Run

```bash
cd simulation_camera_scooter
pytest tests/ -v
```

**Output example**:
```
tests/test_heading.py::TestComputeHeading::test_straight_path_gives_zero_heading PASSED
tests/test_heading.py::TestComputeHeading::test_left_curve_gives_negative_heading PASSED
tests/test_heading.py::TestHeadingToCommand::test_heading_to_command_straight PASSED
...
======================== 104 passed in 2.34s ========================
```

### With Coverage Report

```bash
pytest tests/ --cov=. --cov-report=term-missing
```

**Output shows**:
- Missing lines in each file (lines NOT covered by tests)
- Summary: `TOTAL ... 85% covered`

### Debugging a Failing Test

```bash
# Run with full output (no capture)
pytest tests/test_heading.py::TestComputeHeading::test_straight_path_gives_zero_heading -v -s

# Run with pdb on failure
pytest tests/test_heading.py -v --pdb

# Run with more verbose assertion introspection
pytest tests/test_heading.py -v --tb=long
```

---

## Writing New Tests

### Test Template

```python
"""
test_new_module.py
==================
Tests for new_module.py.

Requirements covered: REQ-01, REQ-02, ...
"""

import pytest
import numpy as np

from new_module import function_to_test


class TestFunctionToTest:
    """Tests for function_to_test()."""

    def test_normal_case(self):
        """Description of normal happy-path behavior."""
        input_data = ...
        result = function_to_test(input_data)
        assert result == expected_value

    def test_edge_case_empty_input(self):
        """Empty input should return default/empty result."""
        result = function_to_test([])
        assert result == [] or result is None

    def test_edge_case_single_element(self):
        """Single element should not crash."""
        result = function_to_test([1])
        assert isinstance(result, float)

    def test_with_fixture(self, straight_bev_mask):
        """Use shared fixture for consistency."""
        result = function_to_test(straight_bev_mask)
        assert result.shape == (220, 220)
```

### Adding a Fixture

**In `conftest.py`**:

```python
@pytest.fixture
def my_test_data():
    """220x220 test mask with specific properties."""
    mask = np.zeros((220, 220), dtype=np.uint8)
    # Create test pattern
    mask[:, 80:140] = 255  # vertical corridor
    return mask
```

**Use in test**:

```python
def test_with_my_data(self, my_test_data):
    """Test behavior with custom fixture."""
    result = process(my_test_data)
    assert result.shape[0] > 0
```

### Documenting Requirements

Tests explicitly reference requirements (e.g., Phase documentation):

```python
def test_hard_block_masks_bev(self, bev_h_matrix, bev_obstacle_mask_500x600, mock_detections):
    """OBS-06: hard-block paints BEV mask black at stop-distance obstacle location."""
    from config import BEV_HARD_BLOCK_DIST_M
    # ... test implementation
```

**Pattern**: `# <REQ-ID>: <description>` in docstring.

---

## Common Test Issues & Fixes

### Issue: Test Passes Locally But Fails in CI

**Cause**: Hardcoded path assumptions, missing fixtures, config state pollution.

**Fix**:
- Use fixtures from `conftest.py` for all setup
- Avoid `os.path.join()` without absolute paths
- Reset config state in `@pytest.fixture(autouse=True)`

**Example**:
```python
@pytest.fixture(autouse=True)
def _reset_config():
    """Reset config to defaults before each test."""
    original_value = config.SOME_CONSTANT
    config.SOME_CONSTANT = default_value
    yield
    config.SOME_CONSTANT = original_value
```

### Issue: Numerical Precision Failures

**Cause**: Floating-point arithmetic differs across platforms/seeds.

**Fix**: Use absolute tolerance in assertions.

```python
# WRONG: exact equality
assert result == 3.14159

# CORRECT: tolerance
assert abs(result - 3.14159) < 1e-5
```

### Issue: Flaky Tests (Pass Randomly)

**Cause**: Uninitialized state, random seeds, timing assumptions.

**Fix**: Seed random number generators, reset state, avoid time dependencies.

```python
import numpy as np
import random

@pytest.fixture(autouse=True)
def _seed():
    """Seed RNG for reproducibility."""
    np.random.seed(42)
    random.seed(42)
    yield
```

### Issue: Test Imports Fail

**Cause**: Module not in path, circular imports.

**Fix**: Ensure `conftest.py` sets up path correctly.

**In `conftest.py`**:
```python
import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

---

## Test Maintenance

### Before Each Commit

```bash
cd simulation_camera_scooter
pytest tests/ -v --cov=. --cov-report=term-missing
```

**Checklist**:
- [ ] All 104 tests pass
- [ ] Coverage ≥ 80%
- [ ] No warnings or skipped tests
- [ ] New tests added for new code

### When Refactoring

1. **Run tests before change** to establish baseline
2. **Make code change**
3. **Run tests after change** to ensure no regression
4. **Update tests if behavior changed** (not implementation details)

### Debugging Philosophy

- **Don't change tests to make them pass** — fix the implementation
- **If test is wrong**, fix it explicitly (document why in docstring)
- **Use `-s` flag** (`pytest -s`) to see print output during debugging
- **Use `--pdb`** to drop into debugger on failure

---

## Performance & Real-Time Testing

### Timing Instrumentation

Many pipeline functions measure execution time:

```python
def process(self, mask: np.ndarray) -> PathPlanResult:
    t0_skel = time.time()
    skel = extract_skeleton(mask)
    t_skeleton_ms = 1000 * (time.time() - t0_skel)

    t0_path = time.time()
    result = self._find_best_path(skel, mask)
    t_path_ms = 1000 * (time.time() - t0_path)

    return PathPlanResult(..., t_skeleton_ms=t_skeleton_ms, t_path_ms=t_path_ms)
```

**Tests can verify timing**:
```python
def test_skeleton_extraction_fast(self, straight_bev_mask):
    """Skeleton extraction should be < 10 ms."""
    extractor = BEVPathExtractor()
    result = extractor.process(straight_bev_mask)
    assert result.t_skeleton_ms < 10.0
```

---

## Test Dependencies

### Required Packages

From `requirements.txt`:
- `pytest` — test framework
- `pytest-cov` — coverage reporting
- `numpy` — numerical testing
- `opencv-python` — image processing in tests
- `torch`, `transformers`, `ultralytics` — optional (model loading tests)

### Installation

```bash
pip install -r requirements.txt
```

Or minimal:
```bash
pip install pytest pytest-cov numpy opencv-python
```

---

## Summary Checklist

Before writing tests:

- [ ] Use class-based organization (`Test*`)
- [ ] Fixtures in `conftest.py` for reusable data
- [ ] Google-style docstrings on every test
- [ ] Type hints on function parameters
- [ ] Assertion messages (optional but helpful for debugging)
- [ ] Test names describe input + expected outcome
- [ ] Edge cases covered (empty, single, boundary values)
- [ ] Error cases handled (None inputs, invalid states)
- [ ] No hardcoded paths; use fixtures and config
- [ ] Coverage target: 80% minimum

Before committing:

- [ ] Run full test suite: `pytest tests/ -v`
- [ ] Check coverage: `--cov` report shows ≥80%
- [ ] No test warnings or errors
- [ ] New code has corresponding tests
- [ ] Docstrings explain what each test validates
