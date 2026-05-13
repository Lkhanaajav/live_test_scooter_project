# Coding Conventions

**Analysis Date:** 2026-03-04

## Naming Patterns

**Files:**
- Script files use snake_case: `fast_road_detector.py`, `live_heading_demo.py`, `realtime_nav_core.py`
- Executable scripts include `#!/usr/bin/env python3` shebang
- Analysis/utility scripts follow same naming: `analyze_log.py`, `camera_waypoint_pipeline.py`

**Functions:**
- Use snake_case for all function names: `get_gpu_memory()`, `process_frame()`, `extract_skeleton_graph()`
- Private/internal functions prefixed with underscore: `_setup_logging()`, `_clean_mask()`, `_thin()`
- Factory/toggle functions use verb-first pattern: `toggle_device()`, `initialize_model()`
- Utility helper functions use prefix notation: `_metric_from_pixel()`, `_polyline_length_m()`, `_clip()`

**Variables:**
- Snake_case for all local and global variables
- Global constants in UPPER_CASE: `ROAD_ID`, `SIDEWALK_ID`, `MODEL_DIR`, `SPEED_MAX`
- Configuration constants group-prefixed: `HEADING_STRAIGHT_THRESH`, `SPEED_TURN`, `OBSTACLE_CLOSE_M`
- Private/temporary variables use leading underscore: `_file`, `_writer`, `_row_count`
- Temporary tuple unpacking: `x1, y1, x2, y2 = box.xyxy[0]`

**Types:**
- Dataclass names use PascalCase: `Config`, `SystemInfo`, `PerformanceMetrics`, `AxisEdge`, `PathCandidate`
- Class names use PascalCase: `FastRoadDetector`, `BEVPathExtractor`, `AdaptivePurePursuitController`
- Type hints used for important function signatures in `realtime_nav_core.py` and `fast_road_detector.py`

## Code Style

**Formatting:**
- No explicit linting configuration found (`.eslintrc`, `.prettierrc`, Ruff config not present)
- Default Python conventions: 4-space indentation
- Line length: Generally kept under 100 characters; some longer lines in camera_waypoint_pipeline.py reach 120+
- String quotes: Mix of single and double quotes, no consistent preference detected

**Linting:**
- No linting configuration detected in codebase
- Code appears to follow PEP 8 informally but inconsistently

## Import Organization

**Order:**
1. Standard library imports (os, sys, cv2, math, time, argparse, threading, csv, json)
2. Third-party imports (numpy, matplotlib, pandas, networkx, torch, transformers, psutil, ultralytics)
3. Local imports (from fast_road_detector, from realtime_nav_core)

**Path Aliases:**
- No path aliases detected; imports use absolute module names
- Relative imports from same directory: `from fast_road_detector import FastRoadDetector, Config`

**Example from `fast_road_detector.py`:**
```python
import os
import time
import cv2
import torch
import numpy as np
import logging
import psutil
import platform
import json
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, List, Union
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
import argparse
```

## Error Handling

**Patterns:**
- Try-except blocks for resource loading: model loading, camera initialization (`fast_road_detector.py`, lines 155-192)
- Graceful degradation with fallback: GPU unavailable → CPU (`fast_road_detector.py`, lines 122-133)
- Try-except with import fallback for optional dependencies: matplotlib (`analyze_log.py`, lines 29-38)
- Try-except for shape mismatches in video processing: dynamic resizing fallback (`camera_waypoint_pipeline.py`, lines 277-283)
- Model loading with local fallback before HuggingFace resolution (`fast_road_detector.py`, lines 160-186)
- GPS serial reading with graceful error handling in `camera_waypoint_pipeline.py` lines 121-136

**Error Logging:**
```python
try:
    self.logger.error(f"Error loading model: {e}")
    raise
except Exception as e:
    print("❌ Error message with emoji indicators")
```

**Exception Types:**
- Custom RuntimeError for fatal conditions: "Cannot open video"
- Exception re-raising after logging
- SystemExit for argument validation failures

## Logging

**Framework:** Python's standard `logging` module

**Patterns:**
- Logger initialized in class `__init__`: `self.logger = logging.getLogger(__name__)`
- Configurable via Config dataclass: `enable_logging` flag
- Fallback to NullHandler when disabled: `logging.NullHandler()`
- Log levels: INFO (startup, progress), WARNING (fallback behavior), ERROR (failures)

**Example from `fast_road_detector.py`:**
```python
def _setup_logging(self):
    if self.config.enable_logging:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
```

**Print-based logging:**
- Simple print() with emoji prefixes in user-facing scripts: `print("✅ Model ready!")`, `print("📹 Opening camera...")`
- Found in: `camera_waypoint_pipeline.py`, `analyze_log.py`

## Comments

**When to Comment:**
- Algorithm explanations: medial axis extraction, pure pursuit controller
- Non-obvious parameters: BEV transformation matrices, smoothing weights
- Module docstrings: each script has detailed module-level documentation
- Complex numerical operations: curvature calculation, perspective transform

**JSDoc/TSDoc:**
- Not used; Python uses Google/NumPy docstring style occasionally
- Module docstrings with usage examples: `live_heading_demo.py` lines 2-20
- Function docstrings are sparse; mainly used for public APIs

**Example from `realtime_nav_core.py`:**
```python
def _polyline_length_m(pts_m: np.ndarray) -> float:
    """Arc-length of polyline in meters."""
    if pts_m is None or len(pts_m) < 2:
        return 0.0
```

## Function Design

**Size:**
- Utility functions: 5-30 lines (e.g., `_odd()`, `_safe_norm()`, `_clip()`)
- Processing functions: 30-100 lines (e.g., `process_frame()`, `_preprocess()`)
- Complex algorithms: 100-300 lines (e.g., `_search_candidates()`, `_build_graph()`)
- No strict size limit enforced; longest function is ~200 lines

**Parameters:**
- Use dataclasses for configuration: `Config`, `PathExtractorConfig`, `PurePursuitConfig`
- Explicit parameters for high-frequency operations: `process_frame()` takes frame and optional processor_size
- Optional parameters with None defaults for fallback logic
- Type hints in signatures where non-obvious: `process_frame(self, frame: np.ndarray, processor_size: Optional[...] = None)`

**Return Values:**
- Tuples for multiple related returns: `(mask, overlay)`, `(skel_bin, dist_m)`
- Dataclass returns for complex results: `PathPlanResult`, `ControlOutput`, `MemoryStats`
- None for optional results: `_fit_regularized_cubic()` returns `Optional[CubicPathModel]`
- Early return pattern for guard clauses: check validity before processing

## Module Design

**Exports:**
- No explicit `__all__` lists detected
- Public API inferred by docstrings and usage
- Main exports from modules: classes and config dataclasses
  - `fast_road_detector.py`: `FastRoadDetector`, `Config`
  - `realtime_nav_core.py`: `BEVPathExtractor`, `AdaptivePurePursuitController`, various Config/Result dataclasses
  - `live_heading_demo.py`: `DataLogger`, `TinyObjectDetector` (and utilities)

**Barrel Files:**
- No barrel file pattern detected (no `__init__.py` with re-exports)
- Top-level scripts import directly from modules

**Internal Structure:**
- Constants defined at module top: `ROAD_ID`, `SIDEWALK_ID`, `MODEL_DIR`
- Dataclasses next: `Config`, `SystemInfo`, `PerformanceMetrics`
- Utility functions: `get_gpu_memory()`, `get_system_info()` (prefix with `_` if internal)
- Main class definitions: `FastRoadDetector`, `BEVPathExtractor`
- Helper functions: `toggle_device()`, `initialize_model()`
- Main entry point: `if __name__ == "__main__": main()`

---

*Convention analysis: 2026-03-04*
