# Phase 2: BEV Calibration and Path Reliability - Context

**Gathered:** 2026-03-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Recalibrate the BEV homography matrix so the warpPerspective transform preserves ≥50% of sidewalk
pixels (down from 7% current), and tune path extraction parameters to achieve has_path ≥ 60% on a
representative sidewalk video clip. Calibration documentation must be written so the procedure can
be repeated when the camera mount changes.

This phase does NOT include: demo visualization overlay (Phase 3), serial command output (Phase 3),
or embedded deployment (Phase 4).

</domain>

<decisions>
## Implementation Decisions

### Camera and calibration tooling
- Existing interactive 4-point click tool (`bev_calibration.py`) is used as-is — no new tool needed
- Calibration is done from a video file: `python live_heading_demo.py --calibrate --video <good_video.mp4>`
- The video must be recorded with the camera in its final mount position (not handheld/tilted)
- Result saved to `bev_calibration.npy` (overwrites current ill-conditioned file)

### Acceptance criteria for calibration
- Quantitative: condition number < 1000 (currently 1.1e+06) — checked by `load_bev_params()`
- Visual: BEV output frame shows a filled sidewalk region, not a thin sliver
- Pixel survival: ≥50% of sidewalk mask pixels survive warpPerspective (measure by comparing
  sidewalk pixel count before and after transform on a representative frame)

### Path tuning approach
- Tier 1 params already applied (min_sidewalk_width_m=0.50, branch_min_len_m=0.30, min_path_len_m=0.80)
- After recalibration, run pipeline on the same test video and measure has_path rate
- If has_path < 60%: adjust PathExtractorConfig params iteratively (no automated sweep needed —
  calibration quality is the dominant variable, not params)
- DT_CORE_THRESH=2.0 is already relaxed from tier1 tuning; can go lower if needed

### Calibration documentation
- Write a short calibration SOP (standard operating procedure) in the phase plan or as a README
  section: what video to record, how to pick 4 points, how to verify, when to redo
- This covers BEV-03 requirement

### Claude's Discretion
- Exact video recording instructions (frame count, duration, lighting conditions)
- Whether to add a pixel-survival measurement utility script or measure manually
- Specific parameter values to try if has_path < 60% after calibration

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `bev_calibration.py:run_calibration()` — interactive tool, already works, just needs a good video input
- `bev_calibration.py:load_bev_params()` — already validates condition number and logs warning
- `realtime_nav_core.py:PathExtractorConfig` — all path params are dataclass fields, easy to tune
- `config.py:DEFAULT_SRC_POINTS/DEFAULT_DST_POINTS/BEV_SIZE` — fallback defaults always present

### Established Patterns
- Tier 1 inline comments: `# was X — tier1 tuning: reason` — follow same style for tier2 changes
- Parameter sweep pattern from Phase 1 (`tune_smoother.py`) — reusable if a path sweep is needed
- `python -m pytest tests/` — existing test suite must still pass after any config changes

### Integration Points
- `live_heading_demo.py --calibrate --video <path>` — entrypoint for calibration
- `bev_calibration.npy` — output file; overwriting it changes behavior for all subsequent runs
- `config.py` constants (DT_CORE_THRESH, TRIM_BOTTOM) and `realtime_nav_core.py` PathExtractorConfig
  are the two files to edit for path tuning

</code_context>

<specifics>
## Specific Ideas

- Root cause is clear: current `bev_calibration.npy` was made with a tilted/handheld iPhone, not
  the mounted camera. Recalibration with proper video is expected to drop condition number by 3+ orders
  of magnitude and raise has_path from ~2% to 60-80%.
- "Good calibration video" means: camera in final mount position, walking straight down a clear
  sidewalk, level (no tilt), good lighting, sidewalk edges clearly visible.
- Measurement baseline: test_video_mar3.MOV is the current test video — use it for before/after
  comparison of has_path rate.

</specifics>

<deferred>
## Deferred Ideas

- Auto-detection of calibration drift (ROB-03 in v2 requirements) — out of scope for Phase 2
- GPS-assisted calibration validation — not needed, visual + condition number is sufficient
- Multi-camera or fisheye support — out of scope

</deferred>

---

*Phase: 02-bev-calibration-and-path-reliability*
*Context gathered: 2026-03-05*
