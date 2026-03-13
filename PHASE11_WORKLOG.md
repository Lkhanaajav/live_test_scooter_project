# Phase 11 Worklog

## 2026-03-12 18:04:40 -05:00

### Scope Check
- Objective confirmed from `simulation_camera_scooter/.planning/ROADMAP.md` and `simulation_camera_scooter/.planning/REQUIREMENTS.md`.
- Phase 11 goal: replace raw centerline-following as the primary path selector with a template-bank path approval layer that scores smooth BEV trajectories against a perceived sidewalk corridor.
- Required outputs: controller-compatible metric path, pixel overlay path, confidence, slowdown guidance, and explicit fallback diagnostics.

### Assumptions
- Phase 11 depends on Phase 7 conceptually, but Phase 7 boundary predictions are not wired into the live loop yet.
- Implementation will therefore start with a corridor derived from the cleaned BEV mask while keeping the corridor contract compatible with future boundary-based inputs.
- The old path planner remains useful as a baseline and fallback, so it will be preserved and measured rather than deleted.

### Git / Environment
- Base branch synced from `origin/main` to commit `5e8f616052a2fefccad0597c541a0d2ad1774493`.
- Local working branch created: `phase11-template-approval`.
- Windows long-path issue blocked the first `git pull`; resolved with `git config core.longpaths true`.
- Current untracked asset directory remains untouched: `simulation_camera_scooter/test_videos/`.

### Files Reviewed
- `simulation_camera_scooter/.planning/PROJECT.md`
- `simulation_camera_scooter/.planning/ROADMAP.md`
- `simulation_camera_scooter/.planning/REQUIREMENTS.md`
- `simulation_camera_scooter/.planning/STATE.md`
- `simulation_camera_scooter/.planning/phases/11-template-path-fitting-inside-segmentation-corridor-with-path-approval-scoring/11-CONTEXT.md`
- `simulation_camera_scooter/.planning/phases/11-template-path-fitting-inside-segmentation-corridor-with-path-approval-scoring/11-RESEARCH.md`
- `simulation_camera_scooter/.planning/phases/11-template-path-fitting-inside-segmentation-corridor-with-path-approval-scoring/11-VALIDATION.md`
- `simulation_camera_scooter/overnight_runs/2026-03-06_path_improvement/VISUAL_FINDINGS.md`
- `simulation_camera_scooter/overnight_runs/2026-03-06_path_improvement/RESEARCH_NOTES.md`
- `simulation_camera_scooter/overnight_runs/2026-03-06_path_improvement/EXPERIMENT_LOG.md`
- `simulation_camera_scooter/realtime_nav_core.py`
- `simulation_camera_scooter/camera_waypoint_pipeline.py`
- `simulation_camera_scooter/boundary_inference.py`
- `simulation_camera_scooter/tests/conftest.py`

### Commands Run
- `git fetch origin main`
- `git pull --ff-only origin main`
- `git switch -c phase11-template-approval`
- `py -m pytest simulation_camera_scooter/tests -q`

## 2026-03-12 18:15:00 -05:00

### External Research Pass
- Reviewed official OpenCV morphology, distance transform, and connected-components references for lightweight corridor extraction.
- Reviewed CMU Pure Pursuit notes for controller-facing path smoothness requirements.
- Reviewed motion-primitive/state-lattice literature to confirm that a compact template bank is the right complexity level here.
- Reviewed temporal lane-video papers for why continuity and hysteresis matter more than raw per-frame optimality.

### Decision
- Chose ego-anchored cubic templates over constant-curvature-only arcs, full lattices, or clothoid optimization.
- Kept skeleton/graph logic as a measurable fallback and benchmark instead of removing it.

## 2026-03-12 18:25:00 -05:00

### Initial Implementation
- Added `simulation_camera_scooter/template_path_planner.py`.
- Added a corridor contract and approval result dataclasses.
- Implemented a first pass of template generation, scoring, confidence, and slowdown behavior.
- Integrated template approval into `simulation_camera_scooter/realtime_nav_core.py`.
- Added replay tooling in `simulation_camera_scooter/scripts/eval_template_planner.py`.
- Added tests in `simulation_camera_scooter/tests/test_template_path_planner.py`.

## 2026-03-12 18:40:45 -05:00

### Initial Replay Evidence
- Ran early June and long June comparisons with the first template planner.
- Those runs showed promise, but the result quality was unstable and visually suspicious.
- This later turned out to be contaminated by a BEV calibration-loading bug rather than only bad path selection.

## 2026-03-13 02:31:59 -05:00

### Root Cause Found: BEV Calibration Path Bug
- I finally inspected the actual single-frame BEV output instead of trusting the metrics.
- Segmentation was fine, but the BEV warp was wrong.
- Cause:
  - two different calibration files existed:
    - `bev_calibration.npy` at repo root
    - `simulation_camera_scooter/bev_calibration.npy`
  - `CALIBRATION_FILE` in `simulation_camera_scooter/config.py` was relative, so running from repo root loaded the wrong file.
- Broken root calibration points were for a different image geometry and fell outside the 1280x720 June frame.
- Debug artifacts created:
  - `simulation_camera_scooter/demo_outputs/bev_debug_sheet.png`
  - `simulation_camera_scooter/demo_outputs/bev_debug_compare.png`

### Fix
- Updated `simulation_camera_scooter/config.py` so `CALIBRATION_FILE` is an absolute path inside `simulation_camera_scooter/`.
- Added a calibration-path test in `simulation_camera_scooter/tests/test_bev_calibration.py`.
- Verification:
  - `py -m pytest simulation_camera_scooter/tests/test_bev_calibration.py -q`
  - Result: `6 passed`

## 2026-03-13 02:38:00 -05:00

### Selector Redesign
- Reframed Phase 11 around a small reusable intent bank instead of a larger family of always-turning templates.
- New template behavior:
  - `straight`
  - `left` with near / mid / late turn onset
  - `right` with near / mid / late turn onset
- Added family reuse / hysteresis:
  - previous family gets a reuse bonus
  - switching requires stronger evidence
  - if the same family still fits reasonably well, the planner can reuse it instead of dropping immediately to fallback
- Added top candidate-path output so the BEV HUD shows a small 3-5 path fan instead of one opaque decision.

### Code Changes In This Step
- `simulation_camera_scooter/template_path_planner.py`
- `simulation_camera_scooter/realtime_nav_core.py`
- `simulation_camera_scooter/tests/test_template_path_planner.py`

## 2026-03-13 02:51:13 -05:00

### Replay Check: Intent Bank Without Reuse Lane
- Command:
  - `py simulation_camera_scooter/scripts/eval_template_planner.py --videos simulation_camera_scooter/test_video_june_03_3.mp4 --max-frames 220 --output-root simulation_camera_scooter/eval_runs/template_planner_intent220 --save-video`
- Result:
  - baseline mean abs heading `1.122 deg`
  - Phase 11 mean abs heading `0.743 deg`
  - template rate `45.5%`
  - low-confidence rate `54.5%`
  - template-family switches `8`
- Observation:
  - candidate bank and family stability were much better
  - fallback / slowdown were still too common because approval remained too binary

## 2026-03-13 02:55:35 -05:00

### Replay Check: Intent Bank With Reuse Lane
- Command:
  - `py simulation_camera_scooter/scripts/eval_template_planner.py --videos simulation_camera_scooter/test_video_june_03_3.mp4 --max-frames 220 --output-root simulation_camera_scooter/eval_runs/template_planner_intent220_reuse --save-video`
- Primary output:
  - `simulation_camera_scooter/eval_runs/template_planner_intent220_reuse/summary.md`
- Frame comparison artifact:
  - `simulation_camera_scooter/eval_runs/template_planner_intent220_reuse/frames/comparison_sheet.png`

### Corrected June Results
- Graph baseline:
  - mean abs heading `1.009 deg`
  - p95 abs heading `3.251 deg`
  - max abs heading `4.830 deg`
  - mean speed `1.500 m/s`
  - path-source switches `44`
- Phase 11:
  - mean abs heading `0.208 deg`
  - p95 abs heading `1.319 deg`
  - max abs heading `1.979 deg`
  - mean speed `1.006 m/s`
  - template rate `59.1%`
  - graph rate `6.4%`
  - fallback rate `34.5%`
  - path-source switches `19`
  - template-family switches `5`
- Key observation:
  - this was the first corrected-calibration run where Phase 11 was both logically aligned with the intended design and quantitatively better than the corrected baseline on path smoothness and stability.

## 2026-03-13 03:05:40 -05:00

### Final Logic Cleanup
- Found one remaining state bug in `realtime_nav_core.py`:
  - `prev_template_family` could survive after graph or centerline fallback took over.
  - that stale family could bias the next template vote even when the active path was no longer template-driven.
- Added `_commit_selected_path(...)` to centralize path-state updates.
- New state policy:
  - preserve `prev_template_family` for `template`
  - preserve it for short `fallback_hold`
  - clear it when `graph`, `fallback_centerline`, or `fallback_skeleton` actually take over
- Added a startup tie-break rule in `template_path_planner.py`:
  - when the corridor is still effectively straight and the best turn template only barely leads, the selector now prefers `straight`
  - this startup straight bias is obstacle-aware, so it does not override a genuinely safer turn candidate

### Files Edited In This Step
- `simulation_camera_scooter/template_path_planner.py`
- `simulation_camera_scooter/realtime_nav_core.py`
- `simulation_camera_scooter/tests/test_template_path_planner.py`
- `simulation_camera_scooter/tests/test_realtime_nav_core.py`

### Verification
- `py -m pytest simulation_camera_scooter/tests/test_template_path_planner.py -q` -> `19 passed`
- `py -m pytest simulation_camera_scooter/tests/test_realtime_nav_core.py -q` -> `10 passed`

## 2026-03-13 03:10:09 -05:00

### Final GUI Validation: Phase 11 Enabled
- Command:
  - workdir `simulation_camera_scooter/demo_outputs/phase11_june_intent_reuse_gui/`
  - `py ..\..\live_heading_demo.py --video ..\..\test_video_june_03_3.mp4 --max-frames 220 --save --log --no-detection --log-dir logs`
- Calibration confirmed at runtime:
  - `Loaded BEV calibration from C:\Users\lhana\OneDrive\Desktop\scootedr\live_test_scooter_project\simulation_camera_scooter\bev_calibration.npy`
- Output artifacts:
  - `simulation_camera_scooter/demo_outputs/phase11_june_intent_reuse_gui/heading_demo_output.mp4`
  - `simulation_camera_scooter/demo_outputs/phase11_june_intent_reuse_gui/logs/run_20260313_031009.csv`
  - `simulation_camera_scooter/demo_outputs/phase11_june_intent_reuse_gui/frame_sheet.png`
  - `simulation_camera_scooter/demo_outputs/phase11_june_intent_reuse_gui/bev_sheet.png`
- Summary from `run_20260313_031009.csv`:
  - mean abs heading `0.707 deg`
  - p95 abs heading `3.508 deg`
  - max abs heading `5.575 deg`
  - mean speed `1.067 m/s`
  - low-confidence rate `37.7%`
  - mean slowdown `0.351`
  - template rate `62.3%`
  - graph rate `5.9%`
  - fallback rate `31.8%`
  - path-source switches `18`
  - template-family switches `3`
- Visual observation:
  - saved BEV frames now show a small candidate fan and a stable straight winner instead of a noisy or obviously broken path selection.

## 2026-03-13 03:11:46 -05:00

### Final GUI Validation: Corrected Baseline
- Command:
  - workdir `simulation_camera_scooter/demo_outputs/baseline_june_intent_gui/`
  - `py ..\..\live_heading_demo.py --video ..\..\test_video_june_03_3.mp4 --max-frames 220 --save --log --no-detection --no-template-planner --log-dir logs`
- Output artifacts:
  - `simulation_camera_scooter/demo_outputs/baseline_june_intent_gui/heading_demo_output.mp4`
  - `simulation_camera_scooter/demo_outputs/baseline_june_intent_gui/logs/run_20260313_031146.csv`
  - `simulation_camera_scooter/demo_outputs/baseline_june_intent_gui/frame_sheet.png`
- Summary from `run_20260313_031146.csv`:
  - mean abs heading `1.190 deg`
  - p95 abs heading `3.724 deg`
  - max abs heading `4.950 deg`
  - mean speed `1.500 m/s`
  - graph rate `75.0%`
  - fallback rate `25.0%`
  - path-source switches `36`
- Final matched-GUI comparison:
  - Phase 11 improved mean abs heading and p95 heading.
  - Phase 11 cut path-source switching from `36` to `18`.
  - Phase 11 kept the candidate set visually cleaner.
  - Phase 11 remained more conservative on speed.

## 2026-03-13 03:12:20 -05:00

### Final Test Status
- `py -m pytest simulation_camera_scooter/tests -q`
- Result: `101 passed`

### Final Files Edited
- `simulation_camera_scooter/config.py`
- `simulation_camera_scooter/template_path_planner.py`
- `simulation_camera_scooter/realtime_nav_core.py`
- `simulation_camera_scooter/live_heading_demo.py`
- `simulation_camera_scooter/heading.py`
- `simulation_camera_scooter/data_logger.py`
- `simulation_camera_scooter/visualization.py`
- `simulation_camera_scooter/scripts/eval_template_planner.py`
- `simulation_camera_scooter/tests/conftest.py`
- `simulation_camera_scooter/tests/test_bev_calibration.py`
- `simulation_camera_scooter/tests/test_realtime_nav_core.py`
- `simulation_camera_scooter/tests/test_template_path_planner.py`
- `PHASE11_WORKLOG.md`
- `PHASE11_RESEARCH.md`
- `PHASE11_RESULTS.md`
- `PHASE11_VIDEO_EVAL.md`

### Current Assessment
- Phase 11 now matches the intended planning model much more closely:
  - small reusable template bank
  - 3-5 candidate paths in BEV
  - stable reuse of the selected family
  - startup straight bias unless turn evidence becomes stronger
  - no stale template-family state leaking through graph or centerline fallback
- Current honest status:
  - works on the calibrated June clip
  - visually and quantitatively better than the corrected graph baseline on that clip
  - still conservative in weak-confidence windows
  - still calibration-limited on the separate phone-video smoke clip
