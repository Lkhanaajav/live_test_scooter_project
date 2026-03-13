# Phase 11 Results

## Status
- Working on the calibrated June test video.
- Meaningfully improved versus the corrected graph baseline on that clip.
- Still partial overall because the planner remains conservative in weak-confidence windows and calibration still limits the separate phone-video smoke test.

## Goal
- Phase 11 is trying to replace graph-first path commitment with a small reusable template selector that:
  - fits smooth ego-anchored paths inside the BEV sidewalk corridor
  - keeps a few plausible candidate paths visible
  - reuses the winning path family instead of flapping every frame
  - preserves graph / centerline fallback when corridor evidence is weak

## Final Implementation Summary

### Core planner changes
- `simulation_camera_scooter/template_path_planner.py`
  - corridor extraction from BEV mask
  - 7-template intent bank:
    - `straight_center`
    - `left_near`, `left_mid`, `left_late`
    - `right_near`, `right_mid`, `right_late`
  - scoring by containment, near-field support, clearance, center alignment, continuity, curvature, and obstacle overlap
  - family reuse / hysteresis
  - startup straight preference when turn evidence is only marginally stronger
  - obstacle-aware tie-break so straight does not override a genuinely better avoidance path
- `simulation_camera_scooter/realtime_nav_core.py`
  - template-first planning path integrated ahead of graph search
  - candidate-path output propagated for BEV visualization
  - `_commit_selected_path(...)` added so template-family state is preserved only when semantically valid
  - stale template-family memory cleared when graph or non-hold fallbacks take over

### Runtime / observability changes
- `simulation_camera_scooter/live_heading_demo.py`
  - Phase 11 toggle retained via `--no-template-planner`
  - max-frame replay support retained
  - GUI + video-save flow used for final validation
- `simulation_camera_scooter/heading.py`
  - planner slowdown still applied to controller speed
- `simulation_camera_scooter/data_logger.py`
  - logs planner confidence, margin, slowdown, selected family, and corridor stats
- `simulation_camera_scooter/visualization.py`
  - BEV HUD shows selected family, slowdown, and candidate fan

### Calibration fix
- `simulation_camera_scooter/config.py`
  - `CALIBRATION_FILE` now resolves to the module-local calibration file instead of accidentally loading the wrong repo-root file
- Validation artifact:
  - `simulation_camera_scooter/demo_outputs/bev_debug_compare.png`

### Tooling and tests
- `simulation_camera_scooter/scripts/eval_template_planner.py`
  - replay/evaluation helper retained for repeatable comparisons
- Tests added/updated:
  - `simulation_camera_scooter/tests/test_bev_calibration.py`
  - `simulation_camera_scooter/tests/test_realtime_nav_core.py`
  - `simulation_camera_scooter/tests/test_template_path_planner.py`
- Final automated result:
  - `py -m pytest simulation_camera_scooter/tests -q`
  - `101 passed`

## Final Validation Artifacts

### Phase 11 GUI replay
- Video:
  - `simulation_camera_scooter/demo_outputs/phase11_june_intent_reuse_gui/heading_demo_output.mp4`
- Log:
  - `simulation_camera_scooter/demo_outputs/phase11_june_intent_reuse_gui/logs/run_20260313_031009.csv`
- Saved frame sheet:
  - `simulation_camera_scooter/demo_outputs/phase11_june_intent_reuse_gui/frame_sheet.png`
- Saved BEV sheet:
  - `simulation_camera_scooter/demo_outputs/phase11_june_intent_reuse_gui/bev_sheet.png`
- Baseline-vs-Phase11 sheet:
  - `simulation_camera_scooter/demo_outputs/phase11_june_intent_reuse_gui/baseline_vs_phase11_sheet.png`

### Matched GUI baseline replay
- Video:
  - `simulation_camera_scooter/demo_outputs/baseline_june_intent_gui/heading_demo_output.mp4`
- Log:
  - `simulation_camera_scooter/demo_outputs/baseline_june_intent_gui/logs/run_20260313_031146.csv`
- Saved frame sheet:
  - `simulation_camera_scooter/demo_outputs/baseline_june_intent_gui/frame_sheet.png`

### Additional replay artifact from the tuning loop
- `simulation_camera_scooter/eval_runs/template_planner_intent220_reuse/summary.md`
- `simulation_camera_scooter/eval_runs/template_planner_intent220_reuse/frames/comparison_sheet.png`

## Final Matched GUI Comparison
- Video: `simulation_camera_scooter/test_video_june_03_3.mp4`
- Frames: `220`
- Detection: disabled for deterministic replay
- Calibration: corrected module-local calibration confirmed at runtime

### Corrected graph baseline
- mean abs heading: `1.190 deg`
- p95 abs heading: `3.724 deg`
- max abs heading: `4.950 deg`
- mean speed: `1.500 m/s`
- graph rate: `75.0%`
- fallback rate: `25.0%`
- path-source switches: `36`

### Final Phase 11
- mean abs heading: `0.707 deg`
- p95 abs heading: `3.508 deg`
- max abs heading: `5.575 deg`
- mean speed: `1.067 m/s`
- low-confidence rate: `37.7%`
- mean slowdown: `0.351`
- template rate: `62.3%`
- graph rate: `5.9%`
- fallback rate: `31.8%`
- path-source switches: `18`
- template-family switches: `3`

### Interpretation
- Phase 11 is now doing the intended kind of work:
  - it usually selects from the small template bank
  - it keeps the candidate fan readable
  - it reuses the same family for long stretches instead of switching constantly
- Relative to the corrected baseline, it is better on the two most important behavior metrics for this phase:
  - lower mean heading error
  - lower path-source switching
- It is still more conservative on speed because weak-confidence windows still trigger slowdown or reuse logic.
- Max heading is not better in every window, so the result is improved but not uniformly dominant.

## Old Path Planning Baseline
- File:
  - `simulation_camera_scooter/camera_waypoint_pipeline.py`
- Role in this phase:
  - structural baseline only
- Reason it is not the final comparison target:
  - it renders multiple Dijkstra branches rather than one approved controller path
  - it does not expose confidence, slowdown, or one final selected trajectory
- Representative prior probe from June frames showed the expected branch fan rather than one path decision.

## What Worked
- The BEV calibration bug is fixed and validated.
- The planner now behaves like a small reusable template selector instead of a noisy template fan.
- The startup tie-break prevents small score noise from forcing early turn selection.
- Candidate paths are visible and interpretable in BEV.
- Family reuse is stable enough to materially cut path-source switching.
- The final GUI replay is visually consistent with the numerical result.

## What Still Does Not Work Well Enough
- The planner still slows down too aggressively in weak-confidence windows.
- Some fallback / hold periods remain longer than ideal in the branch-heavy later part of the June clip.
- The separate phone-video smoke clip remains calibration-limited, so it is not yet evidence that the planner generalizes well.

## Known Limitations
- Corridor evidence still comes from the BEV mask, not the future boundary model.
- Replay metrics are sensitive to temporal smoothing and predictive reuse.
- The final validation is strongest on the calibrated June clip; generalization to new capture conditions is not solved yet.

## How To Run

### Full test suite
- `py -m pytest simulation_camera_scooter/tests -q`

### Final Phase 11 GUI replay with the corrected calibration
- workdir: `simulation_camera_scooter/demo_outputs/phase11_june_intent_reuse_gui`
- command:
  - `py ..\..\live_heading_demo.py --video ..\..\test_video_june_03_3.mp4 --max-frames 220 --save --log --no-detection --log-dir logs`

### Matching baseline GUI replay
- workdir: `simulation_camera_scooter/demo_outputs/baseline_june_intent_gui`
- command:
  - `py ..\..\live_heading_demo.py --video ..\..\test_video_june_03_3.mp4 --max-frames 220 --save --log --no-detection --no-template-planner --log-dir logs`

### Replay/evaluation helper
- `py simulation_camera_scooter/scripts/eval_template_planner.py --videos simulation_camera_scooter/test_video_june_03_3.mp4 --max-frames 220 --output-root simulation_camera_scooter/eval_runs/template_planner_intent220_reuse --save-video`

## Current Conclusion
- Phase 11 now works in the way it was intended to work on the calibrated June clip.
- It is meaningfully improved, understandable, and reproducible.
- It is not finished as a universally robust planner yet.
- Best honest status:
  - works on the calibrated June replay
  - better than the corrected graph baseline there
  - still partial overall
