# Phase 11 Video Evaluation

## Evaluation Notes
- `old path planning` in this document refers to `simulation_camera_scooter/camera_waypoint_pipeline.py`.
- `corrected graph baseline` refers to `simulation_camera_scooter/realtime_nav_core.py` with `template_planner_enabled=False`.
- `new Phase 11` refers to the same live planner with template approval enabled.
- The old planner remains a useful structural baseline, but it is not directly comparable on controller metrics because it renders many Dijkstra branches instead of one approved controller path.
- Final visual validation in this document uses GUI-on saved-video runs, not headless-only summaries.

## Videos Used
- `simulation_camera_scooter/test_video_june_03_3.mp4`
- `simulation_camera_scooter/test_videos/IMG_1876.MOV`

## Video: `simulation_camera_scooter/test_video_june_03_3.mp4`
- Frames evaluated in final matched replay: `220`
- Scene characteristics:
  - calibrated repo clip
  - mostly straight sidewalk with branch / intersection geometry later in the sequence

### Old path planning result
- File:
  - `simulation_camera_scooter/camera_waypoint_pipeline.py`
- Qualitative result:
  - produces a branch fan, not one controller-ready path
- Practical takeaway:
  - still useful as a reference for skeleton extraction behavior
  - not useful as the final decision layer for scooter control

### Corrected graph baseline result
- Video artifact:
  - `simulation_camera_scooter/demo_outputs/baseline_june_intent_gui/heading_demo_output.mp4`
- Log artifact:
  - `simulation_camera_scooter/demo_outputs/baseline_june_intent_gui/logs/run_20260313_031146.csv`
- Saved frame sheet:
  - `simulation_camera_scooter/demo_outputs/baseline_june_intent_gui/frame_sheet.png`
- Summary:
  - mean abs heading `1.190 deg`
  - p95 abs heading `3.724 deg`
  - max abs heading `4.950 deg`
  - mean speed `1.500 m/s`
  - graph rate `75.0%`
  - fallback rate `25.0%`
  - path-source switches `36`
- Visual observation:
  - the baseline generally stays usable on this clip
  - in the branch-heavy part of the replay, the chosen path bends and re-centers more abruptly than the new template selector
  - the BEV view shows one selected graph path but not an explicit reusable intent set

### Final Phase 11 result
- Video artifact:
  - `simulation_camera_scooter/demo_outputs/phase11_june_intent_reuse_gui/heading_demo_output.mp4`
- Log artifact:
  - `simulation_camera_scooter/demo_outputs/phase11_june_intent_reuse_gui/logs/run_20260313_031009.csv`
- Saved frame sheet:
  - `simulation_camera_scooter/demo_outputs/phase11_june_intent_reuse_gui/frame_sheet.png`
- Saved BEV sheet:
  - `simulation_camera_scooter/demo_outputs/phase11_june_intent_reuse_gui/bev_sheet.png`
- Side-by-side sheet:
  - `simulation_camera_scooter/demo_outputs/phase11_june_intent_reuse_gui/baseline_vs_phase11_sheet.png`
- Summary:
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
  - the BEV now shows a small candidate fan rather than an unstable branch fan
  - the selected family stays `straight` through most of the clip
  - path reuse is visible in the later branch region instead of repeated family churn

### Best examples on this video
- Frame 40:
  - Phase 11 shows a clean 5-path fan with a stable straight winner.
  - Baseline is still fine there, but it does not expose the same intent-level structure.
- Frame 100:
  - Phase 11 keeps a straighter, calmer selected path in BEV.
  - Baseline shows a more noticeable bend / correction in the chosen path.
- Frame 140:
  - Phase 11 continues to reuse a consistent straight family while the corridor is partially ambiguous.

### Failure examples on this video
- Later weak-confidence windows still cause Phase 11 to reduce speed more than the baseline.
- Max heading is not lower on every frame; the new selector is better on average, not strictly better everywhere.

### Verdict on this video
- Phase 11 is materially better than the corrected graph baseline on the calibrated June replay for the main goals of this phase:
  - smoother mean path behavior
  - fewer path-source switches
  - interpretable 3-5 candidate paths
  - better reuse / hysteresis behavior
- It is still more conservative on speed.

## Video: `simulation_camera_scooter/test_videos/IMG_1876.MOV`
- Role in this session:
  - smoke test only
- Earlier replay conclusion:
  - current calibration still does not map this clip into a strong corridor
  - both planners degrade toward fallback behavior
- Interpretation:
  - this is not yet a fair benchmark for final planner quality
  - the clip is still dominated by calibration mismatch, not by the Phase 11 selector itself

## Cross-Video Conclusions
- On the calibrated repo video, the new Phase 11 selector is now the better path-selection layer.
- On the phone-video smoke clip, calibration still dominates and hides true planner quality.
- The old legacy planner remains useful as a baseline reference, but it is not sufficient as a final controller-facing planner because it never resolves the branch fan into one approved path.

## Overall Video Conclusion
- Phase 11 now looks logically correct in BEV, not just numerically improved.
- The saved GUI frames and BEV sheets match the design intent:
  - small reusable path bank
  - stable winner
  - visible candidate paths
  - reuse instead of per-frame flapping
- Current honest outcome:
  - good result on the calibrated June clip
  - partial overall result across all capture conditions
