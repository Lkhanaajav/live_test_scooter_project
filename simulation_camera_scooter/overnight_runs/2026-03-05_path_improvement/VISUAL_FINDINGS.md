# Visual Findings

## Baseline (E0)
Artifacts: `baseline/frames/`, `baseline/baseline_mar3_overlay.mp4`
- BEV panel frequently almost empty (black) with only tiny green fragments near bottom.
- No candidate path drawn in BEV; no projected path in camera view.
- Path planner effectively starved (`has_path=0%`).

## E1 (fallback recovery)
Artifacts: `exp1_fallback/frames/`, `exp1_fallback/exp1_mar3_overlay.mp4`
- Path appears consistently (major improvement vs baseline).
- However, path frequently biases to one side and steering settles around +20..+25 deg for long segments.
- Stable but not well-centered.

## E2 (aggressive recenter, rejected)
Artifacts: `exp2_recenter/`
- Visual behavior showed persistent side lock with near-saturated steering.
- Rejected due unrealistic command behavior.

## E3 (ego-anchor rollback)
Artifacts: `exp3_ego_anchor/`
- Improved over E2 but still one-sided steering lock tendency remained.
- Not selected as final.

## E4 (selected best)
Artifacts: `exp4_lateral_clip/frames/`, `exp4_lateral_clip/exp4_lateral_clip_mar3_overlay.mp4`
- Continuous path remains available across frames.
- Steering bias magnitude reduced significantly vs E1/E3.
- No prolonged +/-30 deg lock observed.
- Remaining weakness: projected path still appears left-biased in many frames (likely calibration/projection issue), but practical behavior is markedly better than baseline collapse.

## Priority Failures Remaining
1. Projection/calibration bias still affects absolute centering in camera overlay.
2. Fallback path is conservative and can under-represent true curvature.
3. Need explicit path-confidence overlay (`GRAPH/FALLBACK/HOLD`) for clearer diagnosis.
