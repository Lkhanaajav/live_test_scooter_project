# Research Notes

## Candidate methods for tonight (pre-implementation)
- Robust mask cleanup after segmentation: connected-component filtering, morphology close/open, hole filling.
- BEV region stabilization: ROI weighting and boundary smoothing.
- Candidate scoring upgrades: center-distance reward, clearance reward, curvature penalty, temporal continuity cost.
- Selection stability: hysteresis with switch margin and previous-path consistency.
- Fallback behavior: confidence-triggered hold-last-path and gradual recovery.

## Constraints from project docs
- Focus on practical path reliability and visible behavior in output video.
- Keep solution lightweight and maintainable; avoid model retraining tonight.

## Notes from planning docs
- Existing pipeline already has tier1/tier2 mask cleanup and branch hysteresis, but current behavior can still fail under noisy masks and unstable BEV boundaries.
- Plan docs emphasize calibration as primary root cause; for this overnight task we still optimize the downstream extractor/selector robustness to imperfect masks.
- Available lightweight levers for tonight: _preprocess morphology/CC filtering, candidate scoring terms, hysteresis persistence, discontinuity/hold logic, and debug overlays.

## Candidate methods considered after baseline
- High value now: add robust fallback path generation when graph search returns no candidates (scanline centerline + temporal hold), because current primary failure is path starvation, not path jitter.
- Deferred for later iteration: fine candidate scoring/hysteresis tuning (center/clearance/curvature) after path existence is restored.

## Methods selected vs rejected (overnight)
### Selected
- Graph-path fallback hierarchy (graph -> fallback centerline -> hold previous path).
- Sparse-mask tolerant preprocessing (relaxed width under low occupancy).
- Lateral-clipped fallback output to reduce one-sided command lock.

### Rejected
- Aggressive confidence recentering (E2): produced persistent -30 deg steering lock.

### Deferred
- Full confidence-scored path blending between graph and fallback.
- Camera-space centerline fallback independent of BEV homography.
