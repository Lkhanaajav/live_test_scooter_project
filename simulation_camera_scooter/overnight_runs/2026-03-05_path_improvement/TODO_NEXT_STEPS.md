# TODO Next Steps

## Highest-value technical next steps
- Add explicit `path_confidence` output (graph path vs fallback vs hold) and use it to modulate steering gain/speed.
- Add confidence-aware blend: use graph path when available, blend toward fallback only when graph quality drops.
- Add a simple branch-free centerline mode in camera space as a second fallback when BEV survival is near zero.
- Add per-frame debug text in BEV HUD: `mode=GRAPH/FALLBACK/HOLD`, `cost`, `confidence`, `forward_span_m`.

## When better videos / better calibration are available
- Re-run full baseline vs E4 with mounted-camera calibration and compare:
  - has_path
  - heading jump stats
  - center offset in camera overlay
- Retune `fallback_output_lateral_clip_m` upward once BEV geometry is trustworthy.
- Re-enable stronger center preference in fallback after calibration quality improves.

## Remaining weak points
- Path overlay still shows lateral bias on some segments (projection/calibration sensitive).
- Current fallback can be conservative (shorter path length, limited lateral motion).
- No explicit confidence signal is logged yet; analysis is inferred from behavior.

## Thesis / demo suggestions
- Present a clear ablation figure: baseline (0% path) vs E1 (recover path) vs E4 (recover + stabilize steering).
- Include representative frame triplets in thesis appendix:
  - baseline no-path frame
  - E1 recovered but biased frame
  - E4 stabilized frame
- In demo script, mention graceful degradation strategy: graph extraction -> fallback centerline -> hold-last-path.
