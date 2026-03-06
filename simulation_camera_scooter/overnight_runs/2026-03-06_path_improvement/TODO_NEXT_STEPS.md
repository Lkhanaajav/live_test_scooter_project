# TODO NEXT STEPS

1. Add a dedicated BEV confidence scalar in `PathPlanResult` (not just occupancy) using combined cues:
   - occupancy,
   - forward reach,
   - skeleton continuity,
   - graph edge count.
2. Use confidence-adaptive controller gains:
   - lower steering aggressiveness and stronger center pull when confidence is low.
3. Implement optional inverse-distance path search (lightweight grid DP/A*) as an ablation path selector for low-confidence frames.
4. Add side-by-side comparison rendering utility (baseline vs selected run) for thesis/demo videos.
5. Add automated metric script for collapse windows:
   - lock duration,
   - time-to-recover,
   - source transition counts (`graph/predict/fallback`).
6. Re-validate selected logic on additional videos (especially with harder shake/lighting) when available.
7. For thesis/demo:
   - include path-source + occupancy overlay snapshots to explain robust fallback behavior.

## Added from afternoon deep-dive
- Add per-frame debug export of candidate diagnostics (graph vs skeleton fallback):
  - near-ego lateral at x=1.0m,
  - progress/length,
  - cost terms,
  - final hysteresis decision reason.
- Add explicit near-ego branch-angle penalty term using first 0.8-1.2m segment heading (not only end heading), to suppress tiny-triangle branch capture.
- Evaluate graph extraction robustness improvement:
  - reduce spurious skeleton node fragmentation and preserve connectivity near branch junctions,
  - compare 8-neighbor vs directional continuation rules.
