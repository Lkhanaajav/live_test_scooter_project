# VISUAL FINDINGS

## Baseline Observations
- Primary baseline (predict ON) is mostly stable in normal segments, but fails during low-confidence windows.
- Around frames ~1520-1600:
  - segmentation confidence collapses (`seg_iou` near zero),
  - BEV occupancy drops to near-empty or empty,
  - steering remains latched around `-10.6 deg` too long.
- Paper-target mismatch in these windows: paper figures show a clean central trunk and controlled branch selection; baseline shows stale-path persistence after BEV evidence collapses.

## E1 Observations (Selected)
- During the same collapse window, path source transitions are explicit and sensible: `graph/predict -> fallback_centerline -> fallback_hold -> fallback_centerline/graph`.
- Steering recovers faster and does not stay pinned at large magnitude as long.
- Visual overlays show improved behavior:
  - at `frame_1530`: near-neutral command while BEV occupancy is low,
  - at `frame_1550`: fallback hold active with controlled magnitude,
  - at `frame_1565`: hold has decayed close to center,
  - at `frame_1570+`: fallback centerline/graph reacquire without prolonged lock.

## Remaining Weaknesses
- In severe degradation windows, fallback still can produce short transient bias before full recovery.
- Occasional heading spikes remain in difficult frames with abrupt segmentation changes.
- A dense clearance-optimal planner (e.g., inverse-DT A*) was not fully integrated tonight; current approach remains lightweight by design.

## BEV Assessment
- Is the BEV transform working reasonably?
  - Yes on this June clip. Mean BEV survival in baseline was ~63%, and BEV region is usually geometrically usable.
- Main BEV failure modes:
  - intermittent occupancy collapse to sparse islands/empty mask,
  - temporary fragmentation near difficult lighting/tree-shadow segments,
  - low-confidence windows where topological continuity breaks.
- How much do those failures hurt path planning?
  - High impact. In baseline they directly caused long stale steering lock despite path availability being reported.
- BEV improvements attempted:
  - occupancy-aware predictor guard (`predict_empty` handling),
  - low-evidence-aware fallback hold decay,
  - logging/overlay of `path_source` + `mask_occ_ratio` for explicit diagnosis.
- Which BEV fix helped most?
  - occupancy-aware stale-path suppression plus aggressive low-evidence hold decay (E1).
- Is final BEV more stable/usable than baseline?
  - Yes in practical navigation output: recovery from BEV collapse is faster and less biased.

## Paper/Image Comparison
- Which paper/report images were inspected?
  - `bev_mask_raw.png`, `bev_clean.png`, `bev_skeleton.png`, `skeleton_paths_overlay.png`, `planned_vs_skeleton_overlay.png`, `cam_paths_0001.png`.
- What looked better there than current outputs?
  - cleaner contiguous BEV region,
  - dominant center trunk with meaningful branches,
  - branch selection that stays centered and avoids noisy side artifacts.
- What concrete clues were learned?
  - prioritize medial/center preference and suppress stale branch/path persistence under weak evidence,
  - make low-confidence mode explicit and conservative rather than trusting old paths.
- What changes were attempted because of those clues?
  - center/continuity candidate scoring,
  - occupancy-gated predictor reuse,
  - aggressive low-evidence hold decay,
  - source/confidence debug overlays.
- Did those changes help?
  - Yes. E1 reduced collapse-window lock dramatically and improved overall heading stability metrics.

## Additional Visual Findings (2026-03-06 afternoon)

### Skeleton vs selected-path mismatch (deep check)
- Deep diagnostic at frame ~350 showed:
  - skeleton leaves exist far ahead (multiple leaves near forward x ~= 10m),
  - but selected graph path remained short (progress ~= 1.8m in isolated frame checks), indicating reachable-path mismatch.
- Interpretation: Dijkstra is running, but the graph seen from selected start is under-connected/under-constrained; the selected candidate can fail to represent full visible skeleton branch extent.

### Frame snapshots from `verify_no_predict_skel_fallback`
- Frames inspected: 350, 355, 360.
- BEV showed branch point near ego with selected cyan path taking leftward early branch while trunk continues forward.
- Path source at these frames was still `graph` in full run logs, with large negative heading spikes.

### Frame snapshots from `verify_no_predict_skel_override`
- Frames inspected: 230, 350, 355, 360, 1460, 1530, 1580, 1590.
- Early-false-turn window (350/355/360) still showed graph-selected spikes in full run despite targeted override.
- Late difficult windows (around 1530+) still depend on hold/fallback logic under segmentation collapse.

### BEV Assessment update
- BEV transform remains geometrically reasonable for global layout.
- Main remaining issue is not gross BEV warp; it is branch selection sensitivity near ego branch artifacts and graph under-connectivity under noisy masks.
- Most impactful BEV-adjacent mitigation remains conservative fallback behavior (centerline/skeleton) rather than transform change itself.

### Paper/Image Comparison update
- Paper target indicates stable trunk-following centerline at branch entries.
- Current difficult windows still deviate by selecting a short branch near ego before trunk continuation.
- Added pure-skeleton geodesic fallback directly motivated by this mismatch; it helps in some windows but does not yet fully dominate selection at all failure frames.
