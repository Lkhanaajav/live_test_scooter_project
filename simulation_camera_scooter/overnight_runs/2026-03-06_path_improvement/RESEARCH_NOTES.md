# RESEARCH NOTES

## Candidate Methods
- Considered for tonight:
  - robust low-confidence handling around empty/sparse BEV,
  - center/continuity-aware candidate scoring,
  - occupancy-aware prediction gating,
  - bounded discontinuity-hold in pure pursuit.
- Deferred as too heavy/risky for this session:
  - full dense inverse-distance A* re-planner each frame,
  - new model training/retraining,
  - architecture-level rewrites.

## Similar Work Review

### Source 1: Robust Path Planning in the Wild for Automatic Vehicle Following Using a Single Forward-Facing Camera
- Link: https://www.scitepress.org/publishedPapers/2024/130006/pdf/index.html
- Problem solved: stable center path extraction from noisy drivable segmentation.
- Key idea relevant here:
  - distance-transform-based center path,
  - endpoint/offshoot filtering near ego to reduce branch artifacts.
- Lightweight/heavy: lightweight-moderate classical CV.
- Practical for tonight: yes (adaptation, not full reimplementation).
- Tested tonight: partially (adapted via center/continuity scoring + low-evidence handling).
- Adopt/adapt/reject: adapted.

### Source 2: Pure Pursuit Path Tracking (Coulter, CMU-RI-TR-92-01)
- Link: https://www.ri.cmu.edu/publications/implementation-of-the-pure-pursuit-path-tracking-algorithm/
- Problem solved: geometric path tracking and lookahead behavior.
- Key idea relevant here:
  - stable tracking requires controlled reacquisition when path updates are discontinuous.
- Lightweight/heavy: lightweight.
- Practical for tonight: yes.
- Tested tonight: yes (bounded discontinuity hold/reacquire).
- Adopt/adapt/reject: adapted.

### Source 3: OpenCV morphology / distance transform / connected components docs
- Links:
  - https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html
  - https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html
  - https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
- Problem solved: practical denoising/cleanup and center-support extraction.
- Key idea relevant here:
  - occupancy, morphology, and DT-derived cues are effective lightweight confidence signals.
- Lightweight/heavy: lightweight.
- Practical for tonight: yes.
- Tested tonight: yes (occupancy-based gating + cleanup-aware fallback behavior).
- Adopt/adapt/reject: adopted.

## Adopted vs Rejected Tonight
- Adopted:
  - occupancy-aware predictor guard,
  - low-evidence aggressive hold decay,
  - center/continuity candidate cost terms,
  - bounded discontinuity-hold in controller,
  - path-source/occupancy diagnostics in logs + BEV HUD.
- Rejected:
  - additional low-confidence recenter parameter tuning pass (E2) due regression.

### Similar Work Review (afternoon addendum)
- **Classical skeleton geodesic centerline extraction (graph/BFS on skeleton pixels)**
  - Problem solved: recover a full center route when branch graph extraction is fragmented or over-pruned.
  - Relevant idea: compute path on raw skeleton pixels from ego seed to farthest/most-forward endpoint.
  - Lightweight/heavy: lightweight (no new dependencies, pixel-level Dijkstra/BFS).
  - Practical tonight: yes.
  - Tested: yes (`fallback_skeleton` in `realtime_nav_core.py`).
  - Adopt/adapt/reject: adopted as fallback candidate/path source; helps in some failure windows but does not fully replace graph selection in all hard frames.
