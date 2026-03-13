# Phase 11 Research

## Problem Framing
- With the BEV calibration fixed, Phase 11 should not behave like graph search with extra scoring.
- It should behave like a small reusable intent selector operating directly in BEV space.
- The user-level behavior target is:
  - keep a few pre-created path families
  - fit them against the segmentation corridor
  - show 3-5 plausible paths
  - reuse the current winner until there is strong evidence to switch

## Key Correction In Understanding
- The earlier dramatic planner win was not trustworthy because the BEV transform was wrong.
- After inspecting real frames, the true problem became clearer:
  - segmentation was already acceptable
  - path selection logic was the part that needed redesign
- Once the correct module calibration was loaded, the right question was no longer "why is BEV broken?".
- The right question became "how should a reusable path-intent selector behave when the BEV is finally correct?".

## Repo-Specific Constraints
- The existing controller expects a smooth metric path and already works with cubic path models.
- Runtime has to stay practical on CPU.
- Phase 7 boundary predictions are not yet live in the main loop, so Phase 11 still has to score paths from the cleaned BEV mask.
- The old graph / centerline logic remains valuable as:
  - a fallback when template evidence is weak
  - a baseline for comparison

## Approaches Considered

### Option A: Keep graph-first planning and only improve graph scoring
- Rejected as the primary Phase 11 design.
- Reason:
  - graph search is useful for recovery, but it does not naturally express a small set of reusable motion intents.

### Option B: Large bank of always-turning templates
- Rejected.
- Reason:
  - it produced a noisy BEV candidate fan
  - it encouraged premature turn commitment
  - it did not match the product behavior target

### Option C: Small intent bank with delayed-turn variants
- Selected.
- Reason:
  - matches the intended behavior best
  - keeps the BEV candidate display readable
  - makes reuse / hysteresis straightforward
  - gives a small set of paths that are easy to debug visually

### Option D: Full state-lattice or dense optimization
- Rejected for this phase.
- Reason:
  - more machinery than this codebase currently needs
  - harder to validate visually and incrementally
  - unnecessary before proving the simple template-intent selector

## External Research Findings

### 1. OpenCV morphology, distance transform, and connected components
- Sources:
  - https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html
  - https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html
- Practical takeaway:
  - a lightweight corridor extracted from the BEV mask is sufficient for explainable path scoring
  - row-wise support, width consistency, and forward span are enough to define a usable corridor contract

### 2. Coulter, "Implementation of the Pure Pursuit Path Tracking Algorithm"
- Source:
  - https://www.ri.cmu.edu/publications/implementation-of-the-pure-pursuit-path-tracking-algorithm/
- Practical takeaway:
  - controller-facing path continuity matters more than chasing a new per-frame optimum
  - reusing a good path is a feature, not a compromise

### 3. Motion primitive / state-lattice work (Pivtoraiko, Kelly)
- Source:
  - https://spacefrontiers.org/r/10.1002/rob.20285
- Practical takeaway:
  - a bounded set of feasible path primitives is a legitimate planning abstraction
  - the transferable idea here is the small primitive bank, not a full lattice implementation

### 4. Parametric lane-shape fitting work
- Source:
  - https://openaccess.thecvf.com/content/CVPR2024/html/Xie_End-to-End_Lane_Shape_Prediction_With_Transformers_via_Least_Squares_Fitting_CVPR_2024_paper.html
- Practical takeaway:
  - smooth parametric curves remain a strong representation even when perception is pixel-based

### 5. Temporal lane-video literature
- Source:
  - https://openaccess.thecvf.com/content/ICCV2023/html/Chen_Recursive_Video_Lane_Detection_ICCV_2023_paper.html
- Practical takeaway:
  - temporal continuity and hysteresis are required for stable behavior in video
  - one-frame-only optimality is the wrong objective

## Final Chosen Design

### Corridor model
- Extract a corridor from the cleaned BEV mask.
- Score corridor quality using:
  - valid-row ratio
  - near-field valid ratio
  - forward span
  - width consistency
  - occupancy support

### Template bank
- Keep the bank intentionally small:
  - `straight_center`
  - `left_near`, `left_mid`, `left_late`
  - `right_near`, `right_mid`, `right_late`
- Use delayed turn onset instead of a large family of arbitrary curves.
- Keep the top 3-5 candidates visible in BEV.

### Template scoring
- Score each template by:
  - containment inside corridor bounds
  - near-field containment
  - supported ratio
  - clearance from corridor edges
  - centerline alignment
  - continuity with the previous path
  - curvature feasibility
  - obstacle overlap penalty

### Selection and reuse policy
- Prefer the previously selected family when evidence is similar.
- Require stronger evidence to switch families.
- If the same family still fits reasonably well, allow reuse instead of immediate fallback.
- Add a startup rule:
  - if the scene is still effectively straight and a turn template only barely leads, prefer `straight`
- Make that startup straight rule obstacle-aware so it does not override a genuinely safer avoidance path.

### State hygiene
- Preserve template-family memory only when that memory is still semantically valid.
- Keep family memory across:
  - template reuse
  - short `fallback_hold` reuse of the previous path
- Clear family memory when the planner actually changes authority to:
  - graph
  - fallback centerline
  - fallback skeleton
- Reason:
  - stale intent memory is a real failure mode in reusable-template planners

## Why This Design Was Chosen
- It matches the intended product behavior better than the earlier planner attempts.
- It makes the BEV output understandable to a human reviewer.
- It turns Phase 11 into explicit path approval, not just another path generator.
- It keeps the legacy graph path available without letting it dominate when the corridor is clean.

## Evidence From The Corrected June Replay
- The current selector now behaves like a stable intent system rather than a noisy template fan.
- Final matched GUI replay on `simulation_camera_scooter/test_video_june_03_3.mp4`:
  - baseline mean abs heading `1.190 deg`
  - Phase 11 mean abs heading `0.707 deg`
  - baseline p95 `3.724 deg`
  - Phase 11 p95 `3.508 deg`
  - baseline path-source switches `36`
  - Phase 11 path-source switches `18`
  - template-family switches only `3`
- Visual frame sheets confirm the same story:
  - the BEV shows a small reusable candidate fan
  - the selected path stays straight unless the corridor meaningfully justifies something else

## Remaining Open Questions
- How much more speed should be given back during weak-confidence windows without increasing heading instability?
- Should the turn families get one sharper hook variant each, or is the current onset-shifted bank enough?
- Once Phase 7 boundary predictions are live, should corridor scoring replace or complement the mask-derived corridor?

## Final Research Conclusion
- The best practical Phase 11 design in this repo is not a larger template bank or a more complex optimizer.
- It is a small reusable intent bank with explicit tie-break rules, family reuse, and state hygiene.
- After fixing BEV calibration and rebuilding the selector around that idea, Phase 11 finally behaves like path approval inside the segmentation corridor rather than unstable graph-like search.
