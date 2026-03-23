# Roadmap: Scooter Sidewalk Navigation System

## Overview

The pipeline is substantially built. The two original root blockers, segmentation flicker and unusable BEV calibration, have now been addressed. The main remaining work is live demo integration plus robustness against branch-entry artifacts and low-confidence windows. Phase 4 (Radxa/ONNX) is a stretch goal — the thesis demo is valid on a laptop.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Segmentation Stability** - Eliminate visible per-frame flickering in SegFormer output on real sidewalk video (completed 2026-03-05)
- [x] **Phase 2: BEV Calibration and Path Reliability** - Recalibrate homography and achieve has_path >= 60% on typical sidewalk footage (formalized 2026-03-12)
- [ ] **Phase 3: Demo Integration** - End-to-end demo run with scooter receiving commands, visualization overlay, and graceful degradation
- [ ] **Phase 4: Radxa Deployment (STRETCH)** - Export pipeline to ONNX/TensorRT and run on Rock 5B at >= 5 Hz

## Phase Details

### Phase 1: Segmentation Stability
**Goal**: Segmentation output is stable enough that flickering is not visible to a demo observer and does not corrupt downstream BEV input
**Depends on**: Nothing (first phase)
**Requirements**: SEG-01, SEG-02, SEG-03
**Success Criteria** (what must be TRUE):
  1. Playing back the demo sidewalk video, consecutive segmentation masks look visually stable — no frame-to-frame class flipping visible to the eye
  2. IoU between consecutive frames >= 0.85 on >= 90% of frames, measured by running the pipeline on the representative demo video
  3. SegFormer checkpoint is validated (or fine-tuned) on the actual outdoor demo environment — correct sidewalk class is predicted on representative frames without gross errors
  4. Temporal smoothing alpha is tuned so dynamic obstacles (people walking) still update within 2-3 frames rather than being frozen by over-smoothing
**Plans**: TBD

Plans:
- [x] 01-01: Evaluate SegFormer checkpoints and benchmark temporal stability on demo video
- [x] 01-02: Tune temporal smoothing (EMA alpha, IoU thresholds) to hit stability targets

### Phase 2: BEV Calibration and Path Reliability
**Goal**: BEV homography is accurate for the actual recording setup and the path extractor reliably finds a drivable path
**Depends on**: Phase 1
**Requirements**: BEV-01, BEV-02, BEV-03, PATH-01, PATH-02, PATH-03
**Success Criteria** (what must be TRUE):
  1. Running load_bev_params() does NOT print the ill-conditioned WARNING (condition number < 1e6, down from 1.1e+06 current) — note: cond < 1000 is not achievable with a real perspective warp; pixel survival (criterion 2) is the primary metric
  2. Inspecting BEV output frames, >= 50% of sidewalk pixels survive the warpPerspective transform (visible as a filled region in BEV view)
  3. has_path rate is >= 60% of frames when running the pipeline on a representative sidewalk video clip
  4. The extracted skeleton path follows the sidewalk centerline in BEV — it does not jump to edges, grass artifacts, or reverse direction on a straight sidewalk
  5. Calibration steps are written up so the procedure can be repeated when the camera mount changes
**Plans**: 2/2 complete

Plans:
- [x] 02-01-PLAN.md — Recalibrate BEV homography, verify condition number and pixel survival, write calibration SOP
- [x] 02-02-PLAN.md — Validate path extraction reliability (has_path >= 60%), heading stability, and centerline behavior

### Phase 3: Demo Integration
**Goal**: The full pipeline runs end-to-end during a live sidewalk demo — scooter receives steering and speed commands, the video feed shows a path overlay, and the system never abruptly stops
**Depends on**: Phase 2
**Requirements**: DEMO-01, DEMO-02, DEMO-03, DEMO-04
**Success Criteria** (what must be TRUE):
  1. The camera view displays a colored path overlay projected back from BEV onto the live image — visible and correct during a demo walkthrough
  2. Scooter receives and executes steering and speed commands over serial in real-time during a demo run (commands change as the scooter turns)
  3. End-to-end pipeline runs at >= 8 Hz on the demo laptop throughout the demo run (no slowdowns that drop below threshold)
  4. When segmentation temporarily fails (IoU drop), the scooter reduces speed and holds the last valid heading rather than stopping or oscillating — recoverable without manual intervention
**Plans**: TBD

Plans:
- [ ] 03-01: Integrate and test serial scooter command output during a real or hardware-in-the-loop demo run
- [ ] 03-02: Implement and validate graceful degradation behavior (speed hold-down, fallback decay, no abrupt stops)

### Phase 03.1: YOLO BEV Obstacle Projection (INSERTED)

**Goal:** Project YOLOv8-nano detected bounding boxes onto the BEV plane using the existing homography, converting each detection into a 2-D metric exclusion zone. The BEVPathExtractor penalizes candidate paths that pass through exclusion zones and prefers alternative branches when the primary path is blocked.
**Requirements**: OBS-01, OBS-02, OBS-03, OBS-04, OBS-05, OBS-06, OBS-07, OBS-08, OBS-09
**Depends on:** Phase 3
**Plans:** 4/4 plans complete

Plans:
- [x] 03.1-01-PLAN.md — Wave 0: test stubs for all 9 OBS requirements in tests/test_bev_obstacle.py + conftest fixtures
- [x] 03.1-02-PLAN.md — Wave 1: bev_obstacle.py (project_foot_to_bev, detection_to_metric, ObstacleEMAGrid) + config constants; 5 tests green
- [x] 03.1-03-PLAN.md — Wave 2: integrate into realtime_nav_core.py (_obstacle_penalty) + live_heading_demo.py (projection loop, hard-block, EMA); all 9 tests green
- [x] 03.1-04-PLAN.md — Wave 3: BEV HUD visualization (draw_bev_hud obstacle circles) + human checkpoint

### Phase 4: Radxa Deployment (STRETCH)
**Goal**: Pipeline runs on the Rock 5B ARM64 board at >= 5 Hz with scooter connected — eliminates need to carry a laptop
**Depends on**: Phase 3
**Requirements**: RADXA-01, RADXA-02
**Success Criteria** (what must be TRUE):
  1. SegFormer is exported to ONNX or TensorRT format and runs inference on Rock 5B at >= 5 Hz (measurable with the built-in FPS profiling)
  2. The full pipeline (segmentation + BEV + path + control + serial output) completes a demo run on Rock 5B with the scooter physically connected and responding to commands

**Note: This phase is a stretch goal. The thesis demo is accepted on a laptop. Only begin Phase 4 if Phases 1-3 are complete with > 3 weeks remaining before the demo date.**

**Plans**: TBD

Plans:
- [ ] 04-01: Export SegFormer to ONNX/TensorRT and benchmark on Rock 5B
- [ ] 04-02: Full pipeline end-to-end test on Rock 5B with scooter connected

### Phase 5: Cloud-Offloaded Navigation (STRETCH RESEARCH)
**Goal**: Evaluate cloud-offloaded inference vs. local CPU — quantify the latency/model-quality tradeoff and assess feasibility for real-time scooter control
**Depends on**: Phase 3 (need a working closed-loop baseline to compare against)
**Requirements**: CLOUD-01, CLOUD-02
**Success Criteria** (what must be TRUE):
  1. Cloud variant uses a larger, higher-accuracy segmentation model (e.g., SegFormer-B4/B5 or Mask2Former) running on a GPU instance — segmentation quality is measurably better than B0 local (mIoU or mask stability improvement reported)
  2. Round-trip latency (frame capture → cloud inference → command received) is measured and compared against local CPU latency — with analysis of whether the latency budget allows safe pedestrian-speed control
  3. A tradeoff table is produced: local (low latency, smaller model) vs. cloud (higher latency, better model, network-dependent) with concrete numbers
  4. Results are written up as a thesis section — either demonstrating feasibility or providing a clear quantitative argument for why local compute is preferred

**Note: This is a stretch research phase. Only pursue if Phases 1-3 are complete AND hardware/cloud resources are available. Decision point: after Phase 3 completion. Even if not executed, the tradeoff analysis can be written as a Future Work section in the thesis.**

**Plans**: TBD

Plans:
- [ ] 05-01: Measure local baseline latency end-to-end and set up cloud inference endpoint (GPU instance + streaming protocol)
- [ ] 05-02: Run comparative experiment — local B0 vs cloud B4/B5 — latency, stability, has_path rate, segmentation quality

## Progress

**Execution Order:**
Phases execute in numeric order, with Phase 3 as the next mainline target. Phases 7-10 are architecture-upgrade tracks that follow the working demo baseline rather than replacing the need to complete Phase 3.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Segmentation Stability | 2/2 | Complete   | 2026-03-05 |
| 2. BEV Calibration and Path Reliability | 2/2 | Complete   | 2026-03-12 |
| 3. Demo Integration | 0/2 | Not started | - |
| 3.1 YOLO BEV Obstacle Projection | 4/4 | Complete | 2026-03-09 |
| 4. Radxa Deployment (STRETCH) | 0/2 | Not started | - |
| 5. Cloud-Offloaded Navigation (STRETCH RESEARCH) | 0/2 | Not started | - |
| 6. Path Quality Improvements | 0/2 | Planned | - |
| 7. Lightweight Sidewalk-Boundary Network | 0/1 | In progress | - |
| 8. Boundary-Aware Segmentation Backbone | 0/0 | Not planned | - |
| 9. Shared-Backbone Multitask Perception | 0/0 | Not planned | - |
| 10. Tiny Image-to-Waypoints Student | 0/0 | Not planned | - |
| 11. Template Path Approval Scoring | 0/4 | Planned | - |
| 11.1 GPS-Intent Corridor Waypoint Turn Planner | 1/4 | In Progress|  |

### Phase 6: Path quality improvements: post-selection smoothing, BEV mask morphological closing, stronger temporal continuity weight, and draw fitted cubic on overlay

**Goal:** Improve visible path quality on the camera overlay and reduce frame-to-frame instability: smooth lateral jitter in the extracted path before cubic fitting, fill larger BEV mask gaps with stronger morphological closing, increase temporal continuity weight to reduce branch-flipping, and draw the fitted cubic spline (not raw skeleton pixels) on the camera overlay.
**Requirements**: PATH-SMOOTH-01, MORPH-CLOSE-01, CONT-WEIGHT-01, CUBIC-OVERLAY-01
**Depends on:** Phase 5 (planned after Phase 03.1 completion)
**Plans:** 2 plans

Plans:
- [ ] 06-01-PLAN.md — Wave 1: Write 10 failing test stubs in tests/test_path_quality.py + noisy_path_m fixture in conftest.py
- [ ] 06-02-PLAN.md — Wave 2: Implement all 4 improvements in realtime_nav_core.py + live_heading_demo.py; all 10 tests green

### Phase 7: Lightweight sidewalk-boundary network with analytical centerline and confidence gating

**Goal:** Replace the most fragile part of the path extractor with a lightweight boundary-first representation: predict left/right sidewalk boundaries, derive centerline analytically, and expose path confidence for controller gating.
**Requirements**: BOUNDARY-01, BOUNDARY-02, BOUNDARY-03
**Depends on:** Phase 3 baseline
**Plans:** 1 plan

Plans:
- [ ] 07-01-PLAN.md - Foundation: boundary-target export, tiny baseline model, metric centerline decoder, smoke evaluation

### Phase 8: Real-time boundary-aware segmentation backbone replacement for sidewalk navigation

**Goal:** Upgrade the tiny boundary/perception backbone to a real-time boundary-aware segmenter that preserves sidewalk edges more reliably on small compute while keeping deployment practical.
**Requirements**: TBD
**Depends on:** Phase 7
**Plans:** 4 plans

Plans:
- [ ] TBD (run /gsd:plan-phase 8 to break down)

### Phase 9: Shared-backbone multitask perception for sidewalk, boundaries, and obstacles

**Goal:** Collapse duplicate perception cost into one shared backbone that predicts sidewalk structure and obstacles together, reducing total compute and synchronization complexity.
**Requirements**: TBD
**Depends on:** Phase 8
**Plans:** 4 plans

Plans:
- [ ] TBD (run /gsd:plan-phase 9 to break down)

### Phase 10: Tiny image-to-waypoints student policy for low-compute scooter control

**Goal:** Distill the navigation stack into a tiny image-to-waypoints student that predicts a small set of future path targets directly, keeping classical low-level control but minimizing runtime perception/planning cost.
**Requirements**: TBD
**Depends on:** Phase 9
**Plans:** 0 plans

Plans:
- [ ] TBD (run /gsd:plan-phase 10 to break down)

### Phase 11: Template path fitting inside segmentation corridor with path approval scoring

**Goal:** Replace raw centerline following with a small-compute, GPS intent-conditioned path planner: given a commanded maneuver (`straight`, `left`, `right`), generate only intent-consistent smooth candidate paths in BEV, score how well they fit inside the perceived sidewalk corridor, and approve only paths that stay feasible and well-supported by vision.
**Requirements**: TPL-01, TPL-02, TPL-03, TPL-04
**Depends on:** Phase 7
**Plans:** 0 plans

**Success Criteria** (what must be TRUE):
  1. The planner takes route intent from GPS or route logic and restricts candidate generation to paths consistent with that commanded maneuver instead of free-guessing left vs right
  2. On representative recorded sidewalk frames, the selected path stays inside the segmented/boundary-defined corridor and rejects candidates that visibly exit the sidewalk region, especially in the first 1-2 meters near ego
  3. On turning sequences, path approval is temporally stable for the commanded intent — the planner does not branch-flip frame to frame when corridor evidence is still consistent
  4. On ambiguous or low-evidence frames, the planner outputs low confidence plus slowdown/hold advice instead of selecting a different maneuver than the commanded intent
  5. The final approved path can be exported in the same metric-path format consumed by the existing controller and overlay code

Plans:
- [ ] 11-01-PLAN.md - Corridor abstraction from BEV mask + synthetic corridor tests
- [ ] 11-02-PLAN.md - Intent-conditioned template bank, scoring terms, and approval logic
- [ ] 11-03-PLAN.md - Integrate intent-conditioned template approval into `BEVPathExtractor.process()` and live-loop confidence handling
- [ ] 11-04-PLAN.md - Replay evaluation harness and threshold tuning for intent-conditioned planning

---
*Roadmap created: 2026-03-04*
*Granularity: coarse (3-5 phases)*
*Coverage: 15/15 v1 requirements mapped + 9 OBS requirements (Phase 03.1) + 4 PATH-QUALITY requirements (Phase 6) + 4 TPL requirements (Phase 11)*
*Last updated: 2026-03-13 — Phase 11 reframed as GPS intent-conditioned corridor fitting; Phase 7 boundary-net foundation remains in progress*

### Phase 11.1: GPS-intent corridor waypoint turn planner (INSERTED)

**Goal:** Add a research-backed, GPS-intent-conditioned turn mode that chooses a corridor-supported waypoint target on the commanded side and fits a smooth controller-ready path to that target, avoiding skeleton-first turn selection
**Requirements**: WPT-01, WPT-02, WPT-03, WPT-04
**Depends on:** Phase 11
**Plans:** 1/4 plans executed

**Success Criteria** (what must be TRUE):
  1. With `left` or `right` intent active, the planner chooses a commanded-side waypoint target from the visible corridor instead of selecting a skeleton branch as the primary turn source
  2. The resulting path remains smooth and controller-feasible, with visibly earlier and more stable turn commitment than the current manual/skeleton fallback behavior
  3. During commanded turns, the live path remains maneuver-consistent across consecutive frames and does not flip back to uncommanded `straight` behavior while corridor support remains valid
  4. If the commanded turn is not yet visually supported, the planner emits low confidence and slowdown/hold guidance instead of inventing a different maneuver
  5. The implementation is reversible: baseline `dt_corridor`, Phase 11 template approval, and this new waypoint-turn mode can be compared without deleting existing work

Plans:
- [ ] 11.1-01-PLAN.md - Wave 0 baseline stabilization plus waypoint-turn test scaffolding
- [ ] 11.1-02-PLAN.md - Standalone waypoint-turn core with commanded-side target and dual-gate approval
- [ ] 11.1-03-PLAN.md - Runtime integration with dt_corridor preservation, maneuver lock, and low-confidence hold behavior
- [ ] 11.1-04-PLAN.md - Replay comparison harness and waypoint-turn threshold tuning
