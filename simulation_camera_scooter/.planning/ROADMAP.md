# Roadmap: Scooter Sidewalk Navigation System

## Overview

The pipeline is substantially built. Two root problems block the thesis demo: segmentation flickers frame-to-frame (corrupting BEV input), and the BEV homography is ill-conditioned (losing 93% of sidewalk pixels before path extraction even starts). This roadmap fixes those two problems in order, then hardens the system for a live demo run. Phase 4 (Radxa/ONNX) is a stretch goal — the thesis demo is valid on a laptop.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Segmentation Stability** - Eliminate visible per-frame flickering in SegFormer output on real sidewalk video
- [ ] **Phase 2: BEV Calibration and Path Reliability** - Recalibrate homography and achieve has_path >= 60% on typical sidewalk footage
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
- [ ] 01-01: Evaluate SegFormer checkpoints and benchmark temporal stability on demo video
- [ ] 01-02: Tune temporal smoothing (EMA alpha, IoU thresholds) to hit stability targets

### Phase 2: BEV Calibration and Path Reliability
**Goal**: BEV homography is accurate for the actual recording setup and the path extractor reliably finds a drivable path
**Depends on**: Phase 1
**Requirements**: BEV-01, BEV-02, BEV-03, PATH-01, PATH-02, PATH-03
**Success Criteria** (what must be TRUE):
  1. Running load_bev_params() reports condition number < 1000 (down from 1.1e+06 current)
  2. Inspecting BEV output frames, >= 50% of sidewalk pixels survive the warpPerspective transform (visible as a filled region in BEV view)
  3. has_path rate is >= 60% of frames when running the pipeline on a representative sidewalk video clip
  4. The extracted skeleton path follows the sidewalk centerline in BEV — it does not jump to edges, grass artifacts, or reverse direction on a straight sidewalk
  5. Calibration steps are written up so the procedure can be repeated when the camera mount changes
**Plans**: TBD

Plans:
- [ ] 02-01: Recalibrate BEV homography with well-framed level video and validate condition number
- [ ] 02-02: Tune path extractor parameters (min_sidewalk_width_m, branch_min_len_m, min_path_len_m) to hit has_path >= 60% target

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

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 (4 is stretch only)

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Segmentation Stability | 0/2 | Not started | - |
| 2. BEV Calibration and Path Reliability | 0/2 | Not started | - |
| 3. Demo Integration | 0/2 | Not started | - |
| 4. Radxa Deployment (STRETCH) | 0/2 | Not started | - |

---
*Roadmap created: 2026-03-04*
*Granularity: coarse (3-5 phases)*
*Coverage: 15/15 v1 requirements mapped*
