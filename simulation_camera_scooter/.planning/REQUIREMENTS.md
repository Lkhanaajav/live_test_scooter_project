# Requirements: Scooter Sidewalk Navigation System

**Defined:** 2026-03-04
**Core Value:** Scooter visibly follows sidewalk path in a live thesis demo

## v1 Requirements

### Segmentation Quality

- [x] **SEG-01**: Segmentation output is temporally stable — no visible per-frame flickering on real sidewalk video (IoU between consecutive frames >= 0.85 on >= 90% of frames)
- [x] **SEG-02**: SegFormer model is fine-tuned or validated on representative outdoor sidewalk footage (current demo environment)
- [x] **SEG-03**: Temporal smoothing strategy is tuned to eliminate flickering without over-smoothing dynamic obstacles

### BEV Calibration

- [x] **BEV-01**: BEV homography is recalibrated using well-framed, level sidewalk video and `load_bev_params()` does not print the ill-conditioned warning (`cond(H) < 1e6`)
- [x] **BEV-02**: `warpPerspective` preserves >= 50% of sidewalk pixels after calibration
- [x] **BEV-03**: Calibration procedure is documented so it can be repeated when camera mount changes

### Path Reliability

- [x] **PATH-01**: `has_path` rate >= 60% of frames on representative sidewalk footage after calibration fix
- [x] **PATH-02**: Extracted path is geometrically correct at the phase level — follows the sidewalk centerline rather than gross edge/grass artifacts in validated runs
- [x] **PATH-03**: Path is stable across consecutive frames — no direction reversals on straight sidewalk in validated runs

### Demo Integration

- [ ] **DEMO-01**: Extracted BEV path is projected back onto the camera view as a colored overlay for demo visualization
- [ ] **DEMO-02**: Scooter receives and executes steering + speed commands in real-time during a demo run
- [ ] **DEMO-03**: System runs at >= 8 Hz end-to-end on the demo laptop during live demo
- [ ] **DEMO-04**: System degrades gracefully — holds last valid path and reduces speed on seg failure, no abrupt stops

### YOLO BEV Obstacle Projection (Phase 03.1)

- [x] **OBS-01**: Foot-point of a detection centered in the image projects through H to the correct BEV quadrant (center-bottom of BEV = ego-forward)
- [x] **OBS-02**: Metric coordinate (forward_m, lateral_m) of projected foot matches analytically expected values given a known H matrix
- [x] **OBS-03**: EMA grid decays to near-zero (< 1% max value) after 10 frames with no detection at alpha=0.5
- [x] **OBS-04**: EMA grid shows nonzero value (>= 0.4) at the foot-point location after one update from a zero grid
- [x] **OBS-05**: A candidate path whose points pass through an obstacle zone receives a higher cost than a clear path — planner prefers the clear path when alternatives exist
- [x] **OBS-06**: Hard-block paints BEV mask pixels black within the stop-zone radius for obstacles closer than BEV_HARD_BLOCK_DIST_M
- [x] **OBS-07**: Out-of-bounds projected points (negative or > BEV dimensions) are clamped without raising an exception
- [x] **OBS-08**: No obstacle penalty is applied when obstacle_zones_m is None or empty — process() result is identical to no-obstacle call
- [x] **OBS-09**: Full pipeline (project -> EMA -> hard-block -> process with zones) runs end-to-end on a synthetic BEV mask without crash

### Embedded Deployment (Stretch)

- [ ] **RADXA-01**: SegFormer exported to ONNX or TensorRT format runnable on Rock 5B at >= 5 Hz
- [ ] **RADXA-02**: Full pipeline tested on Radxa board end-to-end with scooter connected

### Template Path Approval (Phase 11)

- [ ] **TPL-01**: A fixed bank of smooth candidate paths can be generated in BEV from the ego pose with bounded curvature suitable for pedestrian-speed scooter control
- [ ] **TPL-02**: Candidate paths are scored against the perceived sidewalk corridor using corridor fit, boundary clearance, center preference, and continuity with the previous approved path
- [ ] **TPL-03**: The approved path remains inside the sidewalk corridor on representative straight and turning sequences and rejects obviously invalid templates that leave the corridor
- [ ] **TPL-04**: When no candidate path scores above approval threshold, the planner emits low confidence and a slowdown/hold recommendation instead of forcing a turn guess

## v2 Requirements

### Advanced Navigation

- **NAV-01**: GPS waypoint route following tested on a real outdoor route
- **NAV-02**: Automatic re-routing when sidewalk path is lost for > 2 seconds
- **NAV-03**: Multi-segment route (turn at intersections) using GPS + vision fusion

### Robustness

- **ROB-01**: Performance validated across lighting conditions (overcast, direct sun, shadows)
- **ROB-02**: Performance validated on textured/uneven sidewalk surfaces
- **ROB-03**: Automatic BEV recalibration detection when calibration degrades

## Out of Scope

| Feature | Reason |
|---------|--------|
| Stereo / depth camera | Single-camera constraint for thesis scope |
| SLAM / map building | Reactive navigation only — no global map required |
| Multi-robot coordination | Single scooter thesis project |
| Mobile app / remote UI | Serial command interface is sufficient for demo |
| Full Radxa deployment | Stretch goal — laptop fallback accepted for thesis |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| SEG-01 | Phase 1 | Complete |
| SEG-02 | Phase 1 | Complete |
| SEG-03 | Phase 1 | Complete |
| BEV-01 | Phase 2 | Complete |
| BEV-02 | Phase 2 | Complete |
| BEV-03 | Phase 2 | Complete |
| PATH-01 | Phase 2 | Complete |
| PATH-02 | Phase 2 | Complete |
| PATH-03 | Phase 2 | Complete |
| DEMO-01 | Phase 3 | Pending |
| DEMO-02 | Phase 3 | Pending |
| DEMO-03 | Phase 3 | Pending |
| DEMO-04 | Phase 3 | Pending |
| OBS-01 | Phase 03.1 | Complete |
| OBS-02 | Phase 03.1 | Complete |
| OBS-03 | Phase 03.1 | Complete |
| OBS-04 | Phase 03.1 | Complete |
| OBS-05 | Phase 03.1 | Complete |
| OBS-06 | Phase 03.1 | Complete |
| OBS-07 | Phase 03.1 | Complete |
| OBS-08 | Phase 03.1 | Complete |
| OBS-09 | Phase 03.1 | Complete |
| RADXA-01 | Phase 4 | Pending |
| RADXA-02 | Phase 4 | Pending |
| TPL-01 | Phase 11 | Pending |
| TPL-02 | Phase 11 | Pending |
| TPL-03 | Phase 11 | Pending |
| TPL-04 | Phase 11 | Pending |

**Coverage:**
- v1 requirements: 28 total (15 original + 9 OBS + 4 TPL)
- Mapped to phases: 28
- Unmapped: 0

---
*Requirements defined: 2026-03-04*
*Last updated: 2026-03-12 — added Phase 11 template-path approval requirements*
