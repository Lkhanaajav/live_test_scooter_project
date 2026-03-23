---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: Completed 11.1-02-PLAN.md (Waypoint-turn core with dual-gate approval)
last_updated: "2026-03-23T23:17:00.000Z"
last_activity: 2026-03-23 — implemented waypoint-turn planner core with BEV mask-scanned target selection and Hermite path fitting
progress:
  total_phases: 13
  completed_phases: 3
  total_plans: 19
  completed_plans: 10
  percent: 53
---

---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: Phase 2 formalized complete; Phase 3 is next execution target
last_updated: "2026-03-12T00:00:00.000Z"
last_activity: "2026-03-12 — Phase 2 formalized complete from validated calibration/log evidence; reusable SOP + validator added"
progress:
  [█████░░░░░] 47%
  completed_phases: 3
  total_plans: 10
  completed_plans: 8
---

---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: Completed 03.1-04-PLAN.md (BEV HUD obstacle visualization — human checkpoint approved)
last_updated: "2026-03-09T17:18:42.664Z"
last_activity: "2026-03-09 — Plan 03.1-04 completed: BEV HUD obstacle visualization human-verified — Phase 03.1 COMPLETE"
progress:
  total_phases: 6
  completed_phases: 2
  total_plans: 8
  completed_plans: 6
---

---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: Completed 03.1-04-PLAN.md (BEV HUD obstacle visualization — human checkpoint approved)
last_updated: "2026-03-09T17:00:00.000Z"
last_activity: "2026-03-05 — Plan 01-02 completed: smoother sweep finds alpha=0.65, c_thresh=0.20 at 99.6% stable — Phase 1 COMPLETE"
progress:
  total_phases: 6
  completed_phases: 1
  total_plans: 8
  completed_plans: 6
  percent: 44
---

---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: Completed 01-02-PLAN.md (smoother sweep + config update) — awaiting Task 3 human verify
last_updated: "2026-03-05T17:30:39.759Z"
last_activity: "2026-03-05 — Plan 01-02 completed: smoother sweep finds alpha=0.65, c_thresh=0.20 at 99.6% stable — Phase 1 COMPLETE"
progress:
  [████░░░░░░] 38%
  completed_phases: 1
  total_plans: 2
  completed_plans: 2
---

---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: Completed 01-02-PLAN.md — Phase 1 COMPLETE (human verification approved)
last_updated: "2026-03-05T00:00:00.000Z"
last_activity: "2026-03-05 — Plan 01-02 completed: smoother sweep finds alpha=0.65, c_thresh=0.20 at 99.6% stable — Phase 1 COMPLETE"
progress:
  total_phases: 4
  completed_phases: 1
  total_plans: 2
  completed_plans: 2
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-12)

**Core value:** Scooter visibly follows sidewalk path in a live thesis demo
**Current focus:** Phase 7 — lightweight sidewalk-boundary network foundation work with metric path decoding and confidence output (while Phase 3 remains the main demo milestone)

## Current Position

Phase: 02 + 03.1 complete; Phase 7 started
Plan: 07-01 foundation plan defined and partially executed
Status: Phase 2 COMPLETE, Phase 03.1 COMPLETE, Phase 7 foundation underway with train/eval/decoder pieces, Phase 3 Demo Integration still remains the main demo milestone
Last activity: 2026-03-12 — added Phase 7 metric path decoder and control-facing confidence outputs on top of the boundary-net baseline

Progress: 3 completed phases (01, 02, 03.1); Phase 7 foundation underway under 07-01; next mainline target is still Phase 3

## Performance Metrics

**Velocity:**
- Total plans completed: 8
- Historical duration metrics are incomplete because Phase 2 was formalized after the work had already been executed
- Current planning data is sufficient for status tracking, not for accurate throughput benchmarking

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-segmentation-stability | 2 | tracked | tracked |
| 02-bev-calibration-and-path-reliability | 2 | formalized from existing evidence | n/a |
| 03.1-yolo-bev-obstacle-projection | 4 | tracked | tracked |

**Recent Trend:**
- Last major update: Phase 2 formalization on 2026-03-12
- Trend: planning state is now aligned with actual repository evidence

*Updated after each plan completion*
| Phase 01-segmentation-stability P02 | 15 | 2 tasks | 2 files |
| Phase 03.1 P03.1-01 | 2 | 2 tasks | 2 files |
| Phase 03.1 P03.1-02 | 7 | 2 tasks | 3 files |
| Phase 03.1 P03.1-03 | 20 | 2 tasks | 3 files |
| Phase 03.1 P03.1-04 | 15 | 2 tasks | 2 files |
| Phase 11.1 P01 | 5 | 2 tasks | 4 files |
| Phase 11.1 P02 | 10 | 2 tasks | 3 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Phase 2]: Current BEV calibration loads without warning at cond~=7.99e5, and representative logs show 59-66% pixel survival with 99-100% has_path
- [Phase 2]: Calibration acceptance is now tied to reusable criteria (`load_bev_params()` warning state, pixel survival, has_path, heading reversals), not the unrealistic `cond < 1000` target
- [Setup]: Segmentation flicker must be fixed first (Phase 1) because it corrupts the BEV input that Phase 2 depends on
- [Setup]: Phase 4 (Radxa) is stretch only — do not start until Phases 1-3 complete and > 3 weeks remain
- [01-01]: my-segformer-road wins benchmark at 99.3% stable frames — 11 points above 90% target, switch from my-segformer-road_new
- [01-01]: Checkpoint-5000 directory is corrupted/incomplete — missing model weights, cannot be used
- [01-01]: TemporalMaskSmoother conservative-blend triggers when IoU<0.5 (not just <consistency_thresh), limiting alpha response speed in blank→obstacle transitions
- [01-02]: alpha=0.65, consistency_thresh=0.20 wins sweep at 99.6% stable — higher alpha preferred on ties for better SEG-03 responsiveness
- [01-02]: consistency_thresh has zero sensitivity at my-segformer-road baseline — all 5 threshold values produce identical pct_stable for same alpha
- [01-02]: Phase 1 COMPLETE — SEG-01 MET (99.6% >= 90%), SEG-03 PASS (alpha=0.65 >= 0.25), human verified 2026-03-05
- [Phase 03.1-01]: bev_h_matrix uses scale matrix (x*0.3, y*0.5) for analytically verifiable projection — not a realistic perspective warp
- [Phase 03.1-01]: test_bev_obstacle.py keeps imports minimal (numpy + pytest only) to avoid ImportError on non-existent bev_obstacle module in Wave 0
- [Phase 03.1-02]: forward_m clamped to >= 0.3 to filter obstacles at/behind ego (foot below BEV bottom)
- [Phase 03.1-02]: bev_obstacle.py has no dependency on realtime_nav_core — clean contract for Wave 2 integration
- [Phase 03.1]: obstacle_zones_m defaults to None (not []) for explicit no-obstacles vs empty-list distinction in process()
- [Phase 03.1]: _obstacle_penalty skipped when len(cands) < 2 to avoid penalizing the only viable path (Pitfall 5 from RESEARCH.md)
- [Phase 03.1-04]: obstacle_zones_m=None default keeps headless callers unaffected; color threshold tied to BEV_HARD_BLOCK_DIST_M so visualization matches planner logic exactly
- [Phase 03.1-04]: Phase 03.1 COMPLETE — OBS-01 through OBS-09 all satisfied, human verified 2026-03-09
- [Phase 11.1]: Fix test expectations not production code: test_metric_conversion used stale config assumptions
- [Phase 11.1]: waypoint_turn_planner.py is purely additive: no runtime integration in Wave 0
- [Phase 11.1-02]: Scan raw BEV mask pixels instead of corridor abstraction for support detection (corridor_from_mask merges runs)
- [Phase 11.1-02]: Asymmetric pixel count as commanded-side support metric avoids false positives on narrow centered corridors
- [Phase 11.1-02]: Decision band widened to 2.0-7.0m to match real BEV turn opening visibility
- [Phase 11.1-02]: Cubic Hermite smoothstep for path fitting avoids polynomial overshoot

### Roadmap Evolution

- Phase 3.1 inserted after Phase 3: YOLO BEV Obstacle Projection (INSERTED 2026-03-09) — project YOLO detections onto BEV as metric exclusion zones for path-avoiding navigation
- Phase 6 added: Path quality improvements — post-selection smoothing, BEV mask morphological closing, stronger temporal continuity weight, draw fitted cubic on overlay
- Phase 2 formalized complete on 2026-03-12 based on validated calibration/log evidence and new reusable calibration artifacts
- Phase 7 added: Lightweight sidewalk-boundary network with analytical centerline and confidence gating
- Phase 8 added: Real-time boundary-aware segmentation backbone replacement for sidewalk navigation
- Phase 9 added: Shared-backbone multitask perception for sidewalk, boundaries, and obstacles
- Phase 10 added: Tiny image-to-waypoints student policy for low-compute scooter control
- Phase 11 added: Template path fitting inside segmentation corridor with path approval scoring
- Phase 11.1 inserted after Phase 11: GPS-intent corridor waypoint turn planner (URGENT) - preserve the Phase 11 template-approval track while planning a waypoint-target alternative for commanded turns
- Phase 7 started: added reusable row-wise boundary target extraction and dataset export pipeline
- Phase 7 advanced: added metric centerline decoding and low-confidence gating outputs for the tiny boundary-net baseline
- Phase 11 planned: added requirements, context, research, validation, and four execution plans for template-path approval scoring
- Phase 11 reframed: GPS or route logic provides maneuver intent; vision only fits intent-conditioned paths inside the corridor and may refuse unsupported turns

### Pending Todos

None yet.

### Blockers/Concerns

- Remaining navigation risk is branch-selection robustness under low-evidence and near-ego artifact windows, not gross BEV calibration failure
- Scooter serial interface (Phase 3) is "almost ready" per PROJECT.md — needs live hardware test to confirm command execution

## Session Continuity

Last session: 2026-03-23T23:17:00.000Z
Stopped at: Completed 11.1-02-PLAN.md (Waypoint-turn core with dual-gate approval)

## Work Log

### 2026-03-12 — Phase 7 foundation start
- Added `boundary_targets.py` with reusable row-wise left/right boundary extraction from binary sidewalk masks
- Added `scripts/export_boundary_targets.py` to export boundary-target JSONL records from existing mask datasets
- Added `tests/test_boundary_targets.py` with synthetic corridor and curve coverage
- Full test suite remains green: 73 passed

### 2026-03-12 — Phase 7 decoder and offline baseline evaluation
- Added `boundary_inference.py` to decode left/right boundary predictions into metric centerline paths, pixel overlays, width estimates, confidence scores, and slowdown recommendations
- Extended `scripts/eval_boundary_net.py` to report control-facing path metrics such as has-path rate, forward span, confidence, and suggested slowdown
- Added `tests/test_boundary_inference.py` to lock down straight-path decoding, low-confidence behavior, and previous-path blending

### 2026-03-12 — Phase 11 planning
- Added Phase 11 requirement IDs (`TPL-01` through `TPL-04`) and roadmap success criteria for template-path approval scoring
- Added `11-CONTEXT.md`, `11-RESEARCH.md`, and `11-VALIDATION.md` for the new phase
- Planned Phase 11 into four execution waves covering corridor abstraction, template approval, runtime integration, and replay evaluation

### 2026-03-13 — Phase 11 spec correction
- Reframed Phase 11 as GPS intent-conditioned template fitting rather than free left/right guessing from vision
- Locked the design rule that vision fits geometry inside the corridor while GPS or route logic provides maneuver intent
- Updated Phase 11 requirements, roadmap goal, context, research notes, validation language, and Wave 2-4 plans to enforce intent-conditioned approval

### 2026-03-09 — Phase 03.1 complete + benchmark evaluation
**Phase 03.1 YOLO BEV Obstacle Projection — ALL 4 PLANS DONE**
- Plan 03.1-04: draw_bev_hud() extended with obstacle_zones_m, orange/red circles, human verified
- 69 tests passing (9 new OBS tests + 60 existing)

**Cross-domain segmentation benchmarks (no GSD plan — exploratory)**
- Created `eval_cityscapes.py` — runs SegFormer on Cityscapes val via HuggingFace streaming
- Created `eval_rugd.py` — runs SegFormer on RUGD outdoor scenes via HuggingFace streaming
- New model discovered: `models/drivable-segformer-b0` (base: Cityscapes-pretrained SegFormer-B0, fine-tuned on drivable_dataset_v1, 92.7% custom mIoU)
- Results: Cityscapes 67.7% road+sw IoU; RUGD 91% recall on asphalt+concrete (50% IoU)
- Decision: benchmarks → 1 paragraph in §6 Discussion only, NOT full results sections
- Key insight: RUGD 71.5% GT pixels are gravel trails (irrelevant for scooter); asphalt+concrete recall=91% is the honest number

**Next session priorities:**
1. Write cross-domain benchmark paragraph into thesis §6
2. Update thesis with drivable-segformer-b0 numbers (92.7%)
3. Plan Phase 3 (Demo Integration) — no hardware needed to plan
