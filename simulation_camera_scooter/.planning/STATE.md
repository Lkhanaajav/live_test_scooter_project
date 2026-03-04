# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-04)

**Core value:** Scooter visibly follows sidewalk path in a live thesis demo
**Current focus:** Phase 1 — Segmentation Stability

## Current Position

Phase: 1 of 4 (Segmentation Stability)
Plan: 1 of 2 in current phase
Status: In progress
Last activity: 2026-03-04 — Plan 01-01 completed: benchmark identifies my-segformer-road as best checkpoint (99.3% stable)

Progress: [█░░░░░░░░░] 10%

## Performance Metrics

**Velocity:**
- Total plans completed: 1
- Average duration: 10 min
- Total execution time: ~0.17 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-segmentation-stability | 1 | 10 min | 10 min |

**Recent Trend:**
- Last 5 plans: 01-01 (10 min)
- Trend: -

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Setup]: BEV homography is the dominant root cause of low has_path (cond=1.1e+06, 93% pixel loss) — Phase 2 is the highest-impact fix
- [Setup]: Segmentation flicker must be fixed first (Phase 1) because it corrupts the BEV input that Phase 2 depends on
- [Setup]: Phase 4 (Radxa) is stretch only — do not start until Phases 1-3 complete and > 3 weeks remain
- [01-01]: my-segformer-road wins benchmark at 99.3% stable frames — 11 points above 90% target, switch from my-segformer-road_new
- [01-01]: Checkpoint-5000 directory is corrupted/incomplete — missing model weights, cannot be used
- [01-01]: TemporalMaskSmoother conservative-blend triggers when IoU<0.5 (not just <consistency_thresh), limiting alpha response speed in blank→obstacle transitions

### Pending Todos

None yet.

### Blockers/Concerns

- BEV recalibration (Phase 2) requires recording a new, well-framed, level sidewalk video — this must be done physically on the actual hardware with the camera in its final mount position
- Scooter serial interface (Phase 3) is "almost ready" per PROJECT.md — needs live hardware test to confirm command execution

## Session Continuity

Last session: 2026-03-04
Stopped at: Completed 01-01-PLAN.md (benchmark + tests + config update)
Resume file: None
