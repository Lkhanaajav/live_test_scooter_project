# Phase 11.1: GPS-Intent Corridor Waypoint Turn Planner

**Date:** 2026-03-24
**Branch:** main

## What Was Done

### Phase 11.1 Implementation (4 plans, 12 commits)

**Plan 01 — Wave 0 Baseline:**
- Fixed pre-existing test failures (`test_metric_conversion`, template planner tests)
- Created `waypoint_turn_planner.py` contract stub
- Added 4 turn-specific test fixtures (commanded-left, commanded-right, false-pocket, unsupported-turn)
- 14 new Wave 0 tests

**Plan 02 — Core Planner:**
- Implemented full `waypoint_turn_planner.py` (320 lines): BEV mask scan, support clustering, target selection, Hermite path fitting, dual-gate approval (target + path containment)
- Added 18 `WAYPOINT_*` constants to `config.py`
- 28 total tests for the planner module

**Plan 03 — Runtime Integration:**
- Wired `_try_waypoint_turn()` into `BEVPathExtractor.process()` in `realtime_nav_core.py`
- Maneuver lock with `WAYPOINT_LOCK_SUSTAIN_FRAMES` / `WAYPOINT_LOCK_RELEASE_FRAMES`
- Low-confidence hold speed gating in `live_heading_demo.py`
- `WAYPOINT_TURN_ENABLED=True` toggle in config
- 9 new integration tests

**Plan 04 — Replay Evaluation:**
- Created `scripts/eval_waypoint_turn_planner.py` for three-mode comparison (baseline vs template vs waypoint-turn)
- Fixed replay manifest at `.planning/phases/11.1-gps-intent-corridor-waypoint-turn-planner/11.1-REPLAY_SET.txt`
- Documented all threshold rationale in config comments

### Post-Implementation Fixes

**Restored `live_heading_demo.py` from origin/main:**
The Phase 11.1 executor agents had stripped features from `live_heading_demo.py` that existed on main:
- Arrow key controls (`KEY_UP`, `KEY_LEFT`, `KEY_RIGHT`, `KEY_DOWN`)
- `--no-stabilization`, `--no-seg-smoothing`, `--no-bev-smoothing`, `--no-ego-anchor`, `--no-ego-connected` flags
- `manual_intent_family` vs `scheduled_intent_family` separation
- `active_intent` display on HUD

Restored the origin/main version, then re-applied only the Phase 11.1 addition (waypoint-turn low-confidence speed gating) and fixed two API mismatches:
- `load_bev_params(frame_size=...)` → `load_bev_params(current_frame_size=...)`
- Removed `manual_intent_override` and `lookahead_scale`/`disable_discont_hold` kwargs (removed by Phase 11.1 from the callee signatures)

**Updated `RUNTIME_RUNBOOK.md`:**
- Fixed model reference: `oneformer_teacher` → `old400_img1931_vid017_020` (best IoU 0.9602)
- Fixed all file paths from codex machine (`/C:/Users/miji0000/...`) to relative paths
- Updated controls section to document both arrow keys and letter keys

## Correct Model

```
outputs/training/binary_segformer_old400_img1931_vid017_020/best_checkpoint
```
Val IoU 0.9602 | 9 videos | Trained 2026-03-19

## Run Command

```powershell
python simulation_camera_scooter\live_heading_demo.py ^
  --video simulation_camera_scooter\test_videos\VID_20260319_155939_00_017.mp4 ^
  --model-dir outputs\training\binary_segformer_old400_img1931_vid017_020\best_checkpoint ^
  --seg-conf-thresh 0.6 ^
  --seg-width 512 --seg-height 288 ^
  --path-scale 0.7 ^
  --stride 10 ^
  --no-detection ^
  --no-stabilization ^
  --no-seg-smoothing ^
  --no-bev-smoothing ^
  --no-ego-anchor ^
  --no-ego-connected ^
  --no-predict ^
  --bev-clean legacy
```

## Test Results

- **206 tests passing**, 0 failures
- **~9-11 FPS** at 512x288, stride 10
- **~6.5 FPS** at 640x360, stride 5
- IoU solid at 1.00 throughout all runs
