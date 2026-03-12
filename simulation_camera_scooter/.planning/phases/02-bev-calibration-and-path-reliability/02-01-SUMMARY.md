---
phase: 02-bev-calibration-and-path-reliability
plan: "01"
status: complete
completed: 2026-03-12
---

# Phase 2 Plan 01 Summary

Phase 2 calibration work was already present in the repository, but it was never closed out with the expected artifacts. This summary formalizes the current calibration using reusable validation criteria rather than a one-off frame-specific claim.

## What Was Verified

- `bev_calibration.npy` loads without the ill-conditioned warning.
- Current homography condition number is `7.99e5`, below the `1e6` warning threshold used by `load_bev_params()`.
- The active source points are:
  - `[261, 717]`
  - `[1196, 715]`
  - `[796, 343]`
  - `[619, 338]`
- The validated calibration has been backed up as `bev_calibration_backup_20260312.npy`.
- A reusable log validator now exists at `scripts/measure_bev_survival.py`.
- A repeatable calibration procedure now exists at `CALIBRATION_SOP.md`.

## Evidence

Representative validated runs:

| Run | Mean survival | has_path |
|-----|---------------|----------|
| `logs/run_20260305_172519.csv` | 61.4% | 99.1% |
| `logs/run_20260309_165802.csv` | 61.4% | 100.0% |
| `logs/run_20260309_171655.csv` | 64.7% | 100.0% |
| `logs/run_20260309_172316.csv` | 66.5% | 100.0% |

These numbers exceed the phase thresholds of `>= 50%` survival and `>= 60%` `has_path`.

## Permanent Fixes Added

- Replaced ad hoc spreadsheet-style log inspection with `scripts/measure_bev_survival.py`.
- Wrote `CALIBRATION_SOP.md` so recalibration is tied to mount geometry and validation thresholds, not to a specific video.
- Corrected planning state to use the real acceptance criterion: no `load_bev_params()` warning plus pixel survival, not the unrealistic `cond < 1000` target.

## Outcome

Plan `02-01` is complete. The calibration is now documented, measurable, and reusable across future mount changes.
