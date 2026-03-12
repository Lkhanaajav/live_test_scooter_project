---
phase: 02-bev-calibration-and-path-reliability
plan: "02"
status: complete
completed: 2026-03-12
---

# Phase 2 Plan 02 Summary

Path reliability validation had already been executed in practice across multiple logged runs. This summary closes the phase using multi-run evidence and the existing visual findings, rather than treating one hand-picked clip as the only proof.

## What Was Verified

- `has_path` exceeds the 60% target on multiple representative runs.
- Heading stability is strong on the same runs, with zero reversals above 90 degrees.
- Visual findings from the March 5-6 investigations show the BEV transform is geometrically usable and that the remaining issue is branch-selection robustness under low evidence, not gross calibration failure.

## Evidence

Validated runs:

| Run | Video/config | has_path | Survival | Max heading jump | Reversals |
|-----|--------------|----------|----------|------------------|-----------|
| `logs/run_20260305_172519.csv` | representative clip | 99.1% | 61.4% | 1.43 deg | 0 |
| `logs/run_20260309_164920.csv` | June clip, no YOLO | 100.0% | 59.2% | 3.75 deg | 0 |
| `logs/run_20260309_165802.csv` | `test_video_june_03_3.mp4`, no YOLO | 100.0% | 61.4% | 5.56 deg | 0 |
| `logs/run_20260309_171655.csv` | `test_video_june_03_3.mp4`, YOLO on | 100.0% | 64.7% | 9.21 deg | 0 |
| `logs/run_20260309_172316.csv` | short representative clip | 100.0% | 66.5% | 2.12 deg | 0 |

Visual/manual evidence used for `PATH-02`:

- `overnight_runs/2026-03-05_path_improvement/VISUAL_FINDINGS.md`
- `overnight_runs/2026-03-06_path_improvement/VISUAL_FINDINGS.md`

Those notes show:

- the BEV region is usable rather than collapsing to a narrow sliver,
- the planner maintains a viable center path in normal conditions,
- the main unresolved issue is branch-entry false turns under noisy masks, which is a robustness problem beyond the original Phase 2 calibration gate.

## Permanent Fixes Added

- Phase validation is now tied to reusable metrics (`has_path`, survival, heading reversals) rather than one-off visual intuition.
- The remaining path-planning work is clearly separated from calibration success: future fixes should target branch-selection robustness and low-confidence behavior, not re-litigate a calibration issue that is already solved.

## Outcome

Plan `02-02` is complete. `PATH-01`, `PATH-02`, and `PATH-03` are satisfied at the phase level, with remaining failures correctly categorized as later-stage robustness work rather than incomplete BEV calibration.
