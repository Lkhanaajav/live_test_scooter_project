---
phase: 2
slug: bev-calibration-and-path-reliability
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-05
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.0.2 |
| **Config file** | none — discovery by convention (`tests/` directory) |
| **Quick run command** | `python -m pytest tests/ -x -q` |
| **Full suite command** | `python -m pytest tests/ -v` |
| **Estimated runtime** | ~0.1 seconds (35 existing tests) |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/ -x -q`
- **After every plan wave:** Run `python -m pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green + manual BEV/path checks complete
- **Max feedback latency:** ~2 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 01 | 1 | BEV-01 | unit | `python -m pytest tests/test_bev_calibration.py -x -q` | ✅ | ⬜ pending |
| 02-01-02 | 01 | 1 | BEV-02 | manual | Measure `bev_mask_pixels/sidewalk_mask_pixels` from CSV log | manual | ⬜ pending |
| 02-01-03 | 01 | 1 | BEV-03 | manual | Human review: SOP written and complete | manual | ⬜ pending |
| 02-02-01 | 02 | 2 | PATH-01 | manual | `df['has_path'].mean()` from CSV log >= 0.60 | manual | ⬜ pending |
| 02-02-02 | 02 | 2 | PATH-02 | manual | Visual inspect BEV output — single clean path on centerline | manual | ⬜ pending |
| 02-02-03 | 02 | 2 | PATH-03 | manual | `heading_smoothed_deg.diff().abs().max()` — no reversals | manual | ⬜ pending |
| all | all | all | BEV+PATH | unit | `python -m pytest tests/ -v` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

None — existing test infrastructure covers all automatable requirements. The BEV-02, PATH-01,
PATH-02, PATH-03 checks are inherently manual (require hardware + new calibration video) and
cannot be automated in pytest without a real sidewalk video file.

*Existing infrastructure covers all phase requirements.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| ≥50% pixel survival | BEV-02 | Requires real camera + new calibration video | Run pipeline with `--enable-logging`, open CSV, compute `df['bev_mask_pixels']/df['sidewalk_mask_pixels']` — target mean >= 0.50 |
| has_path rate ≥60% | PATH-01 | Requires new calibration + real sidewalk video | Run pipeline with `--enable-logging`, open CSV, compute `df['has_path'].mean()` — target >= 0.60 |
| Path follows centerline | PATH-02 | Visual geometry check only | Open BEV display window during run, inspect skeleton path — should be a single spine in center, not edge artifacts |
| No sudden heading reversals | PATH-03 | Requires real run data | Open CSV, compute `heading_smoothed_deg.diff().abs()` — no values > 90 deg on straight sidewalk |
| Calibration SOP complete | BEV-03 | Document review | Read Plan 02-01 SOP section — verify 5-step checklist exists and is complete |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 5s (quick run)
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
