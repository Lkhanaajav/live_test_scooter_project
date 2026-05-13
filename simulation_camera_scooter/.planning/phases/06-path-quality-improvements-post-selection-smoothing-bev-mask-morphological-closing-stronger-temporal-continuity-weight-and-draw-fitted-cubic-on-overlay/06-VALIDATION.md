---
phase: 6
slug: path-quality-improvements-post-selection-smoothing-bev-mask-morphological-closing-stronger-temporal-continuity-weight-and-draw-fitted-cubic-on-overlay
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-11
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (already installed, `requirements.txt`) |
| **Config file** | none — pytest discovers by convention |
| **Quick run command** | `python -m pytest tests/test_path_quality.py -v` |
| **Full suite command** | `python -m pytest tests/ -v` |
| **Estimated runtime** | ~5 seconds (new tests) + ~10 seconds (full suite) |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_path_quality.py -v`
- **After every plan wave:** Run `python -m pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** ~15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 06-W0-01 | 01 | 0 | PATH-SMOOTH-01 | unit | `python -m pytest tests/test_path_quality.py::test_post_selection_smoothing_reduces_lateral_variance -x` | ❌ W0 | ⬜ pending |
| 06-W0-02 | 01 | 0 | PATH-SMOOTH-01 | unit | `python -m pytest tests/test_path_quality.py::test_post_selection_smoothing_preserves_x_coords -x` | ❌ W0 | ⬜ pending |
| 06-W0-03 | 01 | 0 | PATH-SMOOTH-01 | unit | `python -m pytest tests/test_path_quality.py::test_post_selection_smoothing_guard_short_path -x` | ❌ W0 | ⬜ pending |
| 06-W0-04 | 01 | 0 | MORPH-CLOSE-01 | unit | `python -m pytest tests/test_path_quality.py::test_morphological_closing_fills_gap -x` | ❌ W0 | ⬜ pending |
| 06-W0-05 | 01 | 0 | MORPH-CLOSE-01 | unit | `python -m pytest tests/test_path_quality.py::test_config_close_iters_default -x` | ❌ W0 | ⬜ pending |
| 06-W0-06 | 01 | 0 | CONT-WEIGHT-01 | unit | `python -m pytest tests/test_path_quality.py::test_config_continuity_weight_default -x` | ❌ W0 | ⬜ pending |
| 06-W0-07 | 01 | 0 | CONT-WEIGHT-01 | unit | `python -m pytest tests/test_path_quality.py::test_continuity_weight_penalizes_lateral_deviation -x` | ❌ W0 | ⬜ pending |
| 06-W0-08 | 01 | 0 | CUBIC-OVERLAY-01 | unit | `python -m pytest tests/test_path_quality.py::test_cubic_sample_xy_returns_valid_points -x` | ❌ W0 | ⬜ pending |
| 06-W0-09 | 01 | 0 | CUBIC-OVERLAY-01 | unit | `python -m pytest tests/test_path_quality.py::test_cubic_overlay_pixel_coords_in_bounds -x` | ❌ W0 | ⬜ pending |
| 06-W0-10 | 01 | 0 | CUBIC-OVERLAY-01 | integration | `python -m pytest tests/test_path_quality.py::test_cubic_overlay_end_to_end -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_path_quality.py` — 10 test stubs for PATH-SMOOTH-01, MORPH-CLOSE-01, CONT-WEIGHT-01, CUBIC-OVERLAY-01
- [ ] New fixture: `noisy_path_m` — 10-point metric path with synthetic lateral jitter (in conftest.py)
- [ ] No framework install needed — pytest already present
- [ ] Existing fixtures `straight_bev_mask`, `straight_path_model` reusable for closing and cubic tests

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Camera overlay shows smooth cubic arc (not jagged skeleton polyline) | CUBIC-OVERLAY-01 | Requires video playback | Run `python live_heading_demo.py --video test_video_mar3.MOV` and visually confirm the path arc is smooth |
| Frame-to-frame path lateral deviation < 0.15m on straight corridor | PATH-SMOOTH-01 + CONT-WEIGHT-01 | Requires real video data | Check `pp_target_y_m` column in CSV log; compute std dev over 50 straight frames |
| BEV mask occ_ratio increases relative to baseline | MORPH-CLOSE-01 | Depends on calibration state | Compare `bev_mask_occ_ratio` in CSV before/after the change on same test video |

---

## Before/After Metrics

| Metric | Baseline | Target | Source Column |
|--------|----------|--------|---------------|
| Mean frame-to-frame lateral path deviation | ~0.2–0.4m | < 0.15m | `pp_target_y_m` |
| Heading oscillation std dev (deg, straight) | ~3–8 deg | < 3 deg | `heading_raw_deg` |
| BEV sidewalk occ_ratio | ~4–7% | +10–30% relative | `bev_mask_occ_ratio` |
| Camera overlay appearance | Jagged polyline | Smooth curve | Visual |

---

*Phase: 06-path-quality-improvements*
*Validation strategy created: 2026-03-11*
