---
phase: 1
slug: segmentation-stability
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-04
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | none — default discovery |
| **Quick run command** | `python -m pytest tests/ -x -q` |
| **Full suite command** | `python -m pytest tests/ -v` |
| **Estimated runtime** | ~30 seconds (unit tests) + benchmark separately |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/ -x -q`
- **After every plan wave:** Run `python -m pytest tests/ -v` + `python scripts/benchmark_seg_stability.py --video test_video_mar3_1_h264.mp4`
- **Before `/gsd:verify-work`:** Full suite must be green + benchmark >= 90% frames at IoU >= 0.85
- **Max feedback latency:** ~30 seconds (unit tests)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 1-01-01 | 01 | 1 | SEG-01, SEG-02 | integration | `python scripts/benchmark_seg_stability.py --video test_video_mar3_1_h264.mp4` | ❌ W0 | ⬜ pending |
| 1-01-02 | 01 | 1 | SEG-02 | integration | `python scripts/benchmark_seg_stability.py --checkpoint <ckpt>` | ❌ W0 | ⬜ pending |
| 1-02-01 | 02 | 2 | SEG-03 | unit | `python -m pytest tests/test_temporal_smoother.py -x` | ❌ W0 | ⬜ pending |
| 1-02-02 | 02 | 2 | SEG-01, SEG-03 | integration | `python scripts/benchmark_seg_stability.py --video test_video_mar3_1_h264.mp4` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `scripts/benchmark_seg_stability.py` — benchmark script to evaluate checkpoints and measure frame-to-frame IoU stability (SEG-01, SEG-02)
- [ ] `tests/test_temporal_smoother.py` — unit tests for TemporalMaskSmoother edge cases: empty masks, low-alpha response time for dynamic obstacles (SEG-03)
- [ ] `tests/conftest.py` — already exists, extend with segmentation fixtures if needed

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Flickering not visible to demo observer | SEG-01 | Perceptual check, no automated equivalent | Play back processed demo video at normal speed, check for visible class flipping |
| Dynamic obstacle still updates within 2-3 frames | SEG-03 | Requires live/video visual review | Review benchmark output for obstacle frames; confirm mask changes within 2-3 frames of object entering scene |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
