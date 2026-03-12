---
phase: 11
slug: template-path-fitting-inside-segmentation-corridor-with-path-approval-scoring
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-03-12
---

# Phase 11 - Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | none - existing repo pytest layout |
| **Quick run command** | `python -m pytest tests/test_template_path_planner.py -q` |
| **Full suite command** | `python -m pytest tests -q` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_template_path_planner.py -q`
- **After every plan wave:** Run `python -m pytest tests -q`
- **Before `$gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 20 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 11-01-01 | 01 | 1 | TPL-01 | unit | `python -m pytest tests/test_template_path_planner.py -q -k corridor` | ❌ W0 | ⬜ pending |
| 11-02-01 | 02 | 2 | TPL-01,TPL-02 | unit | `python -m pytest tests/test_template_path_planner.py -q -k template` | ❌ W0 | ⬜ pending |
| 11-03-01 | 03 | 3 | TPL-02,TPL-03,TPL-04 | integration | `python -m pytest tests/test_template_path_planner.py -q -k integration` | ❌ W0 | ⬜ pending |
| 11-04-01 | 04 | 4 | TPL-03,TPL-04 | replay | `python scripts/eval_template_planner.py --help` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_template_path_planner.py` - stubs and synthetic corridor fixtures for TPL-01 through TPL-04
- [ ] `tests/conftest.py` - optional reusable BEV corridor fixtures if needed

*Existing infrastructure covers the test runner itself; only phase-specific tests/fixtures are missing.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Overlay path remains visually inside sidewalk corridor on representative turning video | TPL-03 | Final visual acceptability is easier to judge in replay than by a single scalar | Run headless replay on the representative video, save output video, inspect left/right turns and branch-entry windows for visible corridor exit or branch-flip |
| Low-confidence slowdown looks sensible during ambiguous turns | TPL-04 | Human judgment is needed to decide whether slowdown/hold behavior is too timid or too aggressive | Replay a low-evidence sequence, inspect logged `approval_confidence`, `suggested_slowdown`, and resulting speed command trend |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 20s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
