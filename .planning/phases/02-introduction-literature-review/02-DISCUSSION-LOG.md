# Phase 2: Introduction & Literature Review - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-03-30
**Phase:** 02-introduction-literature-review
**Areas discussed:** Introduction opening & hook, Contribution framing, Lit Review structure, Tone & voice

---

## Introduction Opening & Hook

This area turned into a deep system understanding session. User questioned the "image-space vs BEV" framing, which led to a full investigation of:
- The first paper (test.md): seg→BEV→skeleton→graph pipeline at 109ms
- The current code: template planner runs (DT_PLANNER module doesn't exist, import fails silently)
- RUNTIME_RUNBOOK.md confirms template planner on all 1800 frames
- The real story is two findings, not one

| Option | Description | Selected |
|--------|-------------|----------|
| Concrete scenario | Open with vivid picture of a robot on a sidewalk | ✓ |
| Gap-first academic | Open with the research gap directly | |
| Statistics-driven | Open with market size numbers | |

**User's choice:** Concrete scenario, but informed by the real system story (template approval replacing skeleton graph, not just "image-space > BEV")
**Notes:** User did not know the exact details of their own system's findings. The deep dive was necessary to establish what the thesis should actually claim.

---

## Contribution Framing

| Option | Description | Selected |
|--------|-------------|----------|
| Two-finding structure | Benchmarking finding + system design finding | ✓ |
| Design iteration story | 4-iteration chronological narrative | |
| Keep current 7-item list | Rewrite prose around existing 7 contributions | |

**User's choice:** Two-finding structure
**Notes:** Consolidate 7 items down to focus on the two key findings plus supporting contributions.

---

## Lit Review Structure

| Option | Description | Selected |
|--------|-------------|----------|
| Keep 8 sections, sharpen gaps | Keep themes, add per-section gap statements | ✓ |
| Consolidate to 5 sections | Merge related topics | |
| Restructure around two findings | Build toward the findings narratively | |

**User's choice:** Keep 8 sections, sharpen gaps
**Notes:** Current thematic structure already covers the field well.

---

## Tone & Voice

| Option | Description | Selected |
|--------|-------------|----------|
| Confident academic | Active voice, assertive, CVPR-style | |
| Traditional thesis formal | Measured, hedged, passive voice OK | ✓ |
| Match Discussion chapter | Use existing ch:discussion as style target | |

**User's choice:** Traditional thesis formal
**Notes:** Conservative approach for OU committee review.

---

## Key Insight from Discussion

The most important outcome of this discussion was not the gray area answers but the system understanding session. The user discovered:
1. The thesis was framing "image-space > BEV" but the real system uses BEV for corridor extraction
2. The skeleton graph from the first paper was replaced by template approval, not by image-space midpoint
3. The DT planner module doesn't exist — template planner runs by default via silent import failure
4. The two-finding structure (benchmarking + system design) is the correct framing
