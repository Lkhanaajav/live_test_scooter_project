# Phase 1: Structural Reorganization - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-03-30
**Phase:** 01-structural-reorganization
**Areas discussed:** Chapter merge strategy, Cross-reference cleanup, Iteration framing, Section ordering

---

## Chapter Merge Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Single section at end | Add 'System Integration & Control' as final section of System Design | |
| Spread into subsections | Distribute by topic into existing sections | |
| You decide | Claude picks best approach | ✓ |

**User's choice:** Claude's discretion
**Notes:** User trusts Claude to pick the merge strategy that reads best.

---

## Cross-Reference Cleanup

| Option | Description | Selected |
|--------|-------------|----------|
| Fresh labels | Rename all \label{ch:*} to match new chapter numbers | ✓ |
| Keep old labels | Leave labels as-is, LaTeX resolves correctly | |
| You decide | Claude picks | |

**User's choice:** Fresh labels — clean rename of all chapter labels.

---

## Iteration Framing

| Option | Description | Selected |
|--------|-------------|----------|
| Final system + history | Present final system in Methodology, iteration story in Results | |
| Inline chronological | Walk through v1→v2→v3→v4 in Methodology | |
| You decide | Claude picks best approach | ✓ |

**User's choice:** Claude's discretion
**Notes:** Fix "three" to "four" regardless of presentation approach.

---

## Section Ordering

| Option | Description | Selected |
|--------|-------------|----------|
| Pipeline order | Follow data flow: Hardware → Seg → BEV → Planners → GPS → Obstacles → Safety → Temporal | ✓ |
| Importance order | Lead with key contributions first | |
| You decide | Claude picks | |

**User's choice:** Pipeline order — follow the data flow through the system.

---

## Claude's Discretion

- Chapter merge placement (single section vs. distributed)
- Iteration presentation approach (final system vs. chronological)

## Deferred Ideas

None.
