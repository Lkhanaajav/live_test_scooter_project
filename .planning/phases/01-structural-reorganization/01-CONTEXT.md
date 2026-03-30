# Phase 1: Structural Reorganization - Context

**Gathered:** 2026-03-30
**Status:** Ready for planning

<domain>
## Phase Boundary

Transform the current 7-chapter thesis (`thesis/main.tex`) into a clean 6-chapter structure by merging the Closed-Loop Control chapter into System Design, fixing the iteration count inconsistency, and renaming all LaTeX labels to match the new structure. No prose rewriting — structure only.

</domain>

<decisions>
## Implementation Decisions

### Chapter Merge Strategy
- **D-01:** Claude's discretion on where Closed-Loop content lands in System Design. Recommended approach: single "System Integration and Control" section at end of System Design, or distribute by topic — whichever reads best with the pipeline-order structure.

### Cross-Reference Cleanup
- **D-02:** Fresh labels throughout — rename ALL `\label{ch:*}` and `\ref{ch:*}` to match new chapter numbers. Clean slate, no legacy label names.

### Iteration Framing
- **D-03:** Claude's discretion on how to present the 4 design iterations. Options: present final system in Methodology with iteration history in Results, or walk chronologically in Methodology. Pick what reads best.
- **D-04:** Fix inconsistency — consistently say "four design iterations" (not "three").

### Section Ordering
- **D-05:** Follow pipeline/data-flow order within System Design: Hardware → Segmentation → BEV → Planners → GPS/Control → Obstacles → Safety → Temporal Smoothing → Software Architecture.

### Claude's Discretion
- Merge placement strategy (D-01)
- Iteration presentation approach (D-03)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Thesis Source
- `thesis/main.tex` — Current 7-chapter thesis (1,323 lines). The file being restructured.

### Planning Documents
- `.planning/PROJECT.md` — Project context and constraints
- `.planning/REQUIREMENTS.md` — STRUCT-01 (merge chapters) and STRUCT-04 (fix iteration count)
- `.planning/research/ARCHITECTURE.md` — Recommended 6-chapter structure with page targets

</canonical_refs>

<code_context>
## Existing Code Insights

### Current Chapter Structure (thesis/main.tex)
- Ch. 1: Introduction (`\label{ch:introduction}`) — lines 197-283
- Ch. 2: Literature Review (`\label{ch:related_work}`) — lines 288-367
- Ch. 3: System Design and Methodology (`\label{ch:methodology}`) — lines 370-656
- Ch. 4: Closed-Loop Control (`\label{ch:closed_loop}`) — lines 659-737
- Ch. 5: Experiments and Results (`\label{ch:results}`) — lines 740-1211
- Ch. 6: Discussion (`\label{ch:discussion}`) — lines 1214-1274
- Ch. 7: Conclusion (`\label{ch:conclusion}`) — lines 1277-1323

### Key Labels to Rename
- `ch:introduction` → `ch:introduction` (stays Ch. 1)
- `ch:related_work` → `ch:background` (stays Ch. 2)
- `ch:methodology` → `ch:system_design` (stays Ch. 3)
- `ch:closed_loop` → REMOVED (merged into Ch. 3)
- `ch:results` → `ch:evaluation` (becomes Ch. 4)
- `ch:discussion` → `ch:discussion` (becomes Ch. 5)
- `ch:conclusion` → `ch:conclusion` (becomes Ch. 6)

### Cross-References to Update
All `\ref{ch:*}` in Thesis Organization section, chapter intros, and cross-chapter references.

</code_context>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches. User trusts Claude's judgment on merge placement and iteration framing.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-structural-reorganization*
*Context gathered: 2026-03-30*
