# Phase 11: Template path fitting inside segmentation corridor with path approval scoring - Context

**Gathered:** 2026-03-12
**Status:** Ready for planning

<domain>
## Phase Boundary

Replace direct centerline-following as the final path decision with a small-compute path approval layer.

The phase should:
- generate a compact bank of smooth candidate paths from the ego pose in BEV,
- evaluate how well each candidate fits inside the perceived sidewalk corridor,
- approve the best candidate only when evidence is strong enough,
- output controller-ready metric paths plus confidence/slowdown signals.

The phase should not require a heavy learned planner or a large end-to-end policy.
</domain>

<decisions>
## Implementation Decisions

### Final Path Should Be Approved, Not Assumed
- Do not drive the exact middle of the segmentation mask by default.
- Use the perceived corridor as evidence and choose from a bank of feasible path templates.

### Small-Compute Constraint
- Keep the planner geometric and lightweight.
- The candidate bank and scoring must be practical for low-compute scooter hardware.

### Turn Handling
- The planner must explicitly support straight, gentle-turn, medium-turn, and sharper-turn path families.
- Turns should be handled by better template fit and continuity scoring, not by graph branch search alone.

### Low-Confidence Behavior
- If all candidates fit poorly or corridor evidence is ambiguous, emit low confidence and recommend slowdown/hold instead of forcing a turn selection.

### Integration Contract
- Approved outputs must remain compatible with the current controller contract: metric path polyline, overlayable path pixels, and confidence/speed guidance.

### Claude's Discretion
- Exact candidate family representation can be cubic splines, clothoid-like parameterizations, or another lightweight smooth-path primitive.
- Exact scoring weights can be decided during planning, but corridor containment, clearance, center preference, and temporal continuity must all be represented.
</decisions>

<specifics>
## Specific Ideas

- Score each candidate by:
  - percentage of samples inside the sidewalk corridor
  - clearance to corridor edges
  - preference for staying near the corridor center when evidence is good
  - continuity with the previous approved path
  - curvature / feasibility penalty
- Reject paths that leave the corridor early or repeatedly.
- Reuse existing BEV metric coordinates so this planner can sit between perception and controller with minimal integration risk.
</specifics>

<deferred>
## Deferred Ideas

- Do not combine this phase with a heavy learned planner.
- Do not replace obstacle veto logic here; that can remain a separate layer above or beside path approval.
- Do not require full live-control integration inside the planning phase; prioritize offline path approval quality first.
</deferred>

---

*Phase: 11-template-path-fitting-inside-segmentation-corridor-with-path-approval-scoring*
*Context gathered: 2026-03-12 via design discussion*
