# Phase 11: Template path fitting inside segmentation corridor with path approval scoring - Context

**Gathered:** 2026-03-12
**Status:** Ready for planning

<domain>
## Phase Boundary

Replace direct centerline-following as the final path decision with a small-compute, intent-conditioned path approval layer.

The phase should:
- generate a compact bank of smooth candidate paths from the ego pose in BEV,
- condition that candidate bank on route/GPS maneuver intent (`straight`, `left`, `right`),
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

### GPS Chooses Maneuver Intent
- Vision should not semantically guess whether the scooter should turn left or right at an intersection.
- GPS or route logic provides maneuver intent; Phase 11 only decides whether a path consistent with that intent can be approved inside the visible corridor.

### Small-Compute Constraint
- Keep the planner geometric and lightweight.
- The candidate bank and scoring must be practical for low-compute scooter hardware.

### Turn Handling
- The planner must explicitly support straight, gentle-turn, medium-turn, and sharper-turn path families.
- Candidate families must be filtered by the commanded intent before scoring.
- Turns should be handled by better template fit and continuity scoring, not by graph branch search alone.

### Low-Confidence Behavior
- If all intent-consistent candidates fit poorly or corridor evidence is ambiguous, emit low confidence and recommend slowdown/hold instead of forcing a turn selection.
- Do not fall back to a different maneuver direction just because it scores better visually in the current frame.

### Integration Contract
- Approved outputs must remain compatible with the current controller contract: metric path polyline, overlayable path pixels, and confidence/speed guidance.

### Claude's Discretion
- Exact candidate family representation can be cubic splines, clothoid-like parameterizations, or another lightweight smooth-path primitive.
- Exact scoring weights can be decided during planning, but near-field fit, corridor containment, clearance, temporal continuity, and intent consistency must all be represented.
</decisions>

<specifics>
## Specific Ideas

- Score each intent-consistent candidate by:
  - near-field corridor fit in the first 1-2 meters
  - percentage of samples inside the sidewalk corridor
  - clearance to corridor edges
  - preference for staying near the corridor center when evidence is good
  - continuity with the previous approved path
  - curvature / feasibility penalty
- Reject paths that leave the corridor early or repeatedly.
- If GPS says `right` and no right-consistent candidate fits well enough, emit low confidence and slow/hold rather than approving `straight` or `left`.
- Reuse existing BEV metric coordinates so this planner can sit between perception and controller with minimal integration risk.
</specifics>

<deferred>
## Deferred Ideas

- Do not combine this phase with a heavy learned planner.
- Do not replace obstacle veto logic here; that can remain a separate layer above or beside path approval.
- Do not require full live-control integration inside the planning phase; prioritize offline path approval quality first.
- Do not ask vision to decide the semantic maneuver at intersections; that belongs to GPS/route logic.
</deferred>

---

*Phase: 11-template-path-fitting-inside-segmentation-corridor-with-path-approval-scoring*
*Context gathered: 2026-03-12 via design discussion*
