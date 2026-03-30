# Phase 2: Introduction & Literature Review - Context

**Gathered:** 2026-03-30
**Status:** Ready for planning

<domain>
## Phase Boundary

Rewrite Chapter 1 (Introduction) and Chapter 2 (Background and Related Work) to set up the thesis contribution with a compelling narrative. The two-finding structure and design iteration story must be clearly established in the Introduction. The Literature Review must sharpen each section's gap statement to point toward this thesis's contributions.

</domain>

<decisions>
## Implementation Decisions

### Introduction Opening & Hook
- **D-01:** Open with a concrete scenario — a delivery robot or scooter on a cracked sidewalk with no lane markings, no LiDAR, just a cheap camera. Ground the reader in the real-world problem before stating the research question.
- **D-02:** The opening must establish both the practical constraint (embedded hardware, monocular camera) and the intellectual question (do you really need BEV and graph-based planning for sidewalk navigation?).

### Contribution Framing
- **D-03:** Frame contributions around two key findings:
  - **Finding 1 (Benchmarking):** Image-space midpoint planning dominates BEV methods for straight-ahead following — 421x faster, 4.5x lower lateral error. BEV is fragile (99.3% frame failure in one sequence).
  - **Finding 2 (System Design):** Template-approval on BEV corridors replaces fragile skeleton-graph planning. The evolution from expensive discovery (graph construction from noisy pixels) to cheap verification (score 5 pre-drawn arcs against DT corridor) is the engineering contribution.
- **D-04:** Reduce the current 7-item contribution list to focus on these two findings plus supporting contributions (teacher-student training, comprehensive evaluation protocol, GPS-conditioned turn planner).
- **D-05:** The real system uses BEV for corridor extraction and turns — do NOT oversimplify to "BEV bad, image-space good." The argument is about *how* you use BEV (verify templates, not build graphs).

### Literature Review Structure
- **D-06:** Keep the current 8-section thematic structure (Micro-Mobility, Segmentation, BEV, Path Planning, Distance Transform, Skeletonization, Teacher-Student, Embedded Perception).
- **D-07:** Rewrite each section to end with a specific gap statement that this thesis fills. Currently the gaps are only in the final "Summary and Research Gap" section — distribute them.
- **D-08:** The final "Research Gaps" section must connect all 4 identified gaps to the two main findings from D-03.

### Tone & Voice
- **D-09:** Traditional thesis formal tone — measured, hedged language. "Results suggest..." rather than "We demonstrate...". Passive voice acceptable. Err on the side of being conservative for committee review.
- **D-10:** Do NOT match the more assertive conference-paper style. This is a thesis for a conservative committee at the University of Oklahoma.

### Claude's Discretion
- Exact wording and paragraph structure within each section
- How to distribute the 4 research gaps across the 8 lit review sections
- Whether to add a "Thesis Organization" subsection at the end of the Introduction (currently exists, may need updating)
- How many contributions to list (consolidate from 7 down to 4-5, Claude decides exact grouping)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Thesis Source
- `thesis/main.tex` — Current thesis with 6-chapter structure (Phase 1 complete). Introduction at lines 197-283, Background at lines 288-367.

### First Paper (Pipeline v1)
- `test.md` — Original conference paper describing seg→BEV→skeleton→graph pipeline. Contains the v1 pipeline description, contributions, and results that the thesis must supersede.

### System Documentation
- `simulation_camera_scooter/RUNTIME_RUNBOOK.md` — Accepted run configuration, confirms template planner is current system (1800 frames, 100% template path source)
- `simulation_camera_scooter/MUST_READ_TURN_CONTAINMENT.md` — Turn planner validation results and safety architecture

### Planning Documents
- `.planning/PROJECT.md` — Project context, constraints, problems with current draft
- `.planning/REQUIREMENTS.md` — NARR-01 (rewrite Introduction) and NARR-02 (rewrite Lit Review as themed synthesis)
- `.planning/research/ARCHITECTURE.md` — Recommended chapter structure with page targets

</canonical_refs>

<code_context>
## Existing Code Insights

### Current Chapter Structure (after Phase 1)
- Ch. 1: Introduction (`ch:introduction`) — Motivation, Problem Statement, Approach Overview, Contributions, Thesis Organization (~90 lines)
- Ch. 2: Background and Related Work (`ch:background`) — 8 themed sections + Research Gap (~80 lines)

### Key Data Points for Introduction
- 5-planner comparison table (Table 3): image-space midpoint 14.3px/2.2ms vs BEV DT 65.0px/926.8ms
- BEV fragility: 99.3% frame failure rate in one sequence
- Template planner: 1800-frame run, 100% template path source, 9.97 FPS, IoU 0.954
- Teacher-student: OneFormer Swin-L → SegFormer-B0, IoU 0.946 at 11.7ms
- Template bank: 5 arcs (straight, left_gentle, left_sharp, right_gentle, right_sharp)
- System evolution: skeleton graph (109ms/9.1 FPS) → template approval (69ms/10 FPS)

### Design Iteration History
- Iteration 1: BEV skeleton-graph planner with SegFormer-B2 teacher
- Iteration 2: BEV distance-transform corridor planner with enhanced mask refinement
- Iteration 3: Image-space midpoint and DT planners with OneFormer Swin-L teacher
- Iteration 4: Template arc planner with GPS-conditioned waypoint-turn planner

</code_context>

<specifics>
## Specific Ideas

- The key insight is "propose-and-verify replaces discover-from-scratch" — template approval instead of skeleton graph construction
- The image-space vs BEV comparison is a benchmarking result, not the system design. The real system uses BEV for corridor extraction.
- Template bank is extensible — better templates improve the system without changing architecture. This is a future work point.
- User wants to understand and be able to explain every claim in the thesis. No hand-waving.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-introduction-literature-review*
*Context gathered: 2026-03-30*
