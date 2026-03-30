# Phase 3: Methodology & Results Rewrite - Context

**Gathered:** 2026-03-30
**Status:** Ready for planning

<domain>
## Phase Boundary

Rewrite Chapter 3 (System Design) and Chapter 4 (Experimental Evaluation) with design rationale ("why before what") and claim-based evaluation structure. Remove checkpoint benchmark, add iteration progression, consolidate runtime tables, add naive baseline discussion.

</domain>

<decisions>
## Implementation Decisions

### Design Rationale Framing (NARR-03)
- **D-01:** Why-then-what per component. Each component section in System Design starts with 1-2 paragraphs explaining WHY this approach was chosen (constraint, trade-off, lesson from prior iteration), THEN describes WHAT it does.
- **D-02:** Design rationale must cover at minimum: SegFormer-B0 choice (why not bigger model), BEV homography (why not learned BEV), skeleton graph (why it was tried and abandoned), template arc planner (why propose-and-verify instead of discover), image-space midpoint (why it works for straight-ahead), teacher-student framework (why OneFormer Swin-L over SegFormer-B2).

### Checkpoint Table Replacement (EVAL-01)
- **D-03:** Remove tab:checkpoint_benchmark entirely (11-checkpoint comparison adds no scientific value).
- **D-04:** Replace with 2-3 paragraphs describing teacher-student training progression: SegFormer-B2 teacher → OneFormer Swin-L teacher, why the switch improved IoU from 0.758 to 0.946. The existing tab:seg_comparison already has the quantitative comparison.

### Evaluation Restructuring (NARR-04, STRUCT-02, EVAL-02)
- **D-05:** Reorganize Chapter 4 around claim-driven sections:
  - Claim 1: Segmentation quality — teacher-student improves mask quality
  - Claim 2: Planning domain — image-space methods dominate BEV for straight-ahead
  - Claim 3: BEV fragility — monocular BEV is unreliable as primary planning domain
  - Claim 4: Template-approval — propose-and-verify is more stable than skeleton discovery
  - Claim 5: System validation — complete pipeline sustains stable operation
- **D-06:** Each claim section follows: state claim → present evidence (tables/figures) → conclude.
- **D-07:** Add design iteration progression table (v1→v2→v3→v4) showing key metric at each stage. Weave into the planning domain claim or as a summary subsection.

### Runtime Table Consolidation (EVAL-03, EVAL-04)
- **D-08:** Merge tab:runtime_comparison and tab:runtime_offenders into one clean per-module runtime table. Keep tab:runtime_configs as supporting evidence.
- **D-09:** Add 1-2 paragraphs establishing naive baseline (raw mask center-following — conceptual lower bound, no new experiments needed). This is a thought experiment showing what performance looks like without any planning at all.

### Tone & Consistency (carried from Phase 2)
- **D-10:** Traditional thesis formal tone (measured, hedged) — same as Phase 2 decisions D-09, D-10.
- **D-11:** BEV nuance (carried from Phase 2 D-05): don't oversimplify. Real system uses BEV for corridor extraction and turns. The argument is about HOW to use BEV.

### Claude's Discretion
- Exact claim wording and ordering within sections
- How to handle the iteration progression table format (horizontal timeline vs vertical table)
- Where to place the naive baseline discussion (within planning domain claim or separate subsection)
- Whether to keep tab:runtime_configs as-is or integrate it into the merged runtime table
- How much of the current System Design prose can be preserved vs. needs full rewrite

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Thesis Source
- `thesis/main.tex` — Current thesis. System Design at Ch.3 (lines ~370-700), Evaluation at Ch.4 (lines ~740-1195).

### System Documentation
- `simulation_camera_scooter/RUNTIME_RUNBOOK.md` — Accepted run data (1800 frames, template planner, 9.97 FPS)
- `simulation_camera_scooter/MUST_READ_TURN_CONTAINMENT.md` — Turn planner validation (0% containment failure)

### First Paper
- `test.md` — Original pipeline paper (v1 skeleton-based). Contains the results the thesis must frame as Iteration 1.

### Phase 2 Outputs
- `.planning/phases/02-introduction-literature-review/02-CONTEXT.md` — Tone decisions (D-09, D-10), BEV nuance (D-05)

### Planning Documents
- `.planning/REQUIREMENTS.md` — NARR-03, NARR-04, EVAL-01 through EVAL-04, STRUCT-02
- `.planning/PROJECT.md` — Project context and key decisions

</canonical_refs>

<code_context>
## Existing Code Insights

### Current Tables in Evaluation (to be restructured)
- `tab:video_dataset` — Test video summary (keep)
- `tab:seg_comparison` — Segmentation quality, 32 frames (keep, key evidence)
- `tab:fullvideo_replay` — Full-video metrics, 22,679 frames (keep)
- `tab:planner_comparison` — 5-planner comparison (keep, Finding 1 evidence)
- `tab:oracle_comparison` — Oracle mask comparison (keep, BEV fragility evidence)
- `tab:bev_fragility` — BEV reliability stats (keep, Finding 1 evidence)
- `tab:runtime_comparison` — Per-module runtime (MERGE with offenders)
- `tab:runtime_offenders` — Ranked cost (MERGE into runtime_comparison)
- `tab:runtime_configs` — System configs (keep as supporting)
- `tab:checkpoint_benchmark` — 11 checkpoints (REMOVE per D-03)
- `tab:template_eval` — Template vs skeleton (keep, Finding 2 evidence)
- `tab:waypoint_turn_eval` — Turn planner eval (keep)
- `tab:accepted_run` — 1800-frame run (keep, system validation)

### Key Data Points
- Image-space midpoint: 14.3px center error, 2.2ms (421x faster than BEV DT)
- BEV DT: 65.0px center error, 926.8ms
- BEV fragility: 99.3% frame failure
- Template vs skeleton: 40.6% heading error reduction, 50% fewer path switches
- Teacher-student: IoU 0.758 → 0.946, inference 18.9ms → 11.7ms
- Accepted run: 1800 frames, 100% template source, 9.97 FPS, IoU 0.954
- Turn planner: 0% containment failure, 12.9 FPS

</code_context>

<specifics>
## Specific Ideas

- The iteration progression table should show: Iteration | Planner | Domain | Key Metric | What Changed
- The naive baseline is conceptual — "what if you just followed the mask center row-by-row without any planning?" This is approximately what image-space midpoint does, making the 421x speedup result even more striking.
- The checkpoint benchmark removal should be clean — no trace of "11 checkpoints compared" narrative. Replace with the teacher-student story.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-methodology-results-rewrite*
*Context gathered: 2026-03-30*
