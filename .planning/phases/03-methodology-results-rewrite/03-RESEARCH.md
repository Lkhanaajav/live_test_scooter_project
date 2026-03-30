# Phase 3: Methodology & Results Rewrite - Research

**Researched:** 2026-03-30
**Domain:** Academic thesis writing -- System Design (Chapter 3) and Experimental Evaluation (Chapter 4) rewrite
**Confidence:** HIGH

## Summary

This phase rewrites two substantial chapters of a Master's thesis: Chapter 3 (System Design, lines 390-735, approximately 345 lines of LaTeX) and Chapter 4 (Experimental Evaluation, lines 738-1209, approximately 470 lines of LaTeX). Together these chapters constitute approximately 60% of the thesis body. The current System Design chapter describes the pipeline components in a "what-it-does" style without design rationale. The current Evaluation chapter presents results as a flat list of experiments without a coherent claim-driven narrative, includes a scientifically worthless checkpoint benchmark table (11 fine-tuned SegFormer checkpoints compared against each other), has redundant runtime tables, and lacks both an iteration progression view and a naive baseline discussion.

The rewrite must accomplish four things: (1) restructure System Design so each component section starts with 1-2 paragraphs explaining WHY the approach was chosen before describing WHAT it does; (2) replace the checkpoint benchmark with a teacher-student progression narrative; (3) reorganize Evaluation around five claim-driven sections (segmentation quality, planning domain, BEV fragility, template-approval stability, system validation); (4) add an iteration progression table and a conceptual naive baseline discussion, while merging redundant runtime tables.

This is the largest phase in the project (7 requirements spanning two chapters). The Phase 2 plan pattern -- splitting into 2 plans, one per chapter -- worked well and should be replicated. However, given the larger scope, plan granularity within each chapter matters more here.

**Primary recommendation:** Split into 2 plans: Plan 01 for System Design rewrite (NARR-03), Plan 02 for Evaluation restructure (NARR-04, EVAL-01 through EVAL-04, STRUCT-02). Each plan should work on thesis/main.tex with careful line-range awareness. The System Design plan preserves existing technical content while prepending design rationale paragraphs; the Evaluation plan is a heavier restructure that reorders sections, removes/merges tables, and adds new content.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Why-then-what per component. Each component section in System Design starts with 1-2 paragraphs explaining WHY this approach was chosen (constraint, trade-off, lesson from prior iteration), THEN describes WHAT it does.
- **D-02:** Design rationale must cover at minimum: SegFormer-B0 choice (why not bigger model), BEV homography (why not learned BEV), skeleton graph (why it was tried and abandoned), template arc planner (why propose-and-verify instead of discover), image-space midpoint (why it works for straight-ahead), teacher-student framework (why OneFormer Swin-L over SegFormer-B2).
- **D-03:** Remove tab:checkpoint_benchmark entirely (11-checkpoint comparison adds no scientific value).
- **D-04:** Replace with 2-3 paragraphs describing teacher-student training progression: SegFormer-B2 teacher to OneFormer Swin-L teacher, why the switch improved IoU from 0.758 to 0.946. The existing tab:seg_comparison already has the quantitative comparison.
- **D-05:** Reorganize Chapter 4 around claim-driven sections:
  - Claim 1: Segmentation quality -- teacher-student improves mask quality
  - Claim 2: Planning domain -- image-space methods dominate BEV for straight-ahead
  - Claim 3: BEV fragility -- monocular BEV is unreliable as primary planning domain
  - Claim 4: Template-approval -- propose-and-verify is more stable than skeleton discovery
  - Claim 5: System validation -- complete pipeline sustains stable operation
- **D-06:** Each claim section follows: state claim, present evidence (tables/figures), conclude.
- **D-07:** Add design iteration progression table (v1->v2->v3->v4) showing key metric at each stage. Weave into the planning domain claim or as a summary subsection.
- **D-08:** Merge tab:runtime_comparison and tab:runtime_offenders into one clean per-module runtime table. Keep tab:runtime_configs as supporting evidence.
- **D-09:** Add 1-2 paragraphs establishing naive baseline (raw mask center-following -- conceptual lower bound, no new experiments needed). This is a thought experiment showing what performance looks like without any planning at all.
- **D-10:** Traditional thesis formal tone (measured, hedged) -- same as Phase 2 decisions D-09, D-10.
- **D-11:** BEV nuance (carried from Phase 2 D-05): don't oversimplify. Real system uses BEV for corridor extraction and turns. The argument is about HOW to use BEV.

### Claude's Discretion
- Exact claim wording and ordering within sections
- How to handle the iteration progression table format (horizontal timeline vs vertical table)
- Where to place the naive baseline discussion (within planning domain claim or separate subsection)
- Whether to keep tab:runtime_configs as-is or integrate it into the merged runtime table
- How much of the current System Design prose can be preserved vs. needs full rewrite
- Task decomposition within each plan

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| NARR-03 | Add design rationale ("why before what") for every methodology design choice | D-01/D-02 define exact pattern; Architecture Patterns below provide target structure for each section; design rationale source material identified from CONTEXT.md code_context and test.md (v1 pipeline) |
| NARR-04 | Frame all Results sections as claim-evidence-conclusion | D-05/D-06 define 5 claims with claim-evidence-conclusion pattern; Architecture Patterns below map each claim to its evidence tables/figures |
| EVAL-01 | Remove checkpoint benchmark table (Table 7) -- replace with teacher-student comparison narrative | D-03/D-04 define removal and replacement; current table at lines 1069-1086 (tab:checkpoint_benchmark); replacement prose covers B2 teacher to OneFormer switch |
| EVAL-02 | Add design iteration progression table (v1->v2->v3->v4 with key metrics at each stage) | D-07 defines content; iteration data available in CONTEXT.md code_context section; v1 data from test.md |
| EVAL-03 | Consolidate redundant runtime tables (merge current Tables 5+6 into one) | D-08 defines merge: tab:runtime_comparison (lines 986-1006) + tab:runtime_offenders (lines 1022-1034) into one table |
| EVAL-04 | Add naive baseline discussion (raw mask center-following as conceptual baseline) | D-09 defines conceptual approach; no new experiments; 1-2 paragraphs within or adjacent to planning domain claim |
| STRUCT-02 | Restructure Results chapter around claims (claim-evidence-conclusion pattern) | D-05 defines 5 claims; current flat structure must be reordered into claim-driven sections |
</phase_requirements>

## Project Constraints (from CLAUDE.md)

- **LaTeX only** -- all edits are to `thesis/main.tex`; no new files created for content (exception: if a new generated table is needed, it goes in `thesis/tables/generated/`)
- **No new experiments or data** -- all numbers must come from existing evaluation data
- **No new figures** -- reuse existing figures; minor relabeling acceptable
- **60-80 pages** double-spaced target for entire thesis; System Design should be approximately 15-20 pages, Evaluation approximately 15-20 pages
- **OU format** -- 12pt, double-spaced, Times New Roman (mathptmx), 1.5" left margin
- **Existing bibliography** -- `references.bib` has 40+ entries; reuse citation keys
- **Existing cross-references** -- must preserve all `\label{}` identifiers that are referenced from other chapters (Introduction, Discussion, Conclusion)
- **Existing figures** -- all `\begin{figure}` blocks must be preserved; can be moved but not removed
- **Tone** -- measured, hedged thesis tone (D-10), not conference-paper assertive style
- **config.py single source of truth** -- any parameter values cited in the thesis must match config.py

## Architecture Patterns

### Chapter 3: System Design -- Current vs. Target Structure

**Current structure (lines 390-735, ~345 lines):**
```
\chapter{System Design}
  Opening paragraph (no rationale)
  \section{Hardware Platform}
  \section{Segmentation Module and Supervision Strategy}
    \subsection{Teacher--Student Supervision}
    \subsection{Hybrid Training Dataset}
    \subsection{Loss Function}
    \subsection{Training Progression}
  \section{Resolution Trade-Off Analysis}
  \section{Bird's-Eye View Projection}
  \section{BEV Mask Refinement}
  \section{Path Planning Methods}
    \subsection{BEV Skeleton-Graph Planner}
    \subsection{BEV Distance-Transform Ridge Planner}
    \subsection{BEV Template Arc Planner}
    \subsection{Image-Space Midpoint Planner}
    \subsection{Image-Space Distance-Transform Planner}
    \subsection{GPS-Conditioned Waypoint-Turn Planner}
    \subsection{Turn Containment Safety Guard}
  \section{Temporal Smoothing}
  \section{Lightweight Object Detection}
    \subsection{Monocular Distance Estimation}
    \subsection{Speed Modulation}
  \section{GPS Waypoint Navigation}
  \section{Steering and Speed Computation}
  \section{Serial Command Protocol}
  \section{Safety Mechanisms}
  \section{Evaluation Metrics}
  \section{Software Architecture}
```

**Target structure (same sections, but with rationale paragraphs prepended):**
```
\chapter{System Design}
  Opening paragraph: Design philosophy overview + iteration narrative
  \section{Hardware Platform}
    [WHY: constraint-first paragraph explaining what the hardware limits force]
    [WHAT: existing content, minor edits for tone]
  \section{Segmentation Module and Supervision Strategy}
    [WHY: Why SegFormer-B0? Why not bigger? Why teacher-student? Why OneFormer over B2?]
    [Covers D-02 rationale for SegFormer-B0, teacher choice]
    \subsection{Teacher--Student Supervision}  -- rewritten for clarity
    \subsection{Hybrid Training Dataset}
    \subsection{Loss Function}
    \subsection{Training Progression}
  \section{Resolution Trade-Off Analysis}
    [WHY: Why this matters for real-time constraint]
  \section{Bird's-Eye View Projection}
    [WHY: Why homography? Why not learned BEV? What are the assumptions?]
    [Covers D-02 rationale for BEV homography]
    [NUANCE per D-11: BEV is used for corridor extraction and turns, not dismissed]
  \section{BEV Mask Refinement}
    [WHY: Why needed -- perspective warp amplifies seg noise]
  \section{Path Planning Methods}
    [WHY overview: Design iteration narrative -- from discovery to verification]
    [WHY per subsection per D-02:]
    \subsection{BEV Skeleton-Graph} -- [WHY tried, WHY abandoned: noise, cost]
    \subsection{BEV Distance-Transform Ridge} -- [WHY ridge over skeleton, still BEV cost]
    \subsection{BEV Template Arc} -- [WHY propose-and-verify instead of discover]
    \subsection{Image-Space Midpoint} -- [WHY image-space works for straight-ahead]
    \subsection{Image-Space DT} -- [WHY needed as fallback]
    \subsection{GPS-Conditioned Waypoint-Turn} -- same
    \subsection{Turn Containment Safety Guard} -- same
  \section{Temporal Smoothing}
    [WHY: frame-by-frame jitter is unavoidable, smoothing is essential]
  [Remaining sections: minor edits for tone, add WHY where missing]
  \section{Evaluation Metrics} -- same, possibly with naive baseline metric introduced
  \section{Software Architecture} -- same
```

### Design Rationale Source Material per Component

| Component | WHY Source | Key Rationale |
|-----------|-----------|---------------|
| SegFormer-B0 | Config.py constraints, real-time requirement | 3.7M params fits RPi; B2 (24M) too slow; need <50ms on CPU |
| Teacher: OneFormer over B2 | tab:seg_comparison, tab:training_progression | B2 teacher: IoU 0.758; OneFormer teacher: IoU 0.946; switch justified by 25% improvement |
| BEV homography | D-11, sec:bev_fragility_results | Cheap (0.9ms), metric-scale needed for corridor extraction and turns; learned BEV needs GPU+multi-cam |
| Skeleton graph | test.md (v1 pipeline), tab:template_eval | v1 approach: noise-sensitive, 380ms per frame, per-pixel graph construction; abandoned for template approval |
| Template arc planner | tab:template_eval, RUNTIME_RUNBOOK | Propose-and-verify: 5 fixed arcs scored against DT corridor; eliminates graph construction noise; 40.6% heading error reduction |
| Image-space midpoint | tab:planner_comparison | 14.3px error at 2.2ms; 421x faster than BEV DT; works because camera perspective preserves lateral ordering for straight-ahead |
| Image-space DT | tab:planner_comparison | Best fallback: 99.4% inside-GT at 108.1ms; robust to irregular boundaries |
| Turn planner | MUST_READ_TURN_CONTAINMENT.md | BEV needed for metric-scale turn geometry; containment guard ensures safety; 0% failure rate |

### Chapter 4: Evaluation -- Current vs. Target Structure

**Current structure (lines 738-1209, ~470 lines):**
```
\chapter{Experimental Evaluation}
  Opening paragraph (flat listing)
  \section{Experimental Setup}
    \subsection{Data Collection}  -- tab:video_dataset
    \subsection{Hand-Annotated Ground Truth}
    \subsection{Evaluation Protocol}
  \section{Segmentation Results}  -- fig:segmentation_stage, tab:seg_comparison, fig:seg_improvement
    \subsection{Hand-Annotated Evaluation}
    \subsection{Full-Video Temporal Stability}  -- tab:fullvideo_replay
  \section{Planner Comparison Study}  -- tab:planner_comparison, fig:planner_comparison
    \subsection{Key Findings}  (4 items, flat list)
    \subsection{Oracle-Mask Experiment}  -- tab:oracle_comparison
  \section{BEV Skeleton-Graph Pipeline Visualization}  -- fig:skeleton_stage
  \section{BEV Fragility Analysis}  -- tab:bev_fragility, fig:bev_fragility
  \section{System Runtime Analysis}  -- tab:runtime_comparison, tab:runtime_offenders, tab:runtime_configs
  \section{Temporal Smoothing Evaluation}
  \section{Checkpoint Benchmark}  -- tab:checkpoint_benchmark [REMOVE]
  \section{Qualitative Results}  -- fig:seg_comparison_qual, fig:planner_comparison_qual
  \section{Template Arc Planner Evaluation}  -- tab:template_eval
  \section{Waypoint-Turn Planner Evaluation}  -- tab:waypoint_turn_eval
  \section{Full-Length Accepted Run}  -- tab:accepted_run
  \section{Overnight Containment Validation}
```

**Target structure (claim-driven per D-05/D-06):**
```
\chapter{Experimental Evaluation}
  Opening paragraph: introduces the 5 claims to be tested

  \section{Experimental Setup}  [KEEP as-is, minor edits]
    \subsection{Data Collection}
    \subsection{Hand-Annotated Ground Truth}
    \subsection{Evaluation Protocol}
    \subsection{Naive Baseline}  [NEW: 1-2 paras per D-09, conceptual]

  \section{Claim 1: Teacher-Student Training Improves Segmentation Quality}
    [State claim -> present tab:seg_comparison, fig:seg_improvement, fig:segmentation_stage,
     tab:fullvideo_replay, fig:seg_comparison_qual -> conclude]
    [REPLACES: current \section{Segmentation Results}]
    [INCLUDES: teacher-student progression narrative replacing checkpoint benchmark (D-03/D-04)]
    [Qualitative results (fig:seg_comparison_qual) moved here]

  \section{Claim 2: Image-Space Planning Dominates BEV for Straight-Ahead Following}
    [State claim -> present tab:planner_comparison, fig:planner_comparison,
     tab:oracle_comparison -> conclude]
    [REPLACES: current \section{Planner Comparison Study}]
    [INCLUDES: Iteration progression table (D-07, EVAL-02)]
    [INCLUDES: Naive baseline reference (D-09)]
    [Qualitative results (fig:planner_comparison_qual) and fig:skeleton_stage moved here as needed]

  \section{Claim 3: Monocular BEV Projection is Fragile as Primary Planning Domain}
    [State claim -> present tab:bev_fragility, fig:bev_fragility -> conclude]
    [REPLACES: current \section{BEV Fragility Analysis}]
    [Nuance per D-11: BEV is retained for corridor verification and turns]

  \section{Claim 4: Template-Approval Planning is More Stable than Skeleton Discovery}
    [State claim -> present tab:template_eval -> conclude]
    [REPLACES: current \section{Template Arc Planner Evaluation}]

  \section{Claim 5: Complete Pipeline Sustains Stable Real-Time Operation}
    [State claim -> present tab:accepted_run, merged runtime table (D-08),
     tab:runtime_configs, tab:waypoint_turn_eval, overnight validation -> conclude]
    [REPLACES: current \section{System Runtime Analysis}, \section{Waypoint-Turn Planner Evaluation},
     \section{Full-Length Accepted Run}, \section{Overnight Containment Validation}]

  \section{Temporal Smoothing Evaluation}  [KEEP, minor edits]
```

### Table Inventory and Disposition

| Current Table | Label | Lines | Action |
|---------------|-------|-------|--------|
| tab:video_dataset | Video dataset summary | 751-769 | KEEP in Experimental Setup |
| tab:seg_comparison | Segmentation quality, 32 frames | 809-822 | KEEP in Claim 1 |
| tab:fullvideo_replay | Full-video metrics, 22,679 frames | 836-852 | KEEP in Claim 1 |
| tab:planner_comparison | 5-planner comparison | 862-879 | KEEP in Claim 2 |
| tab:oracle_comparison | Oracle mask comparison | 906-919 | KEEP in Claim 2 |
| tab:bev_fragility | BEV reliability stats | 950-964 | KEEP in Claim 3 |
| tab:runtime_comparison | Per-module runtime | 986-1006 | MERGE with offenders -> Claim 5 |
| tab:runtime_offenders | Ranked cost | 1022-1034 | MERGE into runtime_comparison -> Claim 5 |
| tab:runtime_configs | System configs | 1040-1053 | KEEP as supporting in Claim 5 |
| tab:checkpoint_benchmark | 11 checkpoints | 1069-1086 | REMOVE (D-03) |
| tab:template_eval | Template vs skeleton | 1116-1134 | KEEP in Claim 4 |
| tab:waypoint_turn_eval | Turn planner eval | 1148-1164 | KEEP in Claim 5 |
| tab:accepted_run | 1800-frame run | 1177-1192 | KEEP in Claim 5 |

### NEW Tables to Create

**1. Iteration Progression Table (D-07, EVAL-02)**

```latex
\begin{table}[h]
\centering
\caption{Design iteration progression. Each iteration introduces a new planning
strategy while remaining backward-compatible with earlier stages.}
\label{tab:iteration_progression}
\begin{tabular}{clllr}
\toprule
Iter. & Planner & Domain & Key Change & Pipeline FPS \\
\midrule
v1 & Skeleton-Graph & BEV & Guo--Hall thinning + Dijkstra & 9.1 \\
v2 & DT Ridge & BEV & EDT ridge replaces skeleton & 2.4\textsuperscript{*} \\
v3 & Midpoint / DT & Image-space & BEV bypass for straight-ahead & 59.2 \\
v4 & Template Arc & BEV corridor & Propose-and-verify arcs & 9.97 \\
\bottomrule
\end{tabular}
\end{table}
```

Note: v1 FPS (9.1) comes from test.md (Section 4.6: 109.5ms/frame). v2 FPS (2.4) from tab:runtime_comparison BEV Skeleton line (416ms, but BEV DT is slower -- 926.8ms for planner alone, so more accurately ~1.1 FPS for full BEV DT, but the table shows full BEV skeleton pipeline at 2.4 FPS). The executor should verify exact numbers match existing tables. v3 FPS (59.2) from tab:runtime_comparison Image-Space column. v4 FPS (9.97) from tab:accepted_run (full pipeline including stride-4 overhead).

**2. Merged Runtime Table (D-08, EVAL-03)**

Merge tab:runtime_comparison and tab:runtime_offenders into one table. Recommendation: keep tab:runtime_comparison format (module x pipeline architecture) but add a "Recommendation" column from tab:runtime_offenders for the BEV-specific modules. Keep tab:runtime_configs as a separate supporting table.

```latex
\begin{table}[h]
\centering
\caption{Per-module runtime comparison at $640{\times}360$ (CPU-only), with
optimization guidance for BEV-pipeline components.}
\label{tab:runtime_merged}
\begin{tabular}{lccc}
\toprule
Module & BEV Pipeline & Image-Space & Note \\
\midrule
SegFormer Inference   & 11.7\,ms & 11.7\,ms & Shared \\
Mask Refinement       & 8.5\,ms  & 3.0\,ms  & Img skips BEV cleanup \\
BEV Projection        & 0.9\,ms  & ---      & \\
BEV Cleanup           & 14.6\,ms & ---      & \\
Planner               & 380.3\,ms & 2.2\,ms & $173\times$ speedup \\
\midrule
Total                 & 416.0\,ms & 16.9\,ms & \\
FPS                   & 2.4       & \textbf{59.2}  & \\
\bottomrule
\end{tabular}
\end{table}
```

### Design Rationale Writing Pattern (D-01)

Each System Design section should follow this pattern:

```latex
\section{[Component Name]}

% WHY paragraph(s) — 1-2 paragraphs of rationale BEFORE technical description
The design of [component] is driven by [constraint/trade-off/lesson from
prior iteration]. [Specific constraint: e.g., "The real-time requirement
of 10~Hz on a Raspberry Pi 4 limits the parameter budget to roughly 4M
parameters..."]. [What was tried before and why it was insufficient:
e.g., "An initial iteration employed a SegFormer-B2 teacher (24M parameters),
but the resulting student achieved only IoU 0.758..."].

% WHAT — existing technical description, preserved or lightly edited
[Existing content describing what the component does, how it works,
equations, figures, etc.]
```

### Claim-Evidence-Conclusion Pattern (D-06)

Each evaluation section should follow this pattern:

```latex
\section{[Claim Statement as Section Title]}

% STATE the claim (1 paragraph)
This section examines the claim that [specific, testable statement].
If supported, this result would [implication for the thesis argument].

% PRESENT evidence (tables, figures, analysis — bulk of section)
Table~\ref{tab:X} presents [description]...
Figure~\ref{fig:Y} illustrates [description]...
[Analysis paragraphs connecting data to the claim]

% CONCLUDE (1 paragraph)
The evidence presented in this section [supports/partially supports]
the claim that [restate]. [Qualification/limitation]. This finding
[connects to broader thesis argument / other claims].
```

### Naive Baseline Discussion Pattern (D-09)

```latex
\subsection{Naive Baseline}

As a conceptual baseline, consider a system that extracts the traversable
region via segmentation but performs no geometric path planning: for each
row of the mask, the midpoint between the left and right boundaries is
selected, and the vehicle follows this row-by-row center without temporal
smoothing, curvature estimation, or forward look-ahead. This approach is
approximately what the image-space midpoint planner implements, minus the
connected-component filtering and Savitzky--Golay smoothing that make it
a proper planner. The remarkable performance of image-space midpoint
planning (14.3~px center error at 2.2~ms) suggests that for straight-ahead
corridor following, the traversable mask itself already encodes sufficient
geometric information for path extraction, and the dominant contribution
of the planning stage is noise suppression rather than geometric reasoning.
```

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Iteration progression data | Inventing numbers | Extract from existing tables: test.md (v1), tab:runtime_comparison (v2/v3), tab:accepted_run (v4) | All data already exists across existing artifacts |
| Design rationale content | Generic "design choice" prose | Mine specific constraints from CLAUDE.md Architecture section, config.py descriptions, RUNTIME_RUNBOOK.md | Project has detailed documentation of every constraint |
| Claim wording | Inventing new claims | Derive directly from D-05 claim list and existing Key Findings in current sec:planner_comparison | Claims must match existing evidence |
| Naive baseline | New experiments or simulations | Conceptual argument based on image-space midpoint being approximately raw-mask-center-following | D-09 explicitly says "no new experiments needed" |
| Merged runtime table | New format from scratch | Combine columns from tab:runtime_comparison and rows from tab:runtime_offenders | Both tables have consistent data |
| Teacher-student narrative | New analysis | Synthesize from tab:seg_comparison (IoU 0.758 vs 0.946), tab:training_progression (teacher progression), and current sec:teacher_student prose | All data and narrative elements already exist |

## Common Pitfalls

### Pitfall 1: WHY Paragraphs That Are Actually WHAT Paragraphs
**What goes wrong:** The "design rationale" paragraph describes the component's function instead of explaining the decision to use it.
**Why it happens:** The executor has the existing WHAT content readily available and unconsciously paraphrases it instead of writing genuine rationale.
**How to avoid:** Each WHY paragraph must contain at least one of: (a) a constraint that forced the choice, (b) an alternative that was considered and rejected, (c) a lesson from a prior iteration. If the paragraph lacks all three, it is a WHAT paragraph in disguise.
**Warning signs:** A "rationale" paragraph that could be removed without losing any justification for the design choice.

### Pitfall 2: Claim Sections That Are Just Renamed Results Sections
**What goes wrong:** The evaluation sections are renamed to "Claim N: ..." but the content is still a flat presentation of results without the state-claim / present-evidence / conclude structure.
**Why it happens:** Renaming is easy; restructuring the prose to argue a specific point is harder.
**How to avoid:** Each claim section MUST have: (a) an opening paragraph that states the specific testable claim, (b) explicit connection between each table/figure and the claim, (c) a concluding paragraph that evaluates whether the evidence supports the claim.
**Warning signs:** A claim section that starts with "Table X shows..." instead of "This section examines the claim that..."

### Pitfall 3: Losing the BEV Nuance in Claims 2 and 3
**What goes wrong:** Claim 2 (image-space dominates) and Claim 3 (BEV fragility) are written as if BEV should be abandoned entirely, contradicting D-11.
**Why it happens:** The evidence for Claims 2 and 3 is so strong that the prose drifts toward absolute statements.
**How to avoid:** Claim 2 must explicitly scope to "straight-ahead following." Claim 3 must note that BEV is retained for corridor extraction and turn validation. The concluding paragraph of each claim must reference this nuance.
**Warning signs:** Any absolute statement about BEV without "for straight-ahead" or "as primary planning domain" qualification.

### Pitfall 4: Iteration Progression Table With Inconsistent Numbers
**What goes wrong:** FPS or timing numbers in the new iteration table don't match existing tables.
**Why it happens:** v1 data comes from test.md (different pipeline configuration), v2/v3 from tab:runtime_comparison, v4 from tab:accepted_run -- different measurement conditions.
**How to avoid:** Add a table footnote explaining that measurements are not directly comparable across iterations due to different configurations (BEV pipeline vs image-space, stride settings, etc.). The iteration table shows approximate system-level throughput at each design stage, not controlled benchmarks.
**Warning signs:** An FPS number in the iteration table that contradicts a number in another table without explanation.

### Pitfall 5: Checkpoint Benchmark Removal Leaving Ghost References
**What goes wrong:** The checkpoint benchmark section and table are removed but references to "Table 7" or "Section X" remain elsewhere in the document.
**Why it happens:** Other sections may reference the checkpoint benchmark or its results.
**How to avoid:** After removing tab:checkpoint_benchmark, search the entire document for `checkpoint_benchmark`, `tab:checkpoint`, and the section label to find and remove/update all references. Check the temporal smoothing section (line 1060) which references the "best checkpoint" result.
**Warning signs:** LaTeX compilation warnings about undefined references.

### Pitfall 6: System Design Rewrite Accidentally Changes Technical Content
**What goes wrong:** While adding rationale paragraphs, the executor accidentally modifies equations, parameter values, or technical descriptions.
**Why it happens:** Adding text between existing paragraphs requires careful splicing; copy-paste errors or reformulations can alter meaning.
**How to avoid:** The System Design plan should explicitly state "preserve all equations and numerical values." Technical content (equations, tables, figure references) should be checked against the original after each modification.
**Warning signs:** A parameter value that differs between System Design and Evaluation chapters.

### Pitfall 7: Evaluation Chapter Becomes Too Long
**What goes wrong:** Adding claim framing paragraphs, iteration progression table, naive baseline discussion, and teacher-student narrative to an already 470-line chapter pushes it well beyond the target page count.
**Why it happens:** Adding new content without proportionally cutting removed content (checkpoint benchmark is only ~20 lines plus table).
**How to avoid:** The checkpoint benchmark removal saves ~30 lines. The merged runtime tables save ~15 lines. But claim framing adds ~5 paragraphs. Net growth should be modest (~20-30 lines). If the chapter exceeds approximately 500 lines of LaTeX, look for opportunities to tighten existing prose.
**Warning signs:** Evaluation chapter exceeds approximately 520 lines of LaTeX.

### Pitfall 8: Cross-Reference Breakage Between Chapters
**What goes wrong:** Chapter 3 references to evaluation sections, and Chapter 4 references to System Design sections, break because labels were moved or renamed during restructuring.
**Why it happens:** The claim-driven restructuring changes section labels in Chapter 4. System Design sections may reference specific evaluation sections.
**How to avoid:** Before restructuring, catalog all cross-chapter references. After restructuring, verify each one. Key label dependencies:
- Introduction references: `ch:evaluation`, `ch:system_design`
- System Design references: `sec:bev_fragility_results`, `sec:res_seg`, `sec:planner_comparison`
- Discussion references: `tab:oracle_comparison`, `tab:planner_comparison`, `tab:template_eval`, `sec:bev_fragility_results`
- Conclusion references: all major tables
**Warning signs:** LaTeX compilation warnings.

## Code Examples

### Design Rationale Example: Segmentation Module (D-01, D-02)

```latex
\section{Segmentation Module and Supervision Strategy}
\label{sec:segmentation}

% WHY paragraphs (NEW — prepended to existing content)
The real-time constraint of the target platform---10~Hz or better on a
Raspberry Pi~4 class single-board computer---fundamentally limits the
segmentation architecture. At $640{\times}360$ input resolution, a
SegFormer-B0 model (3.7M parameters) achieves sub-50~ms inference on CPU
(Table~\ref{tab:segformer_fps}), leaving approximately 50~ms for all
downstream processing. Larger variants (B2 at 24M parameters, B5 at
84.7M) would consume the entire frame budget on segmentation alone,
leaving nothing for path planning, temporal smoothing, or control output.
The choice of SegFormer-B0 is therefore not a preference but a constraint:
it is the largest model that fits within the latency budget.

Small models trained on limited data, however, often produce noisy masks.
An initial design iteration employed a SegFormer-B2 teacher (24M parameters)
fine-tuned on 300 hand-labeled frames to generate pseudo-labels for the B0
student, but the resulting model achieved only IoU 0.758 on hand-annotated
evaluation frames (Table~\ref{tab:seg_comparison}). Switching to a
high-capacity OneFormer Swin-L teacher~\cite{oneformer}---pre-trained on
ADE20K and capable of universal semantic, instance, and panoptic
segmentation---raised the student's evaluation IoU to 0.946 while
simultaneously reducing inference time from 18.9~ms to 11.7~ms due to
improved model convergence. This improvement motivates the teacher--student
framework described in the following subsections.

% WHAT content (EXISTING — preserved with minor tone adjustments)
The first stage performs pixel-wise segmentation of sidewalk regions
from monocular RGB input. This module is critical for all downstream
steps and must balance segmentation quality with real-time efficiency
on embedded systems.
...
```

### Claim-Evidence-Conclusion Example: Claim 2 (D-05, D-06)

```latex
\section{Image-Space Planning for Straight-Ahead Following}
\label{sec:claim_planning_domain}

% STATE claim
This section examines whether image-space path planning offers
advantages over BEV-domain planning for straight-ahead sidewalk
following. If supported, this result would suggest that the computational
overhead of BEV projection and BEV-domain path extraction can be avoided
for the most common navigation scenario (following a corridor without
turning), reserving BEV processing for situations that require
metric-scale reasoning, such as corridor validation and turn planning.

% PRESENT evidence
Table~\ref{tab:planner_comparison} compares five planning methods on
32~hand-annotated frames...

[Existing content from planner comparison, reorganized to argue the claim]

To isolate the effect of the planning domain from segmentation quality,
Table~\ref{tab:oracle_comparison} presents the same comparison with
oracle (ground-truth) masks...

% Iteration progression context (NEW per D-07)
Table~\ref{tab:iteration_progression} places these results in the
context of the system's design evolution...

% CONCLUDE
The evidence presented in this section supports the claim that
image-space planning offers substantial advantages over BEV-domain
planning for straight-ahead sidewalk following. The image-space
midpoint planner achieves approximately $4.5\times$ lower lateral
error at approximately $421\times$ lower latency than the best
BEV-domain method, and this advantage persists even with oracle masks.
These results do not imply that BEV processing should be abandoned;
as discussed in Section~\ref{sec:claim_bev_fragility}, BEV-domain
reasoning remains valuable for corridor extraction and turn planning.
Rather, they suggest that the default planning domain for straight-ahead
following should be image-space.
```

### Hedging Vocabulary (continued from Phase 2)

```latex
% The same hedging vocabulary from Phase 2 applies:
% Hedging verbs: suggest, indicate, appear to, tend to
% Hedging adverbs: approximately, generally, typically
% Hedging qualifiers: in certain conditions, under tested scenarios
% Distance phrases: "The data suggest...", "Based on the evaluation..."

% ADDITIONAL for claim-driven sections:
% Claim introduction: "This section examines the claim that..."
% Evidence bridge: "The evidence presented in this section..."
% Qualified conclusion: "supports / partially supports the claim"
% Scope limiters: "for straight-ahead following", "under the tested
%   configurations", "on the campus video sequences examined"
```

## Task Decomposition Recommendation

### Plan 01: System Design Rewrite (NARR-03)
**Scope:** Chapter 3 (lines ~390-735)
**Requirements:** NARR-03
**Key work:**
- Add WHY-then-WHAT rationale paragraphs to each major section (D-01, D-02)
- Rewrite chapter opening with design philosophy and iteration narrative
- Preserve all equations, figures, table references, and labels
- Apply thesis formal tone throughout
- Estimated net growth: +30-50 lines (rationale paragraphs added, existing content trimmed for tone)

**Task suggestions:**
- Task 1: Rewrite chapter opening, Hardware Platform, and Segmentation sections (lines 390-470) -- the heaviest rationale work is here (SegFormer choice, teacher-student choice)
- Task 2: Rewrite BEV Projection through Path Planning Methods sections (lines 506-605) -- BEV rationale and planner rationale per D-02, iteration narrative woven into planners overview

### Plan 02: Evaluation Restructure (NARR-04, STRUCT-02, EVAL-01 through EVAL-04)
**Scope:** Chapter 4 (lines ~738-1209)
**Requirements:** NARR-04, STRUCT-02, EVAL-01, EVAL-02, EVAL-03, EVAL-04
**Key work:**
- Restructure from flat results listing to 5 claim-driven sections (D-05)
- Remove checkpoint benchmark (D-03), add teacher-student narrative (D-04)
- Add iteration progression table (D-07)
- Merge runtime tables (D-08)
- Add naive baseline discussion (D-09)
- Apply claim-evidence-conclusion pattern throughout (D-06)
- Estimated net change: approximately +20-30 lines (new content minus removed checkpoint benchmark)

**Task suggestions:**
- Task 1: Restructure chapter opening + Experimental Setup (add naive baseline subsection), rewrite Claim 1 (segmentation quality -- moves current segmentation results + qualitative results, adds teacher-student narrative replacing checkpoint benchmark)
- Task 2: Rewrite Claims 2-5 (planning domain, BEV fragility, template-approval, system validation), add iteration progression table, merge runtime tables, verify all cross-references

### Why 2 Plans, Not 3 or 4

Phase 2 used 2 plans (one per chapter) and completed each in 4-5 minutes. Phase 3 has larger chapters but similar structure: each plan touches one chapter in thesis/main.tex. Splitting further (e.g., 1 plan per claim) would create excessive inter-plan dependencies since the claims reference each other and share the chapter opening. Two plans with 2 tasks each provides enough granularity for checkpoint/verification while keeping dependencies clean.

## Label Preservation Checklist

The following labels are referenced from OTHER chapters and MUST be preserved (or redirected) during restructuring:

**From Introduction (ch:introduction):**
- `\ref{ch:system_design}` -- Chapter 3 label
- `\ref{ch:evaluation}` -- Chapter 4 label
- `\ref{sec:teacher_student}` -- referenced in contribution #3
- `\ref{sec:bev_fragility_results}` -- referenced in contribution #1
- `\ref{sec:template_planner}` -- referenced in contribution #2

**From Discussion (ch:discussion):**
- `\ref{tab:oracle_comparison}` -- oracle mask experiment
- `\ref{tab:planner_comparison}` -- planner comparison
- `\ref{tab:template_eval}` -- template evaluation

**From Conclusion (ch:conclusion):**
- No specific section references, but mentions "Chapter~\ref{ch:evaluation}" generally

**Internal to System Design (must stay consistent):**
- `\ref{sec:bev_fragility_results}` referenced from sec:bev (line 515)
- `\ref{sec:res_seg}` referenced from sec:segmentation (line 467)

**Labels that can be added (new claim sections):**
- `\label{sec:claim_seg_quality}` -- Claim 1
- `\label{sec:claim_planning_domain}` -- Claim 2
- `\label{sec:claim_bev_fragility}` -- Claim 3 (MUST also keep `sec:bev_fragility_results` as alias or redirect)
- `\label{sec:claim_template_approval}` -- Claim 4
- `\label{sec:claim_system_validation}` -- Claim 5

**Critical:** `sec:bev_fragility_results` is referenced from System Design (line 515) and potentially from Introduction/Discussion. When restructuring Evaluation, this label MUST be preserved on the BEV fragility section, even if the section is renamed. Use both labels:
```latex
\section{BEV Coverage Fragility}
\label{sec:claim_bev_fragility}
\label{sec:bev_fragility_results}
```

## Open Questions

1. **Exact FPS for v2 iteration in progression table**
   - What we know: BEV DT Ridge planner alone costs 926.8ms (tab:planner_comparison). The full BEV skeleton pipeline is 416ms/2.4 FPS (tab:runtime_comparison). BEV DT Ridge is slower than skeleton.
   - What's unclear: No existing table shows the full pipeline FPS with BEV DT Ridge as the primary planner. The 926.8ms is planner-only; adding SegFormer + refinement + BEV warp gets approximately 950ms/frame or ~1.1 FPS.
   - Recommendation: Use "~1" FPS for v2 with a footnote explaining this is estimated from component timings. Or use the planner-only time (926.8ms) in the table and note that pipeline overhead adds further.

2. **How much System Design prose can be preserved?**
   - What we know: The current content is technically accurate. The main issue is lack of WHY rationale, not incorrect WHAT descriptions.
   - What's unclear: Whether the tone of the existing WHAT sections already meets the D-10 formal tone requirement, or if significant rewriting is needed beyond adding WHY paragraphs.
   - Recommendation: Default to "preserve existing technical content, add rationale paragraphs, apply light tone editing." Full rewrites only where the existing prose is clearly lab-notes style.

3. **Whether to keep training_progression table or merge into teacher-student narrative**
   - What we know: tab:training_progression (included via \input{tables/generated/training_progression.tex}) shows the 3-iteration teacher progression. D-04 asks for 2-3 paragraphs of narrative replacing the checkpoint benchmark.
   - What's unclear: Whether tab:training_progression stays as-is alongside the new narrative, or is superseded by the new iteration progression table (D-07).
   - Recommendation: Keep tab:training_progression in System Design (it documents the training details). The new iteration progression table (D-07) in Evaluation covers the system-level design evolution, which is a different dimension. No conflict.

## Sources

### Primary (HIGH confidence)
- `thesis/main.tex` lines 390-1209 -- current Chapter 3 and Chapter 4 content, read in full
- `.planning/phases/03-methodology-results-rewrite/03-CONTEXT.md` -- all locked decisions D-01 through D-11
- `.planning/phases/02-introduction-literature-review/02-CONTEXT.md` -- tone decisions D-09, D-10, BEV nuance D-05
- `test.md` -- v1 pipeline paper, source for iteration 1 data (9.1 FPS, skeleton-graph)
- `simulation_camera_scooter/RUNTIME_RUNBOOK.md` -- accepted run configuration (1800 frames, 9.97 FPS)
- `simulation_camera_scooter/MUST_READ_TURN_CONTAINMENT.md` -- turn planner validation data

### Secondary (HIGH confidence)
- `.planning/phases/02-introduction-literature-review/02-RESEARCH.md` -- established patterns for thesis rewriting in this project (hedging vocabulary, anti-patterns, task decomposition)
- `.planning/phases/02-introduction-literature-review/02-01-PLAN.md` -- plan format pattern
- `.planning/REQUIREMENTS.md` -- requirement definitions for NARR-03, NARR-04, EVAL-01-04, STRUCT-02
- `.planning/PROJECT.md` -- project context and constraints
- `thesis/tables/generated/training_progression.tex` -- training iteration data

### Tertiary (not applicable)
No web searches were needed for this phase. All research is grounded in existing project artifacts. This is a thesis writing task, not a technology research task.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- this is a LaTeX-only thesis writing task with no library dependencies
- Architecture: HIGH -- target structure derived directly from locked user decisions (D-01 through D-11) and existing thesis content
- Pitfalls: HIGH -- identified from direct examination of current thesis content, Phase 2 experience, and locked decision constraints

**Research date:** 2026-03-30
**Valid until:** 2026-04-30 (stable -- thesis content does not change externally)
