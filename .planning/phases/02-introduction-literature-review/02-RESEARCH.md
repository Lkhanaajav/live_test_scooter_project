# Phase 2: Introduction & Literature Review - Research

**Researched:** 2026-03-30
**Domain:** Academic thesis writing — Introduction chapter and Literature Review chapter rewrite
**Confidence:** HIGH

## Summary

This phase rewrites two chapters of a Master's thesis: Chapter 1 (Introduction) and Chapter 2 (Background and Related Work). The current Introduction is ~90 lines of LaTeX with a generic motivation section, a list-style problem statement, a serviceable but unfocused approach overview, a 7-item contribution list that dilutes the core findings, and a standard thesis organization paragraph. The current Literature Review is ~80 lines covering 8 thematic sections plus a summary research gap section, but gap statements are concentrated at the end rather than distributed per-section.

The rewrite must accomplish two things: (1) transform the Introduction from a descriptive overview into a narrative that sets up two key findings — image-space dominance for straight-ahead planning and template-approval replacing fragile skeleton-graph planning — with a concrete scenario hook and consolidated contributions; (2) rewrite each of the 8 literature review sections to end with a specific gap statement pointing toward this thesis's contributions, using synthesis rather than survey style.

**Primary recommendation:** Write each chapter as a self-contained narrative document. The Introduction should follow a funnel structure (concrete scenario → general problem → specific gap → approach → contributions). Each lit review section should follow a context-gap-response pattern ending with an explicit bridge to this thesis.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Open with a concrete scenario — a delivery robot or scooter on a cracked sidewalk with no lane markings, no LiDAR, just a cheap camera. Ground the reader in the real-world problem before stating the research question.
- **D-02:** The opening must establish both the practical constraint (embedded hardware, monocular camera) and the intellectual question (do you really need BEV and graph-based planning for sidewalk navigation?).
- **D-03:** Frame contributions around two key findings:
  - **Finding 1 (Benchmarking):** Image-space midpoint planning dominates BEV methods for straight-ahead following — 421x faster, 4.5x lower lateral error. BEV is fragile (99.3% frame failure in one sequence).
  - **Finding 2 (System Design):** Template-approval on BEV corridors replaces fragile skeleton-graph planning. The evolution from expensive discovery (graph construction from noisy pixels) to cheap verification (score 5 pre-drawn arcs against DT corridor) is the engineering contribution.
- **D-04:** Reduce the current 7-item contribution list to focus on these two findings plus supporting contributions (teacher-student training, comprehensive evaluation protocol, GPS-conditioned turn planner).
- **D-05:** The real system uses BEV for corridor extraction and turns — do NOT oversimplify to "BEV bad, image-space good." The argument is about *how* you use BEV (verify templates, not build graphs).
- **D-06:** Keep the current 8-section thematic structure (Micro-Mobility, Segmentation, BEV, Path Planning, Distance Transform, Skeletonization, Teacher-Student, Embedded Perception).
- **D-07:** Rewrite each section to end with a specific gap statement that this thesis fills. Currently the gaps are only in the final "Summary and Research Gap" section — distribute them.
- **D-08:** The final "Research Gaps" section must connect all 4 identified gaps to the two main findings from D-03.
- **D-09:** Traditional thesis formal tone — measured, hedged language. "Results suggest..." rather than "We demonstrate...". Passive voice acceptable. Err on the side of being conservative for committee review.
- **D-10:** Do NOT match the more assertive conference-paper style. This is a thesis for a conservative committee at the University of Oklahoma.

### Claude's Discretion
- Exact wording and paragraph structure within each section
- How to distribute the 4 research gaps across the 8 lit review sections
- Whether to add a "Thesis Organization" subsection at the end of the Introduction (currently exists, may need updating)
- How many contributions to list (consolidate from 7 down to 4-5, Claude decides exact grouping)

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| NARR-01 | Rewrite Introduction with compelling opening hook, sharper problem statement, and stronger contribution framing | Concrete scenario hook (D-01), two-finding structure (D-03), consolidated 4-5 contributions (D-04), hedged formal tone (D-09), funnel structure pattern (Architecture Patterns below) |
| NARR-02 | Rewrite Literature Review as themed synthesis — each section ends with the gap this thesis fills | Keep 8 sections (D-06), distribute gap statements per-section (D-07), connect all 4 gaps to two findings in final section (D-08), synthesis-not-survey style (Architecture Patterns below) |
</phase_requirements>

## Project Constraints (from CLAUDE.md)

- **LaTeX only** — all edits are to `thesis/main.tex`; no new files created for content
- **No new experiments or data** — all numbers cited must come from existing evaluation data
- **No new figures** — reuse existing figures; minor relabeling acceptable
- **60-80 pages** double-spaced target for entire thesis; Introduction should be 6-8 pages, Literature Review 10-14 pages per ARCHITECTURE.md
- **OU format** — 12pt, double-spaced, Times New Roman (mathptmx), 1.5" left margin, centered page numbers
- **Existing bibliography** — `references.bib` has 40+ entries; new citations require adding BibTeX entries
- **Existing cross-references** — must preserve `\label{ch:introduction}`, `\label{ch:background}`, and all `\label{sec:lit_*}` identifiers
- **Existing figures** — `fig:scooter_hw` figure block (lines 266-283) must be preserved at end of Introduction
- **Run tests before commit** — `pytest tests/ -v` in `simulation_camera_scooter/`

## Architecture Patterns

### Chapter 1: Introduction — Target Structure

The Introduction must grow from ~90 lines to approximately 130-180 lines (6-8 double-spaced pages) following this structure:

```
\chapter{Introduction}
\label{ch:introduction}

\section{Motivation}
  - Para 1: CONCRETE SCENARIO HOOK (D-01)
    Delivery robot or scooter on cracked sidewalk, cheap camera only.
    Establish embedded constraint + intellectual question (D-02).
  - Para 2: BROADER CONTEXT
    Why sidewalk navigation matters (economic, social, accessibility).
    Why monocular is the right constraint (cost, power, form factor).
  - Para 3: LIMITATIONS OF EXISTING APPROACHES
    End-to-end = black box + GPU. Learned BEV = multi-camera + heavy.
    Even classical BEV skeleton = fragile with monocular input.

\section{Problem Statement}
  - Para 1: Formal problem specification
    Single camera → traversable region → centered path → safe control.
    Two open questions framing the thesis:
      Q1: Which planning domain (BEV or image-space)?
      Q2: How to make BEV-based planning robust when you need it (turns)?
  - Para 2: Scope and constraints
    Offline evaluation, CPU-only timing, campus sidewalk video.

\section{Approach Overview}
  - Para 1: Pipeline at a glance
    Segmentation → optional BEV → path extraction → temporal smoothing → control.
  - Para 2: Design iteration narrative (4 iterations, one sentence each)
    Key story: from discovery (skeleton graph) to verification (template approval).
  - Para 3: Positioning vs. related work (brief)

\section{Contributions}
  - 4-5 numbered items (consolidated from current 7):
    1. Systematic BEV-vs-image-space comparison (Finding 1)
    2. Template-approval architecture replacing skeleton-graph (Finding 2)
    3. Teacher-student segmentation training (supporting)
    4. Comprehensive offline evaluation protocol (supporting)
    5. GPS-conditioned turn planner with safety guard (supporting, optional merge with #2)

\section{Thesis Organization}
  - Standard roadmap paragraph (update to reflect new chapter labels)

[Figure: fig:scooter_hw — keep at end of chapter]
```

### Chapter 2: Literature Review — Target Structure

The Literature Review must grow from ~80 lines to approximately 200-280 lines (10-14 double-spaced pages) with this pattern per section:

```
\chapter{Background and Related Work}
\label{ch:background}

[Opening paragraph: what this chapter covers and why]

\section{Autonomous Navigation for Micro-Mobility Platforms}
  - Survey of sidewalk/delivery robot navigation
  - Sensor budget distinction from road driving
  - GAP: No prior monocular-only sidewalk navigation system benchmarked
    end-to-end on embedded hardware → connects to Finding 1 (BEV vs image-space)

\section{Semantic Segmentation for Drivable Surfaces}
\label{sec:lit_segmentation}
  - Evolution: FCN → DeepLab → SegFormer → lightweight variants
  - Sidewalk-specific dataset scarcity
  - GAP: Teacher-student training with OneFormer Swin-L teacher + SegFormer-B0
    student has not been applied to sidewalk segmentation → connects to contribution #3

\section{Bird's-Eye View Projection and Monocular BEV}
\label{sec:lit_bev}
  - Classical IPM vs. learned BEV (LSS, BEVFormer, Focus on BEV)
  - GPU/multi-camera assumption of learned approaches
  - GAP: Coverage fragility of monocular BEV has not been quantitatively
    characterized → connects to Finding 1 (BEV fragility)

\section{Path Planning for Mobile Robots}
\label{sec:lit_planning}
  - Grid-based, sampling, potential field, end-to-end
  - Agricultural row-following
  - GAP: No prior systematic comparison of BEV-domain vs. image-space path
    planning for monocular sidewalk navigation → connects to Finding 1

\section{Distance Transform Methods in Navigation}
  - EDT clearance maps, ridge tracing, Dijkstra on cost fields
  - Use in safe corridor extraction
  - GAP: DT corridors used for discovery (extract path from mask) but not
    for verification (score pre-computed templates against corridor) → connects to Finding 2

\section{Skeletonization and Topological Path Extraction}
  - Classical thinning (Zhang-Suen, Guo-Hall), learned skeletonization
  - Noise sensitivity, spurious branches
  - GAP: Skeleton-graph construction is expensive and noise-sensitive; no prior
    work has systematically compared it against simpler template-based
    alternatives on real sidewalk data → connects to Finding 2

\section{Semi-Supervised and Teacher--Student Learning}
\label{sec:lit_teacher_student}
  - Knowledge distillation overview
  - OneFormer as universal segmentation model
  - GAP: OneFormer Swin-L → SegFormer-B0 distillation not previously applied
    to sidewalk binary segmentation → connects to contribution #3

\section{Embedded and Real-Time Perception}
\label{sec:lit_embedded}
  - Hardware constraints (RPi 4, Rock 5B)
  - ONNX Runtime, TensorRT
  - Latency requirements for pedestrian-speed platforms
  - GAP: Few complete perception-to-planning pipelines benchmarked end-to-end
    on CPU-only hardware for sidewalk applications → connects to Finding 1

\section{Summary and Research Gap}
  - Synthesize all 4 gaps into thesis rationale
  - Connect each gap to the two main findings (D-08)
  - Bridge sentence to Chapter 3 (System Design)
```

### Gap Distribution Map

| Gap | Primary Section | Secondary Section | Connected Finding |
|-----|----------------|-------------------|-------------------|
| Gap 1: No BEV-vs-image-space comparison for sidewalk planning | Path Planning (sec 4) | BEV Projection (sec 3) | Finding 1 |
| Gap 2: Monocular BEV fragility uncharacterized | BEV Projection (sec 3) | Micro-Mobility (sec 1) | Finding 1 |
| Gap 3: OneFormer→SegFormer not applied to sidewalk segmentation | Teacher-Student (sec 7) | Segmentation (sec 2) | Contribution #3 |
| Gap 4: Few end-to-end CPU-only sidewalk pipelines | Embedded Perception (sec 8) | Micro-Mobility (sec 1) | Finding 1 |

### Per-Section Gap Statement Pattern

Each literature review section should end with a gap statement following the **context-gap-response** pattern:

```latex
% BAD (current style — gap only in final section):
Prior work has focused on X and Y.

% GOOD (gap distributed per-section):
While [summary of what prior work achieves], [specific limitation or gap].
This thesis addresses this gap by [brief response], as detailed in
Section~\ref{sec:specific_section}.
```

### Anti-Patterns to Avoid

- **Survey-style listing:** "Author A did X. Author B did Y. Author C did Z." Instead, synthesize: "While approaches X and Y share assumption Z, they diverge on..."
- **BEV = bad oversimplification (D-05):** The thesis uses BEV for corridor extraction and turns. The argument is that BEV should be used for verification (scoring templates) not discovery (building graphs from noisy pixels).
- **Assertive tone (D-09, D-10):** Avoid "We demonstrate..." and "Our approach outperforms...". Use "Results suggest...", "The experiments indicate...", "This comparison reveals..."
- **Stacking hedges:** "It might possibly suggest..." is excessive. One hedge per claim suffices.
- **Orphaned gap statements:** A gap statement without a forward reference to the specific section/chapter that addresses it is incomplete.
- **Losing cross-references:** All existing `\label{}` identifiers must be preserved. New sections can add labels but must not rename existing ones.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Page count estimation | Manual line counting | Compile LaTeX and check PDF page count | Double-spacing, figures, and tables make line count unreliable |
| Citation formatting | Manual BibTeX entries | Existing `references.bib` entries with `\cite{}` | 40+ entries already defined; reuse citation keys |
| Hedging language | Ad-hoc phrasing per sentence | Systematic hedging vocabulary (see Code Examples below) | Consistency across two chapters requires a reference vocabulary |
| Gap statements | Writing them from scratch per section | Derive from the 4 defined gaps and map to sections per Gap Distribution Map above | Ensures coherence between individual gaps and final synthesis |

## Common Pitfalls

### Pitfall 1: Introduction Hook Too Generic
**What goes wrong:** Opening with "Autonomous driving is a growing field..." — a funnel so wide it loses the reader before reaching the thesis-specific content.
**Why it happens:** Defaulting to textbook-style broad-to-narrow without a concrete anchor.
**How to avoid:** Start with a specific physical scenario (D-01): a scooter on a cracked sidewalk, cheap camera, no lane markings. Ground the reader in the tangible before abstracting.
**Warning signs:** First sentence contains words like "growing field", "recent advances", or "increasing interest."

### Pitfall 2: Contribution List Inflation
**What goes wrong:** Listing 7+ contributions that overlap or include implementation details masquerading as research contributions.
**Why it happens:** Confusing system engineering work (serial protocol, obstacle integration) with novel research contributions.
**How to avoid:** D-04 mandates consolidation to 4-5 items. Each contribution must be independently verifiable and map to a specific evaluation in Chapter 4.
**Warning signs:** A contribution that starts with "Integration of..." is likely an implementation detail.

### Pitfall 3: BEV Oversimplification
**What goes wrong:** Framing the thesis as "BEV is bad, image-space is good" when the real system uses BEV for corridor extraction and turns.
**Why it happens:** Simplifying the narrative for rhetorical clarity at the expense of accuracy.
**How to avoid:** D-05 is explicit: the argument is about *how* you use BEV (verify templates vs. build graphs), not whether to use BEV at all. The nuanced claim is: for straight-ahead path following, image-space dominates; for turns and corridor validation, BEV remains essential.
**Warning signs:** Any absolute statement about BEV without qualification.

### Pitfall 4: Literature Review as Annotated Bibliography
**What goes wrong:** Each paragraph summarizes one paper without connecting to adjacent work or identifying patterns.
**Why it happens:** Easier to write "Author A did X" than to synthesize "Approaches X and Y share assumption Z."
**How to avoid:** Each paragraph should reference multiple works and identify a pattern, trend, or tension. End each section with a gap statement.
**Warning signs:** Paragraphs starting with author names or "In [year], [Author]..."

### Pitfall 5: Gap Statements Disconnected from Contributions
**What goes wrong:** The gap statements in the literature review identify problems that the thesis does not actually solve, or the final synthesis section introduces gaps not foreshadowed per-section.
**Why it happens:** Gap statements written after the literature review instead of planned in advance.
**How to avoid:** Use the Gap Distribution Map above. Write gap statements first, then build each section's narrative toward that gap.
**Warning signs:** A gap statement that cannot be cross-referenced to a specific contribution number.

### Pitfall 6: Tone Inconsistency with Paper
**What goes wrong:** Matching the assertive conference-paper style from `test.md` instead of the measured thesis tone required by D-09/D-10.
**Why it happens:** The conference paper (`test.md`) is a natural source for prose, and its confident, direct style leaks into the thesis.
**How to avoid:** Explicit tone check: search for "we demonstrate", "we show", "outperforms" and replace with hedged equivalents. Reference the hedging vocabulary below.
**Warning signs:** First-person active voice dominant throughout ("We propose...", "We demonstrate...").

### Pitfall 7: Breaking Existing Cross-References
**What goes wrong:** Renaming or removing `\label{}` identifiers breaks references from later chapters (System Design, Evaluation, Discussion, Conclusion all reference Introduction and Background sections).
**Why it happens:** Restructuring sections during rewrite without checking downstream references.
**How to avoid:** Preserve all existing labels. Search for `\ref{ch:introduction}`, `\ref{ch:background}`, `\ref{sec:lit_*}`, `\ref{sec:bev_fragility_results}` in the full document before removing any label.
**Warning signs:** LaTeX compilation warnings about undefined references.

## Code Examples

### Hedging Vocabulary for Thesis Tone (D-09, D-10)

```latex
% AVOID (assertive, conference-paper style):
We demonstrate that image-space planning outperforms BEV planning.
Our approach achieves a 421x speedup.
We show that BEV is fragile.

% PREFER (measured, hedged, thesis style):
The experiments suggest that image-space planning offers advantages
over BEV-domain planning for straight-ahead sidewalk following.
The observed speedup of approximately 421x indicates substantial
computational savings.
The results indicate that monocular BEV projection may introduce
coverage fragility in certain conditions.

% Hedging verbs: suggest, indicate, appear to, tend to, seem to
% Hedging adverbs: approximately, generally, typically, relatively
% Hedging qualifiers: in certain conditions, under the tested scenarios,
%                     for the configurations examined
% Distance phrases: "The data suggest...", "Based on the evaluation...",
%                   "The comparison reveals..."
```

### Concrete Scenario Hook Pattern (D-01)

```latex
% Pattern: Physical scenario → constraint → question
Consider a battery-powered delivery scooter navigating a university
sidewalk. The path ahead is cracked asphalt bordered by grass on
one side and a brick retaining wall on the other; there are no lane
markings, no curb paint, and no high-definition map of the route.
The vehicle's only forward-looking sensor is a commodity RGB camera,
and its only computer is a single-board ARM processor drawing under
fifteen watts. In this setting, a natural question arises: how
should such a platform perceive the traversable corridor ahead and
plan a safe trajectory through it?
```

### Gap Statement Pattern (D-07)

```latex
% Pattern: context-gap-response with forward reference
While prior work in [topic] has advanced [specific capability],
these approaches typically assume [assumption that does not hold
in our setting]. To the authors' knowledge, no prior study has
[specific gap]. This thesis addresses this gap through [brief
description], with results presented in Section~\ref{sec:relevant}.
```

### Contribution Item Pattern (D-03, D-04)

```latex
% Pattern: Specific, verifiable, connected to evaluation
\item A systematic comparison of five path-planning methods across
  BEV and image-space domains, providing evidence that image-space
  midpoint planning achieves substantially lower lateral error
  ($\sim$14~px vs.\ $\sim$65~px) and latency ($\sim$2~ms vs.\
  $\sim$927~ms) than the best BEV method on the tested sidewalk
  sequences (Chapter~\ref{ch:evaluation},
  Section~\ref{sec:planner_comparison}).
```

### Section Transition Pattern

```latex
% End of Introduction, bridge to Background:
The remainder of this thesis is organized as follows.
Chapter~\ref{ch:background} reviews prior work in [topics] and
identifies the specific research gaps that motivate this work.

% End of Background, bridge to System Design:
The gaps identified in this chapter---[brief list]---motivate the
system design described in Chapter~\ref{ch:system_design}.
```

## Key Data Points for Introduction

These are the verified numbers from the existing evaluation (Chapter 4) that the Introduction must reference accurately:

| Metric | Value | Source |
|--------|-------|--------|
| Image-space midpoint lateral error | 14.3 px | Table 3 (planner comparison) |
| BEV DT lateral error | 65.0 px | Table 3 |
| Image-space midpoint latency | 2.2 ms | Table 3 |
| BEV DT latency | 926.8 ms | Table 3 |
| Speedup ratio | ~421x | 926.8 / 2.2 |
| Lateral error ratio | ~4.5x | 65.0 / 14.3 |
| BEV frame failure rate | 99.3% | BEV fragility analysis |
| Template planner: frames processed | 1800 | Accepted run |
| Template planner: path source | 100% template | Accepted run |
| Template planner: mean FPS | 9.97 | Accepted run |
| Template planner: mean IoU | 0.954 | Accepted run |
| Teacher-student IoU | 0.946 | Segmentation evaluation |
| Teacher-student latency | 11.7 ms | Segmentation evaluation |
| Template bank size | 5 arcs | System design |
| System evolution | skeleton (109ms/9.1 FPS) → template (69ms/10 FPS) | Runtime comparison |
| Design iterations | 4 | Fixed in Phase 1 |

## Existing Citation Keys Available

The following BibTeX keys in `references.bib` are relevant to Chapters 1-2 and can be used directly with `\cite{}`:

| Key | Paper/Source | Relevant Section |
|-----|-------------|-----------------|
| `segformer` | Xie et al. SegFormer | Segmentation, Teacher-Student |
| `oneformer` | Jain et al. OneFormer | Teacher-Student |
| `fcn2015` | Long et al. FCN | Segmentation history |
| `deeplabv3plus` | Chen et al. DeepLabV3+ | Segmentation history |
| `bisenetv2` | Yu et al. BiSeNetV2 | Lightweight segmentation |
| `ddrnet` | Hong et al. DDRNet | Lightweight segmentation |
| `edge_optimized_seg` | MobileNetV3 | Lightweight segmentation |
| `twinlitenet2024` | TwinLiteNet | Lightweight segmentation |
| `cityscapes` | Cordts et al. | Datasets |
| `rugd` | Wigness et al. RUGD | Datasets |
| `mapillary` | Neuhold et al. Mapillary | Datasets |
| `starship2022` | Starship Technologies | Micro-mobility |
| `viteri2024` | Viteri et al. RGB-D scooter | Micro-mobility |
| `machkour2023` | Machkour et al. | Micro-mobility |
| `zhu2017target` | Zhu et al. target-driven nav | Navigation |
| `e2e_nav_policies` | End-to-end policies | Planning |
| `bojarski2016` | Bojarski et al. NVIDIA | Planning |
| `hartley_zisserman` | Hartley & Zisserman MVG | BEV/Homography |
| `lift_splat_shoot` | Philion & Fidler LSS | Learned BEV |
| `bevformer` | Li et al. BEVFormer | Learned BEV |
| `zhao2024bev` | Zhao et al. Focus on BEV | Learned BEV |
| `lavalle2006` | LaValle Planning Algorithms | Path planning |
| `khatib1986` | Khatib potential fields | Path planning |
| `orchard_seg_nav` | Orchard navigation | Row following |
| `row_nav_survey` | Row navigation survey | Row following |
| `felzenszwalb_dt` | Felzenszwalb & Huttenlocher | Distance transform |
| `zhang_suen` | Zhang & Suen | Skeletonization |
| `guo_hall` | Guo & Hall | Skeletonization |
| `flores2025skeleton` | Flores et al. learned skeleton | Skeletonization |
| `hinton2015distilling` | Hinton et al. KD | Teacher-student |
| `pseudolabel_lee2013` | Lee pseudo-labels | Teacher-student |
| `onnxruntime` | ONNX Runtime | Embedded |
| `mental_chronometry_react` | Reaction time | Embedded |
| `pi0_lowfreq` | Low-freq control | Embedded |
| `yolov8` | Ultralytics YOLOv8 | Obstacle detection |
| `shihab2024` | Shihab et al. ensemble | Segmentation |

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| BEV skeleton-graph planner (Iteration 1) | Template arc approval on BEV corridors (Iteration 4) | Phase 11 of development | 109ms → 69ms per frame; eliminates graph construction noise |
| SegFormer-B2 teacher (Iteration 1) | OneFormer Swin-L teacher (Iteration 2+) | Mid-development | Higher-quality pseudo-labels for student training |
| BEV-only path extraction | Image-space midpoint as primary, BEV for corridor validation | Iteration 3 | 421x speedup, 4.5x lower lateral error for straight-ahead |
| 7-item contribution list (current draft) | 4-5 consolidated items focused on two findings | This phase | Clearer thesis narrative |

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | `simulation_camera_scooter/tests/` directory |
| Quick run command | `cd simulation_camera_scooter && pytest tests/ -v -x` |
| Full suite command | `cd simulation_camera_scooter && pytest tests/ -v` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| NARR-01 | LaTeX compiles without errors after Introduction rewrite | manual-only | Compile with `pdflatex thesis/main.tex` | N/A |
| NARR-01 | No broken cross-references | manual-only | Check for `??` in compiled PDF or LaTeX warnings | N/A |
| NARR-02 | LaTeX compiles without errors after Literature Review rewrite | manual-only | Compile with `pdflatex thesis/main.tex` | N/A |
| NARR-02 | Each of 8 lit review sections ends with gap statement | manual-only | Visual inspection of LaTeX source | N/A |

**Note:** This phase modifies only `thesis/main.tex` (LaTeX content). The existing Python test suite in `simulation_camera_scooter/tests/` is not directly affected. However, per CLAUDE.md, `pytest tests/ -v` should still pass after any commit to confirm no accidental breakage.

### Sampling Rate
- **Per task commit:** Visual review of modified LaTeX sections
- **Per wave merge:** Full LaTeX compilation check
- **Phase gate:** LaTeX compiles cleanly; all `\ref{}` resolve; both chapters follow prescribed structure

### Wave 0 Gaps
None — this phase is LaTeX-only content writing with no new test infrastructure needed.

## Open Questions

1. **Exact contribution count: 4 or 5?**
   - What we know: D-04 says consolidate from 7 to "focus on these two findings plus supporting contributions." The supporting contributions are teacher-student training, evaluation protocol, and GPS-conditioned turn planner.
   - What's unclear: Whether the turn planner merits its own contribution item or should be folded into the template-approval contribution.
   - Recommendation: List 5 contributions. The turn planner with safety guard is a distinct engineering contribution validated separately (overnight replay, 0% containment failure). Folding it into the template planner would lose specificity.

2. **Whether to add a "Thesis Organization" subsection**
   - What we know: One currently exists (lines 261-263). It references chapter labels that were updated in Phase 1.
   - What's unclear: Whether to keep, expand, or remove it.
   - Recommendation: Keep and update. It is standard for OU thesis format and committee members use it for navigation. Update to reference new chapter names and content.

3. **Skeletonization section gap statement focus**
   - What we know: The skeletonization section currently describes classical thinning and learned alternatives. The gap needs to bridge to Finding 2 (template-approval).
   - What's unclear: How explicitly to critique skeleton-graph as a planning paradigm within the lit review (vs. saving that for the evaluation chapter).
   - Recommendation: Frame the gap as: skeleton construction from noisy masks is well-known to be fragile, but no prior work has compared it against template-based verification alternatives on real sidewalk data. This sets up Finding 2 without pre-empting the evaluation.

## Sources

### Primary (HIGH confidence)
- `thesis/main.tex` — Current thesis source (1,323 lines), read directly
- `test.md` — Conference paper v1, read directly
- `.planning/phases/02-introduction-literature-review/02-CONTEXT.md` — User decisions, read directly
- `.planning/REQUIREMENTS.md` — Phase requirement definitions
- `.planning/research/ARCHITECTURE.md` — Recommended chapter structure with page targets
- `simulation_camera_scooter/RUNTIME_RUNBOOK.md` — Accepted run configuration and metrics
- `references.bib` — Available BibTeX entries (40+ citations)

### Secondary (MEDIUM confidence)
- [Scribbr: How to Write a Thesis Introduction](https://www.scribbr.com/dissertation/introduction-structure/) — Introduction structure guidance
- [Jenny Hill PhD: How to Write a Literature Gap](https://jennyhillphd.com/how-to-write-a-literature-gap-for-thesis-2025-guide/) — Gap statement patterns
- [Middlebury: Gap Statements](https://sites.middlebury.edu/middsciwriting/overview/organization/gap-statements/) — Context-gap-response pattern
- [PMC: Writing an Effective Literature Review: Mapping the Gap](https://pmc.ncbi.nlm.nih.gov/articles/PMC5807267/) — Systematic gap identification
- [Paperpal: Hedging in Academic Writing](https://paperpal.com/blog/academic-writing-guides/what-is-hedging-in-academic-writing) — Hedging language reference
- [University of Wisconsin: Hedging](https://wisc.pb.unizin.org/esl117/chapter/controlling-tone-through-hedging/) — Hedging strategies with examples

### Tertiary (LOW confidence)
- [OU Graduate College: Steps to Degree](https://www.ou.edu/gradcollege/forms-and-policies/steps-to-degree) — General OU thesis requirements (no specific formatting document retrieved)

## Metadata

**Confidence breakdown:**
- Architecture (chapter structure): HIGH — based on existing draft structure, ARCHITECTURE.md recommendations, and locked user decisions
- Writing patterns (tone, hedging, gap statements): HIGH — well-established academic writing conventions verified with multiple sources
- Data points (numbers for Introduction): HIGH — read directly from existing evaluation tables in main.tex
- Citation coverage: HIGH — read directly from references.bib
- Page targets: MEDIUM — based on ARCHITECTURE.md recommendation; actual page count depends on LaTeX compilation

**Research date:** 2026-03-30
**Valid until:** 2026-04-30 (stable — academic writing conventions do not change)
