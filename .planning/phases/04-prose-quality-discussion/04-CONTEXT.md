# Phase 4: Prose Quality & Discussion - Context

**Gathered:** 2026-03-30
**Status:** Ready for planning

<domain>
## Phase Boundary

Full prose polish across all 6 chapters. Strengthen Discussion (Ch.5) and Conclusion (Ch.6) to publication quality. Rewrite Abstract. Standardize terminology. Make all figure/table captions self-contained.

</domain>

<decisions>
## Implementation Decisions

### Prose Polish Scope (WRIT-01)
- **D-01:** Full pass on all 6 chapters. Even though Phases 2-3 rewrote Ch.1-4 with formal tone, do a complete review pass for remaining passive voice, lab-notes style, and inconsistencies. Discussion (Ch.5) and Conclusion (Ch.6) need the most work since they haven't been rewritten yet.
- **D-02:** Traditional thesis formal tone (carried from Phase 2 D-09/D-10). Measured, hedged language. No assertive constructions.

### Discussion Depth (DISC-01, DISC-02)
- **D-03:** Add failure analysis section (DISC-02): when/why image-space planning fails — sharp turns, very narrow sidewalks, heavy occlusion, perspective distortion at far range. This is an honest assessment, not a weakness to hide.
- **D-04:** Add broader implications section (DISC-01): connect findings to the wider monocular perception field. The geometric coverage argument (why image-space works) applies beyond sidewalk navigation to any monocular corridor-following task.
- **D-05:** Keep existing Discussion sections (Interpretation, Why Image-Space Wins, Segmentation Sufficiency, Template Planning, Limitations, Threats to Validity) but strengthen them. Add the two new sections (failure analysis, broader implications).

### Conclusion Rewrite (DISC-03)
- **D-06:** Rewrite Conclusion to map each key finding back to the numbered contributions from the Introduction. The 5 contributions established in Phase 2 must each have a corresponding conclusion statement.

### Abstract Rewrite (WRIT-04)
- **D-07:** Thesis-comprehensive abstract, ~250-300 words. Structure: problem, approach (modular pipeline, 4 iterations), two key findings with numbers (421x speedup, 14.3px vs 65.0px, 99.3% BEV failure, template-approval), system validation (1800 frames, 9.97 FPS), and one-sentence future direction.

### Figure/Table Captions (WRIT-02)
- **D-08:** All figure and table captions must be self-contained — understandable without reading the body text. Each caption should state what is shown, key takeaway, and conditions (e.g., "32 hand-annotated frames, candidate mask").

### Terminology Standardization (WRIT-03)
- **D-09:** Define canonical terms once in System Design, enforce throughout all chapters. Key terms to standardize:
  - "segmentation mask" (not "road mask", "sidewalk mask" interchangeably)
  - "template-approval planner" (not "template arc planner", "template bank")
  - "image-space midpoint planner" (consistent name)
  - "BEV corridor" (not "BEV mask corridor", "drivable corridor")
  - "design iteration" (not "version", "stage", "phase" to avoid confusion with thesis phases)
  - "teacher-student framework" (not "knowledge distillation" unless in lit review context)

### Claude's Discretion
- Exact wording of failure analysis scenarios
- How to organize broader implications (separate section vs. subsection of Interpretation)
- Whether Abstract should mention all 4 design iterations or just summarize the evolution
- Degree of caption rewriting needed (some may already be adequate)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Thesis Source
- `thesis/main.tex` — Full thesis. All 6 chapters need review. Discussion at Ch.5 (lines ~1258-1317), Conclusion at Ch.6 (lines ~1320-end). Abstract near top of file.

### Prior Phase Contexts
- `.planning/phases/02-introduction-literature-review/02-CONTEXT.md` — Tone decisions (D-09, D-10), contribution framing (D-03, D-04)
- `.planning/phases/03-methodology-results-rewrite/03-CONTEXT.md` — Claim structure, BEV nuance (D-11)

### Planning Documents
- `.planning/REQUIREMENTS.md` — WRIT-01 through WRIT-04, DISC-01 through DISC-03

</canonical_refs>

<code_context>
## Existing Code Insights

### Current Discussion Structure (Ch.5, lines ~1258-1317)
- Interpretation of Key Findings (good, needs tone check)
- Why Image-Space Outperforms BEV (good geometric coverage argument)
- Segmentation as Necessary but Not Sufficient (good)
- Template Planning and Turn Safety (good)
- Limitations (6 paragraphs — comprehensive)
- Threats to Validity (3 paragraphs — adequate)
- MISSING: Failure analysis (DISC-02)
- MISSING: Broader implications / connection to field (DISC-01)

### Current Conclusion Structure (Ch.6, lines ~1320-end)
- Summary of Contributions
- Key Findings
- Future Work
- Needs mapping back to numbered contributions from Introduction

### Abstract Location
- Near top of file, before chapter 1

</code_context>

<specifics>
## Specific Ideas

- The failure analysis should be honest: image-space midpoint fails on sharp turns (that's why the template planner exists), on very narrow sidewalks where boundary detection is unreliable, and when occlusion removes most of the mask.
- Broader implications: the geometric coverage insight (perspective images have dense pixel coverage, BEV has sparse coverage) applies to any monocular navigation task, not just sidewalks.
- The terminology pass should produce a "canonical term → variants to replace" mapping that the executor can use systematically.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 04-prose-quality-discussion*
*Context gathered: 2026-03-30*
