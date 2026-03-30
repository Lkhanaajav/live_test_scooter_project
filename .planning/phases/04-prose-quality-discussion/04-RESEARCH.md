# Phase 4: Prose Quality & Discussion - Research

**Researched:** 2026-03-30
**Domain:** Academic thesis writing -- Prose polish, Discussion/Conclusion rewrite, Abstract rewrite, terminology standardization, caption improvement
**Confidence:** HIGH

## Summary

This phase performs a comprehensive quality pass across the entire thesis (6 chapters plus Abstract) with the heaviest work concentrated on three areas: (1) rewriting Discussion (Ch.5) and Conclusion (Ch.6), which are still in their original draft form from before the Phase 2-3 rewrites; (2) rewriting the Abstract from ~378 words down to ~250-300 words; and (3) systematically standardizing terminology and making figure/table captions self-contained. Chapters 1-4 were rewritten in Phases 2-3 with formal tone, so they need only a review pass for remaining inconsistencies, passive voice cleanup, and terminology alignment. Discussion and Conclusion need substantive new content (failure analysis section, broader implications section, contribution-mapped conclusion).

The scope is broad but shallow on Ch.1-4 (polish only) and deep on Ch.5-6 (rewrite + new sections). The Abstract rewrite is a distinct, bounded task. The terminology standardization and caption improvement are mechanical, systematic tasks that touch all chapters. This suggests a 3-plan split: (1) Discussion and Conclusion rewrite, (2) Abstract rewrite + terminology standardization + caption improvement, (3) full prose review pass across Ch.1-4.

**Primary recommendation:** Split into 3 plans. Plan 01: Discussion (Ch.5) and Conclusion (Ch.6) rewrite -- the deepest work, including two new Discussion sections and a contribution-mapped Conclusion. Plan 02: Abstract rewrite + terminology standardization pass + caption self-containment pass -- systematic, mechanical tasks across the whole document. Plan 03: Prose review pass on Ch.1-4 -- lighter touch, catching remaining tone issues after Phase 2-3 rewrites.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Full pass on all 6 chapters. Even though Phases 2-3 rewrote Ch.1-4 with formal tone, do a complete review pass for remaining passive voice, lab-notes style, and inconsistencies. Discussion (Ch.5) and Conclusion (Ch.6) need the most work since they haven't been rewritten yet.
- **D-02:** Traditional thesis formal tone (carried from Phase 2 D-09/D-10). Measured, hedged language. No assertive constructions.
- **D-03:** Add failure analysis section (DISC-02): when/why image-space planning fails -- sharp turns, very narrow sidewalks, heavy occlusion, perspective distortion at far range. This is an honest assessment, not a weakness to hide.
- **D-04:** Add broader implications section (DISC-01): connect findings to the wider monocular perception field. The geometric coverage argument (why image-space works) applies beyond sidewalk navigation to any monocular corridor-following task.
- **D-05:** Keep existing Discussion sections (Interpretation, Why Image-Space Wins, Segmentation Sufficiency, Template Planning, Limitations, Threats to Validity) but strengthen them. Add the two new sections (failure analysis, broader implications).
- **D-06:** Rewrite Conclusion to map each key finding back to the numbered contributions from the Introduction. The 5 contributions established in Phase 2 must each have a corresponding conclusion statement.
- **D-07:** Thesis-comprehensive abstract, ~250-300 words. Structure: problem, approach (modular pipeline, 4 iterations), two key findings with numbers (421x speedup, 14.3px vs 65.0px, 99.3% BEV failure, template-approval), system validation (1800 frames, 9.97 FPS), and one-sentence future direction.
- **D-08:** All figure and table captions must be self-contained -- understandable without reading the body text. Each caption should state what is shown, key takeaway, and conditions (e.g., "32 hand-annotated frames, candidate mask").
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

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| WRIT-01 | Full prose rewrite -- active voice, professional academic tone, no lab-notes style | D-01/D-02 define scope (all 6 chapters) and tone (traditional thesis formal). Ch.1-4 need review pass only; Ch.5-6 need full rewrite. Architecture Patterns below provide tone guidance and common fixes. |
| WRIT-02 | Self-contained figure and table captions throughout | D-08 defines standard: what is shown, key takeaway, conditions. 32 captions total identified. Caption assessment below categorizes which need rewriting. |
| WRIT-03 | Consistent terminology across all chapters | D-09 defines 6 canonical terms with variants to replace. Terminology Inventory below documents current usage counts and locations for systematic find-and-replace. |
| WRIT-04 | Polish Abstract -- tight, specific, compelling, no redundancy | D-07 defines target: 250-300 words (current is ~378). Structure prescribed: problem, approach, two findings with numbers, validation, future direction. |
| DISC-01 | Strengthen Discussion -- explain WHY findings hold, connect to broader monocular perception field | D-04 defines broader implications content. D-05 requires strengthening existing sections plus adding new Broader Implications section. Architecture Patterns below provide recommended section structure. |
| DISC-02 | Add failure analysis for image-space pipeline (when/why it breaks) | D-03 defines failure modes: sharp turns, narrow sidewalks, heavy occlusion, far-range distortion. Architecture Patterns below provide recommended subsection structure with scenario-evidence-mitigation pattern. |
| DISC-03 | Rewrite Conclusion mapping each finding back to numbered contributions from Introduction | D-06 requires 5 numbered contributions from Introduction each mapped to a concluding statement. Contribution Mapping below provides the 5 contributions with their evidence and recommended conclusion wording. |
</phase_requirements>

## Architecture Patterns

### Current Document Structure (Post-Phase 3)

The thesis currently has 1366 lines of LaTeX across 6 chapters:

```
thesis/main.tex (1366 lines)
  Front matter (lines 1-194)
    Title, Approval, Copyright, Acknowledgments, TOC, Abstract (lines 178-191)
  Ch.1 Introduction (lines 197-271) -- rewritten Phase 2
  Ch.2 Background and Related Work (lines 275-386) -- rewritten Phase 2
  Ch.3 System Design (lines 390-735) -- rewritten Phase 3
  Ch.4 Experimental Evaluation (lines 738-1255) -- rewritten Phase 3
  Ch.5 Discussion (lines 1258-1317) -- ORIGINAL DRAFT, needs full rewrite
  Ch.6 Conclusion and Future Work (lines 1320-1366) -- ORIGINAL DRAFT, needs full rewrite
  Back matter (lines 1360-1367)
```

### Recommended Discussion Structure (Ch.5)

The current Discussion has 6 sections spanning ~60 lines. Per D-05, keep all existing sections and add two new ones. Recommended order:

```
Ch.5 Discussion
  5.1 Interpretation of Key Findings (existing, strengthen)
  5.2 Why Image-Space Outperforms BEV (existing, strengthen)
  5.3 Segmentation as Necessary but Not Sufficient (existing, minor polish)
  5.4 Template Planning and Turn Safety (existing, minor polish)
  5.5 When Image-Space Planning Fails (NEW - DISC-02, failure analysis)
  5.6 Broader Implications for Monocular Navigation (NEW - DISC-01)
  5.7 Limitations (existing, strengthen)
  5.8 Threats to Validity (existing, minor polish)
```

**Rationale for ordering:** Failure analysis (5.5) should come after the positive interpretation (5.1-5.4) but before formal Limitations (5.7). It is an honest technical analysis of boundary conditions, not a weakness admission. Broader implications (5.6) should come after both positive and failure analysis to provide the "zoom out" perspective before the formal limitations.

### Failure Analysis Section Pattern (DISC-02)

Each failure scenario should follow this structure:

```
Scenario: [description of when it fails]
Root cause: [why the image-space approach is geometrically limited here]
Evidence: [reference to existing data or geometric argument]
Mitigation: [what the system does / could do about it]
```

Recommended failure scenarios (from D-03 and geometric reasoning):
1. **Sharp turns (>45 deg):** Image-space midpoint assumes forward path is roughly straight. For sharp turns, the midpoint drifts to the outside of the curve. This is why the template-approval and waypoint-turn planners exist -- they handle this failure mode in the BEV domain.
2. **Very narrow sidewalks (<1m):** Mask boundary detection becomes unreliable when the sidewalk is only a few pixels wide in the camera view. The per-row midpoint has high variance when the mask is thin.
3. **Heavy occlusion (>60% mask loss):** When pedestrians, vehicles, or overhanging vegetation block most of the sidewalk, the mask fragments and per-row processing produces discontinuous midpoints. The temporal smoother can bridge brief occlusions but not sustained loss.
4. **Perspective distortion at far range:** At distances >8m from the camera, the perspective projection compresses the lateral dimension of the sidewalk to just a few pixels, making midpoint extraction imprecise. This primarily affects the lookahead distance rather than near-field accuracy.

### Broader Implications Section Pattern (DISC-01)

Structure recommendation:

1. **Generalization of the geometric coverage insight:** The fundamental finding -- that image-space representations preserve more useful geometric information than BEV for single-camera systems -- is not specific to sidewalks. Any monocular corridor-following task (agricultural row following, hallway navigation, trail following) faces the same BEV coverage problem.
2. **Propose-and-verify as a general planning paradigm:** Template-approval planning (score pre-computed candidates against a geometric surface rather than constructing paths from scratch) is applicable beyond sidewalk navigation. It trades optimality for robustness and computational efficiency.
3. **Embedded perception design implications:** The 59 FPS result on CPU demonstrates that competitive navigation is achievable without GPU acceleration, expanding the deployment envelope for low-cost platforms.

### Conclusion Rewrite Pattern (DISC-03)

The current Conclusion (lines 1320-1366) has three sections: Summary of Contributions, Key Findings (7-item enumeration), Future Work. Per D-06, the rewrite must explicitly map back to the 5 numbered contributions from the Introduction (lines 236-246).

**The 5 contributions to map:**

| # | Contribution (from Introduction) | Evidence to cite in Conclusion |
|---|----------------------------------|-------------------------------|
| 1 | Systematic comparison of 5 planning methods across BEV/image-space | 14.3px vs 65.0px, 421x speedup, Table planner_comparison |
| 2 | Template-approval architecture replacing skeleton-graph | 109ms to 69ms, 40.6% heading error reduction, Table template_eval |
| 3 | Teacher-student segmentation recipe (OneFormer -> SegFormer-B0) | IoU 0.946 at 11.7ms, Table seg_comparison |
| 4 | Comprehensive offline evaluation protocol | 22,679 frames, 32 hand-annotated, 1800-frame accepted run |
| 5 | GPS-conditioned waypoint-turn planner with containment guard | 0% containment failure, 12.9 FPS |

The Conclusion should:
- Open with 1-2 sentences restating the thesis contribution arc (discovery to verification)
- Map each of the 5 contributions to its key evidence
- Retain the current Future Work section (6 items) with minor polish
- End with a forward-looking closing paragraph

### Abstract Rewrite Pattern (WRIT-04)

Current abstract: ~378 words across 5 paragraphs (lines 181-189). Target: 250-300 words.

Recommended structure (following D-07):
1. **Problem** (1-2 sentences): Sidewalk navigation, monocular camera, embedded constraint.
2. **Approach** (2-3 sentences): Modular pipeline, 4 design iterations (brief), 5-method comparison.
3. **Finding 1** (2-3 sentences): Image-space dominates BEV for straight-ahead. Key numbers: 14.3px vs 65.0px, 421x speedup, 99.3% BEV failure.
4. **Finding 2** (1-2 sentences): Template-approval replaces skeleton-graph. 40.6% heading error reduction.
5. **Validation** (1-2 sentences): 1800 frames, 100% template success, 9.97 FPS. Turn containment 0% failure.
6. **One-sentence future direction.**

What to cut from the current abstract:
- The detailed segmentation training description (paragraph 2 detail about pseudo-labels)
- The full-video replay statistics (22,679 frames, instability reduction)
- The obstacle detection/GPS/serial protocol sentence (line 189 -- too implementation-specific)
- Specific mask-path alignment percentages (98.5% vs 98.6%)

### Prose Tone Guide (WRIT-01)

The Phase 2 (D-09/D-10) and Phase 4 (D-02) decisions establish the target tone. Key conventions:

**Hedged, measured language:**
- "Results suggest..." not "We demonstrate..."
- "The data indicate..." not "We prove..."
- "This finding is consistent with..." not "This confirms..."
- "approximately" before exact numbers in prose (not in tables)

**Passive voice acceptable (traditional thesis):**
- "The mask is projected onto..." is fine
- "Five planning methods were compared..." is fine
- But avoid excessive passive chains that obscure the agent

**Common fixes to look for in Ch.1-4 review pass:**
- Remaining instances of "we" (the thesis uses first-person plural in some places -- should be consistent, either always or never)
- Lab-notes style ("Next, we tried..." "The issue was...")
- Overly assertive claims without hedging
- Casual connectors ("Also," "Plus," "So,")

### Terminology Inventory (WRIT-03)

Current usage analysis of the 6 canonical terms from D-09:

| Canonical Term | Variants Found in main.tex | Instances | Action |
|---|---|---|---|
| "segmentation mask" | "sidewalk mask" (3 instances in Ch.3-4), "road mask" (0 confirmed), "binary sidewalk mask" (1) | ~4 non-canonical | Replace variants with "segmentation mask" or "binary segmentation mask" where contextually appropriate |
| "template-approval planner" | "template arc planner" (7+ instances), "template bank" (implicit references) | ~7-10 non-canonical | Replace "template arc planner" with "template-approval planner" everywhere except first introduction in System Design where both names can be given |
| "image-space midpoint planner" | Generally consistent | ~0-1 non-canonical | Verify consistency only |
| "BEV corridor" | "BEV mask corridor" (0-1), "drivable corridor" (0-1), "distance-transform corridor" (contextually different) | ~1-2 non-canonical | Minor cleanup; "distance-transform corridor" is acceptable as it describes the method, not the domain |
| "design iteration" | "Phase~11" in table (1 instance, line 1096) | 1 confirmed | Replace "Phase~11" with "Iteration 4" or "design iteration 4" |
| "teacher-student framework" | "knowledge distillation" (appears in lit review only -- acceptable per D-09) | 0 non-canonical outside lit review | No action needed; already consistent |

**Execution approach:** Build a find-and-replace checklist. Process each term sequentially through the entire document. Verify contextual appropriateness of each replacement (some variants may be correct in specific contexts, e.g., "sidewalk mask" when specifically discussing what the segmentation identifies).

### Caption Assessment (WRIT-02)

There are 32 `\caption{}` commands in main.tex. Assessment of self-containment:

**Already adequate (need minor polish at most):**
- Fig. scooter_hw (line 269): States what, context, and conditions. Good.
- Fig. pipeline (line 402): States what and scope. Good.
- Tab. seg_comparison (line 856/872): Describes data and conditions. Good.
- Tab. planner_comparison (line 960): States dataset, metrics. Good.
- Fig. bev_fragility (line 1068): States outcome and interpretation. Good.
- Tab. accepted_run (line 1211): States conditions. Good.

**Need rewriting (missing key takeaway or conditions):**
- Subfigure captions (lines 259, 266, 835, 839, 843, 1018, 1022, 1026): Very terse ("Scooter platform with mounted camera" -- no context about why shown)
- Tab. resolution_sweep (line 490): Missing conditions and takeaway
- Tab. runtime_configs (line 1160): Missing interpretation
- Tab. template_eval (line 1092): Missing key takeaway (40.6% improvement)
- Tab. waypoint_turn_eval (line 1182): Missing conditions detail

**Principle for self-contained captions:** Each caption should answer three questions:
1. **What is shown?** (data type, visualization type)
2. **Under what conditions?** (dataset, model, configuration)
3. **What is the key takeaway?** (the one thing a skimming reader should learn)

For subfigure captions: the parent caption handles conditions and takeaway; subfigure captions identify the specific panel content.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Terminology consistency | Manual find-replace by memory | Systematic checklist with grep verification | A systematic approach catches all instances; memory-based replacement misses edge cases |
| Caption quality | Ad-hoc rewrites | Three-question template (what/conditions/takeaway) | Consistent quality standard across all 32 captions |
| Contribution mapping | Free-form conclusion | Numbered list matching Introduction contributions | The 1-to-1 mapping is the key structural requirement from D-06 |
| Abstract compression | Line-by-line editing | Start fresh with prescribed 6-element structure | Editing a 378-word abstract down to 250 words is harder than writing a new 250-word abstract from the prescribed structure |

## Common Pitfalls

### Pitfall 1: Hedging that Undermines the Thesis
**What goes wrong:** Over-hedging the central findings ("Results may possibly suggest that image-space planning might be somewhat better...") weakens the scientific contribution.
**Why it happens:** D-02 says "measured, hedged language" -- easy to over-apply.
**How to avoid:** Hedge interpretations and generalizations, not data. "Image-space midpoint planning achieves 14.3 px lateral error" is a fact (no hedge needed). "This result suggests that monocular BEV planning is inherently fragile" is an interpretation (hedge appropriate).
**Warning signs:** More than one hedge word per sentence. Hedging numbers.

### Pitfall 2: Failure Analysis Reads as Self-Criticism
**What goes wrong:** The failure analysis section comes across as apologetic or defensive, undermining confidence in the system.
**Why it happens:** Writing about limitations and failure modes can drift into negativity.
**How to avoid:** Frame failure analysis as boundary characterization: "The image-space midpoint planner is designed for straight-ahead following; its geometric assumptions break down under [conditions]. The template-approval planner and waypoint-turn planner were introduced precisely to address these boundary cases." Each failure mode should connect to a mitigation that the system already provides.
**Warning signs:** Failure scenarios without corresponding mitigation. Apologetic language ("unfortunately", "regrettably").

### Pitfall 3: Terminology Replacement Breaks Meaning
**What goes wrong:** Mechanical find-and-replace of "template arc planner" with "template-approval planner" in a context where "arc" is the semantically important word.
**Why it happens:** The terminology map doesn't account for all contextual uses.
**How to avoid:** Every replacement must be manually verified in context. Particularly watch for: (a) first-introduction passages where both names should appear, (b) table headers with character limits, (c) mathematical descriptions where "arc" is geometric meaning.
**Warning signs:** A sentence reads awkwardly after replacement. The replaced term appears in a table column header that now wraps.

### Pitfall 4: Abstract Loses Thesis-Comprehensive Coverage
**What goes wrong:** The compressed abstract covers only the BEV vs. image-space finding and omits the template planner, teacher-student training, or evaluation scope.
**Why it happens:** Cutting from 378 to 250-300 words forces hard choices, and the BEV finding is the most dramatic.
**How to avoid:** D-07 prescribes the structure explicitly. Follow the 6-element template. Allocate roughly: problem (30 words), approach (50 words), finding 1 (60 words), finding 2 (40 words), validation (40 words), future (20 words) = 240 words with margin.
**Warning signs:** Any of the 5 contributions completely absent from the abstract.

### Pitfall 5: Conclusion Becomes a Results Summary
**What goes wrong:** The Conclusion simply re-lists the 7 key findings from the current enumeration without synthesizing or mapping to contributions.
**Why it happens:** The easiest path is to copy the existing Key Findings list.
**How to avoid:** D-06 requires explicit contribution mapping. Write the conclusion from the 5 contributions outward, not from the findings inward. Each contribution gets one paragraph stating: "Contribution N was [X]. The evidence demonstrates [Y], suggesting [Z]."
**Warning signs:** The conclusion uses the same sentence structure for every finding. No reference to contribution numbers. No synthesis across findings.

### Pitfall 6: Ch.1-4 Review Accidentally Rewrites Phase 2-3 Work
**What goes wrong:** The prose review pass on Ch.1-4 makes substantive content changes (reordering sections, adding new content) rather than just polishing prose.
**Why it happens:** The reviewer finds something they want to "improve" beyond the prose scope.
**How to avoid:** The Ch.1-4 pass should ONLY fix: (a) terminology consistency per D-09, (b) tone issues per D-02, (c) grammatical errors, (d) caption self-containment per D-08. No content additions, no section reorganization, no table modifications.
**Warning signs:** Diff shows more than 5-10 lines changed per section in Ch.1-4.

## Code Examples

### Hedged vs. Unheedged Language

```latex
% WRONG: Unheedged interpretation
This proves that BEV planning is useless for monocular navigation.

% CORRECT: Hedged interpretation of data
These results suggest that BEV-domain planning is fragile under
monocular projection, consistent with the geometric coverage
analysis presented in Section~\ref{sec:bev_fragility_results}.

% CORRECT: Unheedged fact (data, not interpretation)
The image-space midpoint planner achieves \SI{14.3}{\px} lateral
center error at \SI{2.2}{ms} per frame.
```

### Self-Contained Caption Pattern

```latex
% BEFORE: Not self-contained
\caption{Template arc planner vs.\ skeleton-graph baseline on 220
  frames (calibrated June clip, detection disabled).}

% AFTER: Self-contained (what, conditions, takeaway)
\caption{Template-approval planner vs.\ skeleton-graph baseline on
  a 220-frame calibrated evaluation clip (VID\_017, June segment,
  obstacle detection disabled). The template-approval planner reduces
  mean heading error by 40.6\% and halves path-source switches while
  achieving comparable per-frame cost.}
```

### Failure Analysis Paragraph Pattern

```latex
\paragraph{Sharp turns.}
The image-space midpoint planner extracts the lateral center of the
segmentation mask independently in each image row, an assumption
that holds when the sidewalk extends roughly forward from the camera.
At turn angles exceeding approximately 45$^\circ$, the mask bends
laterally across the image and the per-row midpoint drifts toward
the outside of the curve, producing paths that clip the inner
boundary. This failure mode motivates the template-approval and
waypoint-turn planners introduced in Section~\ref{sec:template_planner}
and Section~\ref{sec:waypoint_turn}, which plan in the BEV domain
where curvature is explicitly represented.
```

### Contribution-Mapped Conclusion Pattern

```latex
\textbf{Contribution~1} established a systematic comparison of five
path-planning methods across both BEV and image-space domains
(Chapter~\ref{ch:evaluation}).  The data indicate that image-space
midpoint planning achieves approximately \SI{14.3}{\px} lateral
center error at \SI{2.2}{ms}---an approximately $421\times$ speedup
over the best BEV method (\SI{65.0}{\px} at \SI{926.8}{ms})---while
BEV-only planning fails to produce valid paths on 99.3\% of frames
in the profiled sequence.  This result suggests that, for
monocular sidewalk following, the image-space domain offers a more
favorable accuracy--latency trade-off than the BEV domain.
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Thesis abstract as exhaustive summary | Structured abstract with prescribed elements | Academic convention (ongoing) | Abstracts under 300 words are standard for most MS theses |
| Discussion as results restatement | Discussion as interpretation + implications + limitations | Academic writing standards | Committee expects interpretation, not repetition |
| Conclusion as findings list | Conclusion as contribution synthesis | Thesis writing conventions | Maps findings back to stated contributions for coherence |
| Ad-hoc figure captions | Self-contained captions (what/conditions/takeaway) | Publication standard | Readers often scan figures first; captions must stand alone |

## Open Questions

1. **First-person pronoun convention**
   - What we know: The thesis uses "we" in some places (likely from Phase 2-3 rewrites) and passive voice in others. Traditional thesis convention varies by department.
   - What's unclear: Whether the OU ECE department has a stated preference.
   - Recommendation: Adopt consistent first-person plural ("we") throughout, as it is increasingly accepted in engineering theses and was used in the Phase 2-3 rewrites. If the committee objects, a global find-replace from "we" to passive is trivial. Flag this for the user to confirm.

2. **Broader implications section placement**
   - What we know: D-04 says add it. Claude's Discretion allows choosing between standalone section vs. subsection of Interpretation.
   - What's unclear: How much content warrants a standalone section.
   - Recommendation: Make it a standalone section (5.6) because the geometric coverage generalization argument is substantive enough to merit its own section heading, and it connects to applications beyond sidewalks.

3. **Abstract mention of all 4 design iterations**
   - What we know: D-07 mentions "approach (modular pipeline, 4 iterations)". Claude's Discretion allows summarizing the evolution rather than listing all 4.
   - What's unclear: Whether naming all 4 iterations in the abstract adds value at the 250-300 word budget.
   - Recommendation: Mention "four design iterations" as a phrase but only name the endpoints (skeleton-graph to template-approval). Naming all 4 would consume ~40 words for minimal information value.

## Project Constraints (from CLAUDE.md)

The following CLAUDE.md directives are relevant to this phase:

- **LaTeX only:** All edits are to `thesis/main.tex` (and `references.bib` if needed). No Python or codebase changes.
- **No new experiments or data:** The Abstract, Discussion, and Conclusion must use only existing experimental results.
- **No new figures:** Caption improvements modify text only, not the figures themselves.
- **Safety gates non-negotiable:** When discussing the system in Discussion, do not relax or question safety thresholds.
- **config.py as single source of truth:** When referencing system parameters in Discussion text, use the canonical values from config.py.
- **Testing:** `pytest tests/ -v` should still pass (though this phase modifies only LaTeX, not code).
- **GSD Workflow:** Work through GSD commands, not direct repo edits.

## Sources

### Primary (HIGH confidence)
- `thesis/main.tex` -- Direct analysis of current document state (1366 lines, 32 captions, 6 chapters)
- `.planning/phases/04-prose-quality-discussion/04-CONTEXT.md` -- All 9 locked decisions
- `.planning/phases/02-introduction-literature-review/02-CONTEXT.md` -- Tone decisions D-09/D-10, contribution framing D-03/D-04/D-05
- `.planning/phases/03-methodology-results-rewrite/03-CONTEXT.md` -- BEV nuance D-11, claim structure D-05/D-06

### Secondary (MEDIUM confidence)
- [Scribbr Discussion Section Guide](https://www.scribbr.com/dissertation/discussion/) -- Discussion chapter structure
- [Caltech Figure Caption Handout](https://writing.caltech.edu/documents/27629/HWC-FigureCaptionHandout.1-2024.pdf) -- Self-contained caption standards
- [Wordvice Limitations Guide](https://blog.wordvice.com/how-to-present-study-limitations-and-alternatives/) -- Failure analysis writing approach
- [SFU Abstract Writing Guide](https://www.sfu.ca/~jcnesbit/HowToWriteAbstract.htm) -- Abstract structure for theses
- [Scribbr Abstract Guide](https://www.scribbr.com/dissertation/abstract/) -- Abstract compression techniques

### Tertiary (LOW confidence)
- None -- all findings verified against primary sources or established academic writing conventions.

## Metadata

**Confidence breakdown:**
- Discussion/Conclusion rewrite: HIGH -- current text is directly analyzed; contribution mapping is mechanical from Introduction; failure scenarios are technically grounded in the system's geometry
- Abstract rewrite: HIGH -- current word count verified (~378), target (250-300) is clear, structure prescribed by D-07
- Terminology standardization: HIGH -- grep analysis confirms specific instances to fix; the list is bounded
- Caption improvement: HIGH -- all 32 captions inspected; assessment of self-containment is based on established standards
- Prose review of Ch.1-4: MEDIUM -- hard to predict exactly how much cleanup is needed without reading every line; Phases 2-3 used formal tone but some inconsistencies likely remain

**Research date:** 2026-03-30
**Valid until:** 2026-04-15 (thesis content, stable domain)
