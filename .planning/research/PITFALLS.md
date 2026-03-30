# Pitfalls Research — Common Thesis Mistakes

## Evaluation Pitfalls

### 1. No External Baseline
**Warning signs:** All comparisons are between variants of the same system. No published method or naive approach is compared.
**This thesis:** The checkpoint benchmark (current Table 7) compares 11 fine-tuned models against each other. This shows which checkpoint is best but proves nothing about the approach.
**Fix:** Replace with teacher-student comparison (SegFormer-B2 teacher vs OneFormer teacher) and add a naive baseline (raw mask center following).
**Phase:** Evaluation restructure phase.

### 2. Cherry-Picked Metrics
**Warning signs:** Different metrics used for different experiments. Metrics chosen post-hoc to make results look good.
**This thesis:** Mostly OK — consistent metrics across planners. But segmentation section uses IoU while temporal section uses stability rate.
**Fix:** Define all metrics in one place (already done in Section 3.7). Ensure every experiment reports the same core metrics where applicable.
**Phase:** Methodology rewrite phase.

### 3. Overclaiming Contributions
**Warning signs:** Phrases like "first ever", "novel framework", "state-of-the-art" without rigorous verification.
**This thesis:** Claims "first quantitative characterization of BEV fragility" — may be true but needs literature verification.
**Fix:** Soften to "to our knowledge" unless exhaustive search confirms the claim.
**Phase:** Introduction rewrite phase.

## Narrative Pitfalls

### 4. Lab-Notes Style Writing
**Warning signs:** Sections read like "then we did X, then we tried Y, then we found Z". No argument structure.
**This thesis:** The Results chapter is a list of experiments without a connecting thread.
**Fix:** Restructure around claims: each section states a hypothesis, presents evidence, draws a conclusion.
**Phase:** Results restructure phase.

### 5. Missing Design Rationale
**Warning signs:** "We use SegFormer-B0" without explaining why. "The threshold is 0.60" without explaining how it was chosen.
**Fix:** For every design choice, include: what alternatives existed, why this was chosen, and what the trade-off is.
**Phase:** Methodology rewrite phase.

### 6. Disconnected Sections
**Warning signs:** Sections can be read in any order without losing comprehension. No forward/backward references.
**Fix:** Add transition sentences between sections. End each section with a sentence that motivates the next.
**Phase:** All phases (structural editing pass).

## Writing Pitfalls

### 7. Passive Voice Overuse
**Warning signs:** "The mask was processed", "It was found that", "The results were obtained".
**Fix:** Active voice: "We process the mask", "The results show", "The planner achieves".
**Phase:** Prose rewrite phase.

### 8. Vague Quantifiers
**Warning signs:** "significantly better", "much faster", "substantially improved".
**Fix:** Always quantify: "24.8% higher IoU", "421x faster", "40.6% lower heading error".
**Phase:** Prose rewrite phase.

### 9. Inconsistent Terminology
**Warning signs:** Same thing called different names in different sections (e.g., "mask", "segmentation map", "binary label").
**This thesis:** Generally consistent, but "design iterations" count varies (3 vs 4).
**Fix:** Create a terminology table. Use one term per concept throughout.
**Phase:** Consistency pass.

## Structure Pitfalls

### 10. Wrong Chapter Proportions
**Warning signs:** Introduction is 15 pages but Discussion is 2 pages. Literature Review is 3 pages.
**This thesis:** Closed-Loop chapter is ~2 pages — too thin for its own chapter.
**Fix:** Merge Closed-Loop into System Design. Target proportions: Intro 10%, Lit Review 15%, Method 25%, Results 25%, Discussion 15%, Conclusion 10%.
**Phase:** Restructure phase.

### 11. Redundancy Between Chapters
**Warning signs:** Same information presented in Introduction, Methodology, and Results.
**This thesis:** The "421x speedup" and "99.3% BEV failure" are mentioned in Abstract, Introduction, Results, Discussion, and Conclusion.
**Fix:** State the finding once in Results, reference it elsewhere. Full detail only in Results.
**Phase:** All phases.

## Figure/Table Pitfalls

### 12. Unexplained Figures
**Warning signs:** Figure appears but the text says nothing beyond "as shown in Figure X".
**Fix:** Every figure needs: (1) what it shows, (2) what the reader should notice, (3) what it means.
**Phase:** Results rewrite phase.

### 13. Redundant Tables
**Warning signs:** Multiple tables showing slight variations of the same data.
**This thesis:** Tables 5, 6 (runtime comparison and runtime offenders) show overlapping information. Table 7 (checkpoint benchmark) should be removed entirely.
**Fix:** Consolidate runtime into one table. Remove checkpoint benchmark.
**Phase:** Results restructure phase.

## Discussion Pitfalls

### 14. Restating Results
**Warning signs:** Discussion paragraph starts with "Table X shows that..." — that's a Results paragraph.
**Fix:** Discussion should explain WHY, not WHAT. "The image-space advantage stems from geometric coverage, not algorithmic superiority."
**Phase:** Discussion rewrite phase.

### 15. Missing Connection to Broader Field
**Warning signs:** Discussion only talks about this specific system, not implications for the field.
**Fix:** Generalize findings: "This suggests that for any monocular perception system where the camera covers a limited ground area, image-space reasoning may be preferable to BEV projection."
**Phase:** Discussion rewrite phase.

## CV/Robotics-Specific Pitfalls

### 16. Runtime Reporting Without Hardware Context
**Warning signs:** "Our method runs at 59 FPS" without specifying CPU model, RAM, batch size.
**This thesis:** Hardware is specified (Intel i7 CPU) but could be more precise.
**Fix:** Report full hardware spec, input resolution, batch size=1, and whether inference includes pre/post-processing.
**Phase:** Evaluation setup section.

### 17. No Failure Analysis
**Warning signs:** Only success cases shown. No discussion of when/why the system fails.
**This thesis:** BEV fragility analysis is actually a good failure analysis. Need similar for the image-space pipeline.
**Fix:** Add a subsection on when image-space planning fails (e.g., narrow masks, extreme curves).
**Phase:** Results chapter.
