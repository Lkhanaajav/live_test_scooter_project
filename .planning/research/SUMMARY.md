# Research Summary — Thesis Rewrite

## Key Findings

### Stack
- Existing LaTeX setup is solid (OU template, booktabs, siunitx, natbib)
- Add `cleveref` for consistent cross-referencing
- Use active voice ("we compare") not passive ("was compared")
- Every figure/table needs a self-contained caption and in-text discussion
- Follow the 5-pass revision workflow: structure → content → consistency → style → format

### Table Stakes for a Strong Thesis
1. **Baseline comparison is mandatory** — compare against naive approach AND prior iterations, not just model variants against each other
2. **Claim-evidence-conclusion pattern** for every result section
3. **Design rationale** — explain WHY before WHAT for every choice
4. **Honest limitations** with specific suggestions for what would resolve them
5. **Consistent terminology and metrics** throughout

### Recommended Structure
**6 chapters** (merge current Closed-Loop chapter into System Design):
1. Introduction (6-8 pp) — hook, problem, approach, contributions
2. Background & Related Work (10-14 pp) — themed synthesis, not paper list
3. System Design (14-18 pp) — full pipeline with design rationale
4. Experimental Evaluation (14-18 pp) — organized by claims, not experiments
5. Discussion (6-8 pp) — why, not what; connect to broader field
6. Conclusion & Future Work (4-6 pp) — map back to contributions

**Target: 62-84 pages** including front/back matter.

### Critical Pitfalls to Fix
1. **Remove checkpoint benchmark** (Table 7) — comparing fine-tuned models against each other adds no value
2. **Stop lab-notes style** — restructure Results around claims, not chronological experiments
3. **Add design rationale** — every decision needs a "why"
4. **Merge redundant runtime tables** — Tables 5+6 overlap, consolidate
5. **Fix iteration count inconsistency** — say "four iterations" (it IS four)
6. **Add transition sentences** between all sections
7. **Quantify all improvements** — never say "significantly better" without numbers

### Narrative Arc
The thesis tells this story:
1. **Problem**: Sidewalk navigation needs real-time perception on cheap hardware
2. **Conventional wisdom**: BEV projection enables metric planning
3. **Discovery**: BEV is actually fragile with monocular cameras (99.3% failure)
4. **Solution**: Image-space planning is 421x faster AND more accurate
5. **Refinement**: Template arc planner + turn safety add stability for deployment
6. **Implication**: Geometric simplicity can beat architectural complexity when the sensing geometry favors it

## Actionable Recommendations

### What to Remove
- Checkpoint benchmark table (Table 7)
- Runtime offenders table (Table 6) — merge useful info into runtime comparison
- Configuration comparison table (Table 8) — move to appendix or cut
- Temporal smoothing grid search details — summarize in 1 paragraph

### What to Add
- Naive baseline comparison (raw mask center-following)
- Design iteration progression table showing improvement at each step
- Failure analysis for image-space pipeline (when does it break?)
- Transition sentences between every section
- "Why" paragraphs before every design choice

### What to Restructure
- Results: organize by claims, not by experiments
- Closed-Loop chapter: merge into System Design
- Introduction: stronger opening hook, tighter contribution framing
- Literature Review: end each section with the gap this thesis fills
