# Master's Thesis Rewrite

## What This Is

A full rewrite of the Master's thesis for the University of Oklahoma ECE department on monocular-camera-based autonomous sidewalk navigation. The existing draft (1,323 lines of LaTeX) has solid experimental data and figures but needs restructuring, professional prose, coherent narrative, and a proper evaluation framework (vs. baseline, not model-vs-model). The goal is a publication-quality 60-80 page thesis ready for committee review.

## Core Value

A professional, cohesive thesis that tells a clear scientific story: simple image-space geometry outperforms complex BEV pipelines for monocular sidewalk navigation on embedded platforms — supported by systematic evaluation against proper baselines.

## Requirements

### Validated

- Existing LaTeX template (OU format: 12pt, double-spaced, Times New Roman, 1.5" left margin)
- Existing experimental data: 32 hand-annotated frames, 22,679-frame video replay, 1,800-frame accepted run, runtime profiling data
- Existing figures: pipeline diagrams, planner comparisons, BEV fragility, segmentation examples, runtime breakdowns, scooter hardware photos
- Existing bibliography (references.bib)

### Active

- [x] Complete restructure of thesis organization for logical flow and professional narrative — Validated in Phase 1: Structural Reorganization
- [ ] Full rewrite of all prose — authoritative academic tone, not lab-notes style
- [ ] Rewrite evaluation to show design iteration progression (v1 skeleton → v2 DT → v3 image-space → v4 template arc)
- [ ] Add proper baseline comparison (full pipeline vs. raw segmentation / naive approach)
- [ ] Remove checkpoint-vs-checkpoint benchmark (Table 7) — replace with meaningful evaluation
- [x] Strengthen Introduction with clearer motivation, sharper problem statement, and stronger contribution framing — Validated in Phase 2
- [x] Rewrite Literature Review to better position our work relative to the field — Validated in Phase 2
- [ ] Rewrite Methodology chapter with clearer design rationale (WHY each decision was made)
- [ ] Restructure Results chapter to tell a progression story, not a list of experiments
- [ ] Strengthen Discussion with deeper analysis and clearer implications
- [ ] Ensure all figures are properly referenced and contribute to the narrative
- [ ] Verify 60-80 page target length when compiled
- [ ] Polish Abstract to be tight, specific, and compelling
- [ ] Complete front matter (acknowledgments, committee member placeholder)

### Out of Scope

- New experiments or data collection — use existing data only (deadline too tight)
- Changing the LaTeX template or OU formatting — keep existing template
- Modifying the codebase — thesis only
- Creating new figures from scratch — reuse/reorganize existing figures (minor edits OK)
- Running new evaluations or benchmarks — use existing numbers

## Context

- **Author:** Lkhanaajav Mijiddorj, MS candidate, ECE Department, University of Oklahoma
- **Advisor:** Dr. Binbin Weng
- **Committee:** Dr. Bin Xu + one unnamed member
- **Deadline:** Within 1 week (tight — must prioritize ruthlessly)
- **Current draft:** `thesis/main.tex` (1,323 lines), compiles with existing OU template
- **Existing figures:** `thesis/figures/` directory with pipeline diagrams, comparisons, qualitative results
- **Bibliography:** `thesis/references.bib`
- **Paper source template:** `thesis/paper_src/` (cas-dc class, separate from thesis template)

### Problems with Current Draft

1. **Structure feels like lab notes** — runtime offender tables, configuration comparisons, checkpoint benchmarks read like development logs, not thesis prose
2. **Model-vs-model comparison is meaningless** — comparing 11 fine-tuned SegFormer checkpoints against each other (Table 7) adds no scientific value
3. **Missing "why"** — decisions are presented but rarely motivated; reader doesn't understand the design journey
4. **Results are a list, not a story** — each section reports numbers but doesn't build a coherent argument
5. **Writing quality varies** — some sections are polished (Discussion), others read like first drafts
6. ~~**Four iterations listed but three claimed** — Introduction says "three design iterations" but lists four~~ (Fixed in Phase 1)

### What's Actually Good (Keep/Build On)

- Planner comparison study (5 methods, systematic)
- BEV fragility analysis (novel, quantitative)
- Teacher-student training framework description
- Hardware platform description
- Oracle-mask experiment (isolates planning domain from segmentation quality)
- Most figures are publication-quality
- OU template formatting is correct

## Constraints

- **Timeline:** ~1 week — full rewrite but no new experiments
- **Format:** University of Oklahoma thesis format (already in template)
- **Length:** 60-80 pages double-spaced
- **Data:** Must use existing experimental results — no new runs
- **Figures:** Reuse existing figures — no new data visualizations
- **Tool:** LaTeX only (main.tex + references.bib)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Full rewrite, not heavy edit | Current narrative structure is fundamentally disorganized — patching won't fix the story | -- Pending |
| Baseline = iteration progression + naive baseline | Shows both the design journey AND scientific rigor vs. a proper control | -- Pending |
| Remove checkpoint benchmark table | Comparing fine-tuned checkpoints adds no value; replace with teacher-student comparison | -- Pending |
| Keep all existing data/figures | No time for new experiments; existing data is strong enough | -- Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd:transition`):
1. Requirements invalidated? -> Move to Out of Scope with reason
2. Requirements validated? -> Move to Validated with phase reference
3. New requirements emerged? -> Add to Active
4. Decisions to log? -> Add to Key Decisions
5. "What This Is" still accurate? -> Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check -- still the right priority?
3. Audit Out of Scope -- reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-03-30 after Phase 2 completion*
