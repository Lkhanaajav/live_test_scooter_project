# Roadmap: Master's Thesis Rewrite

**Created:** 2026-03-30
**Phases:** 5
**Requirements:** 20 mapped
**Granularity:** Standard

## Phase Overview

| # | Phase | Goal | Requirements | Success Criteria |
|---|-------|------|--------------|------------------|
| 1 | Structural Reorganization | Restructure chapters into 6-chapter format | STRUCT-01, STRUCT-04 | 2 |
| 2 | Introduction & Literature Review | Rewrite opening chapters with compelling narrative | NARR-01, NARR-02 | 3 |
| 3 | Methodology & Results Rewrite | Rewrite core technical chapters with design rationale and claim-based evaluation | NARR-03, NARR-04, EVAL-01, EVAL-02, EVAL-03, EVAL-04, STRUCT-02 | 5 |
| 4 | Prose Quality & Discussion | Full prose polish, strengthen Discussion and Conclusion | WRIT-01, WRIT-02, WRIT-03, WRIT-04, DISC-01, DISC-02, DISC-03 | 4 |
| 5 | Final Polish & Verification | Transitions, front matter, page count verification | STRUCT-03, FRNT-01, FRNT-02 | 3 |

---

## Phase Details

### Phase 1: Structural Reorganization
**Goal:** Transform the 7-chapter draft into a clean 6-chapter structure before rewriting prose.

**Requirements:** STRUCT-01, STRUCT-04

**Success Criteria:**
1. Closed-Loop chapter (current Ch. 4) content merged into System Design chapter as sections
2. Chapter numbering and cross-references updated consistently
3. "Four design iterations" stated consistently (not "three")

**Rationale:** Structure must be fixed before prose rewrite — otherwise we'd rewrite content that moves to a different chapter.

---

### Phase 2: Introduction & Literature Review
**Goal:** Rewrite the first two chapters to set up the thesis contribution with a compelling narrative.

**Requirements:** NARR-01, NARR-02

**Plans:** 2 plans

Plans:
- [x] 02-01-PLAN.md — Rewrite Introduction (Motivation, Problem Statement, Contributions, Thesis Organization)
- [ ] 02-02-PLAN.md — Rewrite Literature Review (8 sections with distributed gap statements, Summary synthesis)

**Success Criteria:**
1. Introduction opens with a concrete motivating scenario, states contributions clearly, and frames the design iteration story
2. Literature Review is organized by themes with each section ending by identifying a research gap
3. Final "Research Gaps" section explicitly lists the 4 gaps this thesis fills and connects to contributions

**Rationale:** These chapters frame the entire thesis. A strong Introduction makes the reader care; a strong Lit Review makes them trust that the contribution is novel.

---

### Phase 3: Methodology & Results Rewrite
**Goal:** Rewrite the two heaviest chapters — System Design and Experimental Evaluation — with design rationale and claim-based evaluation structure.

**Requirements:** NARR-03, NARR-04, EVAL-01, EVAL-02, EVAL-03, EVAL-04, STRUCT-02

**Success Criteria:**
1. Every methodology section explains WHY before WHAT (design rationale present for all 5 planners, teacher-student choice, BEV parameters)
2. Checkpoint benchmark removed; replaced with teacher-student comparison narrative
3. Design iteration progression table present showing v1->v4 improvement metrics
4. Runtime tables consolidated into one clean table
5. Every Results section follows claim-evidence-conclusion pattern

**Rationale:** This is the meat of the thesis and where most problems live. The methodology needs motivation; the results need argument structure.

**UI hint**: no

---

### Phase 4: Prose Quality & Discussion
**Goal:** Full prose polish across all chapters. Strengthen Discussion and Conclusion to publication quality.

**Requirements:** WRIT-01, WRIT-02, WRIT-03, WRIT-04, DISC-01, DISC-02, DISC-03

**Success Criteria:**
1. Active voice used throughout; no passive "was performed" constructions
2. All figure/table captions are self-contained (understandable without reading body text)
3. Discussion explains WHY image-space outperforms BEV (geometric coverage argument) and connects to broader monocular perception field
4. Conclusion maps each key finding back to a numbered contribution from the Introduction

**Rationale:** Prose quality is what separates a thesis that reads like a draft from one that reads like a publication. Discussion is where the thesis demonstrates intellectual maturity.

---

### Phase 5: Final Polish & Verification
**Goal:** Add transitions, complete front matter, verify page count, final consistency pass.

**Requirements:** STRUCT-03, FRNT-01, FRNT-02

**Success Criteria:**
1. Transition sentences present between every section and chapter
2. Committee member placeholder completed (or marked for user to fill)
3. Compiled document is 60-80 pages with correct OU formatting

**Rationale:** Final pass catches inconsistencies and ensures the document meets university requirements.

---

## Dependency Graph

```
Phase 1 (Structure) -> Phase 2 (Intro/LitRev) -> Phase 3 (Method/Results) -> Phase 4 (Prose/Discussion) -> Phase 5 (Polish)
```

All phases are sequential — each builds on the previous rewrite.

---
*Roadmap created: 2026-03-30*
*Last updated: 2026-03-30 after Phase 2 planning*
