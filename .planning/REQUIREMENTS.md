# Requirements: Master's Thesis Rewrite

**Defined:** 2026-03-30
**Core Value:** A professional, cohesive thesis that tells a clear scientific story supported by proper baseline evaluation

## v1 Requirements

### Structure

- [x] **STRUCT-01**: Merge Closed-Loop chapter into System Design — 6 total chapters
- [x] **STRUCT-02**: Restructure Results chapter around claims (claim-evidence-conclusion pattern)
- [ ] **STRUCT-03**: Add transition sentences between all sections and chapters
- [x] **STRUCT-04**: Fix iteration count — consistently say "four design iterations"

### Narrative

- [x] **NARR-01**: Rewrite Introduction with compelling opening hook, sharper problem statement, and stronger contribution framing
- [x] **NARR-02**: Rewrite Literature Review as themed synthesis — each section ends with the gap this thesis fills
- [x] **NARR-03**: Add design rationale ("why before what") for every methodology design choice
- [x] **NARR-04**: Frame all Results sections as claim-evidence-conclusion

### Evaluation

- [x] **EVAL-01**: Remove checkpoint benchmark table (Table 7) — replace with teacher-student comparison narrative
- [x] **EVAL-02**: Add design iteration progression table (v1→v2→v3→v4 with key metrics at each stage)
- [x] **EVAL-03**: Consolidate redundant runtime tables (merge current Tables 5+6 into one)
- [x] **EVAL-04**: Add naive baseline discussion (raw mask center-following as conceptual baseline)

### Writing Quality

- [ ] **WRIT-01**: Full prose rewrite — active voice, professional academic tone, no lab-notes style
- [ ] **WRIT-02**: Self-contained figure and table captions throughout
- [ ] **WRIT-03**: Consistent terminology across all chapters (standardize terms for mask, path, planner, etc.)
- [ ] **WRIT-04**: Polish Abstract — tight, specific, compelling, no redundancy

### Discussion & Conclusion

- [x] **DISC-01**: Strengthen Discussion — explain WHY findings hold, connect to broader monocular perception field
- [x] **DISC-02**: Add failure analysis for image-space pipeline (when/why it breaks)
- [x] **DISC-03**: Rewrite Conclusion mapping each finding back to numbered contributions from Introduction

### Front Matter

- [ ] **FRNT-01**: Complete committee member name placeholder
- [ ] **FRNT-02**: Verify final document hits 60-80 page target when compiled

## v2 Requirements

### Polish

- **POL-01**: Add appendix with full configuration parameter table
- **POL-02**: Add notation/symbols table if space permits
- **POL-03**: Cross-check all bibliography entries for completeness

## Out of Scope

| Feature | Reason |
|---------|--------|
| New experiments or data collection | Deadline too tight; existing data is sufficient |
| New figures from scratch | Reuse existing figures; minor relabeling OK |
| Changing OU LaTeX template | Template is university-mandated |
| Modifying the codebase | Thesis rewrite only |
| Running new benchmarks | Use existing numbers from prior evaluations |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| STRUCT-01 | Phase 1 | Complete |
| STRUCT-02 | Phase 3 | Complete |
| STRUCT-03 | Phase 5 | Pending |
| STRUCT-04 | Phase 1 | Complete |
| NARR-01 | Phase 2 | Complete |
| NARR-02 | Phase 2 | Complete |
| NARR-03 | Phase 3 | Complete |
| NARR-04 | Phase 3 | Complete |
| EVAL-01 | Phase 3 | Complete |
| EVAL-02 | Phase 3 | Complete |
| EVAL-03 | Phase 3 | Complete |
| EVAL-04 | Phase 3 | Complete |
| WRIT-01 | Phase 4 | Pending |
| WRIT-02 | Phase 4 | Pending |
| WRIT-03 | Phase 4 | Pending |
| WRIT-04 | Phase 4 | Pending |
| DISC-01 | Phase 4 | Complete |
| DISC-02 | Phase 4 | Complete |
| DISC-03 | Phase 4 | Complete |
| FRNT-01 | Phase 5 | Pending |
| FRNT-02 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 20 total
- Mapped to phases: 20
- Unmapped: 0

---
*Requirements defined: 2026-03-30*
*Last updated: 2026-03-30 after initial definition*
