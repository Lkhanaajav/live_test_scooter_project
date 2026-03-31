---
phase: 04-prose-quality-discussion
plan: "02"
subsystem: thesis
tags: [latex, abstract, terminology, captions, prose-quality]

# Dependency graph
requires:
  - phase: 04-prose-quality-discussion
    plan: "01"
    provides: Rewritten Discussion and Conclusion with canonical terminology
provides:
  - Rewritten Abstract with 6-element structure
  - Standardized canonical terminology across entire document
  - Self-contained figure and table captions
affects: [04-03, 05-final-polish]

# Tech tracking
tech-stack:
  added: []
  patterns: [self-contained-caption-pattern, canonical-terminology-standardization]

key-files:
  created: []
  modified: [thesis/main.tex]

key-decisions:
  - "All captions structured with what/conditions/takeaway per D-08"
  - "Canonical terminology enforced in all captions: template-approval planner, segmentation mask, BEV corridor"

patterns-established:
  - "Caption pattern: what is shown + under what conditions + key takeaway"
  - "Canonical terms used consistently: template-approval planner (not template arc), segmentation mask, BEV corridor, design iteration"

requirements-completed: [WRIT-02, WRIT-03, WRIT-04]

# Metrics
duration: 5min
completed: 2026-03-30
---

# Phase 04 Plan 02: Abstract, Terminology, and Captions Summary

**Task 1 (Abstract rewrite + terminology standardization) was completed in a prior session. Task 2 (self-contained captions) completed in this session.**

## Performance

- **Duration:** 5 min
- **Tasks completed this session:** 1 (Task 2: Captions)
- **Tasks completed previously:** 1 (Task 1: Abstract + terminology)
- **Files modified:** 1

## Task 1: Abstract Rewrite and Terminology Standardization (PRIOR SESSION)

Completed in a prior session. The Abstract was rewritten to 250-300 words with the prescribed 6-element structure (problem, approach, finding 1, finding 2, validation, future direction). Six canonical terms were standardized across the entire document:

1. "template-approval planner" replaced all "template arc planner" instances (except first introduction dual-name)
2. "segmentation mask" replaced non-contextual "sidewalk mask" uses
3. "BEV corridor" standardized
4. "design iteration" replaced "Phase~11" reference
5. "image-space midpoint planner" verified consistent
6. "teacher-student framework" already consistent

## Task 2: Self-Contained Captions (THIS SESSION)

### Substantive Rewrites (4 captions)

1. **Tab waypoint_turn_eval** (line ~1180): Expanded from bare "scheduled right-intent window, 300 frames" to include VID_017 conditions, stride~4, and the key takeaway of 0% containment failure rate.

### Already Rewritten by Task 1 (3 captions confirmed adequate)

2. **Tab resolution_sweep** (line ~488): Already includes 640x360 selection and accuracy-latency balance takeaway. No changes needed.
3. **Tab template_eval** (line ~1090): Already uses "template-approval planner", mentions 40.6% heading error reduction and VID_017. No changes needed.
4. **Tab runtime_configs** (line ~1158): Already includes 59 FPS vs 2.4 FPS takeaway. No changes needed.

### Minor Polish (8 captions improved)

5. **Tab video_dataset** (line ~785): Added description of sequence diversity and frame counts.
6. **Tab fullvideo_replay** (line ~897): Added baseline vs candidate teacher context and 77% instability reduction takeaway.
7. **Tab iteration_progression** (line ~932): Added BEV-to-image-space throughput improvement takeaway.
8. **Tab bev_fragility** (line ~1049): Added baseline model context and 0.7% valid path rate takeaway.
9. **Tab accepted_run** (line ~1209): Added template-approval planner 100% path availability and 9.97 FPS takeaway.
10. **Tab planner_comparison** (line ~958): Added candidate segmentation mask context and 14.3 px / 421x takeaway.
11. **Tab oracle_comparison** (line ~999): Added segmentation mask terminology and geometric-vs-segmentation takeaway.
12. **Tab runtime_merged** (line ~1129): Added per-module comparison context and 173x planner speedup driving 2.4 to 59 FPS.

### Additional Figure Caption Polish (2 figures)

13. **Fig seg_improvement** (line ~870): Added campus frames context and IoU/precision/recall metrics named.
14. **Fig planner_comparison_qual** (line ~988): Added "representative campus frames" context and "monocular BEV projection" canonical phrasing.

### Verified Adequate (no changes needed)

- Fig scooter_hw (line ~267): Already has what/conditions/takeaway structure.
- Fig pipeline (line ~400): Already self-contained with domain comparison context.
- Fig segformer_compare (line ~508): Already has resolution sweep context.
- Tab seg_comparison (line ~854): Already has teacher comparison context.
- Fig seg_comparison_qual (line ~886): Already has qualitative comparison context.
- Fig planner_comparison (line ~979): Already has bar chart context with color coding.
- Fig bev_fragility (line ~1066): Already has 99.3% takeaway.
- BEV skeleton subfigures (lines ~833, 837, 841): Parent caption handles conditions/takeaway.
- Segmentation stage subfigures (lines ~1016, 1020, 1024): Parent caption handles conditions/takeaway.
- Fig runtime_breakdown (line ~1150): Already has 173x and 59 FPS takeaway.

## Verification Results

- `grep "caption.*template arc"` returns **0 matches** (PASS)
- Tab template_eval caption mentions **40.6** (PASS)
- Tab waypoint_turn_eval caption mentions **containment** (PASS)
- Tab runtime_configs caption mentions **59 FPS** (PASS)
- All captions contain verbs (no bare noun phrases) (PASS)
- Canonical terminology used in all captions (PASS)

## Files Modified
- `thesis/main.tex` - 12 caption improvements (1 substantive rewrite, 8 minor polish, 3 confirmed adequate)

## Deviations from Plan

None. Three of the four "substantive rewrite" captions had already been improved during Task 1 (terminology standardization) in the prior session; only Tab waypoint_turn_eval required a full rewrite in this session.

## Next Phase Readiness
- Plan 04-03 (prose review pass on Ch.1-4) can proceed
- All captions are self-contained and use canonical terminology
- No table data or figure references were modified

---
*Phase: 04-prose-quality-discussion*
*Completed: 2026-03-30*
