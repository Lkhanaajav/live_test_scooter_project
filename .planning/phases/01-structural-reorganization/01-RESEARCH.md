# Phase 1: Structural Reorganization - Research

**Researched:** 2026-03-30
**Source:** `thesis/main.tex` (1323 lines)

---

## 1. Current Chapter/Section Structure with Line Numbers

### Chapter Boundaries

| # | Chapter Title | Label | Lines | Span |
|---|---------------|-------|-------|------|
| 1 | Introduction | `ch:introduction` | 197-283 | 87 lines |
| 2 | Literature Review | `ch:related_work` | 287-367 | 81 lines |
| 3 | System Design and Methodology | `ch:methodology` | 370-656 | 287 lines |
| 4 | Closed-Loop Control and System Integration | `ch:closed_loop` | 659-737 | 79 lines |
| 5 | Experiments and Results | `ch:results` | 740-1211 | 472 lines |
| 6 | Discussion | `ch:discussion` | 1214-1274 | 61 lines |
| 7 | Conclusion and Future Work | `ch:conclusion` | 1277-1323 | 47 lines |

### Section-Level Structure (All Sections)

**Chapter 1: Introduction (lines 197-283)**
- L200: `\section{Motivation}`
- L210: `\section{Problem Statement}`
- L223: `\section{Approach Overview}`
- L240: `\section{Contributions}`
- L261: `\section{Thesis Organization}`
- L266-283: Figure `fig:scooter_hw` (scooter platform photo)

**Chapter 2: Literature Review (lines 287-367)**
- L294: `\section{Autonomous Navigation for Micro-Mobility Platforms}`
- L301: `\section{Semantic Segmentation for Drivable Surfaces}` `\label{sec:lit_segmentation}`
- L309: `\section{Bird's-Eye View Projection and Monocular BEV}` `\label{sec:lit_bev}`
- L319: `\section{Path Planning for Mobile Robots}` `\label{sec:lit_planning}`
- L327: `\section{Distance Transform Methods in Navigation}`
- L334: `\section{Skeletonization and Topological Path Extraction}`
- L341: `\section{Semi-Supervised and Teacher--Student Learning}` `\label{sec:lit_teacher_student}`
- L349: `\section{Embedded and Real-Time Perception}` `\label{sec:lit_embedded}`
- L357: `\section{Summary and Research Gap}`

**Chapter 3: System Design and Methodology (lines 370-656)**
- L374: Intro paragraph + Figure `fig:pipeline_diagram` (L376-381)
- L385: `\section{Hardware Platform}` `\label{sec:hardware}`
- L392: `\section{Segmentation Module and Supervision Strategy}` `\label{sec:segmentation}`
  - L400: `\subsection{Teacher--Student Supervision}` `\label{sec:teacher_student}`
  - L422: `\subsection{Hybrid Training Dataset}`
  - L435: `\subsection{Loss Function}`
  - L444: `\subsection{Training Progression}` + Table `tab:training_progression`
- L471: `\section{Resolution Trade-Off Analysis}` `\label{sec:resolution}` + Table + Figure
- L504: `\section{Bird's-Eye View Projection}` `\label{sec:bev}`
- L517: `\section{BEV Mask Refinement}` `\label{sec:mask_refinement}`
- L532: `\section{Path Planning Methods}` `\label{sec:planners}`
  - L538: `\subsection{BEV Skeleton-Graph Planner}` `\label{sec:skeleton_planner}`
  - L548: `\subsection{BEV Distance-Transform Ridge Planner}` `\label{sec:dt_planner}`
  - L558: `\subsection{BEV Template Arc Planner}` `\label{sec:template_planner}`
  - L564: `\subsection{Image-Space Midpoint Planner}` `\label{sec:img_midpoint}`
  - L570: `\subsection{Image-Space Distance-Transform Planner}` `\label{sec:img_dt}`
  - L576: `\subsection{GPS-Conditioned Waypoint-Turn Planner}` `\label{sec:waypoint_turn_method}`
  - L592: `\subsection{Turn Containment Safety Guard}` `\label{sec:turn_containment_method}`
- L607: `\section{Temporal Smoothing}` `\label{sec:temporal_smoothing}`
- L627: `\section{Evaluation Metrics}` `\label{sec:metrics}`
- L652: `\section{Software Architecture}` `\label{sec:software_arch}`

**Chapter 4: Closed-Loop Control and System Integration (lines 659-737)**
- L663: Intro paragraph
- L666: `\section{Lightweight Object Detection}` `\label{sec:object_detection}`
  - L671: `\subsection{Monocular Distance Estimation}`
  - L679: `\subsection{Speed Modulation}`
- L692: `\section{GPS Waypoint Navigation}` `\label{sec:gps_navigation}`
- L704: `\section{Steering and Speed Computation}` `\label{sec:steer_speed}`
- L717: `\section{Serial Command Protocol}` `\label{sec:serial_protocol}`
- L727: `\section{Safety Mechanisms}` `\label{sec:safety}`

**Chapter 5: Experiments and Results (lines 740-1211)**
- L747: `\section{Experimental Setup}`
  - L749: `\subsection{Data Collection}` + Table `tab:video_dataset`
  - L774: `\subsection{Hand-Annotated Ground Truth}`
  - L779: `\subsection{Evaluation Protocol}`
- L785: `\section{Segmentation Results}` `\label{sec:res_seg}`
  - L807: `\subsection{Hand-Annotated Evaluation}` + Table + Figure
  - L834: `\subsection{Full-Video Temporal Stability}` + Table
- L858: `\section{Planner Comparison Study}` `\label{sec:planner_comparison}` + Table + Figure
  - L891: `\subsection{Key Findings}`
  - L904: `\subsection{Oracle-Mask Experiment}` + Table
- L925: `\section{BEV Skeleton-Graph Pipeline Visualization}` + Figure
- L947: `\section{BEV Fragility Analysis}` `\label{sec:bev_fragility_results}` + Table + Figure
- L979: `\section{System Runtime Analysis}` `\label{sec:res_runtime}` + Tables + Figure
- L1059: `\section{Temporal Smoothing Evaluation}` `\label{sec:smoother_results}`
- L1066: `\section{Checkpoint Benchmark}` `\label{sec:checkpoint_benchmark}` + Table
- L1092: `\section{Qualitative Results}` + Figures
- L1112: `\section{Template Arc Planner Evaluation}` `\label{sec:template_eval}` + Table
- L1142: `\section{Waypoint-Turn Planner Evaluation}` `\label{sec:waypoint_turn_eval}` + Table
- L1172: `\section{Full-Length Accepted Run}` `\label{sec:accepted_run}` + Table
- L1199: `\section{Overnight Containment Validation}` `\label{sec:overnight_eval}`

**Chapter 6: Discussion (lines 1214-1274)**
- L1220: `\section{Interpretation of Key Findings}`
- L1227: `\section{Why Image-Space Outperforms BEV}`
- L1236: `\section{Segmentation as a Necessary but Not Sufficient Condition}`
- L1241: `\section{Template Planning and Turn Safety}`
- L1250: `\section{Limitations}`
- L1267: `\section{Threats to Validity}`

**Chapter 7: Conclusion and Future Work (lines 1277-1323)**
- L1282: `\section{Summary of Contributions}`
- L1287: `\section{Key Findings}`
- L1302: `\section{Future Work}`

---

## 2. Complete Cross-Reference Inventory

### All `\label{ch:*}` Definitions (Chapter Labels)

| Line | Label | Current Chapter |
|------|-------|-----------------|
| 198 | `ch:introduction` | Ch. 1 |
| 288 | `ch:related_work` | Ch. 2 |
| 371 | `ch:methodology` | Ch. 3 |
| 660 | `ch:closed_loop` | Ch. 4 (TO BE REMOVED) |
| 741 | `ch:results` | Ch. 5 |
| 1216 | `ch:discussion` | Ch. 6 |
| 1278 | `ch:conclusion` | Ch. 7 |

### All `\ref{ch:*}` Usages

| Line | Reference | Context |
|------|-----------|---------|
| 263 | `\ref{ch:related_work}` | Thesis Organization paragraph |
| 263 | `\ref{ch:methodology}` | Thesis Organization paragraph |
| 263 | `\ref{ch:closed_loop}` | Thesis Organization paragraph |
| 263 | `\ref{ch:results}` | Thesis Organization paragraph |
| 263 | `\ref{ch:discussion}` | Thesis Organization paragraph |
| 263 | `\ref{ch:conclusion}` | Thesis Organization paragraph |

**Note:** All six chapter cross-references occur on a single line (L263) in the Thesis Organization section. This is the only location using `\ref{ch:*}` in the entire document.

### All `\ref{sec:*}` Cross-Chapter References

These are references from one chapter to a section in another chapter. They matter because merging Ch.4 into Ch.3 changes which references are "cross-chapter" vs "intra-chapter".

| Line | Reference | From Chapter | To Chapter |
|------|-----------|-------------|------------|
| 316 | `\ref{sec:bev_fragility_results}` | Ch. 2 (Lit Review) | Ch. 5 (Results) |
| 446 | `\ref{sec:res_seg}` | Ch. 3 (Methodology) | Ch. 5 (Results) |
| 513 | `\ref{sec:bev_fragility_results}` | Ch. 3 (Methodology) | Ch. 5 (Results) |
| 1115 | `\ref{sec:template_planner}` | Ch. 5 (Results) | Ch. 3 (Methodology) |
| 1145 | `\ref{sec:waypoint_turn_method}` | Ch. 5 (Results) | Ch. 3 (Methodology) |
| 1168 | `\ref{sec:turn_containment_method}` | Ch. 5 (Results) | Ch. 3 (Methodology) |
| 1238 | `\ref{tab:oracle_comparison}` | Ch. 6 (Discussion) | Ch. 5 (Results) |

### All `\ref{fig:*}` References

| Line | Reference | Context |
|------|-----------|---------|
| 237 | `\ref{fig:scooter_hw}` | Approach Overview |
| 374 | `\ref{fig:pipeline_diagram}` | Ch.3 intro |
| 388 | `\ref{fig:scooter_hw}` | Hardware Platform section |
| 1094 | `\ref{fig:seg_comparison_qual}` | Qualitative Results |
| 1094 | `\ref{fig:planner_comparison_qual}` | Qualitative Results |

### All `\ref{tab:*}` References

| Line | Reference | Context |
|------|-----------|---------|
| 446 | `\ref{tab:training_progression}` | Training Progression |
| 474 | `\ref{tab:segformer_fps}` | Resolution Trade-Off |
| 809 | `\ref{tab:seg_comparison}` | Segmentation Results |
| 836 | `\ref{tab:fullvideo_replay}` | Temporal Stability |
| 861 | `\ref{tab:planner_comparison}` | Planner Comparison |
| 906 | `\ref{tab:oracle_comparison}` | Oracle-Mask Experiment |
| 950 | `\ref{tab:bev_fragility}` | BEV Fragility |
| 982 | `\ref{tab:runtime_comparison}` | Runtime Analysis |
| 1018 | `\ref{tab:runtime_offenders}` | Runtime Offenders |
| 1039 | `\ref{tab:runtime_configs}` | Runtime Configs |
| 1069 | `\ref{tab:checkpoint_benchmark}` | Checkpoint Benchmark |
| 1175 | `\ref{tab:accepted_run}` | Accepted Run |
| 1196 | `\ref{tab:template_eval}` | Accepted Run (back-reference) |
| 1238 | `\ref{tab:oracle_comparison}` | Discussion |

---

## 3. Closed-Loop Chapter Content and Merge Mapping

### Closed-Loop Chapter Sections (Ch.4, lines 659-737)

The chapter is 79 lines and contains 5 sections:

| Section | Lines | Content Summary |
|---------|-------|-----------------|
| L666: Lightweight Object Detection | 666-690 | YOLOv8-nano setup, monocular distance estimation, speed modulation |
| L692: GPS Waypoint Navigation | 692-701 | GNSS module, NMEA parsing, heading blending equation |
| L704: Steering and Speed Computation | 704-715 | Heading angle computation, discrete commands, speed profile |
| L717: Serial Command Protocol | 717-725 | ASCII line protocol, watchdog |
| L727: Safety Mechanisms | 727-737 | 5-item safety checklist (no-path stop, obstacle stop, GPS loss, manual override, seg instability gate) |

### Recommended Merge Placement

Per D-01 and D-05 (pipeline/data-flow order), the Closed-Loop content should be merged into Chapter 3 (System Design) following the existing pipeline flow. The recommended placement is as a new top-level section at the end of Chapter 3, **before** the Software Architecture section.

**Current Ch.3 section order:**
1. Hardware Platform (L385)
2. Segmentation Module (L392)
3. Resolution Trade-Off (L471)
4. Bird's-Eye View Projection (L504)
5. BEV Mask Refinement (L517)
6. Path Planning Methods (L532)
7. Temporal Smoothing (L607)
8. Evaluation Metrics (L627)
9. Software Architecture (L652)

**Proposed Ch.3 section order after merge:**
1. Hardware Platform
2. Segmentation Module
3. Resolution Trade-Off
4. Bird's-Eye View Projection
5. BEV Mask Refinement
6. Path Planning Methods (already includes GPS-Conditioned Waypoint-Turn Planner and Turn Containment Safety Guard as subsections)
7. Temporal Smoothing
8. **Obstacle Detection and Distance Estimation** (from Ch.4 L666-690)
9. **GPS Waypoint Navigation** (from Ch.4 L692-701)
10. **Steering and Speed Computation** (from Ch.4 L704-715)
11. **Serial Command Protocol** (from Ch.4 L717-725)
12. **Safety Mechanisms** (from Ch.4 L727-737)
13. Evaluation Metrics
14. Software Architecture

**Rationale:** This follows pipeline/data-flow order per D-05. The planners produce paths; then obstacle detection modulates speed; then GPS provides global context; then steering/speed are computed from the fused heading; then the serial protocol sends commands; then safety mechanisms gate the output. Evaluation Metrics and Software Architecture are placed last as they are cross-cutting concerns, not pipeline stages.

### Alternative: Distribute by Topic

Some Ch.4 sections could be distributed rather than grouped:
- Obstacle Detection could go after BEV Mask Refinement (since YOLO operates on the camera frame, before planning)
- GPS Waypoint Navigation could go with the GPS-Conditioned Waypoint-Turn Planner subsection

However, grouping all "closed-loop" content together as a block is simpler, preserves the original narrative flow, and avoids interleaving new content with existing sections. **Recommended: group as a block.**

### Content That Already Exists in Ch.3

The GPS-Conditioned Waypoint-Turn Planner (L576-589) and Turn Containment Safety Guard (L592-604) are already subsections of the Path Planning Methods section in Ch.3. The Ch.4 GPS section (L692-701) provides complementary content (GNSS hardware, NMEA parsing, heading blending) that does not overlap with the planner subsection. No deduplication needed.

---

## 4. "Three" Iterations Lines (Must Become "Four")

### Lines Where "three" Refers to Design Iterations

| Line | Text | Fix Required |
|------|------|-------------|
| **181** | Abstract: "...progresses through **three** design iterations---from a skeleton-graph baseline through distance-transform corridor planning to a lightweight image-space architecture---" | Change "three" to "four" and extend the list to include the fourth iteration (template arc planner + GPS-conditioned turn planner) |
| **227** | Approach Overview: "The system evolved through **three** design iterations, each backward-compatible with the previous:" | Change "three" to "four". Note: the bullet list below (L228-233) already lists all four iterations correctly. Only the sentence header is wrong. |
| **245** | Contributions item 1: "...progressing through **three** design iterations with backward-compatible improvements." | Change "three" to "four" |
| **1284** | Conclusion: "...evolves through **three** design iterations, from a skeleton-graph baseline through distance-transform corridor planning to a lightweight image-space architecture." | Change "three" to "four" and extend the description to include the fourth iteration |

### Lines Where "three" Does NOT Refer to Iterations (No Change Needed)

| Line | Text | Reason |
|------|------|--------|
| 43 | `\usepackage{threeparttable}` | LaTeX package name |
| 203 | "widths vary from under one meter to **over three meters**" | Sidewalk width |
| 452, 466, 988, 1007 | `\begin{threeparttable}` / `\end{threeparttable}` | LaTeX environment |
| 535 | "**three** operating in BEV space and two in image space" | Planning method count (correct) |
| 1099 | "across **three** campus frames" | Figure caption (correct) |
| 1203 | "under **three** configurations" | Overnight eval configs (correct) |

---

## 5. LaTeX-Specific Considerations for Chapter Merging

### Label Renaming Plan

Per D-02 (fresh labels throughout), all chapter labels must be renamed to match the new 6-chapter structure:

| Old Label | New Label | Reason |
|-----------|-----------|--------|
| `ch:introduction` | `ch:introduction` | Stays Ch. 1, no change needed |
| `ch:related_work` | `ch:background` | Stays Ch. 2, rename per context doc |
| `ch:methodology` | `ch:system_design` | Stays Ch. 3, rename per context doc |
| `ch:closed_loop` | **REMOVED** | Merged into Ch. 3 |
| `ch:results` | `ch:evaluation` | Becomes Ch. 4, rename per context doc |
| `ch:discussion` | `ch:discussion` | Becomes Ch. 5, no semantic change |
| `ch:conclusion` | `ch:conclusion` | Becomes Ch. 6, no semantic change |

**Section labels:** All `sec:*` labels in the moved sections (from Ch.4) can remain unchanged since they are section-level, not chapter-level. The labels `sec:object_detection`, `sec:gps_navigation`, `sec:steer_speed`, `sec:serial_protocol`, and `sec:safety` will still resolve correctly after the move since LaTeX resolves labels globally.

**No section labels need renaming** because section labels are already semantic (e.g., `sec:object_detection`, not `sec:ch4_detection`).

### Heading Level Changes

None needed. The Closed-Loop content is already `\section`-level within Ch.4. When moved to Ch.3, these sections remain `\section`-level. The subsections (`\subsection`) within them also remain unchanged.

### Page Break Handling

The `\chapter{}` command on L659 automatically inserts a page break. When this is removed (content merged into Ch.3), the page break disappears. This is correct behavior -- no manual `\newpage` should be inserted. The content will flow naturally after the last section of the current Ch.3.

### Chapter Intro Paragraph

The current Ch.4 intro paragraph (L663):
> "This chapter extends the offline perception pipeline to a real-time system capable of autonomous sidewalk navigation. We describe four additions: obstacle detection, GPS waypoint navigation, steering and speed computation, and the serial command protocol."

This paragraph should be **removed or converted to a transition sentence** when merged into Ch.3. The "four additions" framing no longer makes sense as a standalone paragraph within a larger chapter.

### Thesis Organization Section Rewrite

Line 263 contains the entire thesis organization paragraph with all `\ref{ch:*}` references. This must be **completely rewritten** to reflect the new 6-chapter structure:

```
Chapter~\ref{ch:background} reviews related work...
Chapter~\ref{ch:system_design} describes the proposed perception pipeline...
  [merged: including obstacle detection, GPS navigation, and closed-loop control]
Chapter~\ref{ch:evaluation} reports experimental results...
Chapter~\ref{ch:discussion} discusses implications...
Chapter~\ref{ch:conclusion} concludes...
```

### Chapter Title Changes

| Old Title | New Title |
|-----------|-----------|
| `Literature Review` | `Background and Related Work` (or keep as-is, user decision) |
| `System Design and Methodology` | `System Design` (per ARCHITECTURE.md) |
| `Experiments and Results` | `Experimental Evaluation` (per ARCHITECTURE.md) |
| `Conclusion and Future Work` | `Conclusion and Future Work` (no change) |

### Ch.3 Intro Paragraph Update

The current Ch.3 intro (L374) says:
> "This chapter describes the perception pipeline that transforms monocular RGB frames into controller-ready waypoints."

After merging, this should be broadened to include the control components, e.g.:
> "This chapter describes the complete navigation system, from monocular RGB perception through path planning to closed-loop vehicle control."

### Figure Pipeline Diagram Update

The pipeline overview figure (L376-381, `fig:pipeline_diagram`) caption mentions "The system integrates obstacle detection and GPS navigation for closed-loop autonomous control." This still works after the merge since the content is now in the same chapter as the figure. No caption change needed.

---

## 6. Recommended Merge Order and Execution Sequence

### Step 1: Move Closed-Loop Content into Ch.3

1. Remove the `\chapter{Closed-Loop Control and System Integration}` line (L659) and `\label{ch:closed_loop}` (L660)
2. Remove or convert the Ch.4 intro paragraph (L663-664)
3. Insert the 5 sections (L666-737) into Ch.3 between Temporal Smoothing (ends ~L625) and Evaluation Metrics (starts L627)
4. Add a brief transition comment or sentence before the inserted block

### Step 2: Rename Chapter Labels

1. L288: `ch:related_work` -> `ch:background`
2. L371: `ch:methodology` -> `ch:system_design`
3. L741: `ch:results` -> `ch:evaluation`
4. Delete L660: `ch:closed_loop` (already removed in Step 1)

### Step 3: Update All `\ref{ch:*}` References

All on L263 (Thesis Organization section). Rewrite the entire paragraph.

### Step 4: Fix "Three" -> "Four" Iterations

1. L181 (Abstract): "three" -> "four", extend iteration list
2. L227 (Approach Overview): "three" -> "four"
3. L245 (Contributions): "three" -> "four"
4. L1284 (Conclusion): "three" -> "four", extend iteration list

### Step 5: Update Chapter Titles

1. L287: `Literature Review` -> `Background and Related Work`
2. L370: `System Design and Methodology` -> `System Design`
3. L740: `Experiments and Results` -> `Experimental Evaluation`

### Step 6: Update Chapter Intro/Outro Paragraphs

1. L374: Broaden Ch.3 intro to cover control components
2. L744: Optionally update Ch.4 (new) intro if it references "Chapter 4" by number

---

## 7. Potential Issues and Risks

### Issue 1: Duplicate Section on GPS

Ch.3 already has a subsection "GPS-Conditioned Waypoint-Turn Planner" (L576-589) describing GPS intent for turn planning. Ch.4 has a section "GPS Waypoint Navigation" (L692-701) describing the GNSS hardware and heading blending. These are complementary, not redundant, but a reader might perceive overlap. **Mitigation:** A brief forward-reference from the planner subsection to the GPS infrastructure section, or vice versa.

### Issue 2: Ch.3 Becomes Very Long

Current Ch.3 is 287 lines. Adding 79 lines from Ch.4 makes it ~366 lines. This is the longest chapter but is expected for System Design per ARCHITECTURE.md (14-18 pages). The chapter has clear section boundaries so length is manageable.

### Issue 3: Abstract Iteration List

L181 currently describes three iterations. The fourth iteration description exists in the body (L232-233) but not the Abstract. Fixing the Abstract requires carefully adding the template arc planner and waypoint-turn planner without making the Abstract too long.

### Issue 4: Contribution #6 References Closed-Loop

Contribution #6 (L255): "Integration of YOLOv8-nano obstacle detection, GPS waypoint navigation, and a serial command protocol for closed-loop scooter control." This contribution still makes sense after the merge -- no change needed.

---

*Phase: 01-structural-reorganization*
*Research completed: 2026-03-30*
