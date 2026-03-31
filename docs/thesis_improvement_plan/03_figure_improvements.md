# Figure Improvement Plan -- Thesis Visualization Audit

**Date:** 2026-03-31
**Scope:** All figures in `thesis/main.tex` and unused assets in `thesis/figures/`

---

## Part 1: Audit of Existing Figures

### Figure 1 -- Scooter Hardware (fig:scooter_hw)

**Files:** `scooter/scooter-image1.jpeg`, `scooter/input_image.jpeg`
**Current format:** JPEG photos, two subfigures side by side at 0.48\textwidth
**Used in:** Chapter 1 (Introduction), line 281

**Issues:**

- **JPEG artifacts.** Both images are compressed JPEGs. The scooter photo (3.0 MB) is high-resolution and passable, but `input_image.jpeg` (260 KB) shows visible compression rings around high-contrast edges (tree branches against sky).
- **Mismatched aspect ratios.** The scooter photo is portrait-oriented; the camera view is landscape. When placed side by side at equal width, one will have substantially more white space than the other.
- **No annotations on the scooter photo.** The caption mentions "forward-facing monocular camera rigidly mounted to the handlebar stem" but the camera is small and hard to identify without a callout arrow or circle.
- **No hardware spec callout.** For a thesis that emphasizes the hardware constraint (15 W, ARM SBC), the photo could include an inset or label showing the compute board.

**Verdict:** Acceptable with minor fixes. Not a blocker.

**Recommended fixes:**

1. Re-export both as PNG to eliminate JPEG ringing.
2. Add a red circle or arrow annotation pointing to the camera mount on scooter-image1.
3. Equalize subfigure heights (use `[height=5cm]` instead of `[width=\linewidth]`) so both subfigures have consistent vertical extent.


### Figure 2 -- Pipeline Diagram (fig:pipeline_diagram)

**File:** `pipeline/pipeline_graph.jpg`
**Current format:** JPG, 242 KB, rasterized flowchart
**Used in:** Chapter 3 (System Design), line 444

**Issues:**

- **CRITICAL: JPG format for a vector-style diagram.** This is a block-and-arrow flowchart rendered as a lossy JPEG. Text labels ("SegFormer-B0", "11.7 ms") show visible compression blur. This is the single worst figure quality problem in the thesis.
- **Small text at full width.** When printed at `\textwidth`, the small annotation text (e.g., "0.7 vis + 0.3 gps") becomes illegible.
- **Visual density is good.** The diagram itself is well-structured: training pipeline at top, image-space primary path in middle, BEV fallback at bottom. The color coding and legend are clear.
- **Better alternatives exist but are unused.** `pipeline/pipeline_overview.png` and `pipeline/pipeline_detailed.png` are higher-resolution PNGs with the same information. `pipeline_detailed.png` (469 KB) includes timing annotations and teacher-student training detail.

**Verdict:** Must be replaced.

**Recommended fix:**

1. Replace `pipeline_graph.jpg` with `pipeline/pipeline_detailed.png` or `pipeline/pipeline_overview.png` -- both are PNG, higher resolution, and contain more complete information.
2. If the detailed version is too dense at `\textwidth`, use `pipeline_overview.png` for the System Design chapter and add `pipeline_detailed.png` as a second figure or appendix figure.
3. Ideal: re-render as vector (TikZ or PDF export from draw.io). If not feasible, the PNG versions are adequate at 300+ DPI.


### Figure 3 -- SegFormer Resolution Comparison (fig:segformer_compare)

**File:** `resolution/segformer_compare_20251105_154023.png`
**Current format:** PNG, 1.1 MB, grid of 8 resolution pairs (overlay + mask)
**Used in:** Chapter 3 (System Design), line 557

**Issues:**

- **Individual panels are very small at \textwidth.** Eight resolution pairs (16 images) in one figure means each panel is about 1.5 cm wide when printed. The mask detail that the figure is supposed to demonstrate is invisible at this size.
- **Text labels overlap with image content.** The resolution/latency annotation (e.g., "640x360 | 46.0 ms") is rendered on top of the camera image in small white text. Legibility is poor.
- **Green overlay is garish.** The bright green segmentation overlay obscures the underlying image, making it hard to assess boundary quality. A semi-transparent overlay or boundary-only rendering would be more informative.
- **No arrow or highlight showing the selected operating point.** The 640x360 resolution is the chosen one but gets no visual emphasis.

**Verdict:** Needs improvement. Consider restructuring into fewer panels or highlighting the selected resolution.

**Recommended replacement:**

- Reduce to 3--4 key resolutions (160x90, 320x180, 640x360, 1280x720) displayed in a 2x2 grid. Each panel shows the mask boundary overlaid on the image (not a filled green region). Label each panel clearly with resolution, IoU (if available), and latency.
- Highlight the 640x360 panel with a thicker border or a star annotation.
- Alternatively, pair this with a line plot of IoU vs. resolution and latency vs. resolution (dual-axis or separate subplots) that makes the knee point visually obvious.


### Figure 4 -- Segmentation Pipeline Stages (fig:segmentation_stage)

**Files:** `bev_src_points.png`, `seg_mask_raw.png`, `bev_mask_raw.png`
**Current format:** PNG subfigures at 0.32\textwidth each
**Used in:** Chapter 4 (Evaluation), line 904

**Issues:**

- **`bev_src_points.png` has debug overlay text baked in.** The image contains small-font text at the top ("BEV Source Points (Frame 0)") and bottom ("src_points = np.array(...)") that is illegible at subfigure width and looks unprofessional in a thesis.
- **Aspect ratio mismatch.** The RGB input (bev_src_points) is 16:9 landscape; the raw seg mask is also landscape; the BEV mask is roughly square. At equal widths, the BEV mask will appear stretched or will have excess vertical whitespace.
- **BEV mask (bev_mask_raw.png) is very low resolution.** 480x600 pixels, will look pixelated when scaled up in the subfigure.
- **Color inconsistency.** The RGB input has colored calibration points overlaid (yellow circles, blue trapezoid). These are specific to BEV calibration, not to segmentation. The caption says "RGB input" but the image is actually "BEV calibration visualization."

**Verdict:** Needs replacement of subfigure (a).

**Recommended fix:**

1. Replace `bev_src_points.png` with a clean camera frame (e.g., `scooter/input_image.jpeg` re-exported as PNG, or a frame from the evaluation video without debug overlays).
2. Re-render `bev_mask_raw.png` at higher resolution (at least 640x800) by re-running the BEV transform on the same frame.
3. Match aspect ratios by cropping or padding so all three subfigures align vertically.


### Figure 5 -- Segmentation Improvement Bar Chart (fig:seg_improvement)

**File:** `seg_improvement.png`
**Current format:** PNG, 128 KB, four-panel grouped bar chart
**Used in:** Chapter 4 (Evaluation), line 944

**Issues:**

- **Bar chart is the wrong chart type for this comparison.** With only two conditions (Baseline vs. Student), a grouped bar chart wastes space. The reader must visually compare bar heights across four panels. A paired-point plot (Cleveland dot plot) or a compact table is more informative for a 2-condition comparison.
- **No error bars or confidence intervals.** The IoU and other metrics come from 32 frames. Variability should be shown.
- **Y-axis scales are misleading.** All four panels start at 0, but the IoU panel compresses the interesting range (0.75--0.95) into a small band. For inference time and temporal instability, the absolute scale is fine, but the visual impact is that IoU appears to be a modest improvement.
- **Color choice is fine.** Blue vs. orange is colorblind-safe.
- **Annotation text ("+25%", "38%", "77%", "+8%") is useful** but should be in the body text or table, not baked into the figure.

**Verdict:** Consider replacing with a more compact, informative visualization. The same data appears in Table 5 (tab:seg_comparison) and Table 7 (tab:fullvideo_replay), making this figure redundant if the tables are well-formatted.

**Recommended replacement:**

- Option A: Remove this figure entirely. Tables 5 and 7 already contain all the data. Use the figure slot for something more valuable (e.g., training curves).
- Option B: Replace with a Cleveland dot plot (paired dots connected by a line) showing all five metrics on a single panel with a normalized or ranked axis. This is more compact and more informative than four separate bar charts.
- Option C: If keeping a bar chart, add error bars from the 32-frame per-frame distribution and use a broken y-axis for IoU to show the improvement range more clearly.


### Figure 6 -- Qualitative Segmentation Comparison (fig:seg_comparison_qual)

**File:** `seg_comparison_example.jpg`
**Current format:** JPEG, 2.2 MB, 2-row x 3-column grid
**Used in:** Chapter 4 (Evaluation), line 960

**Issues:**

- **JPEG format for a segmentation comparison.** Compression artifacts can obscure the very boundary differences the figure is meant to highlight. Should be PNG.
- **Mask overlay is semitransparent.** The green overlay is somewhat transparent, which is better than fully opaque, but at the small subfigure size (each panel about 4 cm wide) the boundary differences between baseline and student are hard to see.
- **Good frame selection.** Frame 30 (pedestrian occlusion), Frame 190 (straight corridor), Frame 200 (tree shadows). These cover relevant failure/success cases.
- **No boundary-difference highlight.** The most informative visualization would highlight the pixels that differ between the two models (red for baseline-only, blue for student-only) or show boundary traces rather than filled masks.

**Verdict:** Acceptable but could be significantly improved.

**Recommended fix:**

1. Re-export as PNG.
2. Add a third row showing the difference between the two masks (student minus baseline), color-coded to highlight where the student improves (green = student added correct pixels) and where it regresses (red = student lost correct pixels).
3. Alternatively, show only the boundaries (not filled masks) overlaid on the original image, with baseline in red and student in blue/green.


### Figure 7 -- Planner Comparison Bar Chart (fig:planner_comparison)

**File:** `planner_comparison.png`
**Current format:** PNG, 187 KB, three-panel bar chart
**Used in:** Chapter 4 (Evaluation), line 1055

**Issues:**

- **Bar chart is suboptimal for 5 planners.** The three panels (Center Error, Mask Alignment, Latency) each show 5 bars. The center error and latency panels are dominated by BEV methods, compressing the image-space values to tiny bars that are hard to read.
- **No error bars or whiskers.** The metrics come from 32 frames. Variability is critical: does image-space midpoint *always* win, or does it win *on average* while failing on some frames?
- **Latency panel has extreme dynamic range.** BEV DT at 926.8 ms makes the image-space bars (2.2 ms, 108 ms) nearly invisible. A log-scale y-axis would be more informative.
- **Data values in the figure are inconsistent with the table.** The figure shows "52.3" for BEV Skeleton center error, but Table 9 (tab:planner_comparison) says "76.6". The figure shows "BEV DT 65.0" which matches. The BEV Skeleton discrepancy suggests the figure was generated from a different evaluation run than the table. **This is a factual error that must be corrected.**
- **Template Arc is included in the figure but not in Table 9** (which only has five planners). The figure adds a sixth bar that does not correspond to the table. This needs reconciliation.
- **Color coding is good.** BEV methods in blue/orange, image-space in green/red/brown provides clear domain grouping.

**Verdict:** Must be regenerated for data consistency and should use error bars. The data discrepancy is a credibility risk.

**Recommended replacement:**

- Box plots (or box + strip/swarm) for Center Error and Mask Alignment, showing per-frame distributions across the 32 frames. This reveals whether the improvement is consistent or driven by a few frames.
- Log-scale bar chart or side-by-side comparison for latency (since latency has no per-frame "distribution" to show -- it is measured as mean wall time).
- Reconcile all data values with Table 9.
- If Template Arc is included, add it to Table 9 as well.


### Figure 8 -- Qualitative Planner Comparison (fig:planner_comparison_qual)

**File:** `planner_comparison_example.png`
**Current format:** PNG, 2.8 MB, three-panel comparison (BEV skeleton, Image midpoint, Template arc)
**Used in:** Chapter 4 (Evaluation), line 1064

**Issues:**

- **Panels are very small.** Three full-frame images side by side at \textwidth means each is about 5 cm wide. The path overlay details are hard to see.
- **BEV skeleton panel is grayscale/dark.** Shows the BEV mask with sparse white skeleton, which is visually distinct from the camera views in panels (b) and (c). The domain difference makes visual comparison of path quality difficult.
- **Caption text in the image** ("Sparse coverage", "Noisy skeleton") is useful but baked into the figure rather than referenced from the caption.
- **No shared-frame comparison.** Panels (b) and (c) appear to show the same frame, but panel (a) shows a BEV view. A better comparison would show all three planners on the same camera frame.

**Verdict:** Acceptable but could be more informative.

**Recommended fix:**

1. Show all five planners on the same frame. Use two rows: top row = BEV-domain paths overlaid on the BEV mask; bottom row = image-space paths overlaid on the camera view. This provides a direct visual comparison.
2. Or: show the camera view with paths from all planners overlaid in different colors on a single panel, with a legend. This is more compact and directly shows the path differences.


### Figure 9 -- Skeleton Pipeline Stages (fig:skeleton_stage)

**Files:** `bev_skeleton.png`, `skeleton_paths_overlay.png`, `cam_paths_0001.png`
**Current format:** PNG subfigures at 0.32\textwidth each
**Used in:** Chapter 4 (Evaluation), line 1091

**Issues:**

- **`bev_skeleton.png` is very large (866 KB) for a mostly black image.** The skeleton is visible but the image is mostly empty black space with a thin white skeleton and green mask.
- **`cam_paths_0001.png` (837 KB) has debug-style rendering.** Overlaid path with colored segments and small text labels. The text will be illegible at 0.32\textwidth.
- **`skeleton_paths_overlay.png` is tiny (13 KB, 360x288).** This will be visibly pixelated when scaled to the subfigure width.
- **The figure effectively demonstrates the skeleton pipeline** (thin, prune, search, reproject). This is a good teaching figure.

**Verdict:** Needs resolution improvement on subfigure (b). Remove debug text from (c).

**Recommended fix:**

1. Re-render `skeleton_paths_overlay.png` at higher resolution (at least 720x576).
2. Re-render `cam_paths_0001.png` without debug text overlays.
3. Consider adding a fourth panel showing the distance-transform heatmap for comparison with the skeleton approach.


### Figure 10 -- BEV Fragility Analysis (fig:bev_fragility)

**File:** `bev_fragility.png`
**Current format:** PNG, 195 KB, three-panel figure (histogram, pie chart, time series)
**Used in:** Chapter 4 (Evaluation), line 1144

**Issues:**

- **CRITICAL DATA MISMATCH with text.** The thesis text says "99.3% of frames produced no valid BEV path" and "mean BEV mask occupancy ratio was 0.0002." But the figure shows: (a) histogram with median occupancy 0.861, (b) pie chart with 98.3% valid path (295 frames) and only 1.7% no path (5 frames), (c) time series showing occupancy ratio hovering around 0.85. **The figure shows the opposite of what the text claims.** This is either from a different dataset/run than the 4,407-frame profiled run described in the text, or the figure was generated from the candidate model while the text describes the baseline model. Either way, this is a serious inconsistency that must be resolved.
- **Pie chart is a poor choice.** Pie charts are generally discouraged in scientific publications. A stacked bar or waffle chart would be more precise.
- **The histogram is informative** but the x-axis range (0.82--0.87) is very narrow, suggesting the data is from a stable run -- contradicting the fragility narrative entirely.

**Verdict:** MUST be replaced or the text must be corrected. This is the most serious factual discrepancy in the thesis.

**Recommended fix:**

1. **Priority 1: Resolve the data discrepancy.** Determine whether the text or the figure is correct. If 99.3% failure is real (from the 4,407-frame baseline run), regenerate the figure from that data. If 98.3% validity is real (from the candidate model), update the text.
2. Replace the pie chart with a stacked bar or just report the numbers in text (three values do not need a chart).
3. If two different runs are being discussed (baseline = 99.3% failure, candidate = 98.3% success), show both runs side by side as a before/after comparison. This would be much more compelling.
4. The time series panel (c) is the most valuable subpanel. If regenerated from the correct data, it should show the occupancy dropping near zero for the baseline run.


### Figure 11 -- Runtime Breakdown (fig:runtime_breakdown)

**File:** `runtime_breakdown.png`
**Current format:** PNG, 147 KB, two-panel (stacked horizontal bar + pie chart)
**Used in:** Chapter 4 (Evaluation), line 1232

**Issues:**

- **Pie chart is redundant.** It shows the same proportional data as the stacked bar. Publishing two views of the same data wastes space.
- **The stacked horizontal bar is a good choice** for showing cumulative latency breakdown.
- **Only one pipeline configuration is shown.** The text and Table 11 (tab:runtime_merged) compare BEV pipeline vs. image-space pipeline, but the figure only shows a single pipeline configuration (93.3 ms, 10.7 FPS). The figure should show both configurations side by side.
- **"Other" category is ambiguous.** What does "Other (0.4 ms)" include? This should be labeled more specifically.

**Verdict:** Needs improvement for publication quality.

**Recommended replacement:**

- Paired horizontal stacked bars: one for BEV pipeline (416 ms total), one for image-space pipeline (16.9 ms total). This visually demonstrates the 25x speedup claim.
- Remove the pie chart.
- Use a log-scale x-axis or a broken axis if the BEV bar makes the image-space bar invisible. Alternatively, show them on separate axes with aligned module colors.
- Add the 100 ms real-time threshold as a vertical dashed line.


---

## Part 2: Unused Figures -- Inclusion Recommendations

### training_curves.png

**Content:** Two-panel plot. Left: training and validation loss curves over 8 epochs for two training configurations. Right: validation IoU over 8 epochs.
**Quality:** Good. Clean matplotlib rendering, clear legends, distinct line styles (solid vs. dashed), dual-run comparison.

**Issues:**

- Legend text is truncated ("binary_segformer_all6_t400_..."). Should show full readable names ("All-6 Teacher 400", "Old-400 Plus").
- Only 8 epochs shown. The curves have not fully plateaued, which raises questions about training convergence.
- Y-axis range on loss (0.07--0.12) and IoU (0.90--1.00) is well chosen but the IoU subplot is slightly misleading (starts at 0.90, not 0.00).

**Recommendation:** INCLUDE. Add to Section 4.2 (Teacher-Student Training) or Section 3.2 (Segmentation Module). This provides the convergence evidence that reviewers will expect. Fix the legend labels before inclusion.

**Placement:** After Table 3 (tab:training_progression), as evidence that the training was well-behaved and did not overfit.

**LaTeX:**
```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\textwidth]{training_curves.png}
  \caption{Training and validation loss (left) and validation IoU (right)
           over eight epochs for two training configurations.
           Both configurations converge smoothly without overfitting,
           with the expanded-data model (orange) achieving higher IoU
           throughout training.}
  \label{fig:training_curves}
\end{figure}
```


### heading_analysis.png

**Content:** Two-panel plot. Left: raw vs. smoothed heading angle over 300 frames (time series). Right: histogram of frame-to-frame heading jitter.
**Quality:** Clean rendering, clear distinction between raw (gray) and smoothed (blue) lines.

**Issues:**

- **The data is too "clean."** Both raw and smoothed heading are nearly identical (mean jitter = 0.01 deg for both), making the smoothing appear to have no effect. This is because the 300-frame clip is mostly straight-ahead with one turn event (frames 20--30). The heading analysis should ideally show a clip with more heading variation.
- **Histogram is nearly a single spike at 0.** Uninformative because the data does not exercise the smoother.

**Recommendation:** INCLUDE ONLY if regenerated from a more varied video clip (one with curves, turns, and heading variation). In its current form, it demonstrates that the system is stable on straight paths, which is expected and not very informative.

**Alternative:** Regenerate using a clip from IMG_1878 (T-junctions, surface changes) or IMG_1931 (narrow paths, tight turns) to show the smoother's effect on real heading noise.

**Placement:** Section 4.7 (Temporal Smoothing Evaluation).


### latency_violin.png

**Content:** Violin plot showing pipeline latency distributions across seven evaluation runs, with a 100 ms (10 Hz) threshold line.
**Quality:** Good matplotlib rendering with violins, quartile lines, and threshold annotation.

**Issues:**

- **X-axis labels are truncated and cryptic.** "smoke_017_right_igno" is not meaningful to a reader. Should use human-readable labels ("40-frame smoke test", "300-frame final", "300-frame scheduled", etc.).
- **The smoke test runs (first three) dominate visually** with their wide high-latency distributions (400--700 ms), but they are engineering artifacts, not representative of the final system. Including them confuses the message.
- **The final configurations (vid017_right_300_fin, vid017_schedule_300) show tight distributions below 100 ms.** This is the important result.

**Recommendation:** INCLUDE after filtering to show only the final, representative runs. Remove smoke test runs.

**Placement:** Section 4.6 (Runtime Analysis), as a companion to the runtime breakdown figure. The violin plot shows the distribution where the stacked bar shows the mean. Together they tell a complete latency story.

**Regeneration approach:** Filter to 3--4 runs: the final 300-frame runs and the 1800-frame accepted run. Clean up axis labels.


### temporal_stability.png

**Content:** Three-panel time series: segmentation IoU over time, pipeline throughput (FPS) over time, corridor confidence over time. 300 frames.
**Quality:** Adequate matplotlib rendering. The three-panel vertical layout is compact.

**Issues:**

- **Seg IoU panel is boring.** Shows a flat line at 1.0 for the entire sequence. This means the evaluation clip has perfect segmentation stability, which is good but visually uninformative.
- **FPS panel is the most useful.** Shows an initial drop to ~4 FPS in the first 50 frames (model warmup) followed by stable ~12 FPS. The 10 Hz threshold line is clearly shown.
- **Corridor confidence panel** shows subtle variations (0.956--0.960) that are hard to interpret without more context.
- **Figure size.** At the current rendering, text labels are very small.

**Recommendation:** INCLUDE but restructure. The FPS time series is the most publication-worthy component. Consider including only the FPS panel as a standalone figure, or pair it with a heading time series instead of the (uninformative) IoU=1.0 panel.

**Placement:** Section 4.6 (Runtime Analysis) or Section 4.5 (Complete Pipeline Validation).


### gps_intent_architecture.png

**Content:** Block diagram showing GPS intent conditioning architecture with normal mode (straight segments) and turn mode (junction activation). Includes decision band scan, support clustering, target gating, containment gate, and safety guards.
**Quality:** High-quality color-coded block diagram (407 KB PNG). Clean layout with two major sections (Normal Mode, Turn Mode), color-coded by function (blue=normal, orange=turn plan, purple=GPS, green=execute, red=safety).

**Issues:**

- **Dense text in small blocks.** Some annotation text (e.g., "total: 60%, near: 70%", "turn: 0.40, fwd: 0.25") will be illegible at \textwidth.
- **Good conceptual diagram** that corresponds directly to the five-stage waypoint-turn planner description in Section 3.7.

**Recommendation:** INCLUDE. This diagram clarifies the complex turn planner architecture described in Section 3.7 (GPS-Conditioned Waypoint-Turn Planner). It is more useful than a text-only description for understanding the containment gate flow.

**Placement:** Section 3.7 or Section 3.8 (Turn Containment Safety Guard), after the five-stage enumeration.


### bev_transform_process.png

**Content:** Block diagram showing the complete BEV transform pipeline: Camera Seg Mask -> 4-Point Correspondences -> Homography H -> cv2.warpPerspective -> Raw BEV Mask -> Morphological Cleanup Chain (with optional enhanced mode) -> Clean BEV Mask. Includes BEV coordinate system inset and key parameters box.
**Quality:** High-quality color-coded diagram (388 KB PNG). Clear flow with annotations.

**Issues:**

- **Overlaps with the pipeline diagram.** The pipeline diagram already shows the BEV transform step. This figure adds detail on the morphological cleanup chain that is described in Section 3.6.
- **The key parameters box is useful** (BEV resolution, physical extent, scale factor, ego position, origin).

**Recommendation:** INCLUDE. Place in Section 3.5 (Bird's-Eye View Projection) or Section 3.6 (BEV Mask Refinement) to visually complement the text description of the morphological cleanup stages.

**Placement:** Section 3.6, before the three-step enumeration.


### planner_comparison_arch.png

**Content:** Architectural diagram showing all five planners side by side, with processing steps, domain labels (BEV/Image-Space), and key metrics (runtime, lateral error, inside-GT) annotated per planner. Includes selection logic box.
**Quality:** High-quality color-coded diagram (427 KB PNG). Excellent visual summary of all five planners.

**Issues:**

- **Data in the figure does not match Table 9.** The figure shows "Inside: 89.9%" for BEV Skeleton, but Table 9 says 97.1%. The figure shows "Lat err: 76.6 px" for BEV Skeleton, matching the table. "Inside: 97.3%" for BEV DT matches approximately. **Check all values against Table 9 before including.**
- **Template bank says "15 arcs"** but the text says "seven predefined arc templates" (Section 3.5.3). Discrepancy.

**Recommendation:** INCLUDE after correcting data values. This is an excellent companion to Table 9 (tab:planner_comparison), providing the architectural context that the table cannot convey.

**Placement:** Section 3.7 (Path Planning Methods), at the start of the section, before the individual planner descriptions. Or in Chapter 4 alongside Table 9.


### planned_vs_skeleton_overlay.png

**Content:** BEV mask visualization showing template paths vs. skeleton paths overlaid on the BEV mask. Uses distinct colors (cyan, yellow, lime green, magenta) for different path types.
**Quality:** Low resolution (480x600), no labels or legend. Raw debug output.

**Issues:**

- **No legend or axis labels.** The colors are not explained.
- **Low resolution.** Will appear pixelated in print.
- **Raw debug output** -- not publication-ready.

**Recommendation:** DO NOT INCLUDE in current form. The concept is valuable (showing template vs. skeleton paths on the same mask) but needs to be re-rendered with a legend, axis labels, and higher resolution.


### fps_comparison.png

**Content:** Bar chart with error bars showing pipeline FPS across seven evaluation runs. Red dashed line at 10 Hz threshold.
**Quality:** Clean matplotlib rendering with error bars. Color-coded bars.

**Issues:**

- **Same data as latency_violin.png** but in bar chart form. The violin plot is strictly more informative.
- **X-axis labels have the same truncation problem** as the violin plot.
- **Error bars extend below zero** on the first three runs, which is physically impossible for FPS. This suggests the error bars represent standard deviation, which is inappropriate for a non-negative metric. Use IQR or min-max whiskers instead.

**Recommendation:** DO NOT INCLUDE. The violin plot (latency_violin.png) conveys the same information with more detail. Avoid duplicate representations of the same data.


---

## Part 3: New Figures to Create

### NEW-1: Per-Frame Planner Center Error Box Plots

**Purpose:** Replace the current planner_comparison.png bar chart with a visualization that shows the distribution of center error across the 32 hand-annotated frames, not just the mean.

**Chart type:** Grouped box plots with individual data points (strip/swarm overlay).

**Data needed:** Per-frame center error for each of the five planners (32 values per planner).

**Approach:**
```python
# seaborn boxplot + stripplot overlay
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
sns.boxplot(data=df, x='planner', y='center_error_px', palette=domain_colors)
sns.stripplot(data=df, x='planner', y='center_error_px', color='black',
              size=3, alpha=0.4, jitter=True)
ax.set_ylabel('Lateral Center Error (px)')
# Group by domain with a vertical separator
```

**Why better:** Shows whether the image-space advantage is consistent frame-by-frame or driven by a few outlier frames. Reviewers will ask this question. Error bars on bar charts cannot answer it; box plots with individual points can.

**Placement:** Replace fig:planner_comparison in Section 4.3.


### NEW-2: Latency Comparison with Log Scale

**Purpose:** Show planner latency across all five methods on a scale that makes both the 2.2 ms and 926.8 ms values visible.

**Chart type:** Horizontal bar chart with log-scale x-axis, or paired-domain grouped bar chart on log scale.

**Data needed:** Mean latency per planner from Table 9.

**Approach:**
```python
fig, ax = plt.subplots(figsize=(6, 4))
ax.barh(planner_names, latencies, color=domain_colors)
ax.set_xscale('log')
ax.axvline(x=100, color='red', linestyle='--', label='10 Hz threshold')
for i, (name, val) in enumerate(zip(planner_names, latencies)):
    ax.text(val * 1.1, i, f'{val:.1f} ms', va='center')
```

**Why better:** The current linear-scale bar chart makes the image-space latencies invisible. A log scale reveals the full dynamic range.

**Placement:** Section 4.3 or 4.6, as a subplot alongside the box plots.


### NEW-3: Dual-Pipeline Runtime Stacked Bars

**Purpose:** Directly visualize the BEV pipeline (416 ms) vs. image-space pipeline (16.9 ms) runtime composition, corresponding to Table 11.

**Chart type:** Two horizontal stacked bars, one per pipeline, with module-level color coding and a 100 ms threshold line.

**Data needed:** Table 11 (tab:runtime_merged) values.

**Approach:**
```python
modules = ['SegFormer', 'Mask Refinement', 'BEV Projection',
           'BEV Cleanup', 'Planner']
bev_times =   [11.7, 8.5, 0.9, 14.6, 380.3]
img_times =   [11.7, 3.0, 0.0, 0.0,  2.2]
# Horizontal stacked bar with shared y-axis
# Add vertical dashed line at x=100 for 10 Hz threshold
```

**Why better:** The current runtime_breakdown.png shows only one pipeline. The key thesis argument is the BEV-vs-image-space speedup, so visualizing both side by side makes the argument visually immediate.

**Placement:** Replace fig:runtime_breakdown in Section 4.6.


### NEW-4: BEV Fragility Corrected Figure (Before/After)

**Purpose:** Replace the current bev_fragility.png with a figure that matches the text and tells the fragility story clearly.

**Chart type:** Two-column comparison. Left column: baseline model BEV analysis (99.3% no-path). Right column: candidate model BEV analysis (98.3% valid path). Each column shows a histogram of BEV occupancy and a stacked bar of path validity.

**Data needed:** Per-frame BEV occupancy and path validity from both the 4,407-frame baseline run and the 300-frame candidate run.

**Approach:**
```python
fig, axes = plt.subplots(2, 2, figsize=(10, 6))
# Top row: histograms of BEV mask occupancy
# Bottom row: stacked bars of path outcome
# Left column: baseline (4,407 frames, 99.3% no path)
# Right column: candidate (300 frames, 98.3% valid)
```

**Why better:** Resolves the current data discrepancy. Shows both sides of the story -- BEV is fragile under the baseline but recoverable with better segmentation. This actually strengthens the argument.

**Placement:** Section 4.4 (BEV Fragility).


### NEW-5: Per-Video Metric Breakdown (Faceted)

**Purpose:** Show how segmentation IoU, temporal stability, and template success rate vary across the six evaluation videos, providing evidence that the results are not driven by a single easy video.

**Chart type:** Faceted grouped bar or heatmap. Rows = videos, columns = metrics.

**Data needed:** Per-video breakdown of the full-video replay metrics (Table 7 has aggregate; per-video data presumably exists in the evaluation scripts).

**Approach:**
```python
# Heatmap with annotated cell values
fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(df_per_video, annot=True, fmt='.3f', cmap='RdYlGn',
            xticklabels=metrics, yticklabels=video_ids)
```

**Why better:** Table 7 only shows aggregate metrics across 22,679 frames. Reviewers may wonder if one easy video drives the mean. A per-video breakdown addresses this threat to validity.

**Placement:** Section 4.2 (Full-Video Temporal Stability) or as a supplementary figure.


### NEW-6: Design Iteration Timeline / Swim Lane

**Purpose:** Visualize the four design iterations (v1 skeleton, v2 DT ridge, v3 image-space, v4 template approval) as a progression, showing what changed at each step and why.

**Chart type:** Horizontal swim-lane or timeline diagram. Each iteration gets a row showing: planning domain, key algorithm, FPS, and the failure mode that motivated the next iteration.

**Data needed:** Table 8 (tab:iteration_progression) values.

**Approach:** Best done as a TikZ diagram or a carefully designed matplotlib figure with annotated horizontal bars per iteration.

**Why better:** Table 8 is dense and the narrative connection between iterations requires reading multiple paragraphs. A visual timeline makes the progression arc immediately clear.

**Placement:** Chapter 3 opening or Section 4.3 (alongside Table 8).


---

## Part 4: Table Improvements

### Table 3 (tab:training_progression) -- Segmentation Training Progression

**Current:** 4 rows x 5 columns. Clean, well-structured.

**Issues:**
- "SegFormer-B2 (300 hand-labeled)" in the Teacher column is awkward -- the parenthetical describes data, not teacher.
- No column for student architecture (implied constant SegFormer-B0 throughout).

**Recommendation:** Add a "Student" column (all SegFormer-B0) to make the distillation setup explicit. Move data size info to its own column. Minor.


### Table 5 (tab:seg_comparison) -- Segmentation Quality on 32 Frames

**Current:** 3 rows x 6 columns. Compact, effective.

**Issues:**
- The "Cand.+confhold" row is not explained in the text near the table. What is confhold?
- "ms" as a column header is ambiguous -- should be "Inference (ms)".

**Recommendation:** Add a footnote explaining "confhold" (confidence-hold temporal filter). Rename column. Minor.


### Table 7 (tab:fullvideo_replay) -- Full-Video Replay Metrics

**Current:** 6 rows x 4 columns. Clean.

**Issues:**
- Delta column uses mixed notation ("+0.016", "-1.13 pp", "---"). The "pp" (percentage points) convention is correct but may not be familiar to all readers.
- No per-video breakdown.

**Recommendation:** Add a footnote defining "pp" = percentage points. Consider adding a per-video supplementary table.


### Table 8 (tab:iteration_progression) -- Design Iteration Progression

**Current:** 4 rows x 5 columns. Effective summary.

**Issues:**
- The footnote about v2 FPS being "estimated from component timings" is important but easy to miss.
- "Pipeline FPS" for v2 is listed as "~1" -- this is imprecise.

**Recommendation:** Acceptable as-is. Consider adding a "Failure Mode" column showing what motivated each subsequent iteration.


### Table 9 (tab:planner_comparison) -- Planner Comparison on 32 Frames

**Current:** 5 rows x 5 columns (plus 1 sub-row "DT Ridge (near)"). Well-structured with domain grouping.

**Issues:**
- **No standard deviation or IQR.** This is a 32-frame comparison. Variability matters.
- The "DT Ridge (near)" sub-row is confusing -- is it a different planner or a different configuration of the same planner?
- "Inside-GT" should be "Inside-GT (%)" for clarity.

**Recommendation:** Add a variability column (e.g., "Center Err. [px] (mean +/- std)"). Clarify the near-field variant. This is the most-cited table in the thesis and would benefit from being the most rigorous.


### Table 10 (tab:oracle_comparison) -- Oracle Mask Comparison

**Current:** 3 rows x 4 columns. Compact.

**Issues:**
- Only three planners shown (BEV DT, Img DT, Img Midpoint). The skeleton planner and template planner are missing. Including all five would strengthen the oracle comparison.
- No variability measures.

**Recommendation:** Include all five planners for completeness. Add std/IQR.


### Table 11 (tab:runtime_merged) -- Per-Module Runtime Comparison

**Current:** 6 rows x 4 columns. Effective side-by-side comparison.

**Issues:**
- "Note" column is informal. Replace with a footnote or remove.
- Missing: total pipeline overhead (not just module sum -- includes data transfer, Python overhead).

**Recommendation:** Remove "Note" column, add footnotes. Acceptable otherwise.


### Table 13 (tab:accepted_run) -- 1800-Frame Accepted Run

**Current:** 6 rows x 2 columns. Compact summary.

**Issues:**
- Acceptable as-is. Consider adding confidence intervals on IoU and FPS.

**Recommendation:** Minor: add (min, max) ranges for FPS and IoU to show worst-case behavior.


### General Table Recommendations

1. **Use `booktabs` throughout** (already done -- good).
2. **Align decimal points** using `siunitx` S columns for numeric data. This is not currently done; numbers are formatted as plain text.
3. **Bold best values** in comparison tables (partially done for tab:planner_comparison, not for others).
4. **Add `\small` or `\footnotesize`** to dense tables to prevent overflow.
5. **Use threeparttable** for footnotes (package already loaded).


---

## Part 5: Priority Summary

### Critical (must fix before submission)

| ID | Issue | Action |
|----|-------|--------|
| C1 | **Pipeline diagram is JPG** with compression artifacts | Replace `pipeline_graph.jpg` with `pipeline_detailed.png` or `pipeline_overview.png` |
| C2 | **BEV fragility figure contradicts text** (figure shows 98.3% valid; text says 99.3% no-path) | Resolve discrepancy: either regenerate figure from the correct baseline run data, or correct the text |
| C3 | **Planner comparison bar chart has wrong data** (BEV Skeleton shows 52.3 in figure vs 76.6 in table) | Regenerate `planner_comparison.png` from the same data as Table 9 |

### High Priority (strongly recommended)

| ID | Issue | Action |
|----|-------|--------|
| H1 | Planner comparison needs error bars/distributions | Create NEW-1 (box plots) to replace current bar chart |
| H2 | Runtime breakdown shows only one pipeline | Create NEW-3 (dual-pipeline stacked bars) |
| H3 | Training curves unused | Include `training_curves.png` in Section 4.2 after fixing legend labels |
| H4 | Latency violin unused | Include `latency_violin.png` (filtered to final runs) in Section 4.6 |
| H5 | `seg_comparison_example.jpg` is JPEG | Re-export as PNG |
| H6 | GPS intent architecture diagram unused | Include `gps_intent_architecture.png` in Section 3.7 |
| H7 | BEV transform process diagram unused | Include `bev_transform_process.png` in Section 3.6 |

### Medium Priority (improves quality)

| ID | Issue | Action |
|----|-------|--------|
| M1 | Planner comparison architecture diagram unused | Include `planner_comparison_arch.png` after correcting data values |
| M2 | Temporal stability time series unused | Include FPS panel from `temporal_stability.png` in Section 4.6 |
| M3 | Resolution sweep too dense | Reduce to 4 key resolutions or add a latency-vs-resolution line plot |
| M4 | Segmentation stages subfigure (a) has debug text | Replace with clean camera frame |
| M5 | Add per-video metric breakdown | Create NEW-5 (heatmap or faceted bars) |
| M6 | Scooter photo needs camera callout | Add annotation arrow |
| M7 | Add standard deviations to Tables 9, 10 | Regenerate from per-frame data |
| M8 | Create design iteration timeline | Create NEW-6 (swim-lane/timeline diagram) |

### Low Priority (polish)

| ID | Issue | Action |
|----|-------|--------|
| L1 | Skeleton overlay subfigure (b) low resolution | Re-render at higher resolution |
| L2 | `planned_vs_skeleton_overlay.png` unpublishable | Needs full re-render with legend; skip for now |
| L3 | `fps_comparison.png` duplicates violin plot | Do not include |
| L4 | `seg_improvement.png` potentially redundant with tables | Consider removing in favor of training curves |
| L5 | Heading analysis needs more varied data | Regenerate from a turning-heavy clip, or skip |
| L6 | Use siunitx S columns for decimal alignment in tables | Apply globally |


---

## Part 6: Recommended Final Figure List

After implementing all changes, the thesis should contain approximately 15--17 figures:

| # | Figure | Section | Status |
|---|--------|---------|--------|
| 1 | Scooter hardware (2 subfigures) | Ch.1 | Keep (minor fixes: M6) |
| 2 | Pipeline diagram (detailed or overview) | Ch.3 | Replace (C1) |
| 3 | BEV transform process | Ch.3 S3.6 | **NEW inclusion** (H7) |
| 4 | Planner comparison architecture | Ch.3 S3.7 | **NEW inclusion** (M1) |
| 5 | GPS intent architecture | Ch.3 S3.7 | **NEW inclusion** (H6) |
| 6 | Resolution sweep (reduced) | Ch.3 S3.3 | Improve (M3) |
| 7 | Segmentation stages (3 subfigures) | Ch.4 S4.2 | Fix (M4) |
| 8 | Training curves | Ch.4 S4.2 | **NEW inclusion** (H3) |
| 9 | Qualitative seg comparison | Ch.4 S4.2 | Fix format (H5) |
| 10 | Planner comparison box plots | Ch.4 S4.3 | **NEW figure** (H1, replaces C3) |
| 11 | Planner comparison + log-latency | Ch.4 S4.3 | **NEW figure** (NEW-2) |
| 12 | Qualitative planner comparison | Ch.4 S4.3 | Keep (minor fix) |
| 13 | Skeleton pipeline stages | Ch.4 S4.3 | Fix (L1) |
| 14 | BEV fragility (corrected) | Ch.4 S4.4 | Regenerate (C2, NEW-4) |
| 15 | Dual-pipeline runtime comparison | Ch.4 S4.6 | **NEW figure** (H2, NEW-3) |
| 16 | Latency violin (filtered) | Ch.4 S4.6 | **NEW inclusion** (H4) |
| 17 | Temporal stability (FPS panel) | Ch.4 S4.5 | **NEW inclusion** (M2) |

This brings the thesis from 11 figures (some with data errors) to 17 publication-quality figures with consistent data, appropriate chart types, and complete coverage of all major claims.
