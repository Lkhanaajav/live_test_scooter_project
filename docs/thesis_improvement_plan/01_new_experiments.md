# Thesis Improvement Plan: New Experiments, Ablations, and Analyses

Generated: 2026-03-31
Scope: All experiments use existing code, existing videos, and existing models. No new data collection, no new hardware.

---

## 1. New Ablation Studies

### 1.1 Morphological Kernel Size Sweep (Mask Refinement Sensitivity)

- **What:** Sweep `MORPH_GAUSS_SIGMA_PX` over {0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0} and `MORPH_GAUSS_THRESH` over {0.20, 0.30, 0.35, 0.45, 0.55} on all 11 videos. Measure downstream template success rate, mask IoU stability, and heading jitter per combination.
- **Code/data:** `masks.py` (`clean_bev_mask_enhanced()`), `config.py` (MORPH_GAUSS_SIGMA_PX, MORPH_GAUSS_THRESH), `scripts/eval_research_improvements.py` (adapt the sweep loop).
- **Expected outcome:** Identify the Pareto frontier for mask quality vs. template success. Likely shows diminishing returns beyond sigma=1.2 and that under-smoothing produces noisy corridors while over-smoothing erases narrow paths. Strengthens the thesis by justifying the chosen parameters with empirical evidence rather than ad-hoc selection.
- **Priority:** MEDIUM

### 1.2 EMA Alpha Sweep Across All Planners (Not Just Segmentation)

- **What:** The existing smoother tune (`scripts/tune_smoother.py`) sweeps `MASK_SMOOTH_ALPHA` and `MASK_SMOOTH_CONSISTENCY_THRESH` but only measures mask stability. Extend the sweep to also measure downstream path jitter, heading delta, and template success rate for each (alpha, thresh) pair, running all 5 planners on the resulting masks.
- **Code/data:** `scripts/tune_smoother.py` (extend to record planner metrics), `image_path_planner.py`, `template_path_planner.py`, `realtime_nav_core.py` (BEVPathExtractor).
- **Expected outcome:** Reveals whether the optimal EMA alpha for segmentation stability is also optimal for path stability, or whether different planners prefer different smoothing levels. If the optima diverge, this is a strong argument for the dual-domain architecture (image-space + BEV verification) since each domain may need different temporal characteristics.
- **Priority:** HIGH

### 1.3 Resolution Sweep with All Five Planners

- **What:** The thesis currently benchmarks segmentation FPS at different resolutions (Table 3.1, `tab:segformer_fps`) but does not report planner accuracy at each resolution. Run all 5 planners at {320x180, 512x288, 640x360, 960x540} on the 32 hand-annotated frames. Record center error, inside-GT ratio, and runtime for each (resolution, planner) combination.
- **Code/data:** `scripts/eval_hand_annotated_pipeline.py` (modify `SEG_INPUT_RES` per run), `image_path_planner.py`, `realtime_nav_core.py`.
- **Expected outcome:** Quantifies the resolution-accuracy trade-off for each planner. Hypothesis: image-space planners degrade gracefully at lower resolution because midpoint extraction is resolution-invariant, while BEV planners degrade more sharply because the already-sparse BEV projection becomes sparser. This would further strengthen the image-space advantage argument.
- **Priority:** HIGH

### 1.4 Template Bank Size Ablation

- **What:** The template-approval planner uses 8 pre-computed arc templates. Vary the bank size from 1 (straight-only) through {2, 3, 5, 8, 12, 16, 24} arcs and measure template success rate, heading error, and path-source switches on the 220-frame evaluation clip and the 1800-frame accepted run.
- **Code/data:** `template_path_planner.py` (modify the arc generation to produce N arcs), `scripts/eval_template_planner.py`.
- **Expected outcome:** Demonstrates diminishing returns beyond approximately 5-8 arcs. Shows that a small discrete set of candidates suffices for sidewalk geometry, supporting the "verification over discovery" thesis.
- **Priority:** HIGH

### 1.5 Containment Threshold Sensitivity

- **What:** Sweep the turn containment threshold `WAYPOINT_MIN_CONTAINMENT_RATIO` over {0.40, 0.50, 0.60, 0.70, 0.80, 0.90} and `WAYPOINT_NEAR_FIELD_MIN_RATIO` over {0.50, 0.60, 0.70, 0.80}. Measure containment failure rate, turn activation rate, and hold rate on VID_017 with the scheduled intent window.
- **Code/data:** `config.py` (WAYPOINT_MIN_CONTAINMENT_RATIO, WAYPOINT_NEAR_FIELD_MIN_RATIO), `scripts/eval_waypoint_turn_planner.py`.
- **Expected outcome:** Maps the safety-liveness trade-off curve. Stricter thresholds yield 0% failure but suppress valid turns; looser thresholds allow more turns but may pass unsafe paths. The current 0.60 setting should sit at the knee of this curve.
- **Priority:** MEDIUM

### 1.6 Temporal Path Smoother Alpha Isolation

- **What:** Sweep `PATH_SMOOTH_MIN_ALPHA` and `PATH_SMOOTH_MAX_ALPHA` independently over {0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 0.95} while holding the other fixed. Measure heading jitter (degree/frame), path-source switch count, and cubic coefficient variance on 6 full-video replays.
- **Code/data:** `path_smoother.py`, `config.py`, `scripts/eval_research_improvements.py`.
- **Expected outcome:** Isolates the contribution of adaptive alpha vs. fixed alpha. If adaptive alpha (confidence-weighted) consistently outperforms the best fixed alpha, this supports the claim that confidence-gated smoothing is essential for the propose-and-verify paradigm.
- **Priority:** MEDIUM

### 1.7 BEV Grid Resolution Ablation

- **What:** Vary `BEV_SIZE` from (180, 330) to (360, 660) to (540, 990) to (720, 1320). Run all 3 BEV planners (skeleton-graph, DT ridge, template-approval) on the 32 hand-annotated frames and on 3 full videos. Measure path accuracy, BEV occupancy ratio, and runtime.
- **Code/data:** `config.py` (BEV_SIZE), `scripts/eval_hand_annotated_pipeline.py`.
- **Expected outcome:** Higher BEV resolution may improve BEV planner accuracy but worsen the occupancy ratio (same camera FOV, more empty pixels). This would quantify the inherent tension in BEV resolution for monocular systems and further support the fragility argument.
- **Priority:** LOW

### 1.8 Distance Transform Core Threshold Sweep

- **What:** Sweep `DT_CORE_THRESH` over {1.0, 2.0, 3.0, 4.0, 6.0, 8.0}. Measure skeleton connectivity, path availability, and heading error on both the 32-frame benchmark and full-video runs.
- **Code/data:** `config.py` (DT_CORE_THRESH), `skeleton.py`, `realtime_nav_core.py`.
- **Expected outcome:** Shows the sensitivity of skeleton-based planning to the DT threshold. Too low allows noise through; too high prunes valid narrow paths. Documents why the current value (2.0) was chosen.
- **Priority:** LOW

---

## 2. New Comparative Experiments

### 2.1 BEV Fragility with Improved Model (Not Just Baseline)

- **What:** The BEV fragility analysis (Table 4.6, 99.3% failure) uses the baseline segmentation model. Re-run `scripts/measure_bev_survival.py` with the candidate (OneFormer-trained) model on the same 4,407-frame sequence. Also run on all 11 videos.
- **Code/data:** `scripts/measure_bev_survival.py`, candidate model checkpoint, 11 video files.
- **Expected outcome:** If BEV fragility persists even with the improved model (hypothesis: it will, since the root cause is geometric coverage not segmentation quality), this directly strengthens the thesis claim that BEV fragility is geometric rather than segmentation-related. If fragility drops significantly, that is also interesting and nuances the claim.
- **Priority:** HIGH -- this is the single most impactful experiment for the BEV fragility narrative.

### 2.2 Per-Video Planner Comparison Breakdown

- **What:** The current planner comparison (Table 4.3) aggregates across all 32 hand-annotated frames. Stratify results by source video to show whether the image-space advantage is consistent across conditions (straight, curves, T-junctions, shadows, etc.).
- **Code/data:** `scripts/eval_hand_annotated_pipeline.py` (add per-video grouping to the output), 32 hand-annotated frames with their source video labels.
- **Expected outcome:** If image-space dominance is consistent across all video conditions, this strengthens external validity. If BEV planners are competitive on certain conditions (e.g., wide open areas), this provides nuanced discussion material.
- **Priority:** HIGH

### 2.3 Cross-Video Template Planner Evaluation

- **What:** The template vs. skeleton comparison (Table 4.7) uses a single 220-frame clip. Run the same comparison on all 11 videos, reporting per-video template rate, heading error, and fallback rate.
- **Code/data:** `scripts/eval_template_planner.py` (already supports multiple videos), all 11 video files.
- **Expected outcome:** Validates that the 40.6% heading error reduction generalizes beyond a single clip. Per-video breakdown reveals which conditions challenge the template planner most (likely tight turns and narrow paths).
- **Priority:** HIGH

### 2.4 Image-Space Planners Head-to-Head (Midpoint vs DT) Across Full Videos

- **What:** Run both image-space planners (midpoint and DT) on all 11 videos for full-length evaluation. Report heading jitter, path-source stability, and FPS. Currently only evaluated on 32 static frames.
- **Code/data:** Create a lightweight replay script (adapt `scripts/eval_simple_road.py` structure) that runs both image-space planners frame-by-frame.
- **Expected outcome:** Quantifies the temporal stability advantage of each image-space method. Midpoint is faster but may have higher jitter on curves. DT is slower but may produce smoother paths. This justifies the midpoint-primary + DT-fallback architecture.
- **Priority:** MEDIUM

### 2.5 BEV Fragility as a Function of Stride

- **What:** The 1800-frame accepted run uses stride=4 (process every 4th frame). Re-run the BEV fragility measurement at stride {1, 2, 4, 8} to see if temporal subsampling affects the occupancy ratio and path extraction success rate.
- **Code/data:** `scripts/measure_bev_survival.py` (modify frame stride), VID_017.
- **Expected outcome:** If stride=4 improves BEV occupancy (each processed frame has more temporal decorrelation from the previous, reducing mask flicker), this explains part of the 100% template success in the accepted run vs. 62.3% in the 220-frame clip.
- **Priority:** MEDIUM

### 2.6 Oracle-Mask BEV Fragility Experiment

- **What:** Feed oracle (hand-annotated GT) masks into the BEV pipeline and measure BEV occupancy ratio and path extraction success on the 32 frames. This complements Table 4.5 (oracle planner comparison) by explicitly measuring BEV occupancy with perfect input.
- **Code/data:** `scripts/eval_hand_annotated_pipeline.py` (add BEV occupancy measurement), hand-annotated masks.
- **Expected outcome:** If oracle masks still produce low BEV occupancy, this conclusively proves the geometric argument. The 32-frame sample should produce cleaner statistics than the 4,407-frame baseline run since the segmentation variable is perfectly controlled.
- **Priority:** HIGH

### 2.7 Segmentation Comparison with CityScapes mIoU

- **What:** `scripts/cityscapes_miou_segformer_b0.py` already exists. Run both the baseline and candidate model on the CityScapes validation set and report the sidewalk-class IoU alongside the campus-sidewalk IoU.
- **Code/data:** `scripts/cityscapes_miou_segformer_b0.py`, CityScapes validation set (if available locally), baseline and candidate model checkpoints.
- **Expected outcome:** Positions the models against a public benchmark. Even if campus IoU is high, CityScapes IoU may be lower due to domain shift, which is an honest limitation to discuss.
- **Priority:** MEDIUM

---

## 3. New Metrics or Analyses

### 3.1 Path Smoothness / Jitter Metric

- **What:** Define and compute a formal path smoothness metric across all planners: (a) mean frame-to-frame heading delta (deg/frame), (b) heading standard deviation, (c) maximum single-frame heading jump, (d) path curvature distribution (1/radius in meters). Report on full-video runs.
- **Code/data:** Already partially computed in `eval_research_improvements.py` (FrameMetrics has `path_jitter_raw` and `path_jitter_smooth`). Extend to all 5 planners.
- **Expected outcome:** Provides a quantitative temporal stability metric that center error and inside-GT do not capture. Template planner should show lowest jitter; skeleton-graph highest. This addresses the construct validity threat (Section 5.8) by adding a metric that correlates with steering comfort.
- **Priority:** HIGH

### 3.2 Failure Mode Classification and Per-Frame Taxonomy

- **What:** For each frame in the 1800-frame accepted run and the 22,679-frame full-video replay, classify the outcome into one of: {normal, low-confidence, fallback-to-graph, fallback-to-hold, segmentation-failure, BEV-empty, turn-active, turn-held, obstacle-slowdown}. Report the distribution.
- **Code/data:** The CSV logs from `eval_template_planner.py` already contain `path_source`, `planner_low_confidence`, `planner_slowdown`, `bev_mask_occ_ratio`. Parse and classify.
- **Expected outcome:** A failure mode pie chart / stacked bar chart showing what happens in non-normal frames. Strengthens the discussion by providing empirical evidence for the fallback chain effectiveness.
- **Priority:** HIGH

### 3.3 Computational Cost Breakdown Per-Component (Pie Chart Data)

- **What:** Profile each pipeline component separately: {SegFormer inference, mask morphology, BEV warp, BEV cleanup, corridor extraction, template scoring, path smoothing, YOLO inference, BEV obstacle projection, heading computation, visualization}. Report mean and p95 latency for each.
- **Code/data:** Instrument `realtime_nav_core.py` with per-component timing (some already exists in the CSV logs: `seg_ms`, `det_ms`, etc.). Run on VID_017 for 1000 frames.
- **Expected outcome:** A fine-grained runtime breakdown that goes beyond Table 4.8 (which groups into 5 categories). Shows where the 69.5ms per-frame budget is actually spent, identifying optimization headroom.
- **Priority:** MEDIUM

### 3.4 BEV Occupancy Ratio Distribution

- **What:** For each frame in all 11 videos, compute `bev_mask_occ_ratio` (fraction of BEV grid occupied by road pixels) and plot the distribution as a histogram or violin plot. Report mean, median, p5, p95.
- **Code/data:** `realtime_nav_core.py` or `scripts/measure_bev_survival.py` (already computes this), all 11 videos.
- **Expected outcome:** Shows that the BEV occupancy ratio is typically 0.001-0.05 (extremely sparse), providing visual evidence for the fragility claim beyond the single statistic (0.0002 mean) currently reported.
- **Priority:** HIGH

### 3.5 Curvature Distribution Across Videos

- **What:** For each planned path (all 5 planners), compute the curvature at each point and plot the distribution. Report mean curvature, max curvature, and percentage of frames above 0.10 m^-1 (gentle curve), 0.30 m^-1 (sharp turn), 0.50 m^-1 (very sharp).
- **Code/data:** Extend `image_path_planner.py` and `template_path_planner.py` to output per-frame curvature. Or compute from the cubic coefficients: kappa = |2*a2 + 6*a3*x| / (1 + (a1 + 2*a2*x + 3*a3*x^2)^2)^(3/2).
- **Expected outcome:** Characterizes the path geometry distribution, which the thesis currently lacks. Shows that most frames are near-zero curvature (straight), validating the focus on straight-ahead planning.
- **Priority:** MEDIUM

### 3.6 Effective Planning Horizon Analysis

- **What:** For each planner, measure how far ahead the path extends (in pixels and meters). Report the distribution of planning horizon lengths. Compare the 8-meter design target against actual achieved horizon.
- **Code/data:** `template_path_planner.py` (reports `best_path_length_px` in logs), `eval_template_planner.py`.
- **Expected outcome:** Validates the 8-meter design choice and shows whether the planner consistently achieves it. If the horizon frequently falls short (e.g., on narrow paths), this motivates future work on adaptive horizon.
- **Priority:** LOW

### 3.7 Segmentation Boundary Error Analysis

- **What:** For each of the 32 hand-annotated frames, compute the boundary F1 score (BF1, using a tolerance of 2, 5, and 10 pixels) in addition to the global IoU. This measures segmentation quality at the mask edges, which is what directly affects path planning.
- **Code/data:** `scripts/eval_hand_annotated_pipeline.py` (add BF1 computation), 32 hand-annotated frames.
- **Expected outcome:** If BF1 improves proportionally more than IoU from baseline to candidate, this supports the argument that teacher quality improves boundary precision specifically, which is the critical factor for path planning accuracy.
- **Priority:** MEDIUM

### 3.8 Temporal Autocorrelation of Heading Error

- **What:** Compute the autocorrelation function of the heading error time series for each planner across full-video runs. This characterizes whether errors are random (white noise) or persistent (correlated).
- **Code/data:** Full-video CSV logs, numpy/scipy for autocorrelation computation.
- **Expected outcome:** The skeleton-graph planner likely shows low autocorrelation (random jitter), while the template planner shows high autocorrelation (persistent, smooth paths). This provides statistical evidence for the temporal stability claim beyond the mean heading delta.
- **Priority:** LOW

---

## 4. Visualization Improvements

### 4.1 Frame-by-Frame Heading Time Series

- **What:** Plot heading angle as a continuous time series (x=frame number, y=heading degrees) for a representative 300-500 frame segment. Show all 5 planners overlaid on the same plot, or as a stacked panel (one planner per row).
- **Code/data:** Full-video CSV logs from `eval_template_planner.py`. Matplotlib or similar.
- **Expected outcome:** The most intuitive visualization of why template planning is better: viewers see the smooth template curve vs. the noisy skeleton curve. This figure would be more compelling than the current aggregate statistics.
- **Priority:** HIGH

### 4.2 BEV Occupancy Heatmap

- **What:** Aggregate the BEV masks across N frames of a representative video and visualize as a heatmap (brighter = more frequently occupied). Overlay the template arc bank on the heatmap.
- **Code/data:** Extend `scripts/eval_research_improvements.py` or `scripts/measure_bev_survival.py` to accumulate BEV masks.
- **Expected outcome:** Shows the characteristic trapezoidal strip that the monocular camera covers in BEV space. Makes the BEV fragility argument visually immediate for readers who are unfamiliar with monocular projection geometry.
- **Priority:** HIGH

### 4.3 Side-by-Side 5-Planner Overlay Video

- **What:** For a representative 200-frame segment, render all 5 planner paths on the same camera frame (different colors) and output as a comparison video or filmstrip figure.
- **Code/data:** `scripts/eval_hand_annotated_pipeline.py` (already has per-planner rendering), `scripts/make_video_comparison_strips.py`.
- **Expected outcome:** Provides an at-a-glance visual showing that image-space paths are centered and stable while BEV paths wander and jitter. A single compelling figure/video is worth many paragraphs of description.
- **Priority:** HIGH

### 4.4 Template Arc Bank Visualization on BEV Corridor

- **What:** For 4-6 representative frames, show the BEV corridor (distance-transform heatmap) with all 8 template arcs overlaid. Mark the selected arc in a different color. Show which arcs pass/fail the containment check.
- **Code/data:** `template_path_planner.py` (has the arc generation code), BEV corridor from `safe_corridor.py`.
- **Expected outcome:** Makes the "propose-and-verify" paradigm concrete and visual. Readers can see which arcs are approved and which are rejected, understanding the template planner's decision-making process.
- **Priority:** MEDIUM

### 4.5 BEV Projection Geometry Diagram

- **What:** Create a figure showing the camera frustum, the ground plane, and the resulting trapezoidal BEV coverage area. Annotate with camera height, FOV angle, and the percentage of BEV grid that is actually covered.
- **Code/data:** Pure geometry (can compute from `config.py` parameters: DEFAULT_SRC_POINTS, DEFAULT_DST_POINTS, BEV_SIZE). Matplotlib.
- **Expected outcome:** A clean geometric diagram explaining WHY monocular BEV is sparse. This would complement the current Figure 4.4 (BEV fragility bar chart) by explaining the root cause visually rather than just showing the symptom.
- **Priority:** HIGH

### 4.6 Segmentation Quality Across Conditions (Multi-Panel Figure)

- **What:** Create a 4x3 panel figure showing segmentation output across different conditions: {straight, curve, T-junction, shadow, narrow, wide, pedestrian, surface-change}. Show RGB input, baseline mask, candidate mask side by side.
- **Code/data:** Select representative frames from different videos. `scripts/eval_hand_annotated_pipeline.py` (generates overlay images).
- **Expected outcome:** Extends Figure 4.3 (currently 3 examples) to a more comprehensive qualitative comparison that demonstrates robustness across conditions.
- **Priority:** MEDIUM

### 4.7 Turn Planner Activation Sequence

- **What:** Create a filmstrip showing 6-8 consecutive frames during a turn maneuver: pre-turn (template), turn acquisition (transition), turn active (waypoint-turn path), turn hold (sustain), post-turn (return to template). Annotate with GPS intent, containment ratio, and path source.
- **Code/data:** `scripts/eval_waypoint_turn_planner.py` (already renders overlay videos), frames 15-35 from the VID_017 turn sequence.
- **Expected outcome:** Makes the turn planner's state machine visible and concrete. Currently the thesis describes the turn planner in text but does not show what a turn maneuver looks like frame-by-frame.
- **Priority:** MEDIUM

### 4.8 Latency Distribution Violin/Box Plot Per Pipeline Stage

- **What:** Replace or supplement the current bar chart (Figure 4.6) with violin plots showing the full latency distribution for each pipeline stage, revealing tail latency and variance, not just means.
- **Code/data:** Per-frame timing data from CSV logs.
- **Expected outcome:** Shows whether the 69.5ms mean hides occasional 200ms spikes that could violate the 10Hz requirement. If the distribution is tight, this strengthens the real-time claim. If there are outliers, this motivates discussion of worst-case latency.
- **Priority:** MEDIUM

---

## 5. Table Improvements

### 5.1 Add Standard Deviation Columns to All Tables

- **What:** Tables 4.2 (seg comparison), 4.3 (planner comparison), 4.7 (template eval), 4.8 (runtime) currently report only means. Add SD (or IQR) columns where applicable.
- **Code/data:** Recompute from existing per-frame data in CSV logs.
- **Expected outcome:** Indicates measurement reliability. If the SD is large relative to the difference between conditions, the claims need qualification. If small, it strengthens the claims.
- **Priority:** HIGH

### 5.2 Add Per-Video Rows to Full-Video Replay Table

- **What:** Table 4.4 (full-video replay) aggregates 6 videos into a single row per model. Expand to show per-video rows, allowing readers to see whether the improvement is consistent across conditions.
- **Code/data:** The full-video replay already runs per-video; just disaggregate the output.
- **Expected outcome:** Reveals whether certain videos are "easy" (both models perform well) or "hard" (only the improved model succeeds). This is critical for assessing external validity.
- **Priority:** HIGH

### 5.3 Add Effect Size (Cohen's d) to Key Comparisons

- **What:** For the three main comparisons (baseline vs candidate seg, image-space vs BEV planner, template vs skeleton), compute Cohen's d or the Mann-Whitney U effect size. Report alongside p-values.
- **Code/data:** Per-frame metrics from CSV logs. scipy.stats for statistical tests.
- **Expected outcome:** Provides formal statistical evidence for the magnitude of improvements, not just point estimates. A large effect size (d > 0.8) would be strong evidence; a small one (d < 0.2) would require qualification.
- **Priority:** HIGH

### 5.4 Add Confidence Intervals to Planner Comparison

- **What:** For Table 4.3 (planner comparison on 32 frames), add 95% confidence intervals (bootstrap or normal approximation) for center error, inside-GT ratio, and runtime.
- **Code/data:** Per-frame planner results. numpy for bootstrap CIs.
- **Expected outcome:** With only 32 frames, CIs will be informative about whether the differences are statistically meaningful. If the CIs for different planners overlap, the claims need tempering.
- **Priority:** HIGH

### 5.5 Normalize Planner Comparison Metrics to Common Scale

- **What:** Table 4.3 mixes pixels (center error), ratios (inside-GT), and milliseconds (runtime). Add a normalized "efficiency score" column: e.g., (1/center_error) * (1/runtime_ms), or a Pareto ranking.
- **Code/data:** Computed from Table 4.3 data.
- **Expected outcome:** A single composite metric makes the image-space dominance visually immediate. Useful for a radar/spider chart visualization.
- **Priority:** LOW

### 5.6 Add Speedup Ratios to Runtime Table

- **What:** Table 4.8 shows absolute runtimes. Add a column showing speedup relative to the slowest method (BEV DT Ridge at 926.8ms) or relative to the 100ms budget.
- **Code/data:** Simple arithmetic from existing table values.
- **Expected outcome:** Makes the magnitude of the speedup more immediately communicable. "421x faster" is more impactful than "2.2ms vs 926.8ms" for readers scanning the table.
- **Priority:** LOW

---

## 6. New Hypotheses to Test

### 6.1 "Image-Space Advantage Scales with Mask Sparsity"

- **Hypothesis:** The accuracy advantage of image-space over BEV planning increases as the segmentation mask becomes sparser (narrower sidewalks, more occlusion). Conversely, on wide open paths, the gap narrows.
- **Test:** Bin the 32 hand-annotated frames by mask coverage ratio (e.g., low/medium/high). Report center error per bin for each planner. If the image-space advantage is larger in the low-coverage bin, the hypothesis is supported.
- **Code/data:** `scripts/eval_hand_annotated_pipeline.py`, 32 hand-annotated masks, mask coverage computation.
- **Expected outcome:** Supports the thesis argument by showing that the image-space advantage is particularly strong exactly when it matters most (difficult conditions).
- **Priority:** HIGH

### 6.2 "Template Bank Size Has Diminishing Returns Beyond 5 Arcs"

- **Hypothesis:** Increasing the template bank from 5 to 8, 12, or 24 arcs provides less than 2% improvement in template success rate.
- **Test:** See Ablation 1.4 above.
- **Code/data:** `template_path_planner.py`, `scripts/eval_template_planner.py`.
- **Expected outcome:** If confirmed, supports the "verification is cheap" argument and shows that sidewalk geometry is well-covered by a small discrete set.
- **Priority:** HIGH (combined with 1.4)

### 6.3 "BEV Fragility is Camera-Height Dependent"

- **Hypothesis:** The BEV occupancy ratio is a monotonically increasing function of camera mounting height (higher camera = wider BEV coverage).
- **Test:** Cannot change physical camera height, but CAN simulate different effective heights by adjusting the homography source points. Shift `DEFAULT_SRC_POINTS` up/down by {-100, -50, 0, +50, +100} pixels in the image (simulating higher/lower camera mount) and measure BEV occupancy and planner success rate.
- **Code/data:** `config.py` (DEFAULT_SRC_POINTS), `scripts/measure_bev_survival.py`, any test video.
- **Expected outcome:** Shows how sensitive BEV coverage is to camera geometry. If a 100-pixel shift in the homography source points (roughly equivalent to a 10cm height change) significantly changes the occupancy ratio, this quantifies a practical deployment concern.
- **Priority:** MEDIUM

### 6.4 "Segmentation Quality Has Diminishing Returns for BEV Planning"

- **Hypothesis:** Improving segmentation IoU from 0.75 to 0.95 improves BEV planner accuracy less than it improves image-space planner accuracy.
- **Test:** Already partially demonstrated by the oracle mask experiment (Table 4.5). Compute the delta in center error between baseline model and oracle mask for each planner. If delta_BEV < delta_image-space (as a percentage), the hypothesis is supported.
- **Code/data:** Table 4.5 data plus baseline model planner results (interpolate if needed).
- **Expected outcome:** Quantifies the claim made in Discussion section 5.3 ("segmentation quality has diminishing returns for BEV planning because the bottleneck is geometric coverage").
- **Priority:** MEDIUM

### 6.5 "Heading Jitter Correlates with BEV Occupancy"

- **Hypothesis:** Frames with lower BEV occupancy produce higher heading error/jitter in BEV planners, while image-space planners are unaffected by BEV occupancy.
- **Test:** For each frame in a full-video run, compute (bev_occ_ratio, heading_delta) and plot the scatter. Compute the correlation coefficient for each planner.
- **Code/data:** Full-video CSV logs (contain both `bev_mask_occ_ratio` and `heading_smoothed_deg`).
- **Expected outcome:** A strong negative correlation for BEV planners and zero correlation for image-space planners would cleanly demonstrate that BEV sparsity is the root cause of BEV planner instability.
- **Priority:** MEDIUM

### 6.6 "EMA Smoothing Helps Skeleton More Than Template"

- **Hypothesis:** Because the template planner already produces smooth paths (fixed arc bank), adding temporal EMA smoothing provides marginal improvement. But for the skeleton planner, which produces noisy paths, EMA smoothing provides a larger absolute improvement.
- **Test:** Run both planners with and without path smoothing (toggle `PATH_SMOOTH_ENABLED`). Measure heading jitter delta in each case.
- **Code/data:** `path_smoother.py`, `scripts/eval_template_planner.py`, `config.py`.
- **Expected outcome:** If confirmed, this supports the design decision to use smooth templates (eliminating jitter at the source) rather than relying on post-hoc smoothing (masking jitter after the fact).
- **Priority:** LOW

---

## 7. Discussion Section Ideas

### 7.1 Theoretical BEV Coverage Ratio as a Function of Camera Parameters

- **What:** Derive an analytical expression for the BEV coverage ratio as a function of camera height h, tilt angle theta, horizontal FOV phi, and BEV grid dimensions. Plot the coverage ratio surface.
- **Derivation:** The camera projects a trapezoid onto the ground plane. The near edge is at distance d_near = h * tan(theta - phi_v/2) and the far edge is at d_far = h * tan(theta + phi_v/2). The BEV grid covers L_forward x L_lateral meters. The coverage ratio is the area of the projected trapezoid divided by L_forward * L_lateral.
- **Code/data:** Pure math, can be computed in numpy/matplotlib. Use actual camera parameters from `config.py`.
- **Expected outcome:** An analytical formula that predicts BEV coverage and can be used by other researchers to estimate whether their camera setup will suffer from the same fragility. This elevates the thesis from an empirical finding to a generalizable design tool.
- **Priority:** HIGH

### 7.2 Pixel-to-Meter Conversion Error as a Function of Depth

- **What:** Derive and plot the pixel-to-meter conversion factor at different depths from the camera. Show that near-field pixels represent fewer centimeters each (high precision) while far-field pixels represent many centimeters each (low precision). This explains why far-range path planning in image space is imprecise.
- **Derivation:** For a pinhole camera model, 1 pixel at depth d corresponds to d/(f * image_width) * sensor_width meters. Plot for d = {1, 2, 4, 8, 12} meters.
- **Code/data:** Camera intrinsics (approximate from config), matplotlib.
- **Expected outcome:** Provides the theoretical grounding for the 8-meter planning horizon design choice. At 8 meters, 1 pixel might correspond to 2-3 cm; at 12 meters, it might be 5-6 cm, making midpoint extraction imprecise.
- **Priority:** MEDIUM

### 7.3 Information-Theoretic Argument for Image-Space Advantage

- **What:** Frame the image-space vs. BEV comparison as an information loss problem. The BEV warp is a lossy transformation (many image pixels map to one BEV pixel in the far field; near-field pixels are stretched). Compute the effective information (unique pixel count) in image space vs BEV space for representative frames.
- **Code/data:** Count unique occupied pixels in image mask vs BEV mask for the 32 annotated frames.
- **Expected outcome:** Quantifies the information loss inherent in the BEV projection. If the BEV mask has 10x fewer unique occupied pixels than the image mask, this explains why BEV planning is fundamentally less accurate.
- **Priority:** MEDIUM

### 7.4 Comparison with Published Sidewalk Navigation Systems

- **What:** Create a comparison table of this system vs. published sidewalk navigation systems (Starship, Viteri et al., Machkour et al., etc.) on dimensions: sensor count, compute hardware, planning domain, reported FPS, reported accuracy metric, open/closed-loop. Most systems will have "N/A" for several columns since they don't report the same metrics.
- **Code/data:** Literature review (references already in `references.bib`).
- **Expected outcome:** Positions this thesis within the broader landscape. Even though the systems are not directly comparable, the table shows that this is the only monocular-only, CPU-only, BEV-vs-image-space comparison in the literature.
- **Priority:** MEDIUM

### 7.5 When Should You Use BEV? Decision Framework

- **What:** Based on the experimental results, propose a decision framework for practitioners: "Use image-space planning when [conditions]. Add BEV verification when [conditions]. Use full BEV planning when [conditions]."
- **Code/data:** Synthesize from all experimental results.
- **Expected outcome:** Translates the empirical findings into actionable design guidance. This would be a valuable Discussion contribution that goes beyond describing what was found to prescribing what others should do.
- **Priority:** HIGH

### 7.6 Statistical Power Analysis of 32-Frame Benchmark

- **What:** Compute the statistical power of the 32-frame planner comparison to detect the observed differences. How many frames would be needed to achieve power > 0.80 at alpha = 0.05?
- **Code/data:** Standard power analysis formulas, using the observed effect sizes and variances.
- **Expected outcome:** Addresses the limitations section (32 frames is small) with a quantitative answer. If power is already > 0.90 (because the effect sizes are large), the 32-frame sample is adequate despite its small size. If power is low, this motivates the "larger annotated benchmark" future work item.
- **Priority:** MEDIUM

### 7.7 Energy/Compute Cost Comparison

- **What:** Estimate the energy cost (Joules per frame) for the image-space pipeline vs the BEV pipeline, given typical Raspberry Pi 4 power consumption (5-7W). Compare with GPU-based systems.
- **Code/data:** Runtime data from Table 4.8, RPi4 power specifications.
- **Expected outcome:** Shows that image-space planning saves energy by a factor proportional to the speedup, which matters for battery-powered scooters. At 59 FPS vs 2.4 FPS, the image-space pipeline uses approximately 25x less energy per frame.
- **Priority:** LOW

---

## 8. Quick-Win Experiments (Can Be Done in < 1 Hour Each)

These require minimal code changes and produce immediately usable results:

| ID | Experiment | Time Est. | Impact |
|----|-----------|-----------|--------|
| QW1 | Recompute Table 4.3 with SD columns | 30 min | HIGH |
| QW2 | Per-video breakdown of Table 4.4 | 30 min | HIGH |
| QW3 | BEV occupancy histogram across all videos | 30 min | HIGH |
| QW4 | Oracle mask BEV occupancy measurement | 30 min | HIGH |
| QW5 | Count unique pixels in image vs BEV mask | 15 min | MEDIUM |
| QW6 | BEV fragility re-run with candidate model | 45 min | HIGH |
| QW7 | Statistical tests (Cohen's d) for Table 4.3 | 30 min | HIGH |
| QW8 | Heading time series plot for 300 frames | 30 min | HIGH |
| QW9 | Speedup ratio column for runtime table | 10 min | LOW |
| QW10 | Turn planner filmstrip figure | 45 min | MEDIUM |

---

## 9. Recommended Priority Ordering

### Tier 1: Must-Do (directly strengthens or challenges a thesis claim)

1. **2.1** BEV fragility with improved model -- validates the geometric (not segmentation) root cause
2. **2.6** Oracle-mask BEV occupancy -- conclusive evidence for geometric argument
3. **2.2** Per-video planner breakdown -- consistency of image-space advantage
4. **2.3** Cross-video template evaluation -- generalization of 40.6% improvement
5. **5.1** Add SD columns to all tables -- measurement reliability
6. **5.2** Per-video rows in full-video table -- consistency evidence
7. **5.3** Effect sizes (Cohen's d) -- formal statistical evidence
8. **5.4** Confidence intervals for planner comparison -- statistical significance with n=32
9. **3.1** Path smoothness metric -- addresses construct validity gap
10. **4.1** Heading time series plot -- most compelling visualization
11. **4.5** BEV projection geometry diagram -- explains root cause visually
12. **7.1** Theoretical BEV coverage formula -- elevates finding to generalizable tool
13. **7.5** Decision framework for BEV vs image-space -- practical takeaway

### Tier 2: Should-Do (strengthens narrative, fills gaps)

14. **1.2** EMA alpha sweep with all planners
15. **1.3** Resolution sweep with all planners
16. **1.4** Template bank size ablation
17. **3.2** Failure mode classification
18. **3.4** BEV occupancy ratio distribution
19. **4.2** BEV occupancy heatmap
20. **4.3** Side-by-side 5-planner overlay
21. **6.1** Image-space advantage vs mask sparsity hypothesis
22. **7.4** Comparison with published systems table

### Tier 3: Nice-to-Have (deepens analysis, addresses reviewer concerns)

23. **1.1** Morphological kernel sweep
24. **1.5** Containment threshold sensitivity
25. **1.6** Temporal smoother alpha isolation
26. **2.4** Image-space planners head-to-head on full videos
27. **2.5** BEV fragility vs stride
28. **2.7** CityScapes mIoU comparison
29. **3.3** Computational cost breakdown per-component
30. **3.5** Curvature distribution
31. **3.7** Segmentation boundary F1 score
32. **4.4** Template arc bank visualization
33. **4.6** Multi-condition segmentation panel
34. **4.7** Turn planner filmstrip
35. **6.3** BEV fragility vs simulated camera height
36. **6.4** Segmentation diminishing returns for BEV
37. **6.5** Heading jitter vs BEV occupancy correlation
38. **7.2** Pixel-to-meter conversion at depth
39. **7.3** Information-theoretic argument
40. **7.6** Statistical power analysis

### Tier 4: Optional (polish, completeness)

41. **1.7** BEV grid resolution ablation
42. **1.8** DT core threshold sweep
43. **2.5** BEV fragility vs stride
44. **3.6** Effective planning horizon analysis
45. **3.8** Temporal autocorrelation
46. **4.8** Latency violin plots
47. **5.5** Normalized efficiency score
48. **5.6** Speedup ratio column
49. **6.6** EMA helps skeleton more than template
50. **7.7** Energy cost comparison

---

## 10. Scripts Index

Reference for where each experiment's code lives or should be adapted from:

| Script | Purpose | Experiments |
|--------|---------|-------------|
| `scripts/eval_hand_annotated_pipeline.py` | Planner comparison on 32 GT frames | 1.3, 2.2, 2.6, 3.7, 5.1, 5.4, 6.1 |
| `scripts/eval_template_planner.py` | Template vs skeleton comparison | 1.4, 2.3, 3.1, 4.1, 6.6 |
| `scripts/eval_research_improvements.py` | Research improvement metrics | 1.1, 1.6, 3.4 |
| `scripts/eval_simple_road.py` | Simple-road vs baseline | 2.4 |
| `scripts/eval_waypoint_turn_planner.py` | Turn planner validation | 1.5, 4.7 |
| `scripts/tune_smoother.py` | EMA alpha/thresh grid search | 1.2 |
| `scripts/measure_bev_survival.py` | BEV mask coverage analysis | 2.1, 3.4, 4.2, 6.3 |
| `scripts/benchmark_seg_stability.py` | Temporal seg stability | 2.7 |
| `scripts/cityscapes_miou_segformer_b0.py` | Public benchmark comparison | 2.7 |
| `scripts/make_video_comparison_strips.py` | Side-by-side video rendering | 4.3, 4.6 |
| `template_path_planner.py` | Arc generation, corridor scoring | 1.4, 4.4 |
| `image_path_planner.py` | Image-space planners | 2.4, 3.5 |
| `config.py` | All tunable parameters | All ablation studies |
| `realtime_nav_core.py` | Full pipeline orchestrator | 3.3, all system-level runs |
| `path_smoother.py` | Temporal smoothing | 1.6, 6.6 |
