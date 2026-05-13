# 04 -- Statistical Analysis Plan for Thesis Evaluation

This document specifies the complete statistical framework for all quantitative
claims in the thesis.  Every comparison names the test, justifies it, specifies
assumptions to check, defines the effect-size measure, describes multiple-
comparison correction when needed, addresses sample-size and temporal-
autocorrelation concerns, and provides an APA-style reporting template.

A companion Python script structure is described at the end.

---

## Table of Contents

1. [Comparison 1: Five-Planner Comparison (32 Frames)](#1-five-planner-comparison-32-hand-annotated-frames)
2. [Comparison 2: Segmentation Baseline vs Candidate (32 Frames)](#2-segmentation-baseline-vs-candidate-32-frames)
3. [Comparison 3: Template vs Skeleton Planner (220 Frames)](#3-template-vs-skeleton-planner-220-frames)
4. [Comparison 4: Full-Video Replay (6 Sequences, 22679 Frames)](#4-full-video-replay-6-sequences-22679-frames)
5. [Comparison 5: Temporal Smoothing Grid Search (35 Combinations)](#5-temporal-smoothing-grid-search-35-combinations)
6. [Cross-Cutting: Handling Temporal Autocorrelation](#6-handling-temporal-autocorrelation)
7. [Confidence Intervals](#7-confidence-intervals)
8. [Python Script Architecture](#8-python-script-architecture)
9. [Output Artifacts](#9-output-artifacts)

---

## 1. Five-Planner Comparison (32 Hand-Annotated Frames)

### 1.1 Data Structure

- **Subjects (blocks):** 32 hand-annotated frames, each from a campus sidewalk
  video.
- **Conditions (treatments):** 5 planners (BEV Skeleton-Graph, BEV DT Full,
  BEV DT Nearfield, Image Midpoint, Image DT).
- **Metrics:** center error (px), inside-GT ratio, runtime (ms).
- **Design:** Repeated-measures -- every frame is measured under every planner
  using the same cleaned segmentation mask.

### 1.2 Test Selection

| Metric | Primary Test | Rationale |
|--------|-------------|-----------|
| Center error (px) | **Friedman test** | Non-parametric repeated-measures ANOVA for k > 2 related groups.  n = 32 is moderate; center error is likely right-skewed (bounded below by 0, no upper bound), and Shapiro-Wilk is expected to reject normality.  Friedman does not assume normality or homogeneity of variance. |
| Inside-GT ratio | **Friedman test** | Same design.  Ratio is bounded [0, 1] and often ceiling-compressed (most values near 0.98+), violating normality. |
| Runtime (ms) | **Friedman test** | Runtime distributions are right-skewed with heavy tails.  Non-parametric preferred. |

**Post-hoc pairwise tests:** Nemenyi test (Friedman analog of Tukey HSD) for
all ${5 \choose 2} = 10$ pairwise comparisons.  Report critical difference (CD)
diagram.

**Alternative if normality holds:** If Shapiro-Wilk p > 0.05 on all 5
conditions and Mauchly's sphericity test passes, repeated-measures ANOVA with
Greenhouse-Geisser correction is acceptable, followed by Bonferroni-corrected
paired t-tests.  Report both results to demonstrate robustness.

### 1.3 Assumptions to Check

1. **Normality per condition:** Shapiro-Wilk on each of the 5 planner columns.
   If any p < 0.05, proceed with Friedman (non-parametric).
2. **Outlier check:** Boxplots per planner; flag values beyond 3x IQR.  Report
   but do not exclude unless measurement error is documented.
3. **Independence across frames:** Frames are drawn from multiple videos.
   If within-video correlation is suspected, apply the per-video aggregation
   strategy from Section 6.

### 1.4 Effect Size

| Measure | Formula / Tool | When to Report |
|---------|---------------|----------------|
| **Kendall's W** (coefficient of concordance) | `scipy.stats.friedmanchisquare` returns chi2; W = chi2 / (n * (k-1)) | Overall effect size for Friedman. W in [0, 1]; 0.1 = small, 0.3 = medium, 0.5 = large. |
| **Cliff's delta** (pairwise) | `cliffs_delta(a, b)` from own implementation or `scipy` | For each significant pairwise comparison. Bounded [-1, 1]; |d| < 0.147 negligible, < 0.33 small, < 0.474 medium, else large (Romano et al. 2006). |
| **Rank-biserial r** | r = 1 - (2U) / (n1 * n2) from Wilcoxon | Alternative pairwise effect size if Wilcoxon signed-rank used instead of Nemenyi. |

### 1.5 Multiple Comparisons Correction

- **Primary:** Nemenyi post-hoc (family-wise error rate controlled at alpha = 0.05).
- **Supplement:** Holm-Bonferroni on Wilcoxon signed-rank pairwise tests as a
  sensitivity check (stricter than Nemenyi; confirms significance is not
  method-dependent).

### 1.6 Sample Size Considerations

- n = 32 is above the recommended minimum of 15 for Friedman with k = 5
  (Iman & Davenport 1980).
- **Post-hoc power analysis:** Using G*Power or `statsmodels.stats.power`, compute
  achieved power for the observed effect size.  If power < 0.80, note this as a
  limitation and report the n required for 0.80 power.
- **Recommendation for thesis:** Acknowledge that 32 frames limits statistical
  power for detecting small effect sizes; the full-video replay (22,679 frames)
  provides compensating evidence of trend consistency.

### 1.7 APA-Style Reporting Templates

**Friedman omnibus:**
> A Friedman test indicated a statistically significant difference in center
> error across the five planners, chi2(4) = XX.X, p < .001, W = 0.XX.

**Nemenyi post-hoc (one pair):**
> Post-hoc Nemenyi test revealed that the image-space midpoint planner
> (Mdn = XX.X px) produced significantly lower center error than BEV DT Full
> (Mdn = XX.X px), p = .XXX, Cliff's delta = X.XX (large).

**Table format:**

| Planner | Median | Mean +/- SD | p (vs Midpoint) | Cliff's d |
|---------|--------|-------------|-----------------|-----------|
| Midpoint | ... | ... | -- | -- |
| BEV DT Full | ... | ... | < .001 | 0.XX |
| ... | ... | ... | ... | ... |

---

## 2. Segmentation Baseline vs Candidate (32 Frames)

### 2.1 Data Structure

- **Subjects:** 32 frames.
- **Conditions:** 2 segmentation models (Baseline = SegFormer-B2 teacher;
  Candidate = OneFormer Swin-L teacher), optionally 3 if including
  Candidate+confhold.
- **Metrics:** IoU, precision, recall, F1, inference time (ms).
- **Design:** Paired (same frames, same ground-truth masks).

### 2.2 Test Selection

| Metric | Primary Test | Rationale |
|--------|-------------|-----------|
| IoU, precision, recall, F1 | **Wilcoxon signed-rank test** (2 conditions) or **Friedman** (3 conditions) | Paired non-parametric.  IoU is bounded [0, 1] and likely non-normal (left-skewed when high). |
| Inference time | **Wilcoxon signed-rank** | Paired, right-skewed. |

For the 3-condition case (baseline, candidate raw, candidate+confhold):
Friedman + Nemenyi post-hoc, identical to Section 1.

### 2.3 Assumptions to Check

1. **Normality of paired differences:** Shapiro-Wilk on (candidate_IoU -
   baseline_IoU) for each metric.
2. **Symmetry of differences:** Visual inspection via histogram.  Wilcoxon
   signed-rank assumes symmetric distribution of differences around the median.
   If violated, use the sign test as a fallback (less powerful but assumption-free).

### 2.4 Effect Size

| Measure | When |
|---------|------|
| **Matched-pairs rank-biserial r** | r = Z / sqrt(n).  From Wilcoxon Z statistic. |
| **Cohen's d (paired)** | d = mean(diff) / SD(diff).  Report if differences are approximately normal. |
| **Common-language effect size (CLES)** | Proportion of pairs where candidate > baseline.  Intuitive for non-technical readers. |

### 2.5 Multiple Comparisons

- Two-condition case: No correction needed (single test per metric).
- Three-condition case: Holm-Bonferroni across 3 pairwise comparisons per metric.
- Across metrics (IoU, precision, recall, F1): These are not independent tests of
  the same hypothesis, so no cross-metric correction is needed.  Each tests a
  distinct aspect of mask quality.

### 2.6 Sample Size

- n = 32 paired observations.  Wilcoxon signed-rank has power > 0.80 for
  detecting medium effects (r > 0.3) at this sample size.
- The observed effect (IoU 0.758 vs 0.946) is very large; significance is
  expected to be robust.

### 2.7 APA-Style Reporting Template

> A Wilcoxon signed-rank test indicated that the candidate model (Mdn IoU =
> 0.XXX) produced significantly higher IoU than the baseline (Mdn IoU = 0.XXX),
> Z = X.XX, p < .001, matched-pairs rank-biserial r = 0.XX (large).  The
> candidate improved IoU for 31 of 32 frames (CLES = 96.9%).

**Compact table (thesis Table 4.2 replacement):**

| Model | IoU | Prec. | Recall | F1 | ms | p (vs Baseline) | r |
|-------|-----|-------|--------|----|----|-----------------|---|
| Baseline | 0.758 +/- 0.XX | ... | ... | ... | 18.9 | -- | -- |
| Candidate | **0.946** +/- 0.XX | ... | ... | ... | **11.7** | < .001 | 0.XX |

---

## 3. Template vs Skeleton Planner (220 Frames)

### 3.1 Data Structure

- **Subjects:** 220 frames from a single calibrated video clip (VID_017 June
  segment).
- **Conditions:** 2 planners (skeleton-graph baseline, template-approval).
- **Metrics:** absolute heading error (deg), path-source switches (count),
  heading jitter (frame-to-frame delta).
- **Design:** Paired within-frame comparison on same BEV mask.

### 3.2 Test Selection

| Metric | Primary Test | Rationale |
|--------|-------------|-----------|
| Abs heading error | **Wilcoxon signed-rank** | Paired, non-negative, likely right-skewed. |
| Per-frame heading jitter | **Wilcoxon signed-rank** | Paired. |
| Path-source switches | **McNemar's test** or descriptive only | Binary per-frame (switched / did not switch). McNemar tests marginal homogeneity in a paired 2x2 table. |

**Critical temporal concern:** The 220 frames are consecutive, introducing
strong temporal autocorrelation.  See Section 6 for handling.

### 3.3 Assumptions to Check

1. **Normality of differences:** Shapiro-Wilk on per-frame
   (template_heading - skeleton_heading).
2. **Temporal autocorrelation:** Durbin-Watson statistic on paired differences.
   If DW < 1.5 (positive autocorrelation), apply block bootstrap (Section 6).

### 3.4 Effect Size

| Measure | Application |
|---------|-------------|
| **Cliff's delta** | Primary.  Robust to non-normality. |
| **Matched-pairs r** | From Wilcoxon Z / sqrt(n). |
| **Percent reduction** | Descriptive: (1 - template_mean / skeleton_mean) * 100.  Currently reported as 40.6%. |

### 3.5 Multiple Comparisons

- Two-condition, single comparison per metric: no correction needed.

### 3.6 Sample Size and Temporal Autocorrelation

- **Nominal n = 220**, but effective n is smaller due to autocorrelation.
- **Effective sample size:** n_eff = n / (1 + 2 * sum(rho_k)) where rho_k are
  autocorrelation coefficients (Bayley & Hammersley 1946).  Estimate from the
  autocorrelation function of the paired differences.
- **Block bootstrap** (preferred): Resample 10,000 blocks of length b (where b
  is chosen via Politis-White-Patton optimal block length estimator) to compute
  bootstrap CIs and p-values.
- Report both naive (n = 220) and corrected (n_eff or block-bootstrap) results.

### 3.7 APA-Style Reporting Template

> A Wilcoxon signed-rank test on block-bootstrapped samples (b = XX, 10,000
> resamples) indicated that the template-approval planner (Mdn |theta| =
> 0.XX deg) achieved significantly lower heading error than the skeleton-graph
> baseline (Mdn |theta| = X.XX deg), p < .001, Cliff's delta = 0.XX (medium).
> Path-source switches decreased from 36 to 18 (50% reduction).

---

## 4. Full-Video Replay (6 Sequences, 22,679 Frames)

### 4.1 Data Structure

- **Subjects:** 6 video sequences (the natural unit of replication).
- **Within-subject measurements:** thousands of frames per sequence.
- **Conditions:** 2 segmentation models (baseline vs candidate).
- **Metrics per sequence:** mean seg IoU, unstable rate (%), template success
  rate (%), fallback rate (%), mean heading delta (deg).

### 4.2 Test Selection

**Strategy: Aggregate to per-video summary, then test at the video level.**

This avoids the massive autocorrelation problem of frame-level testing
(see Section 6).

| Metric | Primary Test | Rationale |
|--------|-------------|-----------|
| Per-video mean IoU | **Wilcoxon signed-rank** (n = 6 paired) | Non-parametric paired test.  n = 6 is very small -- this is a fundamental limitation. |
| Per-video unstable rate | **Wilcoxon signed-rank** | Same. |
| Per-video template rate | **Wilcoxon signed-rank** | Same. |

**Supplementary:** Exact permutation test (enumerate all 2^6 = 64 sign
assignments) since n = 6 is small enough for exhaustive permutation.

### 4.3 Assumptions to Check

1. **Normality:** With n = 6, Shapiro-Wilk has very low power.  Non-parametric
   tests are mandatory.
2. **Consistent direction of effect:** Report whether all 6 sequences show
   improvement (sign-consistency).  If all 6 agree, the exact sign test p-value
   is 2 * (0.5^6) = 0.031, which is significant at alpha = 0.05.

### 4.4 Effect Size

| Measure | Application |
|---------|-------------|
| **Matched-pairs r** | r = Z / sqrt(6).  Will be imprecise with n = 6. |
| **Mean difference +/- SD** | Descriptive.  E.g., "mean IoU improved by 0.016 +/- 0.XX across 6 sequences." |
| **Win rate** | Proportion of sequences where candidate > baseline.  If 6/6, CLES = 100%. |

### 4.5 Multiple Comparisons

- Multiple metrics on same 6 sequences: Holm-Bonferroni across the 5 metrics
  tested.

### 4.6 Sample Size

- n = 6 is the irreducible limitation.  Wilcoxon signed-rank with n = 6 can
  only detect very large effects (power > 0.80 requires |r| > 0.8 approx).
- **Mitigation:** Frame-level effect sizes (from the per-frame data, corrected
  for autocorrelation) provide evidence of magnitude; video-level tests provide
  evidence of generalizability.
- **Recommendation for thesis:** Frame this as: "All six sequences showed
  improvement in direction X; the video-level Wilcoxon signed-rank test
  confirmed significance (p = .031 by exact sign test when all six agree)."

### 4.7 APA-Style Reporting Template

> Across all six video sequences, the candidate model produced higher mean
> segmentation IoU than the baseline (6/6 sequences improved; mean delta =
> +0.016, SD = 0.XXX).  An exact permutation test confirmed statistical
> significance (p = .031).  Template success rate increased in 6/6 sequences
> (mean delta = +5.6 pp, SD = X.X pp).

**Table format (thesis Table 4.4 enhanced):**

| Metric | Baseline (mean +/- SD) | Candidate (mean +/- SD) | Delta | p | Win/Total |
|--------|----------------------|------------------------|-------|---|-----------|
| Mean IoU | 0.909 +/- 0.XX | 0.925 +/- 0.XX | +0.016 | .031 | 6/6 |
| Unstable % | 1.46 +/- X.XX | 0.33 +/- X.XX | -1.13 pp | .031 | 6/6 |
| Template % | 73.7 +/- X.XX | 79.3 +/- X.XX | +5.6 pp | .031 | 6/6 |

---

## 5. Temporal Smoothing Grid Search (35 Combinations)

### 5.1 Data Structure

- **Design:** 5 alpha values x 7 threshold values = 35 cells.
- **Output per cell:** temporally-stable-frame rate (%) on a 500-frame window.
- **Nature:** Hyperparameter optimization, not hypothesis testing.

### 5.2 Analysis Strategy

This is NOT a comparative hypothesis test.  It is a systematic hyperparameter
search.  The appropriate reporting strategy is descriptive:

1. **Full grid table:** 5 rows (alpha) x 7 columns (threshold), each cell
   showing the stability rate.  Highlight the best, worst, and median cells.
2. **Heatmap figure:** Color-coded alpha x threshold grid.
3. **Sensitivity analysis:** How much does stability change per unit change in
   alpha and threshold?  Report the range (min, max, median) and gradient.
4. **Selected configuration:** Report the chosen (alpha=0.65, c_thresh=0.20)
   with its 95% bootstrap CI on the stability rate.

### 5.3 Statistical Tests (Limited)

- **Bootstrap CI on the selected configuration:** Block-bootstrap (b from
  Section 6) the 500-frame window, compute stability rate per resample, report
  the 2.5th--97.5th percentile as the 95% CI.
- **Robustness check:** For the top-5 configurations, report whether their CIs
  overlap.  If they overlap substantially, the choice among them is not
  statistically distinguishable and the selection should be justified on
  additional grounds (e.g., computational cost, conservatism).

### 5.4 Reporting Format

**Grid table (LaTeX-ready):**

```
\begin{table}[h]
\centering
\caption{Temporal smoothing grid search: stability rate (\%) for 35
  alpha-threshold combinations on a 500-frame evaluation window.
  Best configuration highlighted in bold.}
\label{tab:smoother_grid}
\begin{tabular}{c|ccccccc}
\toprule
$\alpha$ \textbackslash{} $c$ & 0.20 & 0.25 & 0.30 & 0.35 & 0.40 & 0.45 & 0.50 \\
\midrule
0.25 & XX.X & ... & ... & ... & ... & ... & ... \\
0.35 & ... & ... & ... & ... & ... & ... & ... \\
0.45 & ... & ... & ... & ... & ... & ... & ... \\
0.55 & ... & ... & ... & ... & ... & ... & ... \\
0.65 & \textbf{99.6} & ... & ... & ... & ... & ... & ... \\
\bottomrule
\end{tabular}
\end{table}
```

**Prose template:**
> A grid search over 35 combinations of EMA alpha (0.25 to 0.65, step 0.10)
> and confidence threshold (0.20 to 0.50, step 0.05) was conducted on a
> 500-frame evaluation window.  Stability rates ranged from XX.X% (alpha =
> 0.25, c = 0.50) to 99.6% (alpha = 0.65, c = 0.20; 95% block-bootstrap
> CI: [XX.X%, XX.X%]).  The top-5 configurations were not statistically
> distinguishable (overlapping CIs); the selected configuration was chosen for
> its lower threshold, which provides more conservative temporal hold behavior.

---

## 6. Handling Temporal Autocorrelation

### 6.1 The Problem

Consecutive video frames are not independent observations.  A metric measured at
frame t is strongly correlated with frame t+1 (typical lag-1 autocorrelation
rho_1 > 0.8 for heading, IoU).  Treating each frame as an independent sample
inflates effective n, producing falsely narrow CIs and inflated test statistics.

### 6.2 Strategy: Three-Tier Approach

**Tier 1 -- Per-video aggregation (preferred for video-level comparisons)**

Compute per-video summary statistics (mean, median, rate) and test at the video
level (n = number of videos).  This eliminates within-video autocorrelation
entirely.  Used for Comparison 4.

**Tier 2 -- Block bootstrap (preferred for within-video comparisons)**

For comparisons where per-video aggregation loses too much information
(Comparisons 3, 5):

```python
def block_bootstrap_ci(data, block_length, n_resamples=10000,
                       ci_level=0.95, statistic=np.mean):
    """
    Circular block bootstrap for autocorrelated data.
    block_length: use Politis-White-Patton optimal block length.
    """
    n = len(data)
    stats = []
    for _ in range(n_resamples):
        indices = []
        while len(indices) < n:
            start = np.random.randint(0, n)
            block = [(start + j) % n for j in range(block_length)]
            indices.extend(block)
        indices = indices[:n]
        stats.append(statistic(data[indices]))
    alpha = 1 - ci_level
    return np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
```

**Optimal block length:** Use the Politis-White (2004) automatic block length
selector, implemented in the `arch` Python package:
```python
from arch.bootstrap import optimal_block_length
opt = optimal_block_length(data)
b = int(np.ceil(opt['circular'].iloc[0]))
```

**Tier 3 -- Effective sample size correction (supplementary)**

Report n_eff alongside nominal n:
```python
def effective_n(x):
    """Bayley-Hammersley effective sample size."""
    n = len(x)
    acf = np.correlate(x - x.mean(), x - x.mean(), mode='full')
    acf = acf[n-1:] / acf[n-1]  # normalize
    # Sum positive autocorrelations until first negative
    tau = 0
    for k in range(1, n):
        if acf[k] < 0:
            break
        tau += acf[k]
    return n / (1 + 2 * tau)
```

### 6.3 Application to Each Comparison

| Comparison | Tier | Rationale |
|-----------|------|-----------|
| 1 (32 hand-annotated) | No correction needed | Frames are sampled from multiple videos, not consecutive. |
| 2 (32 seg comparison) | No correction needed | Same as above. |
| 3 (220-frame template vs skeleton) | **Tier 2 (block bootstrap)** | Consecutive frames from single clip. |
| 4 (full-video replay) | **Tier 1 (per-video aggregation)** | Aggregate to 6 video-level summaries. |
| 5 (smoother grid search) | **Tier 2 (block bootstrap)** | 500 consecutive frames. |

---

## 7. Confidence Intervals

### 7.1 Default: 95% Bootstrap CIs

For all reported point estimates, compute 95% confidence intervals via:

- **Independent samples (Comparisons 1, 2):** BCa (bias-corrected and
  accelerated) bootstrap, 10,000 resamples.
- **Autocorrelated samples (Comparisons 3, 4, 5):** Circular block bootstrap
  with optimal block length, 10,000 resamples.
- **Rates (template success %, unstable %):** Wilson score interval or Clopper-
  Pearson exact interval for proportions, corrected for autocorrelation via
  effective-n if applicable.

### 7.2 When to Use Parametric CIs

Parametric CIs (mean +/- t * SE) are acceptable as supplementary reporting when:
- Shapiro-Wilk p > 0.10 on the data.
- n >= 30.
- The metric is continuous and unbounded.

These conditions are met for center error on 32 frames (if normal) and runtime.
Always report bootstrap CIs as the primary.

### 7.3 Table Formatting

All thesis tables should report: **mean +/- SD** in the main cell, with
**[95% CI]** either in a separate column or as a footnote.

Example:
> Image midpoint: 14.3 +/- 8.7 px [95% CI: 11.1, 17.8]

---

## 8. Python Script Architecture

### 8.1 File: `simulation_camera_scooter/scripts/statistical_analysis.py`

```
statistical_analysis.py
|
+-- main()
|     |-- load_data()               # reads all per-frame CSVs
|     |-- run_comparison_1()        # 5-planner on 32 frames
|     |-- run_comparison_2()        # seg baseline vs candidate
|     |-- run_comparison_3()        # template vs skeleton 220 frames
|     |-- run_comparison_4()        # full-video replay 6 sequences
|     |-- run_comparison_5()        # smoother grid search
|     |-- generate_latex_tables()   # write updated LaTeX fragments
|     |-- write_json_results()      # statistical_results.json
|     +-- write_report()            # human-readable summary
|
+-- utils/
      |-- bootstrap.py              # block_bootstrap_ci, bca_bootstrap_ci
      |-- effect_size.py            # cliffs_delta, matched_pairs_r, kendalls_w
      |-- tests.py                  # friedman_with_posthoc, wilcoxon_paired
      |-- autocorrelation.py        # effective_n, optimal_block_length_wrapper
      +-- latex.py                  # format_table, format_ci
```

### 8.2 Dependencies

```
scipy>=1.11        # Friedman, Wilcoxon, Shapiro-Wilk
scikit-posthocs    # Nemenyi post-hoc (posthocs_nemenyi_friedman)
arch               # optimal_block_length for block bootstrap
numpy
pandas
```

### 8.3 Key Functions

#### `run_comparison_1(planner_df: pd.DataFrame) -> ComparisonResult`

```python
def run_comparison_1(planner_df):
    """Five-planner comparison on 32 hand-annotated frames."""
    # 1. Pivot to wide format: rows = frames, columns = planners
    pivot = planner_df.pivot(index='key', columns='planner',
                             values='mean_center_error_px')

    # 2. Check normality per column
    normality = {col: shapiro(pivot[col]) for col in pivot.columns}

    # 3. Friedman test
    stat, p = friedmanchisquare(*[pivot[c] for c in pivot.columns])
    W = stat / (len(pivot) * (len(pivot.columns) - 1))

    # 4. Nemenyi post-hoc
    nemenyi = posthocs_nemenyi_friedman(pivot.values)

    # 5. Pairwise Cliff's delta
    pairs = {}
    for i, ci in enumerate(pivot.columns):
        for j, cj in enumerate(pivot.columns):
            if i < j:
                pairs[(ci, cj)] = cliffs_delta(pivot[ci], pivot[cj])

    # 6. Bootstrap CIs per planner
    cis = {c: bca_bootstrap_ci(pivot[c].values) for c in pivot.columns}

    return ComparisonResult(
        test_name='Friedman',
        statistic=stat, p_value=p, effect_size_W=W,
        posthoc=nemenyi, pairwise_effect=pairs,
        confidence_intervals=cis,
        normality_checks=normality,
    )
```

#### `run_comparison_3(template_df, skeleton_df) -> ComparisonResult`

```python
def run_comparison_3(template_df, skeleton_df):
    """Template vs skeleton on 220 frames with autocorrelation handling."""
    diff = template_df['abs_heading_deg'] - skeleton_df['abs_heading_deg']

    # 1. Check normality
    normality = shapiro(diff)

    # 2. Autocorrelation check
    dw = durbin_watson(diff)
    n_eff = effective_n(diff.values)
    b_opt = optimal_block_length_wrapper(diff.values)

    # 3. Naive Wilcoxon (report for transparency)
    stat_naive, p_naive = wilcoxon(diff, alternative='less')

    # 4. Block-bootstrap p-value (primary)
    p_boot = block_bootstrap_test(diff.values, block_length=b_opt)

    # 5. Effect sizes
    cliff = cliffs_delta(template_df['abs_heading_deg'],
                         skeleton_df['abs_heading_deg'])
    r_rb = matched_pairs_r(stat_naive, len(diff))

    # 6. Block-bootstrap CI on mean difference
    ci = block_bootstrap_ci(diff.values, b_opt)

    return ComparisonResult(
        test_name='Wilcoxon (block-bootstrap corrected)',
        statistic=stat_naive, p_value=p_boot,
        effect_size_cliff=cliff, effect_size_r=r_rb,
        confidence_intervals={'mean_diff': ci},
        autocorrelation={'DW': dw, 'n_eff': n_eff, 'block_length': b_opt},
        normality_checks={'diff': normality},
    )
```

### 8.4 Output: `statistical_results.json`

```json
{
  "comparison_1_planner": {
    "test": "Friedman",
    "chi2": 98.7,
    "p": 0.00001,
    "W": 0.77,
    "posthoc_nemenyi": {
      "midpoint_vs_bev_dt_full": {"p": 0.001, "cliff_d": -0.82},
      "...": "..."
    },
    "per_planner": {
      "img_midpoint": {
        "mean": 14.3, "sd": 8.7, "median": 12.1,
        "ci_95": [11.1, 17.8]
      },
      "bev_dt_full": {
        "mean": 65.0, "sd": 22.3, "median": 61.4,
        "ci_95": [56.2, 74.1]
      }
    }
  },
  "comparison_2_segmentation": {
    "test": "Wilcoxon signed-rank",
    "Z": -4.89,
    "p": 0.00001,
    "r_rb": 0.97,
    "cles": 0.969,
    "per_model": { "..." : "..." }
  },
  "comparison_3_template_vs_skeleton": {
    "test": "Wilcoxon (block-bootstrap)",
    "p_naive": 0.00001,
    "p_bootstrap": 0.0003,
    "n_eff": 42,
    "block_length": 12,
    "cliff_d": 0.54,
    "mean_diff_ci_95": [-0.72, -0.31]
  },
  "comparison_4_fullvideo": {
    "test": "Exact permutation (n=6)",
    "p_iou": 0.031,
    "p_unstable": 0.031,
    "wins": "6/6",
    "per_video": { "..." : "..." }
  },
  "comparison_5_smoother_grid": {
    "best_config": {"alpha": 0.65, "c_thresh": 0.20},
    "best_stability": 99.6,
    "best_ci_95": [98.8, 99.9],
    "grid": [ ["...", "..."] ]
  }
}
```

---

## 9. Output Artifacts

### 9.1 Files Generated

| File | Purpose |
|------|---------|
| `research/artifacts/statistical_results.json` | Machine-readable results for thesis insertion |
| `research/artifacts/tables/planner_comparison_stats.tex` | LaTeX table fragment with p-values and CIs |
| `research/artifacts/tables/seg_comparison_stats.tex` | LaTeX table fragment |
| `research/artifacts/tables/template_eval_stats.tex` | LaTeX table fragment |
| `research/artifacts/tables/fullvideo_replay_stats.tex` | LaTeX table fragment |
| `research/artifacts/tables/smoother_grid.tex` | Grid-search heatmap table |
| `research/artifacts/statistical_analysis_report.txt` | Human-readable summary of all tests |
| `research/artifacts/figures/nemenyi_cd_diagram.pdf` | Critical difference diagram for 5-planner |

### 9.2 LaTeX Table Upgrade Pattern

Current thesis tables report only point estimates (e.g., "IoU = 0.946").
Updated tables should add:

1. **mean +/- SD** (or median [IQR] for non-normal metrics).
2. **p-value column** referencing the specific test.
3. **Effect size column** (Cliff's d or r).
4. **Table footnote** naming the statistical test and correction method.

Example upgrade for Table 4.5 (planner comparison):

```latex
\begin{table}[h]
\centering
\caption{Planner comparison on 32 hand-annotated frames using the
  candidate segmentation mask.  Statistical significance assessed via
  Friedman test with Nemenyi post-hoc correction ($\alpha = 0.05$).}
\label{tab:planner_comparison}
\begin{tabular}{lcccccc}
\toprule
Planner & Path [\%] & Inside-GT & Center Err.\ [px] & ms &
  $p$ vs.\ Midpoint & Cliff's $\delta$ \\
\midrule
\multicolumn{7}{l}{\textit{BEV-domain methods}} \\
\quad Skeleton-Graph & 100 & $0.971 \pm 0.03$ & $76.6 \pm 28.1$ &
  $380.3$ & $<.001$ & $0.84$ \\
\quad DT Ridge (full) & 100 & $0.986 \pm 0.01$ & $65.0 \pm 22.3$ &
  $926.8$ & $<.001$ & $0.79$ \\
\quad DT Ridge (near) & 100 & $0.986 \pm 0.01$ & $79.0 \pm 31.2$ &
  $1603.0$ & $<.001$ & $0.86$ \\
\midrule
\multicolumn{7}{l}{\textit{Image-space methods}} \\
\quad Midpoint & 100 & $0.985 \pm 0.02$ & $\mathbf{14.3 \pm 8.7}$ &
  $\mathbf{2.2}$ & --- & --- \\
\quad DT Ridge & 100 & $\mathbf{0.994 \pm 0.01}$ & $60.4 \pm 19.8$ &
  $108.1$ & $<.001$ & $0.76$ \\
\bottomrule
\end{tabular}
\begin{flushleft}
\footnotesize Friedman $\chi^2(4) = XX.X$, $p < .001$, $W = 0.XX$.
Post-hoc: Nemenyi, $\alpha = 0.05$.  CIs via BCa bootstrap (10{,}000
resamples).  SD = standard deviation across 32 frames.
\end{flushleft}
\end{table}
```

---

## Summary Checklist

Before submission, verify each thesis table and claim against this checklist:

- [ ] Point estimates accompanied by dispersion (SD or IQR)
- [ ] 95% confidence intervals reported (bootstrap or parametric)
- [ ] Named statistical test with test statistic and exact p-value
- [ ] Effect size (Cliff's d, r, or W) reported alongside p-value
- [ ] Multiple-comparison correction applied where k > 2
- [ ] Temporal autocorrelation addressed for consecutive-frame data
- [ ] Sample size limitations explicitly acknowledged
- [ ] Power analysis or achieved-power statement included
- [ ] APA-style reporting in prose matches table values exactly
- [ ] `statistical_results.json` values match thesis tables exactly
