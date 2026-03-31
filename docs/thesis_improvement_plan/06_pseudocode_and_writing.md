# Task 6: Algorithm Pseudocode, Appendix Plan, and Prose Improvements

Generated from source code analysis (`template_path_planner.py`, `waypoint_turn_planner.py`,
`image_path_planner.py`) and a close reading of `thesis/main.tex` (1,497 lines).

---

## Part 1: Algorithm Pseudocode (LaTeX)

### Algorithm 1: Template-Approval Planner

Source: `template_path_planner.py`, functions `approve_template_bank()`,
`score_template_against_corridor()`, `_prioritize_candidates()`.

```latex
\begin{algorithm}[t]
\caption{Template-Approval Planner (\textsc{ApproveTemplateBank})}
\label{alg:template_approval}
\begin{algorithmic}[1]
\Require BEV mask $M$ (binary, $H \times W$), previous path $\mathbf{p}_{\mathrm{prev}}$, previous family $f_{\mathrm{prev}}$, GPS intent $\iota$, obstacle zones $\mathcal{O}$
\Ensure \textsc{TemplateApprovalResult}: selected path, approval flag, confidence

\Statex \textbf{--- Phase A: Corridor Extraction ---}
\State $\mathcal{C} \gets \Call{CorridorFromMask}{M}$
    \Comment{Per-row left/right boundary scan}
\For{each sampled row $r$ from bottom to top, step 4\,px}
    \State $\mathbf{x}_s \gets$ indices of nonzero pixels in $M[r]$
    \State Group $\mathbf{x}_s$ into contiguous runs; filter runs with width $< w_{\min}$
    \State Select best run by: $\operatorname{score} = \text{width} - \lvert \text{mid} - \text{ref} \rvert / b_{\text{center}} - \lvert \text{mid} - x_{\text{ego}} \rvert / b_{\text{image}}$
    \State Record $(\ell_r, r_r, c_r)$ as left, right, center for row $r$
    \State Update reference center: $\text{ref} \gets 0.65 \cdot \text{ref} + 0.35 \cdot c_r$
\EndFor
\State Compute corridor confidence $\gamma_{\mathcal{C}}$ from coverage, near-field, span, width consistency, occupancy

\Statex \textbf{--- Phase B: Template Bank Generation ---}
\State $\mathcal{T} \gets \{T_1, \ldots, T_7\}$: straight, left\{gentle, medium, sharp\}, right\{gentle, medium, sharp\}
\For{each template spec (family, turn\_start, end\_heading)}
    \State Build circular arc: straight segment $[0, \text{turn\_start}]$ then arc with $R = \Delta x_{\text{fwd}} / \sin\theta$
    \State Resample at $\Delta s = 0.25$\,m; discard if $\kappa_{\max} > 0.90$\,m$^{-1}$
\EndFor
\If{$\iota \in \{\text{left}, \text{right}\}$}
    \State Filter $\mathcal{T}$ to templates with $\text{family} = \iota$
\EndIf

\Statex \textbf{--- Phase C: Score Each Template Against Corridor ---}
\For{each template $T_k \in \mathcal{T}$}
    \State Interpolate corridor boundaries $(\ell_q, r_q, c_q)$ at each path sample $x$-position
    \State $\rho_{\text{contain}} \gets$ fraction of supported samples inside $[\ell_q, r_q]$
    \State $\rho_{\text{near}} \gets$ containment ratio restricted to $x \le 1.2$\,m
    \State $s_{\text{clear}} \gets$ mean $\min(y - \ell_q,\; r_q - y)$ clamped to $[0, 0.20\text{\,m}]$
    \State $s_{\text{center}} \gets 1 - \text{mean}\bigl(\lvert y - c_q \rvert / (0.5 \cdot w_q)\bigr)$
    \State $s_{\text{cont}} \gets$ continuity score vs.\ $\mathbf{p}_{\mathrm{prev}}$ at probes $\{0.8, 1.5, 2.4\}$\,m
    \State $s_{\text{curv}} \gets 1 - \kappa_{\max}(T_k) / \kappa_{\text{limit}}$
    \State $s_{\text{prog}} \gets \max(x_{\text{inside}}) / L_{\text{horizon}}$
    \State $s_{\text{evid}} \gets 0.55 \cdot \gamma_{\mathcal{C}} + 0.45 \cdot \rho_{\text{support}}$
    \State $\pi_{\text{obs}} \gets$ obstacle overlap penalty from $\mathcal{O}$
    \State $S_k \gets 0.22\,\rho_{\text{contain}} + 0.18\,\rho_{\text{near}} + 0.16\,s_{\text{clear}} + 0.10\,s_{\text{center}} + 0.14\,s_{\text{cont}} + 0.08\,s_{\text{curv}} + 0.06\,s_{\text{prog}} + 0.06\,s_{\text{evid}} - 0.30\,\pi_{\text{obs}}$
    \State $\text{approved}_k \gets \rho_{\text{support}} \ge 0.45 \;\wedge\; \rho_{\text{contain}} \ge 0.60 \;\wedge\; \rho_{\text{near}} \ge 0.72 \;\wedge\; \kappa_{\max} \le 0.90$
\EndFor

\Statex \textbf{--- Phase D: Prioritize with Hysteresis ---}
\State Sort candidates by $S_k$ descending
\If{$f_{\mathrm{prev}}$ is defined}
    \State Add family-reuse bonus $+0.06$ to templates matching $f_{\mathrm{prev}}$
    \If{best candidate $\neq f_{\mathrm{prev}}$ and reuse candidate within switch margin $0.05$}
        \State Promote reuse candidate to top
    \EndIf
\EndIf
\If{previous family was straight and straight candidate within margin $0.03$ of best}
    \State Promote straight candidate to top \Comment{Straight preference}
\EndIf

\Statex \textbf{--- Phase E: Final Decision ---}
\State $T^* \gets$ top-ranked candidate
\State $m \gets S^* - S_{\text{runner-up}}$ \Comment{Approval margin}
\State \textbf{approved} $\gets$ $\text{approved}^* \;\wedge\; S^* \ge 0.52 \;\wedge\; m \ge 0.03$
\If{not approved and $f_{\mathrm{prev}} = T^*.\text{family}$ and reuse conditions met}
    \State $\text{reuse} \gets \text{true}$ \Comment{Family-reuse hysteresis fallback}
\EndIf
\State Compute confidence, slowdown, recommend-hold flags
\State \Return $(\text{path}_{T^*},\; \text{approved},\; \text{confidence},\; T^*.\text{family})$
\end{algorithmic}
\end{algorithm}
```


### Algorithm 2: Waypoint-Turn Planner

Source: `waypoint_turn_planner.py`, functions `plan_waypoint_turn()`,
`_extract_commanded_side_support()`, `_select_target()`, `_build_turn_path()`,
`_gate_path_containment()`.

```latex
\begin{algorithm}[t]
\caption{GPS-Conditioned Waypoint-Turn Planner (\textsc{PlanWaypointTurn})}
\label{alg:waypoint_turn}
\begin{algorithmic}[1]
\Require Corridor $\mathcal{C}$, GPS intent $\iota \in \{\text{left}, \text{right}\}$, BEV mask $M$, previous target $\tau_{\mathrm{prev}}$
\Ensure \textsc{WaypointTurnResult}: active flag, path, confidence, hold recommendation

\If{$\iota \notin \{\text{left}, \text{right}\}$}
    \State \Return inactive result \Comment{Module only activates for commanded turns}
\EndIf

\Statex \textbf{--- Step 1: Decision Band Scan ---}
\State Define forward band $[d_{\min}, d_{\max}] = [2.0, 7.0]$\,m
\State Convert band to pixel rows $[r_{\min}, r_{\max}]$
\For{each row $r$ in band, step 4}
    \State $\mathbf{x}_s \gets$ nonzero columns of $M[r]$
    \State Count pixels on each side of ego: $n_L$, $n_R$
    \If{$\iota = \text{left}$ and $n_L > n_R + \delta_{\text{asym}}$ and $n_L \ge 5$}
        \State Record supported row with lateral midpoint $= 0.5 \cdot x_{L,\text{bound}}$
    \ElsIf{$\iota = \text{right}$ and $n_R > n_L + \delta_{\text{asym}}$ and $n_R \ge 5$}
        \State Record supported row with lateral midpoint $= 0.5 \cdot x_{R,\text{bound}}$
    \EndIf
\EndFor
\State $\sigma \gets \lvert\text{supported rows}\rvert / \lvert\text{band rows}\rvert$
    \Comment{Support score}

\Statex \textbf{--- Step 2: Support Clustering ---}
\State Sort supported rows by forward distance
\State Group into clusters: new cluster when forward gap $> 0.5$\,m
\State Rank clusters by size; best cluster midpoint $\to (\hat{f}, \hat{l})$

\Statex \textbf{--- Step 3: Target Gating with Hysteresis ---}
\If{$\sigma < \sigma_{\text{acquire}} = 0.40$}
    \State \Return hold result with low confidence \Comment{Target gate fails}
\EndIf
\State $\text{apex} \gets (\hat{f}, \hat{l})$
\State $\text{exit}.\text{fwd} \gets \min(\hat{f} + 2.0,\; L_{\text{horizon}})$
\State $\text{exit}.\text{lat} \gets 0.4 \cdot \hat{l} + 0.6 \cdot c_{\text{corridor}}(\text{exit}.\text{fwd})$
    \Comment{Rejoin toward center}

\Statex \textbf{--- Step 4: Circular Arc Fitting ---}
\State $\theta \gets 2 \arctan\!\bigl(\lvert l_{\text{end}} \rvert / f_{\text{end}}\bigr)$, \quad clamp to $[0.001, 1.35]$\,rad
\State $R \gets f_{\text{end}} / \sin\theta$, \quad clamp $R \ge 0.25$\,m
\State Sample $n$ points along arc: $x_i = R\sin\phi_i$, \; $y_i = \pm R(1 - \cos\phi_i)$
\State Scale to match endpoint exactly; enforce monotonic forward

\Statex \textbf{--- Step 5: Containment Gating ---}
\State Interpolate corridor $\ell(x)$, $r(x)$ at each path sample
\State $\rho_{\text{all}} \gets$ fraction of samples with $\ell(x) - 0.05 \le y \le r(x) + 0.05$
\State $\rho_{\text{near}} \gets$ fraction of near-field samples ($x \le 1.2$\,m) inside corridor
\If{$\rho_{\text{all}} \ge 0.60$ and $\rho_{\text{near}} \ge 0.70$}
    \State $\gamma \gets 0.35\sigma + 0.30\rho_{\text{all}} + 0.20\rho_{\text{near}} + 0.15\gamma_{\mathcal{C}}$
    \State \Return active result with path, confidence $\gamma$
\Else
    \State \Return hold result \Comment{Path gate fails; safe hold with slowdown}
\EndIf
\end{algorithmic}
\end{algorithm}
```


### Algorithm 3: Image-Space Midpoint Planner

Source: `image_path_planner.py`, class `CameraMidpointPlanner`, functions
`_preprocess_mask()`, `_connected_center_component()`, `_row_run_midpoint()`, `_smooth_cols()`.

```latex
\begin{algorithm}[t]
\caption{Image-Space Midpoint Planner (\textsc{CameraMidpointPlan})}
\label{alg:img_midpoint}
\begin{algorithmic}[1]
\Require Camera-space segmentation mask $M_{\text{img}}$ ($H_o \times W_o$, binary)
\Ensure Path $\mathbf{p}$ (ordered pixel coordinates), heading $\theta$, confidence $\gamma$

\Statex \textbf{--- Preprocessing ---}
\State Resize $M_{\text{img}}$ to work size $(W_w, H_w)$ via nearest-neighbor interpolation
\State Apply morphological close (elliptical kernel) then open (smaller elliptical kernel)

\Statex \textbf{--- Connected-Component Filtering ---}
\State Compute connected components with 8-connectivity
\State Define ego band: bottom 10\% of rows
\For{each component $C_i$}
    \State $s_i \gets \text{area}(C_i) + 0.5 \cdot \text{area}(C_i) \cdot \mathbb{1}[\text{touches ego band}] + 0.35 \cdot \text{area}(C_i) \cdot (1 - \lvert c_{x,i} - W_w/2 \rvert / W_w)$
\EndFor
\State Retain only the component $C^*$ with highest score $s$

\Statex \textbf{--- Per-Row Boundary Detection ---}
\State $x_{\text{ref}} \gets W_w / 2$ \Comment{Initial reference center}
\For{each row $y$ from $(H_w - 1)$ to $\lceil 0.22 \cdot H_w \rceil$, step 2}
    \State $\mathbf{x}_s \gets$ nonzero pixel indices in $C^*[y]$
    \If{$\lvert \mathbf{x}_s \rvert = 0$} \textbf{continue} \EndIf
    \State Group $\mathbf{x}_s$ into contiguous runs
    \State Filter runs with width $\ge w_{\min} = 6$\,px
    \State Select run closest to $x_{\text{ref}}$: $\; x_L, x_R \gets$ run endpoints
    \State $x_{\text{mid}}(y) \gets (x_L + x_R) / 2$
    \State Record width $w_y \gets x_R - x_L + 1$
    \State Update $x_{\text{ref}} \gets 0.65 \cdot x_{\text{ref}} + 0.35 \cdot x_{\text{mid}}(y)$
\EndFor

\Statex \textbf{--- Savitzky--Golay Smoothing ---}
\If{$\lvert \text{valid rows} \rvert \ge 9$}
    \State $\hat{x}_{\text{mid}} \gets \Call{SavGolFilter}{x_{\text{mid}},\; \text{window}=9,\; \text{poly}=2}$
\Else
    \State $\hat{x}_{\text{mid}} \gets$ 3-tap triangular smoothing of $x_{\text{mid}}$
\EndIf

\Statex \textbf{--- Output ---}
\State $\mathbf{p} \gets \{(\hat{x}_{\text{mid}}(y),\; y) : y \in \text{valid rows}\}$
\State Scale $\mathbf{p}$ back to original resolution $(W_o, H_o)$
\State $\theta \gets \arctan\!\bigl(\Delta x / \Delta y\bigr)$ from ego to 40\%-along point
\State $\gamma \gets$ function of forward span, mean clearance, coverage
\State \Return $(\mathbf{p}, \theta, \gamma)$
\end{algorithmic}
\end{algorithm}
```

---

## Part 2: Appendix Structure Plan

### Appendix A: Algorithm Pseudocode

Content: The three algorithms above plus brief prose connecting each to its
thesis section.

| Algorithm | Lines | Referenced Section |
|-----------|------:|-------------------|
| Template-Approval Planner (Alg.~1) | ~45 | Sec.~3.8.3 (`\ref{sec:template_planner}`) |
| Waypoint-Turn Planner (Alg.~2) | ~35 | Sec.~3.8.6 (`\ref{sec:waypoint_turn_method}`) |
| Image-Space Midpoint Planner (Alg.~3) | ~30 | Sec.~3.8.4 (`\ref{sec:img_midpoint}`) |

Each algorithm should be preceded by a 2-3 sentence paragraph noting:
(a) the source module, (b) which thesis section describes the method in prose,
and (c) any simplifications made for clarity (e.g., error-handling paths omitted).


### Appendix B: Configuration Parameters

A table of the key `config.py` constants organized by subsystem. The table should
use `threeparttable` for footnotes. Suggested columns:
`Parameter | Value | Unit | Description | Section`.

Recommended subsystem groupings:

| Group | Example Parameters | Count |
|-------|-------------------|------:|
| Segmentation | `SEG_INPUT_RES`, `SEG_IOU_FAIL`, `SEG_IOU_WARN`, `SEG_FAIL_HOLD_FRAMES` | 6 |
| BEV Projection | `BEV_SIZE`, `NAV_BEV_FORWARD_M`, `NAV_BEV_LATERAL_M`, `BEV_EGO_X_FRAC` | 5 |
| Template Planner | `path_horizon_m`, `min_containment_ratio`, `min_near_containment_ratio`, `approval_threshold`, `family_reuse_bonus`, `straight_preference_margin` | 10 |
| Waypoint-Turn | `WAYPOINT_DECISION_BAND_*`, `WAYPOINT_ACQUIRE_SUPPORT_MIN`, `WAYPOINT_SUSTAIN_SUPPORT_MIN`, `WAYPOINT_PATH_CONTAINMENT_MIN`, `WAYPOINT_NEAR_CONTAINMENT_MIN` | 12 |
| Obstacle Detection | `YOLO_CONF_THRESH`, `OBSTACLE_CLOSE_M`, `OBSTACLE_STOP_M`, `BEV_OBSTACLE_PENALTY_WEIGHT` | 6 |
| Speed Profile | `SPEED_MAX`, `SPEED_TURN`, `SPEED_SHARP_TURN`, `SPEED_OBSTACLE_NEAR`, `SPEED_STOP` | 5 |
| Temporal Smoothing | `MASK_SMOOTH_ALPHA`, `PATH_SMOOTH_MIN_ALPHA`, `PATH_SMOOTH_MAX_ALPHA`, `HEADING_SMOOTH_ALPHA`, `HEADING_SMOOTH_RESET_DEG` | 8 |
| Safety Gates | `SEG_IOU_FAIL`, `SPEED_SEG_UNSTABLE`, `TURN_PATH_IGNORE_FIRST_SAMPLES` | 4 |
| **Total** | | **~56** |


### Appendix C: Temporal Smoothing Grid Search Results

A 35-row table (7 alpha values x 5 threshold values) from Sec.~5.7.

| Column | Type |
|--------|------|
| $\alpha$ | float (0.25, 0.30, ..., 0.65) |
| $c_{\text{thresh}}$ | float (0.20, 0.25, ..., 0.50) |
| Stable Frame Rate [%] | float |
| Mean Temporal IoU | float |
| Unstable Count | int |

Highlight the selected operating point ($\alpha = 0.65$, $c_{\text{thresh}} = 0.20$) with bold.
Add a brief paragraph interpreting the sensitivity surface: alpha dominates,
threshold has secondary effect, and the selected point sits on a plateau.


### Appendix D: Per-Video Evaluation Breakdown

A table with one row per video (6 evaluation videos + 4 newer VID sequences = up to 10 rows).

| Column | Description |
|--------|-------------|
| Video ID | e.g., IMG\_1876 |
| Frames | count |
| Mean Seg IoU | float |
| Unstable Rate [%] | float |
| Template Rate [%] | float |
| Fallback Rate [%] | float |
| Mean Heading $\Delta$ [deg] | float |
| Mean FPS | float |

This table supports the claim that trends are consistent across sequences
rather than driven by a single favorable video.

---

## Part 3: Prose Improvement Findings

### 3.1 Numbers Repeated Verbatim 4+ Times

The following numeric phrases appear 4 or more times in the thesis body text.
Line numbers refer to `thesis/main.tex`.

| Repeated Value | Occurrences | Lines (representative) | Recommendation |
|---------------|:-----------:|------------------------|----------------|
| `14.3~px` (lateral center error) | 6 | 212, 264, 644, 1060, 1110, 1460 | Define `\newcommand{\midpointCenterErr}{14.3}` and use `\SI{\midpointCenterErr}{\px}` everywhere. Reduces maintenance risk if the number changes. |
| `421` / `$421\times$` (speedup factor) | 5 | 212, 264, 644, 1060, 1460 | Define `\newcommand{\bevSpeedup}{421}`. |
| `926.8~ms` (BEV DT runtime) | 5 | 212, 624, 630, 1043, 1460 | Define `\newcommand{\bevDtMs}{926.8}`. |
| `65.0~px` (BEV DT center error) | 4 | 212, 264, 1043, 1460 | Define `\newcommand{\bevDtCenterErr}{65.0}`. |
| `2.2~ms` (midpoint runtime) | 5 | 212, 264, 644, 1060, 1460 | Define `\newcommand{\midpointMs}{2.2}`. |
| `99.3\%` (BEV failure rate) | 5 | 212, 343, 421, 1124, 1153 | Define `\newcommand{\bevFailRate}{99.3}`. |
| `0.946` (student IoU) | 5 | 210, 268, 395, 423, 956 | Define `\newcommand{\studentIoU}{0.946}`. |
| `11.7~ms` (student inference) | 5 | 210, 268, 395, 466, 925 | Define `\newcommand{\studentMs}{11.7}`. |
| `40.6\%` (heading error reduction) | 4 | 214, 636, 1190, 1413 | Define `\newcommand{\headingImprove}{40.6}`. |
| `0\%` containment failure | 4 | 214, 272, 1274, 1328 | Less critical (it is a qualitative result), but could use `\newcommand{\containFailRate}{0}`. |
| `3.7M parameters` (SegFormer-B0) | 5 | 210, 321, 465, 471, 956 | Define `\newcommand{\segformerParams}{3.7M}`. |
| `22679` / `22,000` frames | 4 | 241, 854, 968, 1466 | Already uses `\numprint{22679}` in places; standardize to one form everywhere. |

**Action:** Create a block of `\newcommand` definitions in the preamble (before
`\begin{document}`) for all values appearing 4+ times. This prevents copy-paste
inconsistencies and simplifies future updates.


### 3.2 Paragraphs Exceeding 150 Words with Multiple Ideas

| Location | Approx. Words | Ideas Covered | Suggested Split |
|----------|:------------:|---------------|-----------------|
| Lines 230-235 (Motivation, para 2: "The economic and social...") | ~170 | (a) economic applications, (b) technical requirements, (c) monocular camera advantages | Split after "...small electric vehicle." Start new paragraph with "Among available sensor modalities..." |
| Lines 234 (Motivation, para 3: "Existing approaches...") | ~175 | (a) end-to-end limitations, (b) learned BEV limitations, (c) classical BEV fragility | Split after "...cannot provide." Start new paragraph with "Even classical BEV approaches..." |
| Lines 240-241 (Problem Statement, para 1) | ~155 | (a) problem definition, (b) two open questions, (c) scope constraints | Split after the two open questions. Move "The scope of this work..." to its own paragraph. |
| Lines 246-256 (Approach Overview) | ~185 | (a) pipeline stages, (b) four iterations list, (c) comparison with prior work | Split before "This work complements..." into a separate paragraph. |
| Lines 311-313 (Sec. 2.1, para 2: "A key distinction...") | ~105 but dense | Acceptable length; flagged for density only. | No action needed. |
| Lines 321-328 (Sec. 2.2, para 2-3: "Several lightweight architectures...") | ~160 | (a) lightweight architectures, (b) ensemble approaches, (c) design space trade-off | Split after "...resource-constrained hardware." |
| Lines 337-342 (Sec. 2.3, para 2: "Recent work...") | ~165 | (a) LSS, (b) BEVFormer, (c) Focus on BEV, (d) monocular depth | Split after "...impractical." to separate the MiDaS discussion. |
| Lines 349-357 (Sec. 2.4, para 2: "Agricultural row-following...") | ~160 | (a) agricultural analog, (b) sidewalk differences, (c) open question about domain | Split before "One open question..." |
| Lines 1060-1061 (Sec. 4.3.2, results paragraph) | ~185 | Multiple quantitative comparisons interwoven | Split after the midpoint result. Start new paragraph with "Image-space DT serves as a robust fallback..." |
| Lines 1351-1353 (Discussion, opening para) | ~160 | (a) central finding, (b) iteration progression, (c) design evidence | Split after "...planning strategies, segmentation models, and evaluation scales." |


### 3.3 Passive Voice Candidates for Active Voice

| Line(s) | Passive Construction | Suggested Active Revision |
|---------|---------------------|---------------------------|
| 153 | "His expertise...was invaluable in shaping this work" | Fine in acknowledgments; leave as-is. |
| 210 | "image-space midpoint planning achieves" | Already active; no change. |
| 246 | "This thesis develops a modular pipeline that maps..." | Already active. |
| 329 | "the specific combination...has not been explored" | "No prior work has explored the specific combination..." |
| 341 | "A less-studied limitation of monocular BEV is coverage fragility" | "Researchers have paid less attention to a limitation of monocular BEV: coverage fragility." |
| 377 | "this method is computationally expensive due to per-pixel graph construction" | "per-pixel graph construction makes this method computationally expensive" |
| 389 | "Knowledge distillation transfers representational capacity" | Already active. |
| 406 | "Runtime optimization frameworks...provide cross-platform model deployment" | Already active. |
| 456 | "The experimental platform is an electric scooter equipped with..." | "An electric scooter equipped with...serves as the experimental platform." |
| 466 | "A second challenge arises from the limited training data" | Already active. |
| 484 | "For each unlabeled image...the teacher predicts pixel-wise logits" | Already active. |
| 520 | "Table X summarizes the segmentation model iterations" | Already active. |
| 568 | "A four-point planar homography...is applied" | "The system applies a four-point planar homography..." |
| 587 | "A multi-stage cleanup pipeline is applied" | "The system applies a multi-stage cleanup pipeline" |
| 624 | "Rather than skeletonize, this method computes the EDT" | Already active. |
| 636 | "Evaluation on 220 calibrated frames indicates a 40.6% reduction" | Already active. |
| 698 | "An EMA is applied to the cubic polynomial coefficients" | "The system applies an EMA to the cubic polynomial coefficients" |
| 720 | "YOLOv8-nano was selected for its 3.2M-parameter footprint" | "We selected YOLOv8-nano for its 3.2M-parameter footprint" (or keep passive for objectivity) |
| 837 | "A centralized configuration module serves as the single source of truth" | Already active. |
| 884 | "Metrics are computed per-frame and averaged" | "The evaluation computes metrics per-frame and averages them" |
| 925 | "Table X compares the baseline and improved segmentation models" | Already active. |
| 1315 | "An overnight automated evaluation systematically tested..." | Already active. |

**Summary:** Most of the thesis is already in active voice. About 6-8 sentences
could be strengthened by converting from passive to active. The thesis maintains
appropriate scientific register; aggressive active-voice conversion would be
counterproductive in certain places (e.g., methodology descriptions where the
agent is less important than the action).


### 3.4 Terminology Inconsistencies

| Term Variant A | Term Variant B | Lines (examples) | Recommendation |
|---------------|---------------|-------------------|----------------|
| "template-approval planner" | "template arc planner" | 636 ("also referred to as"), 1018 | Pick one. "Template-approval planner" is more descriptive of the algorithm's nature. Use "template arc planner" only when referring specifically to the arc geometry. |
| "skeleton-graph planner" | "skeleton-graph baseline" / "graph baseline" | 612, 614, 1171, 1190 | Use "skeleton-graph planner" for the method name and "skeleton-graph baseline" only when explicitly comparing. |
| "image-space midpoint planner" | "midpoint planner" / "Img Midpoint" | 641, 1047, 1084 | Use full "image-space midpoint planner" on first mention in each section; "midpoint planner" thereafter. |
| "BEV distance-transform ridge planner" | "BEV DT Ridge" / "DT Ridge (full)" / "DT ridge planner" | 621, 1043-1044 | Standardize prose to "BEV DT ridge planner"; tables can abbreviate to "BEV DT Ridge". |
| "OneFormer Swin-L" | "OneFormer Swin-L teacher" / "high-capacity OneFormer teacher" | 268, 329, 423, 481, 925 | Always use "OneFormer Swin-L" on first reference per section; "OneFormer teacher" for subsequent. |
| "SegFormer-B0" | "SegFormer-B0 student" / "student model" / "compact student" | 210, 465, 471, 956, 1464 | "SegFormer-B0" when referring to architecture; "SegFormer-B0 student" when in distillation context. |
| "campus sidewalk video" | "campus video" / "campus sidewalk sequences" / "University of Oklahoma campus sidewalk video" | 854, 991, 1455 | Standardize to "campus sidewalk video" for brevity; use full university name only in Chapter 4 setup. |
| "containment failure rate" | "containment failure" / "path-outside ratio" | 272, 1274, 1323, 1468 | "Containment failure rate" is the metric; "path-outside ratio" is the measured quantity. Keep both but be consistent about which you mean. |
| "hand-annotated frames" | "hand-labeled frames" | 241, 877, 480 | Standardize to "hand-annotated frames" (used more frequently). |
| "22,679 frames" | "over 22,000 frames" | 241, 854, 968 | Use `\numprint{22679}` everywhere for the exact count; "over 22,000" only in the abstract. |
| "1,800-frame" | "1800-frame" | 5, 216, 1288, 1328 | Standardize to "1,800-frame" (with comma) in prose; "1800" in tables. |
| "v1/v2/v3/v4" | "Iteration 1/2/3/4" | 248-254, 1003-1027 | Use "Iteration N" on first mention, then "vN" in parentheses. Be consistent thereafter. |


### 3.5 Discussion Sections That Restate Rather Than Interpret

| Section | Lines | Issue | Recommendation |
|---------|-------|-------|----------------|
| Sec. 6.1 "Interpretation of Key Findings" (lines 1350-1353) | 1350-1354 | The opening paragraph restates the 14.3 px / 2.2 ms / 421x numbers without adding new interpretive depth beyond what Ch. 4 already said. | Replace repetition with a higher-level framing: *why* this finding matters for the field. E.g., "This result challenges the common assumption in mobile robotics that metric-scale BEV reasoning is a prerequisite for accurate path planning." |
| Sec. 6.2 "Why Image-Space Outperforms BEV" (lines 1357-1365) | 1359-1361 | The coverage numbers (0.02%, 600x600 grid) are new and interpretive --- this section is well done. | No change needed. This is the strongest section of the Discussion. |
| Sec. 6.3 "Segmentation as Necessary but Not Sufficient" (lines 1370-1373) | 1370-1373 | Mostly restates findings: "Better segmentation helps all planners" repeats Ch. 4 claims. Only the "diminishing returns for BEV" point is genuinely new. | Expand the diminishing-returns argument. Add a concrete hypothetical: "Even a perfect segmentation model (IoU = 1.0) would not resolve the geometric coverage deficit, as the oracle-mask experiments demonstrate." |
| Sec. 6.4 "Template Planning and Turn Safety" (lines 1377-1384) | 1377-1381 | First paragraph restates Table 5 numbers (40.6%, 50%). The insight about restricting solution space is interpretive but brief. | Expand the interpretive point: why does constraining the solution space help? Connect to the broader propose-and-verify paradigm. Mention computational predictability as a deployment advantage. |
| Sec. 6.4 cont. (lines 1381-1384) | 1381-1383 | "The 1800-frame accepted run demonstrates..." restates Sec. 4.6.3 findings. | Replace with: "The robustness observed over 1,800 frames suggests that the template-approval architecture is not overfitting to the 220-frame calibration clip but rather reflects a stable operating regime." |
| Sec. 6.6 "Broader Implications" (lines 1408-1416) | 1408-1416 | This section is genuinely interpretive and does not restate. | No change needed. One of the best sections. |

**Overall Discussion Assessment:** Sections 6.1, 6.3, and the first half of 6.4
lean toward restating results rather than interpreting them. The recommended fix
is to replace each numeric restatement with either (a) a causal explanation,
(b) a connection to broader literature, or (c) a counterfactual analysis.
Sections 6.2 and 6.6 are strong examples of genuine interpretation and should
serve as models for the weaker sections.

---

## Summary of Recommended Actions

| Priority | Action | Effort |
|----------|--------|--------|
| 1 (High) | Add `\newcommand` macros for 10+ repeated numbers | 15 min |
| 2 (High) | Insert the 3 algorithm pseudocode blocks into Appendix A | 30 min (LaTeX formatting) |
| 3 (Medium) | Split 8-10 long paragraphs as identified in Sec. 3.2 | 20 min |
| 4 (Medium) | Rewrite 3 Discussion subsections to interpret rather than restate | 30 min |
| 5 (Medium) | Standardize 12 terminology pairs per Sec. 3.4 | 20 min |
| 6 (Low) | Create Appendix B (config parameters table) | 30 min |
| 7 (Low) | Create Appendix C (smoothing grid search table) | 15 min |
| 8 (Low) | Create Appendix D (per-video breakdown table) | 15 min |
| 9 (Low) | Convert 6-8 passive-voice sentences | 10 min |
