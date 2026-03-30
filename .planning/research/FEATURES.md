# Features Research — What Makes a Strong CV/Robotics Thesis

## Table Stakes (must have)

### 1. Clear Problem Statement
- Define the gap explicitly: "No prior work compares BEV vs image-space planning for monocular sidewalk navigation"
- Scope boundaries: what the thesis covers and what it doesn't
- Must be falsifiable: the reader should know what result would disprove the thesis

### 2. Proper Literature Review
- Not a list of papers — a **synthesis** that builds the case for the contribution
- Organize by themes/gaps, not by paper
- End with explicit research gaps that this thesis addresses
- Every cited method should be connected to the thesis contribution (supports, contradicts, or leaves a gap)

### 3. Systematic Evaluation Methodology
- Same test conditions across all compared methods
- Controlled variables: same masks, same hardware, same metrics
- Proper train/test split with no data leakage
- Metrics defined precisely before presenting results

### 4. Baseline Comparisons
**Critical for this thesis:**
- **Naive baseline**: raw segmentation mask with no path planning (just "follow the mask center")
- **Prior art baseline**: closest published approach (if available)
- **Ablation baseline**: each component removed one at a time to show its contribution
- **Cross-domain baseline**: same algorithm in BEV vs image-space (already done — the planner comparison)

### 5. Statistical Rigor
- Report variance/standard deviation, not just means
- For 32 frames: acknowledge small sample size, show consistency across videos
- For 22K frames: aggregate metrics strengthen the case
- Don't overclaim significance from small differences

### 6. Honest Limitations
- The thesis already has a good Limitations section — strengthen it
- Separate "limitations of the approach" from "limitations of the evaluation"
- Each limitation should suggest what would resolve it

### 7. Clear Contribution Statement
- Number contributions (already done)
- Each contribution should be verifiable from the results
- Don't claim contributions you haven't demonstrated

## Differentiators (what separates great from adequate)

### 1. Compelling Narrative Arc
The thesis should tell a **story**:
- **Problem**: Sidewalk navigation is hard, existing approaches are too complex/expensive
- **Hypothesis**: Maybe BEV isn't necessary — image-space planning might be better
- **Investigation**: Systematic comparison of 5 methods across 2 domains
- **Discovery**: Image-space wins decisively — and here's why (geometric coverage)
- **Implication**: Simpler architectures can outperform complex ones when the geometry favors it

### 2. Ablation Studies
- **Segmentation ablation**: oracle mask vs predicted mask (already done)
- **Domain ablation**: same algorithm in BEV vs image-space (already done)
- **Component ablation**: remove temporal smoothing, remove morphological cleanup
- **Teacher ablation**: SegFormer-B2 teacher vs OneFormer teacher (already done)

### 3. Design Iteration Narrative
Show the progression as a scientific journey:
- v1 (skeleton): worked but slow → motivated v2
- v2 (DT corridor): better but still BEV-dependent → motivated domain comparison
- v3 (image-space): breakthrough — faster AND more accurate → led to key finding
- v4 (template arc): refined for stability and junction handling

### 4. Reproducibility
- Config.py as single source of truth (already exists)
- All hyperparameters documented
- Training recipe fully specified
- Evaluation protocol described in enough detail to replicate

### 5. Strong Discussion
- Don't just restate results — explain **why** the results are what they are
- Connect to broader themes: when is geometric simplicity better than learned complexity?
- Implications for other monocular perception systems beyond sidewalks

## Anti-Features (things to avoid)

### 1. Model-vs-Model Without Baseline
Comparing 11 fine-tuned checkpoints against each other tells you which checkpoint is best but provides no scientific insight. **Remove and replace with teacher-student comparison.**

### 2. Development Logs as Results
Tables like "Runtime Offenders Ranked by Cost" read like debugging notes. **Restructure as actionable engineering recommendations in the Discussion.**

### 3. Cherry-Picked Qualitative Examples
Show both success AND failure cases. Include the hard frames, not just the pretty ones.

### 4. Overclaiming
- "First quantitative characterization of BEV fragility" — is it really? Verify with literature search
- "421x speedup" — report this but acknowledge the methods solve slightly different problems
- Don't claim generalization you haven't demonstrated

### 5. Inconsistent Counting
The thesis says "three design iterations" but lists four. Pick one number and be consistent.
