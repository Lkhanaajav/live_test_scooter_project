# Architecture Research — Thesis Structure Patterns

## Recommended Chapter Structure for This Thesis

### Option A: Standard Structure (Recommended)

| Chapter | Pages | Content |
|---------|-------|---------|
| 1. Introduction | 6-8 | Motivation, problem statement, approach overview, contributions, organization |
| 2. Background & Related Work | 10-14 | Literature review, research gaps |
| 3. System Design | 14-18 | Full pipeline: segmentation, BEV, planning methods, temporal smoothing, control |
| 4. Experimental Evaluation | 14-18 | Setup, segmentation results, planner comparison, BEV fragility, runtime, accepted run |
| 5. Discussion | 6-8 | Interpretation, why image-space wins, limitations, threats to validity |
| 6. Conclusion & Future Work | 4-6 | Summary, key findings, future directions |
| Front/Back matter | 8-12 | Abstract, TOC, lists, bibliography |
| **Total** | **62-84** | |

### Why This Structure

- **Merge Closed-Loop Control into System Design** — the current Chapter 4 (Closed-Loop) is only 2 pages of methodology. It belongs in the System Design chapter as a section.
- **Single Experiments chapter** — avoids the current fragmentation where results are scattered across sections
- **Discussion as its own chapter** — the current Discussion chapter is strong; keep it separate from Results

### Chapter-by-Chapter Recommendations

#### Chapter 1: Introduction (6-8 pages)

**Structure:**
1. **Opening hook** (1 paragraph) — concrete scenario that motivates the work
2. **Problem context** (1-2 paragraphs) — why sidewalk navigation is hard, why monocular
3. **Limitations of existing approaches** (1-2 paragraphs) — what's wrong with BEV, end-to-end, LiDAR
4. **Approach overview** (1-2 paragraphs) — high-level description of the pipeline and design iteration story
5. **Contributions** (numbered list) — specific, verifiable claims
6. **Thesis organization** (1 paragraph)

**Key principle:** The Introduction sets up the contribution. By the end, the reader should know exactly what the thesis claims and why it matters.

#### Chapter 2: Background & Related Work (10-14 pages)

**Structure by theme, not by paper:**
1. Autonomous sidewalk/micro-mobility navigation
2. Semantic segmentation for drivable surfaces
3. Bird's-eye view projection (classical and learned)
4. Path planning for mobile robots (grid-based, sampling, potential field, end-to-end)
5. Distance transform methods in navigation
6. Teacher-student / knowledge distillation for segmentation
7. Embedded perception and real-time constraints
8. **Research gaps and positioning** (critical closing section)

**Key principle:** Each section should end by connecting to a gap that this thesis fills. The final section synthesizes all gaps into the thesis rationale.

#### Chapter 3: System Design (14-18 pages)

**Structure as a design narrative:**
1. System overview and hardware platform
2. Segmentation module and teacher-student training
3. BEV projection and mask refinement
4. Path planning methods (all 5, with design rationale for each)
5. GPS navigation and intent conditioning
6. Obstacle detection and safety mechanisms
7. Temporal smoothing
8. Closed-loop control and serial protocol
9. Software architecture overview

**Key principle:** Present each component with the "why" before the "what". The reader should understand why each design choice was made, not just what the choice was.

**Design iteration framing:** Don't present all 4 iterations sequentially. Instead, present the final system and note where design iterations informed the current approach. The iteration story belongs in the Results chapter as the evaluation progression.

#### Chapter 4: Experimental Evaluation (14-18 pages)

**Structure by claim, not by experiment:**
1. Experimental setup (data, hardware, protocol)
2. Segmentation quality (teacher-student vs baseline → claim: OneFormer teacher significantly improves student quality)
3. Planning domain comparison (BEV vs image-space → claim: image-space is faster and more accurate)
4. BEV fragility analysis (→ claim: monocular BEV is fundamentally unreliable)
5. Design iteration progression (v1→v4 → claim: each iteration improved measurably)
6. Template arc planner evaluation (→ claim: reduces heading jitter and path switching)
7. Turn safety validation (→ claim: containment guard prevents unsafe paths)
8. End-to-end stability (1800-frame run → claim: system is deployment-ready)
9. Runtime analysis (→ claim: meets 10 Hz real-time requirement)

**Key principle:** Each section should state a claim, present evidence, and conclude whether the claim is supported. This makes the Results chapter an argument, not a data dump.

#### Chapter 5: Discussion (6-8 pages)

**Structure:**
1. Summary of key findings
2. Why image-space outperforms BEV (geometric analysis)
3. Implications for monocular perception systems
4. Template planning and safety trade-offs
5. Limitations (approach + evaluation)
6. Threats to validity (internal, external, construct)

**Key principle:** Discussion adds insight that isn't in the Results. If you're just restating numbers, cut it.

#### Chapter 6: Conclusion & Future Work (4-6 pages)

**Structure:**
1. Summary of contributions (map back to Introduction)
2. Key findings (numbered, concrete)
3. Recommended architecture (one paragraph)
4. Future work (prioritized, specific)

## Structural Patterns

### The "Claim-Evidence-Conclusion" Pattern
For each major result:
1. **Claim**: "Image-space midpoint planning achieves lower lateral error than BEV DT planning"
2. **Evidence**: Table X, Figure Y
3. **Conclusion**: "This supports the hypothesis that BEV projection introduces geometric distortion that degrades planning accuracy"

### The "Iteration Progression" Pattern
Show measurable improvement at each stage:
| Iteration | Change | Key Metric Before | Key Metric After |
|-----------|--------|-------------------|------------------|
| v1→v2 | Skeleton → DT corridor | slow, noisy | better paths, still slow |
| v2→v3 | BEV → image-space | 926ms/frame | 2.2ms/frame |
| v3→v4 | Fixed path → template arc | heading jitter | 40% less jitter |

### Chapter Transitions
Every chapter should end with a 1-sentence preview of the next chapter. Every chapter should begin with a 1-sentence summary of what it covers and why.
