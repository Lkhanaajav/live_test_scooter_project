# Phase 11 Research: Template path fitting inside segmentation corridor with path approval scoring

## Design Correction

This research was initially written assuming the planner might decide among left, right, or straight alternatives directly from vision. That is no longer the intended design.

Locked correction:

- GPS or route logic provides maneuver intent (`straight`, `left`, `right`)
- vision/perception defines the feasible sidewalk corridor
- Phase 11 only approves or rejects paths that are consistent with the commanded intent
- if intent-consistent paths are not well-supported by the corridor, the planner must emit low confidence and slowdown/hold instead of selecting a different maneuver

The rest of this document should be read through that lens: corridor fitting is the geometric job, maneuver choice is external.

## What This Phase Is Solving In This Repo

The current live stack still chooses a path mainly by extracting a skeleton/graph from the BEV sidewalk mask and then scoring graph-derived candidates in `realtime_nav_core.py`. That works when the mask is clean, but the repo history already documents the remaining failure mode: near-ego branch artifacts and low-evidence collapse windows can still produce false turn commitment or stale path lock. Phase 11 should move final path choice from "follow the extracted centerline/graph" to "approve one of a small set of smooth ego-anchored path templates only when corridor evidence supports the commanded intent."

This phase should not replace the controller contract. The approved output still needs to look like the existing `PathPlanResult` contract used by `live_heading_demo.py`: metric path polyline, pixel path polyline, a fitted `CubicPathModel`, and control-facing confidence/slowdown guidance.

## Current Pipeline Integration Points

### Current mask-based path stack

`live_heading_demo.py` currently does:

1. segment sidewalk in camera view
2. warp to BEV and clean the mask
3. optionally paint hard-block obstacle holes into the BEV mask
4. call `BEVPathExtractor.process(bev_sidewalk, obstacle_zones_m=...)`
5. feed `nav_out.path_model` into `AdaptivePurePursuitController`
6. log and render `candidate_paths_*`, `best_path_*`, `path_source`, and `mask_occ_ratio`

Current control limitation:

- the live loop currently gates speed mostly on `has_path`, controller validity, obstacle distance, and segmentation instability; it does not yet consume a planner-native confidence scalar for slowdown/hold behavior
- the live loop also lacks a clean notion of external maneuver intent driving path-family selection

`BEVPathExtractor.process()` currently owns:

- BEV mask preprocessing
- skeleton extraction and graph building
- candidate generation via Dijkstra/DFS over skeleton edges
- candidate scoring with progress, curvature, end-heading continuity, branch hysteresis, center bias, and obstacle overlap
- fallback centerline / fallback skeleton / fallback hold
- conversion into `PathPlanResult`

### Current boundary-based path stack

`boundary_inference.py` already provides a second, cleaner control-facing contract:

- row-wise left/right boundary predictions are decoded into a metric centerline path
- it emits `confidence`, `is_low_confidence`, and `suggested_slowdown`
- it already uses previous-path blending and near-field validity checks

Important repo-specific constraint: this boundary decoder exists, but it is not wired into `live_heading_demo.py` yet. Phase 11 depends on Phase 7 conceptually, but planning should not assume a fully integrated boundary-net live path source already exists.

### Reuse vs replace

Reuse:

- `BEVPathExtractor._preprocess()` and the BEV metric/pixel conversion helpers
- `PathPlanResult` and `CubicPathModel` as the external output contract
- `AdaptivePurePursuitController` unchanged
- existing obstacle-zone interface and obstacle overlap penalty idea
- existing temporal memory: `prev_best_path_m`, `prev_end_heading`, `no_path_counter`
- existing logging/visualization flow in `live_heading_demo.py`
- boundary confidence ideas from `boundary_inference.py`: near-field support, path continuity, slowdown guidance

Replace or demote:

- `_search_candidates_dijkstra()` and `_search_candidates_dfs()` should stop being the primary final path selector for this phase
- graph-specific hysteresis based on `first_edge_sig` should become template-family hysteresis or be retained only for graph fallback
- current candidate scoring is not corridor-fit aware enough; it scores graph geometry, not whether a smooth controller-ready path stays inside the full corridor
- fallback centerline should remain as emergency recovery, not the normal planner

Recommended insertion point:

- keep `BEVPathExtractor.process()` as the stable public entry point
- insert a corridor-to-template approval stage after `_preprocess()` and before graph fallback
- treat graph search as fallback/ablation when template approval fails or when corridor extraction is too incomplete to define useful boundaries

## Recommended Internal Architecture

The cleanest repo-specific design is to add a corridor abstraction and a template planner without breaking the rest of the stack.

Recommended internal split:

1. `corridor_from_mask(mask_255)`:
   - derive corridor center and left/right limits from the cleaned BEV mask
   - reuse row-wise ideas from `boundary_inference.py`: per-row corridor width, valid rows, near-field support, width consistency

2. `corridor_from_boundaries(...)`:
   - accept Phase 7-style left/right boundary predictions later without changing the planner contract
   - produce the same corridor object as the mask-derived path

3. `generate_template_bank(...)`:
   - create a small fixed set of ego-anchored metric candidate paths
   - return per-template polyline points in metric space, plus a template id/family

4. `score_template_against_corridor(...)`:
   - sample each candidate at fixed forward steps
   - score containment, edge clearance, center preference, continuity, curvature feasibility, obstacle overlap, and evidence support

5. `approve_template(...)`:
   - require both an absolute score threshold and a score-margin threshold over the runner-up
   - if no candidate passes, emit low confidence and slowdown/hold instead of forcing a turn

6. `graph_or_hold_fallback(...)`:
   - preserve current graph/fallback recovery logic as a fallback path source, not as the default planner

This structure keeps the current public API stable while allowing both segmentation-mask and future boundary-net inputs to drive the same approval logic.

## Candidate Path Primitive Choices

### Option 1: Constant-curvature arcs

Form:

- start at ego origin
- fixed horizon
- one curvature value per template

Pros for this repo:

- trivial to generate
- explicit curvature bound
- very cheap to sample and score
- naturally matches straight / gentle / medium / sharp turn families

Cons in this repo:

- awkward when the visible corridor starts straight and then bends
- weak fit for corridors whose turn emerges gradually
- harder to stay centered in variable-width or asymmetric masks
- does not match the repo's existing cubic-path controller representation as directly

Verdict:

- useful as a baseline or ablation
- probably too rigid as the main Phase 11 primitive

### Option 2: Ego-anchored cubic lateral polynomials `y(x)` / cubic Hermite templates

Form:

- parameterize by forward distance `x`
- enforce ego anchor near `(0, 0)` with near-zero initial heading
- discretize end lateral offset and end heading, then sample the cubic into `path_m`

Pros for this repo:

- closest match to the existing `CubicPathModel` and controller assumptions
- easy to export as metric polylines and refit with existing cubic utilities
- cheap enough for a fixed small bank
- flexible enough to represent straight and gradual turns inside noisy corridors
- easy to compare to `prev_best_path_m` at the same `x` probes already used in current continuity scoring

Cons in this repo:

- curvature is implicit, so bad parameter combinations must be filtered by sampled curvature checks
- sharper turns can overshoot laterally if endpoint families are too aggressive

Verdict:

- best default choice for this repo
- lowest integration risk

### Option 3: Clothoid-like / curvature-ramp templates

Form:

- curvature starts near zero and ramps over distance
- better matches steering-feasible turn entry

Pros for this repo:

- physically nicer turn initiation
- reduces abrupt heading change near ego

Cons in this repo:

- more implementation and tuning cost than Phase 11 likely needs
- harder to explain and validate quickly in this thesis repo
- not clearly necessary before the current demo integration work is complete

Verdict:

- good future refinement if cubic templates still branch-flip on turn entry
- not the best first implementation

## Recommended Template Bank For This Repo

Use ego-anchored cubic templates as the main family.

Recommended bank shape:

- 1 straight template
- 2 gentle-left + 2 gentle-right
- 2 medium-left + 2 medium-right
- 1 sharp-left + 1 sharp-right

That gives 11 templates total, which is small enough for per-frame scoring on the current CPU-oriented stack.

Recommended parameterization:

- fixed horizon near the current planner horizon (`path_horizon_m`)
- discrete end headings and end lateral offsets rather than free optimization
- sample spacing aligned with current planner resampling (`path_sample_ds_m` or slightly denser for scoring)
- reject any template whose sampled curvature exceeds the existing controller-friendly bound already implied by `spline_kappa_max_m_inv`

Important planning detail:

- precompute normalized template shapes once, then scale/clip to runtime horizon as needed
- use a stable template id/family string so hysteresis can operate on template families instead of graph edge signatures

## Corridor Representation Needed For Scoring

The planner should not score templates directly against raw occupied pixels only. It needs a compact corridor representation that can come from either the mask or future boundary predictions.

Recommended corridor fields:

- `rows_forward_m`
- `center_lateral_m`
- `left_lateral_m`
- `right_lateral_m`
- `width_m`
- `valid_row_mask`
- `near_field_valid_count`
- `valid_ratio`
- `width_cv`
- `forward_span_m`
- `mask_occ_ratio` or boundary confidence

For mask-derived corridors:

- derive left/right boundary per row from the cleaned BEV mask
- ignore rows with too few occupied pixels
- sort rows in forward metric order
- carry over the same width sanity checks already used in `boundary_inference.py`

This lets Phase 11 share one approval/scoring path across both segmentation and boundary-net sources.

## Recommended Scoring Terms

The current repo already learned that center preference and temporal continuity help, but Phase 11 needs stronger corridor-fit terms and explicit approval logic.

Recommended per-template terms:

- `containment_ratio`
  - fraction of sampled template points inside the corridor
  - hard reject if near-field containment fails early

- `edge_clearance_score`
  - average normalized distance to the nearest corridor edge
  - reject templates that run too close to edges for too many samples

- `center_alignment_score`
  - compare template lateral position to corridor center where corridor width is trustworthy
  - reduce this weight when corridor width confidence is low so one-sided evidence does not force the wrong center

- `continuity_score`
  - reuse the current `prev_best_path_m` probing idea at fixed `x` positions
  - compare both lateral offset and heading to the previous approved path

- `curvature_feasibility_score`
  - penalize mean or peak curvature beyond a configured bound
  - keep this as a soft penalty before a hard reject threshold

- `progress_score`
  - favor templates that remain valid further forward inside the corridor
  - similar purpose to current `j_prog`, but tied to corridor support, not only raw length

- `obstacle_penalty`
  - reuse the current obstacle-zone overlap logic against template samples

- `evidence_support_score`
  - combine corridor valid ratio, near-field support, width consistency, and occupancy/confidence
  - this is the repo's missing explicit path-approval confidence signal

Recommended approval rule:

- compute a final normalized score for each template
- require:
  - best score >= approval threshold
  - best score - second-best score >= margin threshold
  - minimum containment ratio
  - minimum near-field support

If any of those fail:

- mark low confidence
- recommend slowdown/hold
- optionally keep previous path briefly using existing hold logic

## Likely Failure Modes In This Repo

### 1. Near-ego false branch capture

This is the known repo failure. A template can still commit too early if scoring does not weight the first 0.8-1.2 m heavily enough. Planning should include a dedicated near-field containment term, not just whole-path average containment.

### 2. Corridor collapse / fragmentation

The overnight notes already show occupancy collapse windows. When the cleaned BEV mask loses forward reach, all template scores should drop and approval should fail cleanly. This is where explicit low confidence matters.

### 3. Over-centering on one-sided evidence

If only one boundary is visible, the planner can hallucinate the center and drift outward. Center preference must be gated by corridor confidence or width consistency.

### 4. Template oscillation between neighboring families

Once templates replace graph branches, the analogous risk is left-gentle vs left-medium or straight vs gentle-left flapping. The current repo's branch hysteresis idea should be adapted to template families plus score-margin gating.

### 5. Curvature-valid but corridor-invalid sharp turns

A template may be physically smooth but exit the segmented sidewalk early. Containment and clearance must dominate over pure smoothness.

### 6. Graph fallback masking planner failure

If graph fallback is too eager, Phase 11 may appear to work while silently reverting to the old selector. Logging must expose which source was actually approved.

### 7. Skip-frame inconsistency

`live_heading_demo.py` currently runs the planner on predicted BEV on skip frames, and those skip-frame calls do not currently pass obstacle zones. Planning should decide whether template approval on skip frames uses cached obstacles or intentionally ignores them and records that fact.

### 8. Contract drift with Phase 7

Phase 7 confidence outputs and Phase 11 approval outputs can diverge if they use different confidence semantics. Planning should define one control-facing meaning for confidence and slowdown.

## What Should Be Logged And Exposed

The repo already logs `planner_mode`, `path_source`, and `mask_occ_ratio`. Phase 11 should extend diagnostics rather than invent a separate ad hoc debug path.

Recommended new `PathPlanResult` diagnostics:

- `approval_confidence`
- `is_low_confidence`
- `suggested_slowdown`
- `approval_score`
- `approval_margin`
- `template_id`
- `template_family`
- `candidate_scores`
- `candidate_debug` with per-template containment, clearance, continuity, and obstacle penalty

Recommended live-loop follow-through:

- `live_heading_demo.py` should consume planner confidence and suggested slowdown, not just `has_path`, when deciding reduced-speed / hold behavior

Recommended `path_source` values:

- `template`
- `template_hold`
- `graph_fallback`
- `fallback_centerline`
- `fallback_skeleton`
- `fallback_hold`

This is important because the overnight notes already called out the need for candidate diagnostics and an explicit BEV confidence scalar.

## Recommended Phase Split Into Executable Plans/Waves

### Wave 1: Corridor abstraction + synthetic tests

Goal:

- define a repo-stable corridor representation from cleaned BEV masks
- add unit tests for straight and turning synthetic corridors

Deliverables:

- mask-to-corridor helper
- synthetic corridor fixtures
- tests for width extraction, valid rows, near-field support, and confidence gating

Why first:

- this removes ambiguity between mask-derived and future Phase 7 boundary-derived inputs

### Wave 2: Template bank + offline scorer

Goal:

- generate the fixed cubic template bank
- score templates against corridor geometry offline without touching the live loop yet

Deliverables:

- template generator
- per-template scoring function
- tests for template containment, edge rejection, obstacle penalty, and approval-threshold behavior

Why second:

- this isolates the core planner logic before live integration

### Wave 3: Integrate into `BEVPathExtractor.process()`

Goal:

- make template approval the primary selection path while preserving `PathPlanResult`

Deliverables:

- `process()` calls template approval after `_preprocess()`
- graph search becomes fallback
- new diagnostics added to `PathPlanResult`

Why third:

- this is the smallest-risk integration path because the controller, predictor, logger, and visualization already consume `PathPlanResult`

### Wave 4: Replay validation + threshold tuning

Goal:

- tune thresholds on representative recorded videos and existing logs

Deliverables:

- replay script or test harness using existing video/log workflow
- metrics for approval rate, low-confidence rate, template-switch rate, and heading spikes

Why fourth:

- most risk is threshold selection, not template generation

### Optional Wave 5: Boundary-source adapter

Goal:

- feed the same planner from Phase 7 boundary outputs when that runtime path exists

Deliverables:

- boundary-to-corridor adapter
- parity tests against mask-derived corridors

Why optional:

- Phase 11 depends on Phase 7 conceptually, but the repo does not yet have a live boundary path integration point in `live_heading_demo.py`

## Validation Architecture

Validation should be measurable, lightweight, and reuse the repo's existing replay/logging style.

### 1. Unit validation on synthetic corridors

Create synthetic BEV corridors for:

- straight centered sidewalk
- gentle left bend
- gentle right bend
- medium turn with narrowing width
- fragmented corridor with missing near-field rows
- false side pocket near ego

Measure:

- approved template id matches expected family
- approved path stays inside corridor for at least 90-95% of samples
- invalid templates are rejected
- low-confidence is emitted when corridor support drops below threshold

### 2. Contract validation against existing planner output

For integration tests on `BEVPathExtractor.process()`:

- `PathPlanResult` still returns metric and pixel paths
- `nav_out.path_model` remains valid for `AdaptivePurePursuitController`
- `live_heading_demo.py` still renders overlays and logs without special-case branches

Measure:

- no regressions in `has_path` contract
- new diagnostics populated when template mode is active

### 3. Offline replay validation on representative videos

Reuse the existing headless replay pattern from `live_heading_demo.py`.

Measure per run:

- template approval rate
- low-confidence rate
- fallback rate by `path_source`
- template-family switch count per minute
- max and p95 absolute heading
- collapse-window recovery time after occupancy drop

Key success signal:

- fewer false turn spikes than current graph-only selection during known branch-entry windows

### 4. Requirement-mapped validation

Map tests directly to TPL requirements:

- `TPL-01`: template bank size, curvature bound, and controller-ready output
- `TPL-02`: score terms present and exercised in tests
- `TPL-03`: representative straight/turning replay keeps approved path inside corridor
- `TPL-04`: ambiguous frames emit low confidence plus slowdown/hold

### 5. Debuggability validation

Phase 11 should be easy to inspect during thesis runs.

Add validation that logs expose:

- approved template id
- approval score and margin
- whether the result came from template approval or fallback
- candidate diagnostics for at least the best and runner-up templates

Without this, threshold tuning will be guesswork.

## Planning Decisions To Lock Before Execution

Before writing the implementation plan, lock these decisions:

1. Primary primitive family: cubic `y(x)` templates, not clothoids
2. Public API stability: keep `BEVPathExtractor.process()` and `PathPlanResult`
3. Fallback policy: graph search remains fallback, not co-equal primary selector
4. Confidence semantics: one planner confidence that can coexist with Phase 7 boundary confidence
5. Threshold policy: absolute approval threshold plus best-vs-second-best margin
6. Corridor adapter policy: support mask-derived corridors first, boundary-derived corridors through the same interface later

If those six decisions are made up front, Phase 11 can be planned as a sequence of small executable waves instead of a broad planner rewrite.
