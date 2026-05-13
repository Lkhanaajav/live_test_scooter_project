# Final Architecture Recommendation

## Recommended Architecture

### 1. Segmentation
- Replace the shipped runtime checkpoint with the best externally validated binary checkpoint.
- Keep the freshly refreshed mixed-data checkpoint as the first next candidate on new-video-heavy runs:
  - `outputs/training/binary_segformer_old400_img1931_vid017_020/best_checkpoint`
- Start from threshold `0.60`
- Keep binary output only
- Keep lightweight topology cleanup:
  - morphology
  - connected-component selection
  - optional confidence hold / hysteresis when runtime flicker is visible

Rationale:
- best measured mask quality in this repo came from the externally validated candidate
- the new four-video refresh reached `0.9602` validation IoU on its mixed split and is the freshest runtime candidate
- faster than the current shipped baseline
- no architecture rewrite required

### 2. Primary planner
- Use **image-space boundary midpoint planning** as the default path extractor.

Rationale:
- best speed/quality trade in the hand-labeled study
- center error dropped to about `14 px` on the candidate cleaned mask
- runtime stayed near `2.2 ms`
- directly monocular-compatible

### 3. Fallback planner
- Use **image-space DT planning** as the fallback when:
  - the midpoint corridor becomes discontinuous
  - valid rows drop too low
  - corridor width becomes irregular

Rationale:
- more robust than pure midpoint on irregular masks
- still much faster than BEV DT
- better inside-mask containment than the BEV baseline in the sampled evaluation

### 4. Optional BEV branch
- Keep BEV only for:
  - short-range metric visualization
  - optional obstacle projection
  - diagnostic overlays
  - future controller tuning

Do **not** make BEV the only planning domain.

If BEV is used:
- crop to near field
- mask out unstable far field
- do not run full BEV DT as the default planner

## What To Retire From The Primary Path

### Retire as primary planner
- `DtPathPlanner` on full BEV
- old BEV graph planner as first choice

### Gate aggressively
- CPU object detection
- high segmentation resolution above the proven useful range
- any BEV cleanup path when the active planner is image-space only

### De-prioritize
- ONNX optimization on GPU
- large learned monocular BEV models

## Recommended Runtime Modes

### Thesis demo / best overall mode
1. candidate SegFormer checkpoint
2. image midpoint planner
3. image DT fallback
4. predictor enabled
5. detection disabled unless obstacle experiments are active

### Obstacle-aware mode
1. same segmentation
2. image midpoint primary
3. image DT fallback
4. sparse or GPU-gated detection
5. optional cost map over image-space or near-field BEV

### Future robust robotics mode
1. same segmentation front-end
2. cost-map planner such as A* / Smac for obstacle-rich scenes
3. optional kinematic smoothing or Hybrid-A* if steering constraints become important

## Why This Architecture Is Better For The Thesis
It supports a stronger thesis claim than "BEV DT worked after tuning":

- a monocular scooter can follow sidewalks robustly **without** relying on a fragile BEV transform
- simpler image-space geometry can outperform the heavier BEV path stack
- BEV still has value, but only as an auxiliary representation

That is a better engineering result and a better thesis result.

## Concrete Cut List For FPS
Based on `research/artifacts/tables/fps_offenders_summary.csv`, the first things to cut or gate are:

1. CPU YOLO detection when planner experiments do not need it
2. disabling the predictor
3. `512x288` segmentation input as a default
4. always-on BEV warp/cleanup in image-space planning mode
5. full BEV DT / graph planners as default runtime planners

From `research/artifacts/tables/new_model_flag_sweep_summary.csv`, `--headless` and `--save` are not first-order FPS levers here. The pipeline stayed around `606-610 ms/frame` because pathing alone consumed about `515-520 ms/frame`.

## Final Answers

### Segmentation
Current segmentation is **not** good enough. The exact substitute that helps is:
- `binary_segformer_oneformer_teacher/best_checkpoint`
- threshold around `0.60`
- connected-component cleanup
- optional confidence-gated hold if temporal flicker matters more than raw mask recall

### BEV
BEV is **conditionally useful** but **too fragile to stay primary** in this monocular scooter case.

### DT pathing
DT pathing is **not sufficient as the current primary BEV planner**. It should be replaced by:
- image midpoint primary
- image DT fallback
- optional future cost-map + A* / Smac for obstacle-rich expansion
