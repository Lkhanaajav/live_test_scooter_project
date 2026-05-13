# Experiment Plan

## Goal
Compare the current pipeline against practical alternatives with reproducible scripts and saved artifacts.

## Baselines

### Segmentation baseline
- Model: `simulation_camera_scooter/models/my-segformer-road`
- Runtime usage: thresholded binary mask through `FastRoadDetector`

### Planning baseline
- Full BEV pipeline
- `BEVPathExtractor`
- `DtPathPlanner` enabled

## Candidate Methods Tested

### Segmentation
- `candidate_raw`
  - model: `outputs/training/binary_segformer_oneformer_teacher/best_checkpoint`
  - threshold: `0.60`
- `candidate_confhold`
  - same model
  - threshold band hysteresis + morphology + connected-component cleanup

### Pathing
- `bev_dt_full`
- `bev_dt_nearfield`
- `bev_graph`
- `img_dt`
- `img_midpoint`

## Data Used

### Full-video replay artifacts already present in repo
- `outputs/evaluation/binary_model_replay_full/...`
- six videos:
  - `IMG_1876`
  - `IMG_1877`
  - `IMG_1878`
  - `IMG_1921`
  - `IMG_1922`
  - `IMG_1924`

### Hand-labeled research sample
- root: `outputs/hand_annotations/v1`
- evaluation sample used here:
  - `8` evenly sampled frames per video
  - four videos with annotations in this pass
  - total `32` frames

## Metrics

### Segmentation metrics
- IoU
- precision
- recall
- F1
- inference time
- frame-to-frame mask stability IoU

### Planner metrics
- `has_path`
- inside-GT ratio
- mean center error in pixels
- mean clearance inside GT
- forward span
- heading
- runtime

### Runtime / FPS metrics
- FPS
- total stage time
- segmentation time
- detection time
- BEV time
- pathing time

## Executed Commands

### Unit tests
```powershell
pytest simulation_camera_scooter/tests/test_image_path_planner.py `
  simulation_camera_scooter/tests/test_dt_path_planner.py `
  simulation_camera_scooter/tests/test_boundary_inference.py -q
```

Observed result:
- `16 passed`

### Hand-annotated evaluation
```powershell
python simulation_camera_scooter/scripts/eval_hand_annotated_pipeline.py --per-video-limit 8
```

Outputs saved under:
- `research/artifacts/tables/`
- `research/artifacts/images/`
- `research/artifacts/videos/`

## Artifact Targets

### Required markdown
- `research/00-current-pipeline-summary.md`
- `research/01-current-system-failure-analysis.md`
- `research/02-literature-review-segmentation.md`
- `research/03-literature-review-bev.md`
- `research/04-literature-review-pathing.md`
- `research/05-candidate-selection-and-rationale.md`
- `research/06-experiment-plan.md`
- `research/07-results-comparison.md`
- `research/08-final-architecture-recommendation.md`
- `research/99-work-log.md`

### Tables
- `research/artifacts/tables/segmentation_hand_annotations_summary.csv`
- `research/artifacts/tables/segmentation_hand_annotations_by_video.csv`
- `research/artifacts/tables/planner_hand_annotations_summary.csv`
- `research/artifacts/tables/planner_hand_annotations_by_video.csv`
- `research/artifacts/tables/fps_offenders_summary.csv`

### Images
- `research/artifacts/images/planner_bev_vs_img_dt_contact_sheet.png`
- `research/artifacts/images/planner_compare_IMG_1922_frame_001124.png`
- `research/artifacts/images/planner_compare_IMG_1924_frame_001428.png`
- `research/artifacts/images/planner_compare_IMG_1924_frame_003378.png`
- `research/artifacts/images/segmentation_compare_img_1878.jpg`
- `research/artifacts/images/segmentation_compare_img_1922.jpg`

### Videos
- `research/artifacts/videos/segmentation_compare_img_1878_side_by_side.mp4`
- `research/artifacts/videos/planner_compare_slideshow.mp4`

## Expected Decision Logic
- If segmentation improves substantially with no runtime penalty, replace baseline model.
- If BEV fails often or costs too much relative to no-BEV planners, demote BEV to optional use.
- If DT is only competitive after BEV, but image-space midpoint / DT are faster and cleaner, replace primary planner.
