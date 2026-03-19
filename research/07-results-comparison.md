# Results Comparison

## Direct Answers

### Is current segmentation good enough?
**No.**

Evidence:
- hand-labeled IoU improved from `0.7583` to `0.9464` by switching to the better checkpoint
- the better checkpoint was also faster (`11.67 ms` vs `18.91 ms`)
- full-video replay summary showed lower unstable rate and lower fallback usage

### Is BEV truly useful in this monocular scooter case?
**Only as an optional near-field geometric aid. It is too fragile to remain the only planning domain.**

Evidence:
- one profiled run had `4377 / 4407` frames with `path_source = none`
- valid image-space planners often beat BEV planners on the hand-labeled sample
- recent monocular BEV literature itself spends major effort on self-calibration and learned view transformation

### Is DT pathing sufficient?
**Not as the current primary planner.**

Evidence:
- BEV DT was often around `0.9-1.6 s/frame` on populated masks
- image-space midpoint achieved the best center error at about `2.2 ms/frame`
- image-space DT gave comparable or better path containment at about `108-115 ms/frame`

## Segmentation Results

### Hand-labeled sample
From `research/artifacts/tables/segmentation_hand_annotations_summary.csv`:

| Case | IoU | Precision | Recall | F1 | Infer ms |
|---|---:|---:|---:|---:|---:|
| `baseline_raw` | 0.7583 | 0.9100 | 0.8354 | 0.8508 | 18.91 |
| `candidate_raw` | 0.9464 | 0.9833 | 0.9617 | 0.9721 | 11.67 |
| `candidate_confhold` | 0.9030 | 0.9866 | 0.9143 | 0.9467 | 13.14 |

Interpretation:
- `candidate_raw` is the best pure segmentation substitute
- `candidate_confhold` trades some recall for cleaner topology

### Full-video replay summary already in repo
From `outputs/evaluation/binary_model_replay_full/summary.md`:

| Metric | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| Mean seg IoU | 0.9088 | 0.9247 | +0.0159 |
| Unstable rate | 1.46% | 0.33% | -1.12 pp |
| Has-path rate | 100.0% | 100.0% | 0.0 pp |
| Mean heading delta | 0.2091 deg | 0.2010 deg | -0.0081 deg |
| Corridor confidence | 0.8576 | 0.8661 | +0.0085 |
| Fallback rate | 18.98% | 14.27% | -4.71 pp |

Interpretation:
- the segmentation improvement is real
- the downstream pathing improvement is positive overall, but not perfectly uniform on unseen videos
- segmentation was not the only bottleneck in the old stack

## Planner Results

### Candidate cleaned mask case
From `research/artifacts/tables/planner_hand_annotations_summary.csv`:

| Planner | Has path | Inside-GT ratio | Mean center error px | Mean runtime ms |
|---|---:|---:|---:|---:|
| `bev_dt_full` | 1.000 | 0.9857 | 65.0 | 926.8 |
| `bev_dt_nearfield` | 1.000 | 0.9857 | 79.0 | 1603.0 |
| `bev_graph` | 1.000 | 0.9706 | 76.6 | 380.3 |
| `img_dt` | 1.000 | 0.9942 | 60.4 | 108.1 |
| `img_midpoint` | 1.000 | 0.9845 | 14.3 | 2.19 |

Interpretation:
- `img_dt` is the best geometric fallback
- `img_midpoint` is the best primary planner
- BEV-based planners are not competitive on runtime

### Oracle mask case
Even with the ground-truth mask, BEV is not clearly superior:

| Planner | Inside-GT ratio | Mean center error px | Mean runtime ms |
|---|---:|---:|---:|
| `bev_dt_full` | 0.9803 | 69.2 | 894.7 |
| `img_dt` | 0.9890 | 62.5 | 109.0 |
| `img_midpoint` | 0.9841 | 15.2 | 2.39 |

Interpretation:
- the BEV planner is not losing only because of segmentation
- the planning domain itself is part of the problem

## Frame-Level Failures
Three sampled baseline-mask frames where `img_dt` beat `bev_dt_full` by more than `0.2` in inside-GT ratio:

- `IMG_1922/frame_001124`
- `IMG_1924/frame_001428`
- `IMG_1924/frame_003378`

Relevant artifacts:
- [planner_compare_IMG_1922_frame_001124.png](artifacts/images/planner_compare_IMG_1922_frame_001124.png)
- [planner_compare_IMG_1924_frame_001428.png](artifacts/images/planner_compare_IMG_1924_frame_001428.png)
- [planner_compare_IMG_1924_frame_003378.png](artifacts/images/planner_compare_IMG_1924_frame_003378.png)
- [planner_bev_vs_img_dt_contact_sheet.png](artifacts/images/planner_bev_vs_img_dt_contact_sheet.png)

Large center-error reductions for `img_midpoint` over `bev_dt_full` on the candidate cleaned mask included:
- `IMG_1924/frame_003378`: `192.2 px -> 16.5 px`
- `IMG_1924/frame_002884`: `142.9 px -> 16.6 px`
- `IMG_1922/frame_000036`: `150.8 px -> 26.1 px`

## BEV Fragility Result
From `outputs/profiling/runtime_512/full/run_20260319_024759.csv`:

| Metric | Value |
|---|---:|
| Mean `has_path` | 0.006807 |
| Mean `bev_mask_occ_ratio` | 0.000205 |
| `path_source = none` | 4377 frames |
| `path_source = dt_ridge` | 10 frames |
| `path_source = dt_ridge_hold` | 20 frames |

Interpretation:
- the old pathfinding averages understate the real planner cost
- most frames in that run were not successful planning frames

## FPS / Runtime Findings
From `research/artifacts/tables/fps_offenders_summary.csv` and existing profiling reports:

### Clear performance offenders
- CPU YOLO detection: biggest steady-state penalty
- disabling predictor: catastrophic FPS drop
- high segmentation resolution: poor return on latency
- BEV warp + cleanup: wasted overhead if planning stays in image space
- BEV DT and graph planners: not viable as primary real-time planners

### Not worth thesis focus right now
- ONNX on GPU
  - local benchmark at `256x144` showed no practical gain over the existing PyTorch GPU path

## Visual Artifacts

### Segmentation
- [segmentation_compare_img_1878.jpg](artifacts/images/segmentation_compare_img_1878.jpg)
- [segmentation_compare_img_1922.jpg](artifacts/images/segmentation_compare_img_1922.jpg)
- [segmentation_compare_img_1878_side_by_side.mp4](artifacts/videos/segmentation_compare_img_1878_side_by_side.mp4)

### Pathing
- [planner_bev_vs_img_dt_contact_sheet.png](artifacts/images/planner_bev_vs_img_dt_contact_sheet.png)
- [planner_compare_slideshow.mp4](artifacts/videos/planner_compare_slideshow.mp4)

## Final Interpretation
The best overall result from this research pass is not "a slightly better BEV DT planner." It is a stronger architectural conclusion:

- fix segmentation with the better binary checkpoint
- stop forcing planning through BEV
- use image-space geometry first
- keep BEV only where it clearly adds value
