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

### New teacher-student refresh with the four new test videos
New frames were extracted from:
- `simulation_camera_scooter/test_videos/VID_20260319_155939_00_017.mp4`
- `simulation_camera_scooter/test_videos/VID_20260319_160039_00_018.mp4`
- `simulation_camera_scooter/test_videos/VID_20260319_160139_00_019.mp4`
- `simulation_camera_scooter/test_videos/VID_20260319_160240_00_020.mp4`

Training set composition:
- `699` older strong image/mask pairs from the previous mixed dataset
- `720` new teacher-labeled frames from the four new videos
- `1419` total image/mask pairs

Training result:
- new checkpoint: `outputs/training/binary_segformer_old400_img1931_vid017_020/best_checkpoint`
- initializer: `outputs/training/binary_segformer_old400_plus_img_1931_t300/best_checkpoint`
- best validation IoU: `0.9602`
- tuned threshold: `0.60`

Interpretation:
- the mixed old+new refresh trained cleanly and slightly exceeded the previous initializer on the internal validation split
- this makes it the freshest runtime candidate for the new videos
- however, the hand-labeled benchmark above is still the stronger external evidence base, so the new checkpoint should be treated as the next candidate to validate rather than silently declared the new champion

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

### New model flag sweep on a fresh video
From `research/artifacts/tables/new_model_flag_sweep_summary.csv` using:
- video: `simulation_camera_scooter/test_videos/VID_20260319_160139_00_019.mp4`
- model: `outputs/training/binary_segformer_old400_img1931_vid017_020/best_checkpoint`
- threshold: `0.60`
- settings held constant except `--headless` and `--save`

| Case | FPS est | Total ms | Seg ms | BEV ms | Path ms | Other ms |
|---|---:|---:|---:|---:|---:|---:|
| `gui_nosave` | 1.65 | 606.1 | 32.9 | 19.5 | 514.6 | 38.8 |
| `gui_save` | 1.64 | 610.2 | 32.1 | 19.0 | 519.9 | 38.9 |
| `headless_nosave` | 1.65 | 607.5 | 32.9 | 18.7 | 517.5 | 38.2 |
| `headless_save` | 1.64 | 608.9 | 32.9 | 18.7 | 518.5 | 38.5 |

Interpretation:
- `--headless` barely changed throughput in this pipeline
- `--save` barely changed throughput in this pipeline
- the dominant cost is the planner itself, not rendering or video writing
- pathing consumed about `515-520 ms/frame`, dwarfing segmentation at about `32-33 ms/frame`
- BEV still cost about `19 ms/frame` even before the planner cost is counted
- the right optimization target is replacing the BEV DT planner, not debating GUI or save flags

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
- keep the new mixed-data checkpoint as the freshest candidate for the new videos
- stop forcing planning through BEV
- use image-space geometry first
- keep BEV only where it clearly adds value
- do not expect `--headless` or `--save` toggles to rescue runtime while BEV DT remains the primary planner

## Four New Videos With Warped-Back Planner Overlays
Using the mixed-data checkpoint `outputs/training/binary_segformer_old400_img1931_vid017_020/best_checkpoint`, I ran all four March 19 videos through the planner comparison pipeline after patching it to save camera-overlay videos with the planner path warped back onto the source frame.

Run details:
- output root: `outputs/path_planner_eval_new_model_all4_step3/planner_comparison/`
- videos: `VID_20260319_155939_00_017`, `VID_20260319_160039_00_018`, `VID_20260319_160139_00_019`, `VID_20260319_160240_00_020`
- processed sampling: every `3rd` frame
- saved artifacts per video:
  - `comparison.mp4`
  - `camera_overlays/<planner>.mp4`
  - `bev_overlays/<planner>.mp4`

Aggregate result from `outputs/path_planner_eval_new_model_all4_step3/planner_comparison/PATH_PLANNER_COMPARISON_REPORT.md`:

| Planner | Valid% | Hdg Std | Path Jump | Lat Std | Speed (ms) | Confidence |
|---|---:|---:|---:|---:|---:|---:|
| `dt_ridge_baseline` | 100.0% | 0.93d | 0.17d | 1.397m | 1288.2 | 0.96 |
| `vectorized_dt` | 100.0% | 29.26d | 3.71d | 3.328m | 82.0 | 1.00 |
| `weighted_centroid` | 100.0% | 14.09d | 2.42d | 1.082m | 2.7 | 0.99 |
| `potential_field` | 100.0% | 40.03d | 5.77d | 2.104m | 7.0 | 1.00 |
| `skeleton_hybrid` | 100.0% | 34.66d | 11.29d | 2.314m | 11.6 | 0.79 |

Interpretation:
- `dt_ridge_baseline` still gives the smoothest and most stable path, but at `~1.29 s/frame` it is only an oracle/reference planner.
- `weighted_centroid` is still the best fast deployment option across these four videos.
- `vectorized_dt` is materially faster than full DT but still too unstable to replace the baseline directly.
- `potential_field` is fast but too jittery on these clips.
- `skeleton_hybrid` remains visually brittle despite acceptable runtime.

Saved-output note:
- the full batch produced `44` MP4 files with total size about `3.48 GB`
- each video directory contains the camera overlays the user asked for, with the path rendered back onto the original camera view
