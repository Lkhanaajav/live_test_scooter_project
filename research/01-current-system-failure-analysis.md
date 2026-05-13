# Current System Failure Analysis

## Executive Summary
The current pipeline has three separate weaknesses:

1. **The shipped segmentation checkpoint is not strong enough** on difficult sidewalk boundaries.
2. **The monocular BEV stage is fragile** to calibration, resolution, and far-field mask errors.
3. **DT-on-BEV is not sufficient as the primary planner** because it is both slow and geometrically brittle when occupancy is valid.

## Failure Mode 1: Baseline Segmentation Misses Narrow / Ambiguous Sidewalk
Hand-annotated evaluation on 32 sampled frames across `IMG_1878`, `IMG_1921`, `IMG_1922`, and `IMG_1924` shows that the current baseline is materially weaker than the better checkpoint.

From `research/artifacts/tables/segmentation_hand_annotations_summary.csv`:

| Case | IoU | Precision | Recall | F1 | Mean infer ms |
|---|---:|---:|---:|---:|---:|
| `baseline_raw` | 0.7583 | 0.9100 | 0.8354 | 0.8508 | 18.91 |
| `candidate_raw` | 0.9464 | 0.9833 | 0.9617 | 0.9721 | 11.67 |
| `candidate_confhold` | 0.9030 | 0.9866 | 0.9143 | 0.9467 | 13.14 |

Most striking per-video failure:

| Video | Baseline IoU | Candidate raw IoU |
|---|---:|---:|
| `IMG_1878` | 0.6801 | 0.9265 |
| `IMG_1921` | 0.8463 | 0.9612 |
| `IMG_1922` | 0.7389 | 0.9472 |
| `IMG_1924` | 0.7679 | 0.9507 |

Interpretation:
- The baseline model still overreacts to grass, texture, and narrow sidewalk boundaries.
- The better checkpoint is not just more accurate, it is also faster in this repo.

## Failure Mode 2: BEV Can Collapse To Almost Empty
The strongest evidence that the monocular BEV stage is fragile comes from the runtime profiling log at:

- `outputs/profiling/runtime_512/full/run_20260319_024759.csv`

On that run:
- mean `has_path = 0.006807`
- mean `bev_mask_occ_ratio = 0.000205`
- `path_source = none` on `4377 / 4407` frames
- `dt_ridge` only appeared on `10` frames
- `dt_ridge_hold` only appeared on `20` frames

This means the planner looked fast mostly because it usually had **nothing valid to plan on**.

Interpretation:
- a calibration / scale / cleanup mismatch can wipe out the BEV mask almost completely
- once that happens, the end-to-end pipeline degrades into "segmentation + empty warp + no path"
- this is a structural weakness of the monocular homography stage, not just a tuning issue

## Failure Mode 3: When BEV Works, DT Pathing Is Too Slow
The hand-annotated planner comparison is the opposite regime: valid masks are present, so the planner really has to solve a path.

From `research/artifacts/tables/planner_hand_annotations_summary.csv`:

### Baseline mask case

| Planner | Inside-GT ratio | Mean center error px | Mean runtime ms |
|---|---:|---:|---:|
| `bev_dt_full` | 0.9232 | 108.9 | 954.1 |
| `bev_dt_nearfield` | 0.9516 | 107.0 | 1610.9 |
| `bev_graph` | 0.9163 | 110.3 | 895.4 |
| `img_dt` | 0.9253 | 105.3 | 114.8 |
| `img_midpoint` | 0.9321 | 83.0 | 2.18 |

### Candidate cleaned mask case

| Planner | Inside-GT ratio | Mean center error px | Mean runtime ms |
|---|---:|---:|---:|
| `bev_dt_full` | 0.9857 | 65.0 | 926.8 |
| `img_dt` | 0.9942 | 60.4 | 108.1 |
| `img_midpoint` | 0.9845 | 14.3 | 2.19 |

Interpretation:
- the current BEV DT path stage is not a cheap step; it becomes **very expensive** when occupancy is healthy
- image-space planners are dramatically faster
- the simplest image-space midpoint planner produced the lowest center error in the sampled evaluation

## Failure Mode 4: BEV Geometry Can Bend Good Masks Into Bad Paths
Frame-level failures where image-space planners beat `bev_dt_full` by a wide margin:

- `IMG_1922/frame_001124`
- `IMG_1924/frame_001428`
- `IMG_1924/frame_003378`

Representative artifacts:
- `research/artifacts/images/planner_compare_IMG_1922_frame_001124.png`
- `research/artifacts/images/planner_compare_IMG_1924_frame_001428.png`
- `research/artifacts/images/planner_compare_IMG_1924_frame_003378.png`
- `research/artifacts/images/planner_bev_vs_img_dt_contact_sheet.png`

These examples show a repeated pattern:
- the camera-space mask is still reasonable
- the BEV transform amplifies shape distortions or edge bias
- the DT ridge then follows the wrong corridor center

## Failure Mode 5: Some FPS Losses Are Self-Inflicted
The measured "trash" hurting runtime is summarized in:

- `research/artifacts/tables/fps_offenders_summary.csv`

The most important offenders are:

| Component | Evidence | Impact |
|---|---|---:|
| CPU YOLO detection | `outputs/profiling/runtime_512/summary.md` | +42.274 ms over no-detection |
| Predictor disabled | `outputs/profiling/runtime_512/summary.md` | FPS drops from `25.27` to `8.94` |
| High seg resolution `512x288` vs `256x144` | `outputs/comparisons/resolution_benchmark_img1931_1024w30/summary.md` | total time rises `70.8 -> 126.3 ms` for only `+0.0038` seg IoU |
| BEV warp + cleanup | profiling + local timing | pure overhead if planning stays in image space |
| BEV DT / graph planners | planner summary | `~0.9-1.8 s` on populated masks |
| ONNX on GPU as a focus item | `outputs/comparisons/onnx_vs_pytorch_img1931_1024w30_256x144/summary.md` | no meaningful win over current PyTorch GPU path |

## Bottom-Line Failure Diagnosis
- **Segmentation is fixable and worth fixing.**
- **BEV is the fragile middle layer.**
- **DT is not the right primary planner in the current BEV formulation.**

The strongest thesis result from this pass is that the current pipeline is not failing because "path planning is a little noisy." It is failing because the chosen planning domain and planner cost too much while still being sensitive to monocular warp errors.
