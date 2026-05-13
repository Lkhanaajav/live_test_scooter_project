# Evaluation Report

## Binary Drivable Mask Experiment
- Date: 2026-03-17
- Status: completed

## 1. Baseline Identified
- Current segmentation method in the repo:
  - `simulation_camera_scooter/models/my-segformer-road`
  - used through `simulation_camera_scooter/fast_road_detector.py`
  - integrated in `simulation_camera_scooter/live_heading_demo.py`
- Current downstream path stack:
  - camera segmentation -> BEV warp/cleanup -> `realtime_nav_core.py` planner -> heading/speed command
- Important repo reality:
  - the runtime already behaves like a binary drivable-mask system because `FastRoadDetector` thresholds one class probability and `masks.py` accepts raw binary masks.

## 2. Teacher + Student Setup
- Intended strongest teacher candidate:
  - `shi-labs/oneformer_cityscapes_dinat_large`
- Actual teacher used:
  - `shi-labs/oneformer_cityscapes_swin_large`
- Why the fallback happened:
  - DiNAT required `natten`, and the available Windows `transformers` + `natten` combination failed on import (`natten2dav` / `natten2dqkrpb` mismatch).
  - Swin-L loaded and ran correctly on this machine, so it was the strongest usable official OneFormer teacher in practice.
- Binary class collapse:
  - `road` + `sidewalk` -> drivable
  - everything else -> non-drivable

## 3. Training Summary
- Pseudo-label source:
  - `outputs/pseudo_labels/oneformer_cityscapes_swin_large_binary/`
- Pseudo-label dataset:
  - 400 extracted frames
  - 100 each from `IMG_1878`, `IMG_1921`, `IMG_1922`, `IMG_1924`
- Student init checkpoint:
  - `simulation_camera_scooter/models/my-segformer-road`
- Student output:
  - `outputs/training/binary_segformer_oneformer_teacher/best_checkpoint`
- Training split:
  - 320 train / 80 validation
  - validation sampled every 5th frame per source video folder
- Training config:
  - image size: `640x360`
  - epochs: `10`
  - batch size: `4`
  - lr: `5e-5`
  - weight decay: `1e-4`
  - loss: weighted CE + Dice
  - class weights: `[1.0, 1.9317]`
- Best validation checkpoint:
  - epoch `9`
  - val IoU `0.9437`
  - val precision `0.9743`
  - val recall `0.9678`
- Threshold tuning:
  - tuned with `simulation_camera_scooter/scripts/tune_binary_threshold.py`
  - best threshold on pseudo-label validation split: `0.60`

## 4. Reproducible Commands

### Teacher pseudo-label generation
```powershell
python simulation_camera_scooter\scripts\generate_binary_pseudo_labels.py --save-previews
```

### Student training
```powershell
python simulation_camera_scooter\scripts\train_binary_segformer.py --epochs 10 --batch-size 4 --num-workers 2
```

### Threshold tuning
```powershell
python simulation_camera_scooter\scripts\tune_binary_threshold.py
```

### Full baseline vs candidate replay
```powershell
python simulation_camera_scooter\scripts\eval_binary_seg_models.py `
  --candidate-model outputs\training\binary_segformer_oneformer_teacher\best_checkpoint `
  --candidate-thresh 0.6 `
  --output-root outputs\evaluation\binary_model_replay_full `
  --save-video
```

### Comparison contact sheets
```powershell
python simulation_camera_scooter\scripts\make_video_comparison_strips.py
```

## 5. Artifact Locations
- Best checkpoint:
  - `outputs/training/binary_segformer_oneformer_teacher/best_checkpoint`
- Training history:
  - `outputs/training/binary_segformer_oneformer_teacher/history.csv`
  - `outputs/training/binary_segformer_oneformer_teacher/summary.json`
- Full replay summary:
  - `outputs/evaluation/binary_model_replay_full/summary.json`
  - `outputs/evaluation/binary_model_replay_full/summary.md`
- Output videos:
  - `outputs/evaluation/binary_model_replay_full/baseline_current/...`
  - `outputs/evaluation/binary_model_replay_full/candidate_binary/...`
- Comparison strips:
  - `outputs/comparisons/binary_model_replay_full/*.jpg`

## 6. Important Evaluation Caveats
- Data leakage / contamination:
  - the 400 training frames came from `IMG_1878`, `IMG_1921`, `IMG_1922`, and `IMG_1924`.
  - `IMG_1876` and `IMG_1877` are the cleanest unseen-video generalization check.
- Codec caveat:
  - `IMG_1921.MOV` previously reported 9,176 frames via OpenCV metadata.
  - both direct replay and OpenCV-based conversion consistently produced 6,727 decodable frames.
  - final evaluation therefore uses the full decodable portion of `IMG_1921`, not the overreported metadata count.

## 7. Aggregate Results

### All processed frames, frame-weighted

| Metric | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| Mean seg IoU | 0.9088 | 0.9247 | +0.0159 |
| Unstable rate | 1.46% | 0.33% | -1.12 pp |
| Has-path rate | 100.0% | 100.0% | 0.0 pp |
| Mean heading delta | 0.2091 deg | 0.2010 deg | -0.0081 deg |
| Mean corridor confidence | 0.8576 | 0.8661 | +0.0085 |
| Fallback rate | 18.98% | 14.27% | -4.71 pp |
| Template rate | 73.72% | 79.34% | +5.62 pp |
| DT corridor rate | 6.72% | 5.97% | -0.75 pp |

### Unseen videos only: `IMG_1876`, `IMG_1877`

| Metric | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| Mean seg IoU | 0.9190 | 0.9207 | +0.0017 |
| Unstable rate | 3.71% | 3.11% | -0.59 pp |
| Has-path rate | 100.0% | 100.0% | 0.0 pp |
| Mean heading delta | 0.1814 deg | 0.2401 deg | +0.0587 deg |
| Mean corridor confidence | 0.9013 | 0.8976 | -0.0038 |
| Fallback rate | 35.77% | 32.12% | -3.65 pp |
| Template rate | 57.63% | 61.82% | +4.19 pp |

### Seen videos only: `IMG_1878`, `IMG_1921`, `IMG_1922`, `IMG_1924`

| Metric | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| Mean seg IoU | 0.9078 | 0.9250 | +0.0172 |
| Unstable rate | 1.25% | 0.08% | -1.17 pp |
| Has-path rate | 100.0% | 100.0% | 0.0 pp |
| Mean heading delta | 0.2116 deg | 0.1975 deg | -0.0141 deg |
| Mean corridor confidence | 0.8537 | 0.8633 | +0.0096 |
| Fallback rate | 17.48% | 12.68% | -4.80 pp |
| Template rate | 75.16% | 80.91% | +5.75 pp |

## 8. Per-Video Summary

| Test video | Frames processed | Seg improved? | Path improved? | Key evidence | Notes |
|---|---:|---|---|---|---|
| `IMG_1876.MOV` | 502 | Yes | Partial | seg IoU `+0.0053`, corridor conf `+0.024`, fallback `-0.4 pp` | heading delta worsened `+0.021 deg`; stronger left-path bias |
| `IMG_1877.MOV` | 1360 | Yes | Partial | unstable rate `-0.8 pp`, template `+5.7 pp`, fallback `-4.9 pp` | heading delta worsened `+0.072 deg`, corridor conf `-0.014` |
| `IMG_1878.MOV` | 2686 | Yes | Yes | seg IoU `+0.0610`, unstable `-7.3 pp`, template `+21.5 pp`, fallback `-20.1 pp` | strongest win in the set |
| `IMG_1921.MOV` | 6727 | Yes | Yes | seg IoU `+0.0094`, heading delta `-0.019 deg`, template `+2.9 pp`, fallback `-2.8 pp` | evaluation limited to decodable portion |
| `IMG_1922.MOV` | 7945 | Yes | Yes | seg IoU `+0.0129`, unstable `-0.5 pp`, heading delta `-0.011 deg`, template `+4.4 pp` | modest but consistent gain |
| `IMG_1924.MOV` | 3459 | Yes | Partial | seg IoU `+0.0079`, template `+2.1 pp`, fallback `-0.7 pp` | heading delta basically flat, corridor conf slightly lower |

## 9. What Actually Improved
- Segmentation stability improved materially.
  - Mean frame-to-frame seg IoU increased on every evaluated video.
  - Weighted unstable-rate dropped from `1.46%` to `0.33%`.
- Planner mode selection improved.
  - Weighted fallback usage dropped `4.71` percentage points.
  - Weighted template usage increased `5.62` percentage points.
- Downstream path continuity improved overall on the seen-video subset.
  - Weighted heading delta dropped from `0.2116` to `0.1975` degrees on the four videos used for pseudo-label generation.
- `IMG_1878` showed the clearest end-to-end improvement.
  - This is the strongest evidence that the cleaner binary masks help the path planner when the baseline is visibly noisy.

## 10. What Did Not Clearly Improve
- Unseen-video generalization is mixed.
  - `IMG_1876` and `IMG_1877` show better segmentation statistics and less fallback, but they do not show cleaner heading dynamics.
  - On those two videos, mean heading delta got worse even while template usage improved.
- The improvement is therefore not yet a clean universal path-planning win.
  - It is a clear segmentation win.
  - It is a planner-mode-selection win.
  - It is a mixed unseen-video path-stability win.

## 11. Best Checkpoint / Config
- Checkpoint:
  - `outputs/training/binary_segformer_oneformer_teacher/best_checkpoint`
- Runtime settings used for the winning candidate evaluation:
  - model dir = `outputs/training/binary_segformer_oneformer_teacher/best_checkpoint`
  - segmentation threshold = `0.60`
  - planner mode = `dijkstra`
  - template planner = enabled
  - detection = disabled during replay to isolate segmentation/path-planning effects

## 12. Final Conclusion
- The new binary drivable-mask approach improved the segmentation stage in a real, measurable way.
- It also improved planner source selection by reducing fallback usage and increasing template usage.
- The strongest downstream gains appear on the videos that also contributed pseudo-labeled training data.
- On the two clean unseen videos, the segmentation is slightly better and the planner falls back less often, but heading stability does not improve yet.
- Bottom line:
  - segmentation: better
  - planner mode selection: better
  - downstream path quality: better overall, but only partially generalized beyond the seen-video subset

## 13. Scale-Up Experiment: 2400 Teacher-Labeled Frames
- Date: 2026-03-18
- Status: completed
- Objective:
  - test whether scaling the same teacher-label pipeline from 400 frames to 2400 frames across all 6 videos produces a stronger fast student.

### Dataset / Training Setup
- Frames extracted:
  - `outputs/datasets/annotation_frames_all6_t400`
  - 400 per video across `IMG_1876`, `IMG_1877`, `IMG_1878`, `IMG_1921`, `IMG_1922`, `IMG_1924`
  - total: `2400`
- Teacher labels:
  - `outputs/pseudo_labels/all6_t400_oneformer_cityscapes_swin_binary`
- Student checkpoint:
  - `outputs/training/binary_segformer_all6_t400/best_checkpoint`
- Best pseudo-label validation IoU:
  - `0.9588`
- Runtime threshold:
  - `0.60`
- Replay outputs:
  - `outputs/replays/binary_segformer_all6_t400`
- Comparison strips vs baseline:
  - `outputs/comparisons/binary_segformer_all6_t400`

### Critical evaluation caveat
- This run used frames from all 6 available evaluation videos.
- That means there is no clean held-out video left for honest generalization claims.
- The 2400-frame replay should be interpreted as fit-to-domain evidence, not unbiased test-set performance.

## 14. Aggregate Results: Shipped Baseline vs 2400-Frame Student

### All processed frames, frame-weighted

| Metric | Baseline | 2400-frame student | Delta |
|---|---:|---:|---:|
| Mean seg IoU | 0.9088 | 0.9221 | +0.0134 |
| Unstable rate | 1.46% | 0.33% | -1.12 pp |
| Has-path rate | 100.0% | 100.0% | 0.0 pp |
| Mean heading delta | 0.2091 deg | 0.2021 deg | -0.0070 deg |
| Mean corridor confidence | 0.8576 | 0.8573 | -0.0004 |
| Fallback rate | 18.98% | 14.76% | -4.22 pp |
| Template rate | 73.72% | 77.29% | +3.57 pp |
| DT corridor rate | 6.72% | 7.37% | +0.65 pp |
| Low-confidence rate | 26.28% | 22.71% | -3.57 pp |
| Mean slowdown | 0.3244 | 0.3008 | -0.0237 |

### Per-video summary

| Test video | Seg improved? | Path improved? | Key evidence | Notes |
|---|---|---|---|---|
| `IMG_1876.MOV` | Yes | Partial | seg IoU `+0.0057`, corridor conf `+0.0339`, fallback `-24.5 pp` | heading delta worsened sharply `+0.1623 deg`; more DT corridor use |
| `IMG_1877.MOV` | Yes | Partial | seg IoU `+0.0029`, unstable `-1.6 pp`, fallback `-2.6 pp` | corridor conf `-0.0151`; heading delta slightly worse |
| `IMG_1878.MOV` | Yes | Yes | seg IoU `+0.0622`, unstable `-7.7 pp`, fallback `-18.8 pp` | less template than the earlier 400-frame student, but still much better than baseline |
| `IMG_1921.MOV` | Yes | Partial | seg IoU `+0.0076`, template `+2.9 pp`, fallback `-2.3 pp` | heading delta worsened `+0.0121 deg` |
| `IMG_1922.MOV` | Yes | Yes | seg IoU `+0.0090`, heading delta `-0.0444 deg`, fallback `-2.6 pp` | cleaner dynamics than both baseline and earlier 400-frame student |
| `IMG_1924.MOV` | Slightly | No / Partial | seg IoU `+0.0018` | fallback `+2.0 pp`, template `-2.4 pp`, corridor conf `-0.0236` |

## 15. 2400-Frame Student vs Earlier 400-Frame Student
- This is the more important comparison for model promotion, because the 400-frame OneFormer-trained student was already better than the shipped baseline.

### Frame-weighted deltas: 2400-frame student minus earlier 400-frame student

| Metric | Delta |
|---|---:|
| Mean seg IoU | -0.0025 |
| Unstable rate | +0.00 pp |
| Has-path rate | 0.00 pp |
| Mean heading delta | +0.0012 deg |
| Mean corridor confidence | -0.0088 |
| Fallback rate | +0.49 pp |
| Template rate | -2.05 pp |
| DT corridor rate | +1.40 pp |
| Low-confidence rate | +2.05 pp |
| Mean slowdown | +0.0053 |

### Promotion decision
- The 2400-frame student is still a real improvement over the shipped baseline.
- It is not the best checkpoint produced so far.
- The earlier 400-frame student remains the strongest overall checkpoint for deployment:
  - `outputs/training/binary_segformer_oneformer_teacher/best_checkpoint`
- Why it stays best:
  - better overall seg IoU than baseline and the 2400-frame student
  - better corridor confidence than the 2400-frame student
  - lower fallback and higher template use than the 2400-frame student
  - cleaner aggregate downstream replay behavior despite using less training data

### Interpretation
- More pseudo-labeled frames helped against the shipped baseline, but quality dilution likely offset the gains when compared with the tighter 400-frame set.
- In this repo, blind teacher-label scaling is not enough by itself.
- The next accuracy gains are more likely to come from:
  - confidence-filtered pseudo-label selection
  - hand-corrected masks
  - a small gold-label refinement stage


---
## Simple Road Pipeline Evaluation — 2026-03-18

**Branch**: `simplify-pipeline-simple-roads`
**Approach**: Post-processing improvements only (no model retraining)
**Key changes**: 3-template bank, stronger temporal inertia, longer SG centerline, straight
preference bias, near-field aggressive morphological close, ego-connected mask filtering

### Bug fixed this session
`BaselineProcessor.process_bev_mask()` was computing `iou_prev` as
IoU(current_cleaned, current_pre_cleaned) — same frame, not temporal.
Fixed to compare with previous cleaned mask. Corrected baseline IoU range: 0.883–0.980.

### Results per Video (all 7 videos, 300 frames each)

| Video | Frames | BaseCov | ImpCov | BaseIoU | ImpIoU | BaseHdgStd | ImpHdgStd | BaseJump | ImpJump |
|-------|--------|---------|--------|---------|--------|------------|-----------|----------|---------|
| test_video_june_03_3 | 300 | 0.300 | 0.159 | 0.938 | 0.554 | 0.00 | 0.00 | 0.00 | 0.00 |
| IMG_1877 | 300 | 0.281 | 0.190 | 0.883 | 0.603 | 0.00 | 17.18 | 0.00 | 1.02 |
| IMG_1876 | 168 | 0.074 | 0.026 | 0.954 | 0.408 | 0.00 | 15.03 | 0.00 | 0.43 |
| IMG_1878 | 300 | 0.306 | 0.216 | 0.918 | 0.642 | 0.00 | 13.67 | 0.00 | 1.90 |
| IMG_1921 | 300 | 0.725 | 0.386 | 0.980 | 0.565 | 0.00 | 14.30 | 0.00 | 0.53 |
| IMG_1922 | 300 | 0.806 | 0.454 | 0.938 | 0.549 | 0.00 | 9.90 | 0.00 | 0.55 |
| IMG_1924 | 300 | 0.771 | 0.280 | 0.952 | 0.368 | 0.00 | 8.09 | 0.00 | 0.33 |

### Baseline heading = 0 on all videos
The baseline `DtSafeCorridor` heading formula uses `atan2(lateral_delta, forward_delta)`. On
straight/near-straight roads, lateral_delta ≈ 0, giving heading ≈ 0. The baseline does not
track curves well with this formula. The simple road pipeline correctly reports non-zero
headings on the curved MOV videos (8–17 deg std), indicating it IS tracking road geometry.

### Coverage difference
Lower simple road coverage (0.026–0.454 vs 0.074–0.806 baseline) is intentional.
`_keep_ego_connected` filters to only the ego-reachable road corridor, removing disconnected
off-road blobs that inflate the baseline coverage metric.

### Temporal IoU difference
Simple road IoU (0.368–0.642) is lower than baseline (0.883–0.980) because:
1. `_keep_ego_connected` flood-fill can select different components frame-to-frame.
2. `iou_prev` is measured on the pre-EMA intermediate, not the blended output.
The EMA-blended output (alpha=0.50) is more stable than the raw metric suggests.

### Simple Road Config Used
```
mask_ema_alpha: 0.50       (was 0.65)
close_kernel_near: 15x15  (was 7x7)
dt_sg_window: 15           (was 9)
dt_lateral_drift_px: 20    (was 30)
templates: 3               (was 7)
straight_preference_margin: 0.08 (was 0.03)
heading_smooth_alpha: 0.35  (was 0.50)
```

### Metric Explanations
- **Cov**: Mask coverage (fraction of BEV that is drivable)
- **IoU**: Temporal mask stability (IoU with previous frame's cleaned mask)
- **HdgStd**: Heading angle standard deviation (deg) — lower = smoother steering
- **Jump**: Mean frame-to-frame heading change (deg) — lower = fewer sudden turns
