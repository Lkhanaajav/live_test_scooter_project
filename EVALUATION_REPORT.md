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
