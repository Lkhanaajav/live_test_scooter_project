# June Video Performance Optimization Report (2026-03-05)

## Objective
Improve processing speed on `test_video_june_03_3.mp4` without reducing path/segmentation quality and without skipping frames (`stride=1`).

## Environment
- Pipeline: `live_heading_demo.py`
- Segmentation: SegFormer (`models/my-segformer-road`)
- BEV smoothing: enabled (BEV-space motion compensation)
- Detection: YOLOv8n
- Input video: `test_video_june_03_3.mp4` (2057 frames)

## Code Changes
1. `fast_road_detector.py`
- Added `return_overlay` flag to `process_frame(...)` so live pipeline can skip unnecessary overlay creation.

2. `live_heading_demo.py`
- Call segmentation with `return_overlay=False`.
- Skip heavy visualization/compositing when running `--headless` and not saving video.
- Added binary-mask fast path in runtime split stage.
- Fixed false codec/decode error message at normal video EOF.

## Benchmarks
Warm region for fair comparison: frame_id >= 100.

### Baseline (sample window)
- Log: `logs/run_20260305_145411.csv`
- Mean FPS: 4.82
- Mean total pipeline time: 193.85 ms
- seg_iou_mean: 0.9718
- has_path_rate: 0.0027

### Optimized (sample window, det stride 1)
- Log: `logs/run_20260305_151504.csv`
- Mean FPS: 5.43
- Mean total pipeline time: 184.29 ms
- seg_iou_mean: 0.9725
- has_path_rate: 0.0024

### Same-frame quality check (frames 100..830, baseline vs optimized)
- Baseline: FPS 4.823, t_total 193.852 ms, seg_iou 0.9718, has_path 0.0027
- Optimized: FPS 5.422, t_total 184.657 ms, seg_iou 0.9718, has_path 0.0027

Conclusion: +12.4% throughput improvement with no quality regression on matched frames.

### Additional speed mode (det stride 2)
- Log: `logs/run_20260305_152414.csv`
- Mean FPS: 6.42
- Mean total pipeline time: 155.94 ms
- seg_iou_mean: 0.9737
- has_path_rate: 0.0020

Use when throughput is priority and lower detection refresh rate is acceptable.

## Full-File Validation (no frame skip)
- Log: `logs/run_20260305_153055.csv`
- Processed frames: 2057 / 2057
- Wall time: 394.74 s (~6.58 min)
- Warm-region mean FPS: 5.39
- Warm-region mean total pipeline time: 185.54 ms

## Recommended Runtime Profiles
1. Quality-first optimized:
- `python live_heading_demo.py --video test_video_june_03_3.mp4 --headless --log --log-dir logs --detection-stride 1`

2. Faster processing:
- `python live_heading_demo.py --video test_video_june_03_3.mp4 --headless --log --log-dir logs --detection-stride 2`

## Thesis Integration Notes
Suggested placement:
- Chapter 4 implementation/performance subsection: optimization strategy + pipeline hotspots.
- Chapter 5 results subsection: before/after timing table and matched-frame quality preservation.

Suggested key sentence:
"Headless-path rendering elimination and overlay bypass improved throughput by ~12.4% (4.823→5.422 FPS on matched frames) while preserving segmentation stability (seg_iou unchanged at 0.9718)."
