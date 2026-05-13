# BEV Calibration SOP

Use this procedure whenever the camera mount, pitch, height, or resolution changes. The goal is a calibration that generalizes across runs, not one that only looks good on one frame.

## Prerequisites

- Mount the camera in its final scooter position before recording.
- Use the same runtime resolution you will use in `live_heading_demo.py`.
- Pick a straight sidewalk segment with both left and right edges clearly visible.
- Record in stable lighting when possible; avoid glare, heavy shadow bands, and camera motion.

## Record The Calibration Video

1. Park the scooter so the camera is level relative to the sidewalk.
2. Record 5-10 seconds of stationary video.
3. Make sure both sidewalk edges are visible from the bottom of the frame into the far field.
4. Keep the frame wide. A narrow central strip causes low BEV pixel survival.
5. Save the file into the project root or another stable location you can reference later.

## Back Up The Current Calibration

Run:

```powershell
Copy-Item bev_calibration.npy bev_calibration_backup_YYYYMMDD.npy
```

Keep the backup until the new calibration is validated on logs.

## Run Interactive Calibration

Run:

```powershell
python live_heading_demo.py --calibrate --video <your_video>.mp4
```

Click four points in this exact order:

1. Bottom-Left: near-field left sidewalk edge
2. Bottom-Right: near-field right sidewalk edge
3. Top-Right: far-field right sidewalk edge
4. Top-Left: far-field left sidewalk edge

Rules:

- Make the trapezoid as wide as the real sidewalk allows.
- Use actual sidewalk edges, not shadows, grass boundaries, or curb artifacts.
- Do not pick a narrow center strip just to make the transform look tidy.
- Press `r` to reset if the geometry looks wrong.
- Press `s` to save only after all four points outline a realistic sidewalk region.

## Validate The Calibration

First check that the calibration loads cleanly:

```powershell
python -c "from bev_calibration import load_bev_params; load_bev_params()"
```

This should not print the ill-conditioned warning.

Then run a representative logged clip:

```powershell
python live_heading_demo.py --video <representative_video>.mp4 --save --headless --log
python scripts/measure_bev_survival.py logs/run_YYYYMMDD_HHMMSS.csv
```

Acceptance criteria:

- `cond(H) < 1e6` so `load_bev_params()` does not warn
- mean pixel survival >= 50%
- `has_path >= 60%`
- no heading reversals > 90 degrees
- visual BEV path stays on the sidewalk centerline instead of edges or grass

## When To Recalibrate

Redo calibration if any of these happen:

- camera mount position or pitch changes
- runtime camera resolution changes
- `load_bev_params()` starts warning again
- mean BEV survival drops below 40%
- previously good videos lose path reliability

## Troubleshooting

- Low survival with no warning: calibration points are probably too narrow.
- Good near field but bad far field: top points are too low, too central, or on the wrong edges.
- Good calibration on one run only: you likely tuned to a single frame instead of the real sidewalk geometry.
- Sudden regression after a hardware change: verify camera height, tilt, and resolution before changing planner parameters.
