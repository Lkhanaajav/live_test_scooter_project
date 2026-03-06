# Cityscapes mIoU Task (Todo #2)

Date: 2026-03-05

## What was implemented
- Added script: `scripts/cityscapes_miou_segformer_b0.py`
- Script computes:
  - Cityscapes val mIoU (19-class train IDs)
  - Per-class IoU table
- Outputs:
  - `metrics/cityscapes_miou_segformer_b0.json`
  - `metrics/cityscapes_miou_segformer_b0_per_class.csv`

## Run command
```bash
python scripts/cityscapes_miou_segformer_b0.py \
  --cityscapes-root "<PATH_TO_CITYSCAPES_ROOT>" \
  --split val \
  --batch-size 1 \
  --num-workers 2
```

Optional quick sanity run:
```bash
python scripts/cityscapes_miou_segformer_b0.py \
  --cityscapes-root "<PATH_TO_CITYSCAPES_ROOT>" \
  --split val \
  --max-images 50
```

## Current blocker
- Cityscapes dataset root was not found automatically on local drives (no `leftImg8bit/` + `gtFine/` discovered).

## Next action
- Point `--cityscapes-root` to the dataset location and run the script.
- Paste resulting `mean_iou` and per-class IoU table into thesis Chapter 5 Section 5.1.2.
