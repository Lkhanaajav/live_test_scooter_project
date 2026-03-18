# Training Log

## Session: 2026-03-17 Binary Drivable Mask Student

## Environment
- Python: `3.11.9`
- PyTorch before upgrade: `2.7.1+cpu`
- PyTorch after upgrade: `2.10.0+cu128`
- GPU used: `NVIDIA GeForce RTX 5070`

### Environment command
```powershell
python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
```

## Teacher Label Generation

### Intended teacher
- `shi-labs/oneformer_cityscapes_dinat_large`

### Actual teacher used
- `shi-labs/oneformer_cityscapes_swin_large`

### Why
- DiNAT was attempted first but failed in this environment because the available `natten` backend did not expose the functions expected by `transformers`.
- Swin-L loaded and ran reliably on CUDA, so it became the practical teacher for this session.

### Teacher command
```powershell
python simulation_camera_scooter\scripts\generate_binary_pseudo_labels.py --save-previews
```

### Teacher output
- Root: `outputs/pseudo_labels/oneformer_cityscapes_swin_large_binary`
- Images processed: `400`
- Mean drivable ratio: `0.3416`
- Mean runtime: `635.1 ms / frame`
- Collapse mapping:
  - `road`
  - `sidewalk`

## Student Training Run

### Initialization
- Init checkpoint: `simulation_camera_scooter/models/my-segformer-road`

### Training command
```powershell
python simulation_camera_scooter\scripts\train_binary_segformer.py --epochs 10 --batch-size 4 --num-workers 2
```

### Dataset
- Images root: `simulation_camera_scooter/annotation_frames`
- Masks root: `outputs/pseudo_labels/oneformer_cityscapes_swin_large_binary/masks`
- Total pairs: `400`
- Train count: `320`
- Validation count: `80`
- Validation scheme: every 5th frame per source video folder

### Model / Loss / Size
- Architecture: SegFormer
- Target size: `640x360`
- Loss: weighted cross-entropy + Dice
- Class weights: `[1.0, 1.9317]`
- LR: `5e-5`
- Weight decay: `1e-4`
- Epochs: `10`
- Batch size: `4`

### Epoch History

| Epoch | Train loss | Val loss | Val IoU | Val Precision | Val Recall |
|---|---:|---:|---:|---:|---:|
| 1 | 0.2781 | 0.1844 | 0.9123 | 0.9639 | 0.9446 |
| 2 | 0.1834 | 0.1519 | 0.9306 | 0.9835 | 0.9454 |
| 3 | 0.1527 | 0.1314 | 0.9390 | 0.9732 | 0.9640 |
| 4 | 0.1330 | 0.1244 | 0.9426 | 0.9803 | 0.9609 |
| 5 | 0.1260 | 0.1151 | 0.9432 | 0.9740 | 0.9675 |
| 6 | 0.1152 | 0.1104 | 0.9407 | 0.9714 | 0.9674 |
| 7 | 0.1089 | 0.1105 | 0.9422 | 0.9726 | 0.9678 |
| 8 | 0.1045 | 0.1108 | 0.9436 | 0.9760 | 0.9661 |
| 9 | 0.1040 | 0.1104 | 0.9437 | 0.9743 | 0.9678 |
| 10 | 0.1046 | 0.1097 | 0.9420 | 0.9705 | 0.9698 |

### Best checkpoint
- Epoch: `9`
- Val IoU: `0.9437`
- Path: `outputs/training/binary_segformer_oneformer_teacher/best_checkpoint`

### Last checkpoint
- Path: `outputs/training/binary_segformer_oneformer_teacher/last_checkpoint`

### Training runtime
- `308.1 s`

## Threshold Tuning

### Command
```powershell
python simulation_camera_scooter\scripts\tune_binary_threshold.py
```

### Results

| Threshold | IoU | Precision | Recall |
|---|---:|---:|---:|
| 0.35 | 0.9387 | 0.9628 | 0.9741 |
| 0.40 | 0.9409 | 0.9671 | 0.9721 |
| 0.45 | 0.9426 | 0.9709 | 0.9700 |
| 0.50 | 0.9437 | 0.9743 | 0.9678 |
| 0.55 | 0.9442 | 0.9774 | 0.9653 |
| 0.60 | 0.9444 | 0.9803 | 0.9627 |
| 0.65 | 0.9441 | 0.9830 | 0.9598 |
| 0.70 | 0.9432 | 0.9855 | 0.9565 |

### Selected runtime threshold
- `0.60`

## Output Files
- `outputs/training/binary_segformer_oneformer_teacher/history.csv`
- `outputs/training/binary_segformer_oneformer_teacher/summary.json`
- `outputs/training/binary_segformer_oneformer_teacher/best_checkpoint/`
