# Experiment Log

---

## Session: 2026-03-18 — Simple Road Pipeline Simplification (Branch: simplify-pipeline-simple-roads)

### Goal
Simplify the navigation pipeline to work reliably on clean simple-road scenes.
Priority: stable path > low-noise mask > smooth frame-to-frame behavior.
NOT targeting complex urban edge cases.

### Environment
- GPU: NVIDIA RTX 5070 (CUDA 12.8, PyTorch 2.10.0+cu128)
- Python, scipy 1.16.0, OpenCV 4.12.0

### Test Videos
| Video | Resolution | FPS | Duration | Priority |
|-------|-----------|-----|----------|----------|
| test_video_june_03_3.mp4 | 1280x720 | 30 | 68.6s | HIGH — main dev video |
| IMG_1877.MOV | 1920x1080 | 30 | 45.4s | HIGH — cleanest unseen |
| IMG_1876.MOV | 1080x1920 | 30 | 16.7s | MED — portrait |
| IMG_1921.MOV | 1920x1080 | 30 | 306.2s | LOW — long |
| IMG_1878/1922/1924.MOV | 1920x1080 | 30 | varies | MED |

### Baseline Noise Sources (Simple Road Context)
1. 7 templates active — sharp turn templates win on noisy frames even on straight roads
2. Mask EMA α=0.65 — current frame dominates, noise propagates quickly
3. DT SG window=9 — short, centerline has grid quantization jitter
4. straight_preference_margin=0.03 — barely penalizes switching to turn templates
5. Path EMA min_alpha=0.35, max_alpha=0.85 — fast tracking, limited smoothing
6. Heading EMA alpha=0.50 — moderate only

### Simple Road Improvements (Hypothesis)
- Reduce to 3 templates (straight, gentle-left, gentle-right)
- Increase straight_preference_margin to 0.08
- Lower mask EMA alpha to 0.50 (more inertia)
- Increase DT SG window to 15
- Lower path EMA max_alpha to 0.72
- Lower heading EMA alpha to 0.35
- Add stronger near-field morphological close for road gap filling

### Plan
1. [x] Inspect repo, understand baseline
2. [x] Locate test videos
3. [x] Write `simple_road_pipeline.py`
4. [x] Write `scripts/eval_simple_road.py`
5. [ ] Run on test_video_june_03_3.mp4, IMG_1877.MOV
6. [ ] Compare baseline vs improved metrics
7. [ ] Update EVALUATION_REPORT.md

---

## Session: 2026-03-17 Binary Drivable Mask Upgrade

### Context
- Goal: replace or improve the current scooter segmentation stage with a more stable binary drivable-mask model for downstream BEV/path planning.
- Final target schema:
  - `drivable = 1`
  - `non-drivable = 0`
- Constraint from repo reality: the current deployed runtime already behaves as a binary mask pipeline even though some legacy code paths still mention `ROAD_ID=1` and `SIDEWALK_ID=2`.

### Baseline Identified
- Baseline segmentation runtime:
  - `simulation_camera_scooter/fast_road_detector.py`
  - `simulation_camera_scooter/live_heading_demo.py`
  - current default checkpoint from `simulation_camera_scooter/config.py`: `models/my-segformer-road`
- Baseline segmentation behavior:
  - `FastRoadDetector.process_frame()` upsamples logits, takes softmax for class `road_id`, thresholds it, and emits a binary `uint8` mask.
  - `simulation_camera_scooter/masks.py` already treats `{0,255}` masks as a valid binary-input case.
- Baseline downstream planner:
  - `simulation_camera_scooter/live_heading_demo.py`
  - `simulation_camera_scooter/realtime_nav_core.py`
  - BEV projection + BEV cleanup + graph/template planner + controller.

### Data Inventory
- Test videos found in `simulation_camera_scooter/test_videos/`:
  - `IMG_1876.MOV`
  - `IMG_1877.MOV`
  - `IMG_1878.MOV`
  - `IMG_1921.MOV`
  - `IMG_1922.MOV`
  - `IMG_1924.MOV`
- Extracted annotation frames found in `simulation_camera_scooter/annotation_frames/`:
  - 400 JPGs total
  - 100 each from `IMG_1878`, `IMG_1921`, `IMG_1922`, `IMG_1924`
- Important evaluation caveat:
  - training on these 400 frames contaminates 4 of the 6 available evaluation videos.
  - I will still evaluate on all 6 because the user explicitly requested it, but I need to report `IMG_1876` and `IMG_1877` as the cleanest unseen-video check.

### Teacher Model Decision
- Chosen teacher: `shi-labs/oneformer_cityscapes_dinat_large`
- Why:
  - official OneFormer repo reports Cityscapes semantic `mIoU = 83.1` for DiNAT-L vs `83.0` for Swin-L.
  - both official Hugging Face checkpoints are available in the current environment, but DiNAT-L is the slightly stronger accessible semantic teacher.
  - local verification showed the class metadata loads correctly and includes Cityscapes `road` and `sidewalk` labels.
- Binary collapse plan:
  - teacher semantic classes mapped into binary drivable mask.
  - primary first-pass mapping hypothesis: `road`, `sidewalk`, and `parking` count as drivable; everything else is non-drivable.
  - this mapping will be audited visually before student training.

### Risks Identified Up Front
- Current PyTorch install is CPU-only even though the machine has an RTX 5070.
- No maintained dense segmentation training script exists in the repo for the current binary runtime path.
- The current live demo hardcodes `MODEL_DIR`, which makes baseline-vs-new comparison less reproducible than it should be.
- Teacher pseudo-labels may over-segment road-adjacent regions or miss narrow sidewalk structure in scooter footage.
- All available videos are iPhone-style mobile footage; scene motion and rolling-shutter jitter may dominate gains from segmentation alone.

### First Concrete Experiment Plan
1. Build reproducible experiment scaffolding:
   - create root logs
   - create organized `outputs/` tree
   - make baseline/new-model evaluation scriptable by model path
2. Establish reproducible baseline:
   - run current pipeline on all 6 videos
   - save per-video logs, overlays, and summary metrics
3. Generate teacher pseudo-labels for all 400 annotation frames:
   - use OneFormer DiNAT-L semantic output
   - collapse to binary drivable masks
   - record class mapping and any confidence heuristics
4. Fine-tune a practical student:
   - start from existing SegFormer binary checkpoint family in repo
   - train for stable binary mask output, not multi-class output
   - prefer simple reproducible training loop over new framework complexity
5. Evaluate new checkpoint on all videos:
   - compare baseline segmentation stability
   - compare downstream planner behavior
   - save structured before/after artifacts
6. Write final `EVALUATION_REPORT.md` and `NEXT_STEPS.md`

### Commands Run Before Plan Lock
- `git pull --no-edit origin main`
- `git status --short --branch`
- repo/documentation inspection with `Get-ChildItem`, `Get-Content`, and `rg`
- video inventory and decode checks with OpenCV
- environment inspection:
  - `nvidia-smi`
  - Python package checks for `torch`, `transformers`, `huggingface_hub`
- teacher availability checks:
  - Hugging Face model/processor load for official OneFormer Cityscapes checkpoints
  - class metadata validation for road/sidewalk indices

### Initial Hypotheses
- Hypothesis 1:
  - a stronger teacher-generated binary mask set will reduce spurious drivable regions better than the current runtime checkpoint.
- Hypothesis 2:
  - even a modest segmentation improvement will matter in low-occupancy videos because the BEV/path stack is sensitive to missing or fragmented corridor support.
- Hypothesis 3:
  - the largest immediate improvement may come from student thresholding/post-processing stability, not only from raw training loss.

### Files Changed So Far
- `EXPERIMENT_LOG.md` created
- `TRAINING_LOG.md` created
- `NEXT_STEPS.md` created
- `outputs/` folder tree created

### Current Status
- Stage: planning complete, execution starting.
- Next action: patch the runtime/evaluation path so the model checkpoint can be swapped cleanly, then run the all-video baseline.

## Execution Notes

### 2026-03-17 19:05 to 19:10 Central
- Action:
  - patched `simulation_camera_scooter/live_heading_demo.py`
- Why:
  - baseline vs candidate comparison was not reproducible while `MODEL_DIR` and output video path were hardcoded.
- Changes:
  - added runtime `model_dir`
  - added runtime `output_video_path`
  - later added runtime `seg_conf_thresh`
  - disabled internal detector logging from the live replay path

### 2026-03-17 19:11 to 19:16 Central
- Action:
  - upgraded PyTorch from CPU-only to CUDA
- Command:
  - `python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio`
- Result:
  - `torch 2.10.0+cu128`
  - CUDA available on `NVIDIA GeForce RTX 5070`

### 2026-03-17 19:16 to 19:19 Central
- Action:
  - attempted teacher validation with `shi-labs/oneformer_cityscapes_dinat_large`
- Result:
  - blocked by `natten` compatibility mismatch on Windows (`natten2dav` / `natten2dqkrpb` import failure)
- Decision:
  - switched to `shi-labs/oneformer_cityscapes_swin_large`
  - rationale: official checkpoint, clean CUDA execution, no broken backend chain

### 2026-03-17 19:19 to 19:27 Central
- Action:
  - created and ran `simulation_camera_scooter/scripts/generate_binary_pseudo_labels.py`
- Commands:
  - smoke: `python simulation_camera_scooter\scripts\generate_binary_pseudo_labels.py --limit 5 --save-previews`
  - full: `python simulation_camera_scooter\scripts\generate_binary_pseudo_labels.py --save-previews`
- Result:
  - 400 masks generated
  - outputs under `outputs/pseudo_labels/oneformer_cityscapes_swin_large_binary`
  - mean drivable ratio `0.3416`
  - mean runtime `635 ms/frame`

### 2026-03-17 19:27 to 19:32 Central
- Action:
  - created and ran `simulation_camera_scooter/scripts/train_binary_segformer.py`
- Command:
  - `python simulation_camera_scooter\scripts\train_binary_segformer.py --epochs 10 --batch-size 4 --num-workers 2`
- Result:
  - best validation IoU `0.9437`
  - best checkpoint at epoch `9`
  - output checkpoint: `outputs/training/binary_segformer_oneformer_teacher/best_checkpoint`

### 2026-03-17 19:32 to 19:33 Central
- Action:
  - tuned binary threshold with `simulation_camera_scooter/scripts/tune_binary_threshold.py`
- Result:
  - best threshold on pseudo-label validation split: `0.60`

### 2026-03-17 19:34 to 19:35 Central
- Action:
  - created and smoke-tested `simulation_camera_scooter/scripts/eval_binary_seg_models.py`
- Purpose:
  - confirm baseline/candidate replay wiring before full run

### 2026-03-17 19:36 to 21:48 Central
- Action:
  - full all-video replay for baseline and candidate
- Command:
  - `python simulation_camera_scooter\scripts\eval_binary_seg_models.py --candidate-model outputs\training\binary_segformer_oneformer_teacher\best_checkpoint --candidate-thresh 0.6 --output-root outputs\evaluation\binary_model_replay_full --save-video`
- Result:
  - baseline and candidate processed on all 6 videos
  - rendered videos saved
  - summary saved to:
    - `outputs/evaluation/binary_model_replay_full/summary.json`
    - `outputs/evaluation/binary_model_replay_full/summary.md`

### 2026-03-17 21:49 Central
- Action:
  - generated visual comparison strips
- Command:
  - `python simulation_camera_scooter\scripts\make_video_comparison_strips.py`
- Result:
  - comparison JPGs saved under `outputs/comparisons/binary_model_replay_full`

## Key Findings
- The candidate model improved segmentation stability on every test video.
- Weighted all-frame unstable rate dropped from `1.46%` to `0.33%`.
- Weighted fallback rate dropped from `18.98%` to `14.27%`.
- Weighted template rate increased from `73.72%` to `79.34%`.
- Generalization is mixed:
  - unseen videos improved in segmentation statistics and planner mode selection
  - unseen heading dynamics did not improve

## Files Added During Session
- `simulation_camera_scooter/scripts/generate_binary_pseudo_labels.py`
- `simulation_camera_scooter/scripts/train_binary_segformer.py`
- `simulation_camera_scooter/scripts/tune_binary_threshold.py`
- `simulation_camera_scooter/scripts/eval_binary_seg_models.py`
- `simulation_camera_scooter/scripts/make_video_comparison_strips.py`
- `EXPERIMENT_LOG.md`
- `TRAINING_LOG.md`
- `EVALUATION_REPORT.md`
- `NEXT_STEPS.md`

## Final Experimental Decision
- Best checkpoint from this session:
  - `outputs/training/binary_segformer_oneformer_teacher/best_checkpoint`
- Best runtime threshold from this session:
  - `0.60`
- Deployment recommendation:
  - candidate checkpoint is worth keeping as the stronger binary segmentation baseline
  - but I would still want a small human-corrected unseen-scene fine-tune before calling it the final general solution
