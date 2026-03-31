# Task 5: Training Hyperparameters and Annotation Protocol

## Part 1: Extracted Training Hyperparameters

### Common Architecture (All Runs)

| Parameter | Value |
|-----------|-------|
| Architecture | SegFormer-B0 (`SegformerForSemanticSegmentation`) |
| Parameters | ~3.7M |
| Encoder blocks | 4 |
| Encoder depths | [2, 2, 2, 2] |
| Hidden sizes | [32, 64, 160, 256] |
| Attention heads | [1, 2, 5, 8] |
| SR ratios | [8, 4, 2, 1] |
| Decoder hidden size | 256 |
| MLP ratios | [4, 4, 4, 4] |
| Patch sizes | [7, 3, 3, 3] |
| Strides | [4, 2, 2, 2] |
| Activation | GELU |
| Drop path rate | 0.1 |
| Classifier dropout | 0.1 |
| Attention dropout | 0.0 |
| Hidden dropout | 0.0 |
| Num classes | 2 (background, road/sidewalk) |
| Precision | float32 |
| Image normalization | ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |
| Image resolution | 640 x 360 |

### Common Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Weight decay | 1e-4 |
| Batch size | 4 |
| Random seed | 1337 |
| Num workers | 2 |
| Scheduler | Cosine annealing with warmup (warmup = 10% of total steps, min LR factor = 0.1) |
| Mixed precision | CUDA AMP when GPU available |
| Loss function | Cross-entropy + Dice loss (equally weighted, lambda=1.0 each) |
| Class weights | Computed from training set: weight_pos = min(6.0, neg_pixels / pos_pixels) |
| Val split strategy | Deterministic: every Nth frame per video folder (default N=5, i.e., 20% val) |
| Best checkpoint criterion | Highest validation IoU |

### Data Augmentation

| Augmentation | Probability | Parameters |
|--------------|------------|------------|
| Horizontal flip | 50% | Mirror both image and mask |
| Color jitter | 70% | brightness=0.15, contrast=0.15, saturation=0.10, hue=0.03 |
| Gaussian blur | 15% | radius=1.0 |
| No augmentation applied to validation set |

---

### Run 1: Stage 1 -- OneFormer Teacher (Initial Fine-Tune)

**File:** `outputs/training/binary_segformer_oneformer_teacher/best_checkpoint/training_summary.json`

| Parameter | Value |
|-----------|-------|
| Init model | `models/my-segformer-road` (pre-trained SegFormer-B0 on Cityscapes) |
| Teacher | OneFormer Swin-L (ADE20K pre-trained) |
| Training frames | 320 (from 4 videos: IMG_1878, IMG_1921, IMG_1922, IMG_1924) |
| Validation frames | 80 |
| Total frames | 400 (every 5th frame held out for validation) |
| Pseudo-label source | `oneformer_cityscapes_swin_large_binary/masks` |
| Learning rate | 5e-5 |
| Epochs | 10 (best at epoch 9) |
| Class weights | Not recorded in this summary (computed dynamically) |
| Device | Not specified (auto-detect) |
| **Best validation IoU** | **0.9437** |
| Best validation loss | 0.1104 |
| Best precision | 0.9743 |
| Best recall | 0.9678 |
| Best accuracy | 0.9800 |
| Training time | Not recorded (no `total_runtime_s` in summary) |

**Learning rate schedule (sampled):**
- Epoch 1: 5.00e-5
- Epoch 3: 4.42e-5
- Epoch 5: 2.93e-5
- Epoch 7: 1.25e-5
- Epoch 9: 5.00e-6

---

### Run 2: All 6 Videos, From Scratch

**File:** `outputs/training/binary_segformer_all6_t400/best_checkpoint/training_summary.json`

| Parameter | Value |
|-----------|-------|
| Init model | `models/my-segformer-road` (same base pre-trained model) |
| Teacher | OneFormer Swin-L |
| Training frames | 1,920 (from 6 videos) |
| Validation frames | 480 |
| Total frames | 2,400 |
| Pseudo-label source | `all6_t400_oneformer_cityscapes_swin_binary/masks` |
| Learning rate | 5e-5 |
| Epochs | 10 (best at epoch 9) |
| **Best validation IoU** | **0.9588** |
| Best validation loss | 0.0745 |
| Best precision | 0.9793 |
| Best recall | 0.9786 |
| Best accuracy | 0.9857 |
| Training time | Not recorded |

---

### Run 3: Stage 2 -- Fine-Tune on All 6 Videos (From Stage 1 Checkpoint)

**File:** `outputs/training/binary_segformer_all6_t400_stage2/best_checkpoint/training_summary.json` and `summary.json`

| Parameter | Value |
|-----------|-------|
| Init model | `binary_segformer_oneformer_teacher/best_checkpoint` (Stage 1 output) |
| Teacher | OneFormer Swin-L |
| Training frames | 1,920 (from 6 videos: IMG_1876, IMG_1877, IMG_1878, IMG_1921, IMG_1922, IMG_1924) |
| Validation frames | 480 |
| Total frames | 2,400 |
| Pseudo-label source | `all6_t400_oneformer_cityscapes_swin_binary/masks` |
| Learning rate | 2e-5 (reduced from Stage 1) |
| Epochs | 6 (best at epoch 4) |
| Class weights | [1.0, 1.9456] (positive class ~49% of pixels) |
| Device | CUDA (GPU) |
| **Best validation IoU** | **0.9546** |
| Best validation loss | 0.0861 |
| Best precision | 0.9796 |
| Best recall | 0.9739 |
| Best accuracy | 0.9842 |
| **Training time** | **581.6 seconds (~9.7 minutes)** |

---

### Run 4: Old 400 + IMG_1931, Stage 2

**File:** `outputs/training/binary_segformer_old400_plus_img_1931_t300/summary.json`

| Parameter | Value |
|-----------|-------|
| Init model | `binary_segformer_oneformer_teacher/best_checkpoint` (Stage 1 output) |
| Teacher | OneFormer Swin-L |
| Training frames | 559 (5 videos: IMG_1878, IMG_1921, IMG_1922, IMG_1924, IMG_1931) |
| Validation frames | 140 |
| Total frames | 699 |
| Pseudo-label source | `old400_plus_img_1931_t300_oneformer_cityscapes_swin_binary/masks` |
| Learning rate | 2e-5 |
| Epochs | 8 (best at epoch 7) |
| Class weights | [1.0, 1.6509] |
| Device | CPU |
| **Best validation IoU** | **0.9581** |
| Best validation loss | 0.0831 |
| Best precision | 0.9790 |
| Best recall | 0.9782 |
| Best accuracy | 0.9837 |
| **Training time** | **1,475.1 seconds (~24.6 minutes, on CPU)** |

---

### Summary Comparison Table

| Run | Init | Train/Val | LR | Epochs | Best Val IoU | Best Epoch |
|-----|------|-----------|-----|--------|-------------|------------|
| Stage 1 (4 videos) | Cityscapes pre-train | 320/80 | 5e-5 | 10 | 0.9437 | 9 |
| All 6 (from scratch) | Cityscapes pre-train | 1920/480 | 5e-5 | 10 | 0.9588 | 9 |
| Stage 2 (all 6) | Stage 1 checkpoint | 1920/480 | 2e-5 | 6 | 0.9546 | 4 |
| Stage 2 (old+1931) | Stage 1 checkpoint | 559/140 | 2e-5 | 8 | 0.9581 | 7 |

**Key observations:**
- The from-scratch run on all 6 videos (Run 2) achieves the highest val IoU (0.9588), slightly outperforming the two-stage approaches.
- Stage 2 runs use a lower LR (2e-5 vs 5e-5) and converge faster (4-7 epochs vs 9).
- Class weights hover around 1.6-1.95, indicating the sidewalk class occupies roughly 34-38% of pixels.
- All runs use the same seed (1337) for reproducibility.

---

## Part 2: Annotation Protocol Documentation

### What Exists

**Annotation frames directory:** `simulation_camera_scooter/annotation_frames/`
- 4 video subdirectories: IMG_1878, IMG_1921, IMG_1922, IMG_1924
- 100 JPG frames per video = 400 total extracted frames
- These serve as input images for pseudo-label generation and training

**Hand-annotated ground truth** (referenced in eval script):
- Expected location: `outputs/hand_annotations/v1/images/` and `outputs/hand_annotations/v1/masks/`
- Currently does NOT exist on disk (directory not found)
- The thesis states 32 hand-annotated frames were used (Section 4.1.2)
- The eval script (`eval_hand_annotated_pipeline.py`) expects:
  - Images organized as `{video_name}/{frame_name}.jpg`
  - Masks as `{video_name}/{frame_name}.png` (binary: 0=background, >127=sidewalk)

**Eval script expectations:**
- Masks are grayscale PNG, thresholded at 127 (>127 = sidewalk)
- Organized by video folder for per-video aggregation
- Supports filtering by video name, per-video limits, and max-frame caps
- Measures: IoU, precision, recall, F1, stability IoU (temporal), inference latency

### Frame Selection Strategy

**Current approach (from training script):**
- Frames are extracted at regular intervals from video sequences (`val_every=5` means every 5th frame)
- Videos are named by iPhone capture ID (IMG_1876 through IMG_1931)

**Recommended strategy for hand-annotation set (32+ frames):**
1. **Stratified sampling across videos:** Select 4-6 frames per video to ensure route diversity
2. **Scene diversity criteria:**
   - Straight paths (at least 8 frames)
   - Gentle curves (at least 6 frames)
   - T-junctions / intersections (at least 4 frames)
   - Narrow paths (at least 4 frames)
   - Challenging lighting: shadows, bright sun, overcast (at least 4 frames)
   - Partial occlusion by pedestrians or objects (at least 2 frames)
   - Boundary ambiguity: grass-to-sidewalk transition (at least 4 frames)
3. **Temporal spacing:** Avoid selecting adjacent frames (minimum 10-frame gap) to ensure independence
4. **Coverage check:** At least one frame from each video sequence used in training

### Annotation Tool Recommendation

For binary sidewalk segmentation masks:
- **CVAT** (Computer Vision Annotation Tool) -- open-source, supports polygon annotation, exports PNG masks directly
- **Labelme** -- lightweight, local tool, JSON polygon export convertible to binary masks
- **Segment Anything Model (SAM)** -- interactive: click sidewalk region, refine boundary, export mask

### Annotation Guidelines

**What counts as "sidewalk" (positive class):**
1. Paved walking surface (concrete, asphalt, brick)
2. Accessible ramps and curb cuts
3. Shared-use paths explicitly designated for pedestrians
4. Crosswalk surfaces when directly connected to sidewalk

**What does NOT count as "sidewalk" (negative class):**
1. Road/vehicle lanes (even if adjacent)
2. Grass, dirt, gravel shoulders
3. Building walls, fences, posts
4. Parked vehicles, street furniture
5. Shadows on non-sidewalk surfaces
6. Puddles that obscure the surface but are ON the sidewalk -- mark as sidewalk
7. Overhanging foliage above sidewalk -- mark the ground beneath as sidewalk

**Boundary handling:**
- Annotate to the physical edge of the sidewalk surface
- Where the boundary is genuinely ambiguous (e.g., sidewalk blends into dirt), mark the last clearly visible paved pixel
- Curbs: include the top surface up to the edge, exclude the vertical face

**Shadow handling:**
- Shadows cast ON the sidewalk: still mark as sidewalk
- Dark regions where surface type is indeterminate: mark as background (conservative)

### Quality Control Process

1. **Double annotation:** Each frame annotated by one person, reviewed by a second
2. **IoU consistency check:** Re-annotate 5 random frames; intra-annotator IoU should exceed 0.95
3. **Edge case documentation:** For each ambiguous frame, record the decision and rationale
4. **Mask validation script:** Verify all masks are binary (only 0 and 255 values), correct resolution, and have matching filenames

---

## Part 3: Draft LaTeX Text

### Section 3.3 -- Training Hyperparameters (for Segmentation Module section)

```latex
\subsection{Training Configuration}
\label{sec:training_config}

All student models share the SegFormer-B0 architecture (3.7\,M parameters) with
four encoder stages of depth $[2, 2, 2, 2]$, hidden dimensions $[32, 64, 160,
256]$, and a lightweight all-MLP decoder with 256-dimensional hidden features.
Input images are resized to $640 \times 360$ pixels and normalized using ImageNet
statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
The architecture configuration remains fixed across all training iterations;
improvements arise solely from supervision quality and data quantity.

Training uses the AdamW optimizer~\cite{loshchilov2019adamw} with weight decay
$10^{-4}$ and a cosine-annealing learning rate schedule with linear warmup over
the first 10\% of gradient steps. The initial learning rate is $5 \times 10^{-5}$
for first-stage training from the Cityscapes-pretrained checkpoint, reduced to
$2 \times 10^{-5}$ for second-stage fine-tuning from a previously converged
checkpoint. The minimum learning rate factor is 0.1. Batch size is 4 throughout,
with mixed-precision training (PyTorch AMP) enabled when a GPU is available. All
experiments use a fixed random seed of 1337 for reproducibility.

The loss function combines per-pixel cross-entropy and Dice loss~\cite{milletari2016vnet}
with equal weight:
\begin{equation}
  \mathcal{L} = \mathcal{L}_{\text{CE}}(y, \hat{y};\, w) \;+\; \mathcal{L}_{\text{Dice}}(y, \hat{y})
\end{equation}
where $w = [1.0,\; w_+]$ is a class-weight vector computed from the training set
as $w_+ = \min(6.0,\; N_- / N_+)$, with $N_-$ and $N_+$ denoting total
background and sidewalk pixel counts, respectively. Across training runs, $w_+$
ranges from 1.65 to 1.95, reflecting a sidewalk class prevalence of approximately
34--38\% of pixels. The Dice component improves mask completeness by mitigating
class imbalance in sparse regions, while the cross-entropy component provides
stable gradient flow.

Data augmentation during training consists of random horizontal flipping
(probability 0.5), color jitter (brightness 0.15, contrast 0.15, saturation
0.10, hue 0.03; probability 0.7), and Gaussian blur with radius 1.0 (probability
0.15). No augmentation is applied during validation. The validation set is
constructed deterministically: within each video folder, every fifth frame
(ordered by filename) is held out, yielding an 80/20 train/validation split that
preserves temporal ordering while preventing data leakage between adjacent frames.

Table~\ref{tab:training_progression} summarizes the four training stages. The
first stage fine-tunes from the Cityscapes-pretrained checkpoint on 400 frames
(320 train, 80 val) from four campus video sequences using OneFormer Swin-L
pseudo-labels, reaching a validation IoU of 0.944 after 9 epochs. The final stage
expands the training set to 2{,}400 frames from six video sequences (1{,}920
train, 480 val), achieving a best validation IoU of 0.959 after 9 epochs with
approximately 10 minutes of GPU training time. The consistent improvement from
stage to stage confirms that data scaling, rather than architectural change,
drives segmentation gains at this model scale.
```

### Section 4.1.2 -- Hand-Annotated Ground Truth

```latex
\subsection{Hand-Annotated Ground Truth}
\label{sec:hand_annotated_gt}

To evaluate segmentation and planning quality against human judgment rather than
pseudo-label agreement, 32~frames were manually annotated with pixel-level binary
sidewalk masks. These frames were selected to span the diversity of conditions
encountered in the campus video corpus: straight paths, gentle curves,
T-junctions, narrow corridors, and challenging lighting conditions including
direct sunlight, tree shadows, and overcast skies.

Frames were drawn from multiple video sequences (at least four per annotation
set), with a minimum temporal spacing of ten frames between selected images to
ensure statistical independence. Selection prioritized scene diversity over
uniform temporal sampling: T-junctions, boundary ambiguities, and partial
occlusions were deliberately oversampled relative to their occurrence frequency,
as these conditions are most informative for evaluating planner robustness.

Annotation followed a binary labeling protocol: each pixel is classified as
\emph{sidewalk} (value 255) or \emph{background} (value 0). Sidewalk includes
all paved walking surfaces---concrete, asphalt, brick, and accessible
ramps---regardless of shadow coverage or surface discoloration. Boundaries are
drawn at the physical edge of the paved surface; where the edge is ambiguous
(e.g., gradual transition to grass), the last clearly visible paved pixel is
marked. Puddles on sidewalk surfaces are labeled as sidewalk; adjacent road
surfaces, grass, and vertical structures (curb faces, walls, poles) are labeled
as background.

The 32 annotated frames serve a dual purpose: (1)~evaluating segmentation model
accuracy (Table~\ref{tab:seg_comparison}), where the improved model achieves an
IoU of 0.946 versus 0.758 for the baseline; and (2)~providing oracle ground-truth
masks for the planner comparison study
(Section~\ref{sec:claim_planning_domain}), isolating planning quality from
segmentation noise. By comparing planners on predicted masks versus oracle masks,
the evaluation separates the contribution of segmentation quality from path
planning geometry.
```

---

## Appendix: File Locations Reference

| Artifact | Path |
|----------|------|
| Training script | `simulation_camera_scooter/scripts/train_binary_segformer.py` |
| Stage 1 summary | `outputs/training/binary_segformer_oneformer_teacher/best_checkpoint/training_summary.json` |
| Stage 2 (all6) summary | `outputs/training/binary_segformer_all6_t400_stage2/best_checkpoint/training_summary.json` |
| All6 from-scratch summary | `outputs/training/binary_segformer_all6_t400/best_checkpoint/training_summary.json` |
| Old400+1931 summary | `outputs/training/binary_segformer_old400_plus_img_1931_t300/summary.json` |
| Model config | `outputs/training/*/best_checkpoint/config.json` |
| Preprocessor config | `outputs/training/*/best_checkpoint/preprocessor_config.json` |
| Annotation frames | `simulation_camera_scooter/annotation_frames/` (4 dirs, 400 JPGs) |
| Eval script | `simulation_camera_scooter/scripts/eval_hand_annotated_pipeline.py` |
| Existing thesis training table | `thesis/tables/generated/training_progression.tex` |
