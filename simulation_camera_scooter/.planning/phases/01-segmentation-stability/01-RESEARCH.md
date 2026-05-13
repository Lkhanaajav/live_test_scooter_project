# Phase 1: Segmentation Stability - Research

**Researched:** 2026-03-04
**Domain:** SegFormer temporal stability, EMA smoothing, checkpoint evaluation
**Confidence:** HIGH

## Summary

Phase 1 must eliminate visible segmentation flickering that currently corrupts BEV input. Analysis of the March 2026 logs reveals the problem quantitatively: only 62.9% of frames achieve IoU >= 0.85 (target: 90%), and 15.3% of frames drop to IoU = 0.0 -- meaning complete segmentation loss for runs of 10-20 frames. The median IoU is 0.934 (good when stable), confirming the issue is intermittent catastrophic drops rather than consistently poor segmentation.

Two levers exist: (1) selecting the best SegFormer checkpoint from the 10 available (checkpoint-500 through checkpoint-5000), since the current model (`my-segformer-road_new`) may not be the most temporally stable; and (2) tuning the `TemporalMaskSmoother` parameters (currently alpha=0.45, consistency_thresh=0.3) to bridge over short flicker episodes without over-smoothing dynamic obstacles.

**Primary recommendation:** Systematically benchmark all checkpoints on the demo video measuring per-frame IoU, then tune EMA parameters to close the gap from 62.9% to 90%+ frames at IoU >= 0.85.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| SEG-01 | Temporal stability: IoU >= 0.85 on >= 90% of frames | Checkpoint evaluation + EMA tuning; current baseline is 62.9% -- gap of ~27 percentage points |
| SEG-02 | SegFormer validated/fine-tuned on demo environment | 10 checkpoints available (500-5000 steps); evaluate each on representative frames for sidewalk class accuracy |
| SEG-03 | Temporal smoothing tuned: dynamic obstacles update within 2-3 frames | EMA alpha and consistency_thresh tuning; lower alpha smooths more but risks freezing obstacle regions |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| transformers | 4.52.2 | SegFormer inference via `SegformerForSemanticSegmentation` | Already in use, custom-trained model |
| PyTorch | 2.6.0+cu118 | Backend for SegFormer inference | Already in use |
| OpenCV | 4.12.0 | Mask operations, morphology, connected components | Already in use |
| NumPy | 2.1.1 | Array math for IoU computation, EMA blending | Already in use |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pandas | 2.2.2 | Log analysis of seg_iou column | Benchmarking checkpoint quality |
| pytest | (installed) | Validate smoothing behavior | Test mask smoother edge cases |

**No new libraries needed.** All work uses the existing stack.

## Architecture Patterns

### Existing Code Structure (DO NOT CHANGE)
```
fast_road_detector.py    # FastRoadDetector.process_frame() -- SegFormer inference
stabilization.py         # TemporalMaskSmoother -- EMA smoothing with IoU consistency
config.py                # MASK_SMOOTH_ALPHA, MASK_SMOOTH_CONSISTENCY_THRESH, SEG_IOU_*
live_heading_demo.py     # Orchestrator -- calls detector then smoother in sequence
```

### Pattern 1: Checkpoint Evaluation Script
**What:** Standalone script that loads each checkpoint, runs inference on the demo video, and logs per-frame IoU + sidewalk pixel count.
**When to use:** Plan 01-01 (evaluation).
**Key detail:** `FastRoadDetector` takes a `Config` with `model_dir` -- point it at each `models/checkpoint-XXXX` directory. The `process_frame()` method returns `(mask, overlay)`. Compare consecutive masks with `TemporalMaskSmoother._iou()`.

### Pattern 2: EMA Parameter Sweep
**What:** Run the pipeline on demo video with different (alpha, consistency_thresh) combinations, record IoU >= 0.85 rate.
**When to use:** Plan 01-02 (tuning).
**Key detail:** `TemporalMaskSmoother.__init__` accepts `alpha` and `consistency_thresh` parameters. The `smooth()` method with `return_iou=True` gives per-frame IoU. Sweep alpha in [0.2, 0.3, 0.4, 0.5, 0.6] and consistency_thresh in [0.2, 0.3, 0.4, 0.5].

### Anti-Patterns to Avoid
- **Retraining SegFormer from scratch:** 10 checkpoints already exist across 500-5000 training steps. Evaluate them first -- retraining takes hours and may not be needed.
- **Multi-frame voting / heavy temporal fusion:** Adds latency and complexity. EMA is already implemented and lightweight; tune it before adding new approaches.
- **Changing SegFormer input resolution during this phase:** Resolution affects both quality and speed. Keep at current SEG_INPUT_RES (640x360) for now; BEV calibration (Phase 2) is the bigger pixel-loss issue.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| IoU computation | Custom pixel counting | `TemporalMaskSmoother._iou()` | Already tested, handles edge cases (empty masks, float normalization) |
| Model loading | Custom weight loading | `AutoImageProcessor + SegformerForSemanticSegmentation.from_pretrained(checkpoint_dir)` | HuggingFace handles config/weights automatically |
| Temporal smoothing | New smoothing class | Existing `TemporalMaskSmoother` with tuned params | Already integrated in pipeline, has consistency check logic |

## Common Pitfalls

### Pitfall 1: Checkpoint Directory Structure
**What goes wrong:** Checkpoint directories (e.g., `models/checkpoint-500`) may lack `preprocessor_config.json` needed by `AutoImageProcessor.from_pretrained()`.
**Why it happens:** HuggingFace Trainer saves model weights but the preprocessor config lives in the parent model directory.
**How to avoid:** When loading a checkpoint, use the parent model's processor (`models/my-segformer-road_new`) and only swap the model weights from the checkpoint directory. Or copy `preprocessor_config.json` into each checkpoint dir.
**Warning signs:** `OSError: preprocessor_config.json not found` when loading checkpoint.

### Pitfall 2: IoU = 0.0 Runs Masking True Performance
**What goes wrong:** The 15.3% of frames with IoU = 0.0 may be complete segmentation failures (empty masks) rather than "different but valid" masks. These need different treatment.
**Why it happens:** If SegFormer outputs zero sidewalk pixels on a frame, IoU with any non-empty previous mask is 0.0.
**How to avoid:** Distinguish between "empty mask" (seg failure) and "low IoU" (mask shift). For empty masks, the smoother should hold the previous mask entirely. Check `sidewalk_mask_pixels` column in logs -- if 0, that's a seg failure.
**Warning signs:** `sidewalk_mask_pixels = 0` in log rows where `seg_iou = 0.0`.

### Pitfall 3: Over-Smoothing Hides Dynamic Obstacles
**What goes wrong:** Setting alpha too low (e.g., 0.15) makes the mask very stable but a person walking into the frame takes 8-10 frames to appear in the smoothed mask.
**Why it happens:** Low alpha = slow response to real changes.
**How to avoid:** SEG-03 requires dynamic obstacles to update within 2-3 frames. At alpha=0.45 (current), a new region reaches 50% blending in ~1.2 frames. At alpha=0.25, it takes ~2.4 frames. Don't go below alpha=0.25.
**Warning signs:** Obstacle overlay lags visibly behind the person's actual position in video playback.

### Pitfall 4: Confidence Threshold Interaction
**What goes wrong:** The `conf_thresh=0.6` in FastRoadDetector creates a hard binary mask. Lowering it may increase pixel count but also increase noise.
**Why it happens:** Marginal pixels (0.4-0.6 confidence) are the ones that flicker frame-to-frame.
**How to avoid:** Consider evaluating conf_thresh alongside checkpoint selection. A better checkpoint may produce higher-confidence predictions that are naturally more stable.

## Code Examples

### Loading a checkpoint for evaluation
```python
# Use parent model's processor, swap model weights from checkpoint
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

processor = AutoImageProcessor.from_pretrained("models/my-segformer-road_new", local_files_only=True)
model = SegformerForSemanticSegmentation.from_pretrained(
    "models/checkpoint-3000",  # or any checkpoint-XXXX
    local_files_only=True
).to("cuda").eval()
```

### Computing per-frame IoU between consecutive masks
```python
# From stabilization.py TemporalMaskSmoother._iou
def compute_iou(mask_a_255, mask_b_255):
    a = mask_a_255 > 127
    b = mask_b_255 > 127
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / max(union, 1.0)
```

### Sweeping EMA alpha
```python
from stabilization import TemporalMaskSmoother

for alpha in [0.25, 0.35, 0.45, 0.55, 0.65]:
    smoother = TemporalMaskSmoother(alpha=alpha, consistency_thresh=0.3)
    ious = []
    for mask in all_raw_masks:
        smoothed, iou = smoother.smooth(mask, return_iou=True)
        ious.append(iou)
    pct_stable = sum(1 for x in ious if x >= 0.85) / len(ious) * 100
    print(f"alpha={alpha}: {pct_stable:.1f}% frames >= 0.85 IoU")
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| No temporal smoothing | EMA with consistency check (stabilization.py) | Feb 2026 | Reduced mild flicker but IoU=0.0 drops persist |
| Single checkpoint | `my-segformer-road_new` (best of training) | Training time | 10 checkpoints available but never systematically compared for temporal stability |

## Quantitative Baseline (from March 2026 logs)

| Metric | Current Value | Target |
|--------|---------------|--------|
| Frames with IoU >= 0.85 | 62.9% | >= 90% |
| Frames with IoU = 0.0 | 15.3% | < 2% |
| Median IoU (when stable) | 0.934 | -- |
| Mean IoU (all frames) | 0.715 | -- |
| Frames with IoU < 0.22 (SEG_IOU_FAIL) | 19.7% | < 5% |
| EMA alpha | 0.45 | TBD (sweep) |
| Consistency threshold | 0.30 | TBD (sweep) |
| Confidence threshold | 0.60 | TBD (evaluate) |

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (installed) |
| Config file | none (default discovery) |
| Quick run command | `python -m pytest tests/ -x -q` |
| Full suite command | `python -m pytest tests/ -v` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SEG-01 | IoU >= 0.85 on >= 90% of frames | integration (benchmark script) | `python scripts/benchmark_seg_stability.py --video test_video_mar3_1_h264.mp4` | No -- Wave 0 |
| SEG-02 | Checkpoint produces correct sidewalk class | integration (benchmark script) | Same benchmark script per checkpoint | No -- Wave 0 |
| SEG-03 | Dynamic obstacles update within 2-3 frames | unit | `python -m pytest tests/test_temporal_smoother.py -x` | Partial (stabilization tested indirectly) |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/ -x -q`
- **Per wave merge:** `python -m pytest tests/ -v` + benchmark script on demo video
- **Phase gate:** Full suite green + benchmark shows >= 90% frames at IoU >= 0.85

### Wave 0 Gaps
- [ ] `scripts/benchmark_seg_stability.py` -- benchmark script to evaluate checkpoints and measure IoU stability
- [ ] `tests/test_temporal_smoother.py` -- unit tests for TemporalMaskSmoother with edge cases (empty masks, low-alpha response time)

## Open Questions

1. **Which checkpoint is most temporally stable?**
   - What we know: 10 checkpoints exist (500-5000 steps). `my-segformer-road_new` is the currently used one (likely checkpoint-4940 or checkpoint-5000 based on directory listing).
   - What's unclear: Whether earlier checkpoints (less overfitting) produce more stable frame-to-frame predictions.
   - Recommendation: Benchmark all 10 on demo video. This is Plan 01-01's core task.

2. **Are IoU = 0.0 frames caused by empty masks or totally wrong masks?**
   - What we know: Log column `sidewalk_mask_pixels` exists alongside `seg_iou`. If pixels = 0, it's an empty mask.
   - What's unclear: Haven't cross-referenced these columns yet.
   - Recommendation: First task in Plan 01-01 should correlate `seg_iou` with `sidewalk_mask_pixels`.

3. **Can confidence threshold adjustment help?**
   - What we know: Current conf_thresh = 0.6. Marginal pixels (0.4-0.6) likely flicker.
   - What's unclear: Whether lowering to 0.5 reduces flicker or increases noise.
   - Recommendation: Include conf_thresh in the sweep alongside alpha.

## Sources

### Primary (HIGH confidence)
- `stabilization.py` -- TemporalMaskSmoother source code, verified EMA logic and IoU computation
- `fast_road_detector.py` -- FastRoadDetector source code, verified inference pipeline and conf_thresh usage
- `config.py` -- all segmentation-related constants verified
- `logs/run_20260304_*.csv` -- quantitative IoU baseline from actual pipeline runs (2686 frames each)

### Secondary (MEDIUM confidence)
- `models/` directory listing -- confirmed 10 checkpoints available plus 2 named model directories
- HuggingFace transformers checkpoint loading -- standard pattern from transformers docs

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new libraries, everything already in use
- Architecture: HIGH -- code inspected directly, patterns verified from source
- Pitfalls: HIGH -- derived from actual log data analysis and code inspection
- Quantitative baseline: HIGH -- computed directly from pipeline CSV logs

**Research date:** 2026-03-04
**Valid until:** 2026-04-04 (stable domain, no external dependencies changing)
