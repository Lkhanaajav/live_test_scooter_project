# Literature Review: Segmentation

## Question
For a monocular scooter sidewalk pipeline, should the repo keep the current segmentation stack, clean it up, or replace it?

## What The Repo Is Actually Using
The runtime path is already close to a **binary drivable-mask** system:

- `FastRoadDetector` thresholds class-1 probability into a binary mask.
- `masks.py` and `live_heading_demo.py` are already happy with binary 0/255 masks.
- The main real decision is therefore not "multi-class vs binary." It is:
  - which binary model?
  - which cleanup?
  - how much temporal hysteresis is helpful before planning?

## Paper / Code Review

### 1. SegFormer
- Paper: [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)
- Code: [NVlabs/SegFormer](https://github.com/NVlabs/SegFormer)

Relevant method details:
- hierarchical transformer encoder
- no positional encoding, which helps resolution flexibility
- lightweight all-MLP decoder

Why it matters here:
- the repo is already built around a SegFormer-style checkpoint
- the architecture is still a good fit for binary sidewalk masking
- the local benchmark confirms the family is fast enough when the checkpoint is good and the input size is sane

Assessment:
- **keep the SegFormer family**
- do not spend thesis time replacing the runtime model class unless CPU-only deployment becomes the main constraint

### 2. OneFormer
- Paper: [OneFormer: One Transformer to Rule Universal Image Segmentation](https://arxiv.org/abs/2211.06220)
- Code: [SHI-Labs/OneFormer](https://github.com/SHI-Labs/OneFormer)

Relevant method details:
- task-conditioned universal segmentation
- uses task prompts / tokens to switch between semantic, instance, and panoptic outputs
- strong label quality, but heavier runtime stack

Why it matters here:
- it is an excellent **offline teacher**
- it is a poor fit as the on-scooter runtime model in this repo

Repo evidence:
- the best checkpoint already present in this repo, `outputs/training/binary_segformer_oneformer_teacher/best_checkpoint`, came from this teacher-student direction
- the candidate checkpoint improved hand-labeled IoU from `0.7583` to `0.9464`

Assessment:
- **use OneFormer as a label generator / teacher**
- **do not use OneFormer as the runtime model**

### 3. PIDNet
- Paper: [PIDNet: A Real-time Semantic Segmentation Network Inspired by PID Controllers](https://arxiv.org/abs/2206.02066)
- Code: [XuJiacong/PIDNet](https://github.com/XuJiacong/PIDNet)

Relevant method details:
- explicitly separates detail, context, and boundary refinement branches
- designed to improve real-time accuracy / speed tradeoff on street-scene segmentation

Why it matters here:
- if the thesis later needs a **CPU-first runtime segmentation substitute**, PIDNet is a more sensible next candidate than a heavier transformer or learned BEV model
- boundary preservation is especially relevant for sidewalk edge quality

Assessment:
- **not implemented in this pass**
- **worth keeping as the next runtime-model candidate if 2D segmentation becomes the main bottleneck**

## Post-Processing Literature That Actually Helps This Repo
The most useful segmentation-side ideas were not giant new models. They were classical cleanup operators:

- connected-component filtering
- hole filling
- remove-small-holes / remove-small-objects style cleanup
- temporal hysteresis around the threshold

Official reference for the cleanup family:
- [scikit-image morphology documentation](https://scikit-image.org/docs/stable/api/skimage.morphology)

Why this matters:
- the scooter problem is dominated by mask topology and boundary stability
- small cleanup decisions often matter more for path quality than marginal mIoU gains on paper benchmarks

## What I Implemented And Tested

### Candidate model replacement
- Existing baseline: `simulation_camera_scooter/models/my-segformer-road`
- Candidate: `outputs/training/binary_segformer_oneformer_teacher/best_checkpoint`

### Cleanup variants evaluated
- `baseline_raw`
- `candidate_raw`
- `candidate_confhold`

`candidate_confhold` uses:
- threshold band hysteresis
- camera-space morphology
- center-weighted connected-component selection

## Results-Based Interpretation
The measured results support four conclusions:

1. **The shipped baseline is not good enough.**
   - `0.7583` IoU on the hand-labeled sample is not thesis-grade when a materially better checkpoint already exists in the same repo.

2. **The better SegFormer checkpoint is the correct immediate substitute.**
   - It is both more accurate and faster in local testing.

3. **Binary mask cleanup matters.**
   - Confidence hold reduced raw-mask IoU relative to `candidate_raw`, but it produced a very strong planning case when paired with the new no-BEV planners.

4. **The main segmentation work should stay pragmatic.**
   - Better labels + modest cleanup beat a full runtime architecture jump here.

## Recommendation
- Replace the shipped baseline with `binary_segformer_oneformer_teacher/best_checkpoint`.
- Keep the pipeline binary.
- Keep cleanup lightweight and topology-aware.
- Only pursue a new runtime architecture such as PIDNet if strict CPU-only deployment becomes the dominant requirement.

## Sources
- SegFormer paper: https://arxiv.org/abs/2105.15203
- SegFormer code: https://github.com/NVlabs/SegFormer
- OneFormer paper: https://arxiv.org/abs/2211.06220
- OneFormer code: https://github.com/SHI-Labs/OneFormer
- PIDNet paper: https://arxiv.org/abs/2206.02066
- PIDNet code: https://github.com/XuJiacong/PIDNet
- scikit-image morphology docs: https://scikit-image.org/docs/stable/api/skimage.morphology
