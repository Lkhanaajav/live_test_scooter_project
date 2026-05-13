# Literature Review: BEV

## Question
Is BEV genuinely useful in this monocular scooter setting, or is it an attractive but fragile middle layer?

## What The Repo Does Today
The current system performs a classical inverse-perspective style warp:

- camera mask -> homography -> `600 x 600` BEV -> morphology -> BEV planner

This gives nice metric intuition when it works. The problem is that the repo has only one front monocular camera and a hand-maintained calibration. That is the hardest setting for a static BEV warp to stay reliable.

## Paper / Code Review

### 1. Lift-Splat-Shoot
- Paper: [Lift, Splat, Shoot: Encoding Images From Arbitrary Camera Rigs by Implicitly Unprojecting to 3D](https://arxiv.org/abs/2008.05711)
- Code: [nv-tlabs/lift-splat-shoot](https://github.com/nv-tlabs/lift-splat-shoot)

Relevant method details:
- learns depth distributions
- lifts image features into 3D and splats them into BEV
- designed for arbitrary camera rigs, not just one manually calibrated front camera

Implication for this repo:
- strong BEV results are usually bought by **learning the view transformation**, not by trusting a static homography
- that makes sense for autonomous driving fleets, not for a single-scooter thesis with limited labeled data

### 2. BEVFormer
- Paper: [BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers](https://arxiv.org/abs/2203.17270)
- Code: [fundamentalvision/BEVFormer](https://github.com/fundamentalvision/BEVFormer)

Relevant method details:
- BEV queries
- spatial cross-attention into multi-camera features
- temporal self-attention across frames

Implication for this repo:
- modern high-performing BEV systems use **multi-camera + temporal fusion + heavy transformers**
- they are not evidence that a static monocular homography is reliable
- they are evidence that monocular / sparse-view BEV is hard enough to require learned view reasoning

### 3. GitNet
- Paper: [GitNet: Geometric Prior-based Transformation for Birds-Eye-View Segmentation](https://arxiv.org/abs/2204.07733)

Relevant method details from the abstract:
- explicitly states monocular BEV has a **spatial gap**
- proposes geometry-guided pre-alignment plus a ray-based transformer

Implication:
- this is directly aligned with the local failure mode
- the repo's static homography does not address the spatial gap; it assumes it away

### 4. FocusBEV
- Paper: [Focus on BEV: Self-calibrated Cycle View Transformation for Monocular Birds-Eye-View Segmentation](https://arxiv.org/abs/2410.15932)

Relevant method details:
- self-calibrated cross-view transformation
- temporal fusion with ego-motion
- explicitly tries to suppress BEV-irrelevant image regions

Implication:
- recent monocular BEV work is moving toward **self-calibration and temporal consistency**
- that is a strong hint that a fixed one-shot homography is too brittle in practice

### 5. SkyEye
- Paper / repo: [SkyEye: Self-Supervised Bird's-Eye-View Semantic Mapping Using Monocular Frontal View Images](https://github.com/robot-learning-freiburg/SkyEye)

Relevant code-level observation:
- the official repo expects Linux, CUDA, PyTorch, and dedicated BEV datasets

Implication:
- monocular front-view BEV is possible as a research direction
- it is not a drop-in replacement for this scooter repo

### 6. MonoScene
- Paper: [MonoScene: Monocular 3D Semantic Scene Completion](https://arxiv.org/abs/2112.00726)
- Code: [astra-vision/MonoScene](https://github.com/astra-vision/MonoScene)

Relevant code-level observation:
- the official repo is volumetric, dataset-heavy, and trains on multiple GPUs

Implication:
- full learned monocular occupancy / BEV stacks are outside the realistic implementation budget for this thesis pass

## What The Local Experiments Say

### BEV is useful when:
- the calibration is well aligned
- the near-field mask is dense
- the metric geometry is needed for obstacle projection or controller tuning

### BEV is fragile when:
- the segmentation edges are noisy
- the runtime frame size does not match the calibration well
- far-field regions dominate the warp
- the system depends on BEV as the **only** planning domain

The strongest local evidence against BEV-as-primary:
- in one profiled run, `path_source = none` on `4377 / 4407` frames because BEV occupancy nearly vanished
- on hand-labeled frames, image-space planners were often equal or better than BEV planners even when the underlying mask was good

## Thesis-Useful Interpretation
For this scooter problem, BEV is not worthless. It is just **overpromoted** in the current stack.

Best interpretation:
- **use BEV as an optional geometric aid**
- **do not force all planning through BEV**

Most defensible thesis statement:
- in a monocular scooter sidewalk pipeline, a static BEV homography is helpful for short-range geometry and visualization, but too fragile to be the sole planning representation

## Recommendation
- Keep BEV only as an optional near-field module.
- Crop or ignore unstable far-field regions when BEV is used.
- Do not invest thesis time in replacing the current homography with a full learned monocular BEV network unless the thesis scope becomes "learning BEV" rather than "robust path planning from monocular video."

## Sources
- Lift-Splat-Shoot paper: https://arxiv.org/abs/2008.05711
- Lift-Splat-Shoot code: https://github.com/nv-tlabs/lift-splat-shoot
- BEVFormer paper: https://arxiv.org/abs/2203.17270
- BEVFormer code: https://github.com/fundamentalvision/BEVFormer
- GitNet paper: https://arxiv.org/abs/2204.07733
- FocusBEV paper: https://arxiv.org/abs/2410.15932
- SkyEye repo: https://github.com/robot-learning-freiburg/SkyEye
- MonoScene paper: https://arxiv.org/abs/2112.00726
- MonoScene code: https://github.com/astra-vision/MonoScene
