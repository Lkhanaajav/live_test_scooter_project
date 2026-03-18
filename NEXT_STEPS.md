# Next Steps

## Highest-Value Follow-Up
- Manually correct a focused subset of pseudo-labels from the two unseen-style videos.
  - The current checkpoint clearly improved segmentation stability, but unseen-video path dynamics are still mixed.
  - The most direct next step is to add hand-corrected supervision from scenes like `IMG_1876` and `IMG_1877`.

- Add a small human-reviewed validation set that is not drawn from the same four pseudo-labeled source videos.
  - Right now the strongest downstream gains are on the videos that also supplied pseudo-label training frames.
  - A true held-out evaluation set is needed before treating this model as fully generalized.

- Compare `0.55` vs `0.60` threshold on the unseen videos only.
  - `0.60` won on the pseudo-label validation split.
  - Unseen-video path stability might benefit from a slightly less conservative threshold even if pseudo-label IoU drops slightly.

## Practical Pipeline Improvements
- Keep the new checkpoint path and threshold configurable at runtime.
  - The experiment exposed that `MODEL_DIR` and threshold should not stay hardcoded if model comparisons are going to remain reproducible.

- Consider a second training round with a curated hard-example subset.
  - Priority examples:
    - narrow turn entries
    - scenes where the candidate becomes too aggressive left/right
    - scenes with road/sidewalk ambiguity near the image bottom

- Preserve the new comparison workflow.
  - The following scripts are now useful ongoing tooling:
    - `simulation_camera_scooter/scripts/generate_binary_pseudo_labels.py`
    - `simulation_camera_scooter/scripts/train_binary_segformer.py`
    - `simulation_camera_scooter/scripts/tune_binary_threshold.py`
    - `simulation_camera_scooter/scripts/eval_binary_seg_models.py`
    - `simulation_camera_scooter/scripts/make_video_comparison_strips.py`

## Remaining Bottlenecks
- Teacher quality is still pseudo-label quality.
  - The Swin-L teacher is usable and productive, but it is not perfect supervision for scooter-specific scenes.

- `IMG_1921.MOV` remains codec-limited.
  - OpenCV only decodes 6,727 frames consistently even though metadata reports 9,176.
  - If a better transcoding path becomes available, re-run that video to recover the missing tail.

- Segmentation improved more than path behavior.
  - Planner source selection improved.
  - Unseen-video heading dynamics still need work.
