# Candidate Selection And Rationale

## Selection Criteria
The user requested decisions based on:

- robustness
- monocular compatibility
- scooter / sidewalk suitability
- thesis value
- visual quality
- realistic implementation effort

I scored each candidate direction against those criteria qualitatively and then narrowed to the options with the best evidence-to-effort ratio inside this repo.

## Selected Candidates

| Candidate | Status | Why it was selected |
|---|---|---|
| Better binary SegFormer checkpoint | implemented + evaluated | immediate accuracy gain, already trainable in this repo, no architecture rewrite |
| Confidence-gated cleanup | implemented + evaluated | cheap way to reduce mask chatter and enforce topology |
| Connected-components filtering | implemented + evaluated | directly addresses mask fragments and off-center blobs |
| Image-space midpoint planner | implemented + evaluated | strongest speed / simplicity / quality trade in local tests |
| Image-space DT planner | implemented + evaluated | robust fallback when midpoint assumptions weaken |
| BEV kept as optional near-field tool | evaluated conceptually | still useful for metric visualization / obstacle projection, but not as primary planning domain |

## Rejected Or De-Prioritized Candidates

| Candidate | Decision | Why it was not prioritized |
|---|---|---|
| Full learned monocular BEV network | rejected for this pass | too much data / training / integration cost for the thesis objective |
| BEVFormer / Lift-Splat-Shoot class models | rejected for runtime substitution | designed for richer sensor setups and far heavier compute budgets |
| Keep current BEV DT as primary | rejected | too slow when valid and too fragile when invalid |
| Full graph-based centerline learner | rejected | interesting research value, but poor implementation-effort match for this repo |
| ONNX optimization as the main thesis axis | de-prioritized | local GPU benchmark did not show a meaningful win |

## Decision Matrix

| Direction | Robustness | Monocular fit | Runtime fit | Thesis value | Implementation effort | Final decision |
|---|---|---|---|---|---|---|
| Candidate SegFormer checkpoint | high | high | high | medium | low | keep |
| Confidence hold + CC cleanup | medium-high | high | high | medium | low | keep |
| Image midpoint planner | high | high | very high | high | low | keep |
| Image DT planner | high | high | medium-high | high | medium | keep |
| BEV near-field only | medium | medium | medium | high | low | keep as optional |
| BEV DT full pipeline | low-medium | low | low | medium | already paid | baseline only |
| Learned monocular BEV | unknown here | high in principle | low here | high | very high | defer |
| Cost-map + A* / Smac | high | high | medium | high | medium | future upgrade |

## Concrete Implementations Added

### New planner module
- `simulation_camera_scooter/image_path_planner.py`

Implemented classes:
- `CameraMidpointPlanner`
- `CameraDtPlanner`

### New evaluation harness
- `simulation_camera_scooter/scripts/eval_hand_annotated_pipeline.py`

Outputs:
- segmentation metrics
- planner metrics
- comparison images
- summary tables

### New tests
- `simulation_camera_scooter/tests/test_image_path_planner.py`

Also revalidated alongside:
- `simulation_camera_scooter/tests/test_dt_path_planner.py`
- `simulation_camera_scooter/tests/test_boundary_inference.py`

## Why This Selection Is Defensible
The selected candidates are not just "simpler." They are better matched to the problem:

- one front camera
- sidewalk corridor following
- thesis need for interpretable failure analysis
- real runtime constraints

The key thesis insight is that **monocular scooter pathing does not automatically benefit from forcing everything into BEV first**. Once that assumption is relaxed, much cheaper and more robust planners become viable.
