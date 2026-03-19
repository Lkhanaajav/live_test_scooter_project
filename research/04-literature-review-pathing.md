# Literature Review: Pathing

## Question
Is the current DT-based path extraction enough, or should the planner move to a different formulation?

## Current Repo Planner Family
The current stack plans mostly in BEV and relies on:

- BEV mask cleanup
- distance transform / safe corridor logic
- DT ridge extraction
- optional older graph / template fallback logic

This is a classical centerline-from-occupancy approach.

## Method Families Reviewed

### 1. Medial-axis / DT centerline extraction
- Official implementation reference: [scikit-image morphology docs](https://scikit-image.org/docs/stable/api/skimage.morphology)

Why it is attractive:
- no training needed
- intuitive "maximize clearance" behavior
- matches sidewalk following well when the mask is topologically clean

Why it fails:
- very sensitive to holes, thin breaks, and warped geometry
- centerline topology becomes unstable at intersections, splits, or bad boundaries
- runtime can become high if the search space stays dense

Local interpretation:
- still useful
- not reliable enough to be the only planner

### 2. Learned graph extraction
- Paper: [CenterLineDet: CenterLine Graph Detection for Road Lanes with Vehicle-mounted Sensors by Transformer for HD Map Generation](https://arxiv.org/abs/2209.07734)
- Project page: https://tonyxuqaq.github.io/projects/CenterLineDet/

Relevant method details:
- iterative graph growth
- DETR-like transformer
- designed for complicated topology such as intersections

Why it matters:
- it shows that graph reasoning is the right abstraction when topology is the hard part

Why it does not fit this thesis pass:
- built for HD-map lane graphs
- uses a much richer sensor / dataset setting than this repo
- too large a jump for the current scooter stack

### 3. Cost-map + A* / Hybrid-A*
- Official docs: [Nav2 Smac Planner](https://docs.nav2.org/configuration/packages/configuring-smac-planner.html)
- Code: [nav2_smac_planner](https://github.com/ros-navigation/navigation2/tree/main/nav2_smac_planner)

Relevant method details:
- optimized `2D A*`, `Hybrid-A*`, and `State Lattice` planners
- explicitly designed for robot planning over costmaps

Why it matters:
- if obstacle reasoning becomes more important, cost-map search is the most standard and defensible upgrade path
- Hybrid-A* is especially attractive if the scooter steering model needs to be baked into the path

Why it was not the first implementation here:
- the current repo already has mask-based centerline logic, so the lowest-risk test was to stay geometric
- a cost map without a good front-end mask still inherits segmentation failures

### 4. Boundary-midpoint centerline
This is the simplest method reviewed, but it was not useless baseline engineering. It was a strong candidate:

- find the usable corridor in each image row
- pick the midpoint of the valid boundary pair
- smooth the row-wise centerline

Why it is powerful here:
- no BEV required
- directly respects what the camera actually sees
- very low runtime
- naturally monocular-compatible

This method is simple enough that it is often ignored in papers, but it matches the scooter sidewalk geometry unusually well.

## What I Implemented
New image-space planners were added in:

- `simulation_camera_scooter/image_path_planner.py`

Implemented methods:
- `CameraMidpointPlanner`
- `CameraDtPlanner`

These planners intentionally avoid BEV and run directly on the camera mask.

## What The Results Say
From `research/artifacts/tables/planner_hand_annotations_summary.csv`:

### Baseline mask case
- `img_midpoint` beat `bev_dt_full` on both center error and runtime
- `img_dt` matched or slightly beat `bev_dt_full` quality while being about `8x` faster

### Candidate cleaned mask case
- `img_midpoint` had the best center error by a large margin
- `img_dt` had the best inside-GT ratio
- both were drastically faster than BEV planners

This leads to a clear interpretation:

- **DT itself is not the enemy**
- **DT in the current BEV formulation is the problem**

## Recommendation
Use a **hybrid geometric planner stack**:

1. image-space midpoint planner as the primary path extractor
2. image-space DT planner as the fallback when corridor shape becomes irregular
3. optional future upgrade to cost-map + A* / Smac if obstacle reasoning becomes central

Do not keep BEV DT as the primary planner.

## Sources
- scikit-image morphology docs: https://scikit-image.org/docs/stable/api/skimage.morphology
- CenterLineDet paper: https://arxiv.org/abs/2209.07734
- CenterLineDet project page: https://tonyxuqaq.github.io/projects/CenterLineDet/
- Nav2 Smac Planner docs: https://docs.nav2.org/configuration/packages/configuring-smac-planner.html
- Nav2 Smac Planner code: https://github.com/ros-navigation/navigation2/tree/main/nav2_smac_planner
