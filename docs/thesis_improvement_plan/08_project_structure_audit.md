# 08 -- Project Structure Audit

**Date:** 2026-03-31
**Scope:** Full audit of `C:/Users/lhana/OneDrive/Desktop/scootedr/live_test_scooter_project/`
**Total project size:** 21 GB

---

## 1. Top-Level Inventory

| Item | Type | Size | Purpose | Issue? |
|------|------|------|---------|--------|
| `CLAUDE.md` | File | 24 KB | Claude Code project instructions | OK -- large but functional |
| `.gitignore` | File | 280 B | Git exclusions | INCOMPLETE -- see Section 12 |
| `.mcp.json` | File | 222 B | MCP server config (CARL) | OK |
| `simulation_camera_scooter/` | Dir | 6.0 GB | Main Python package | BLOATED -- data mixed with code |
| `thesis/` | Dir | 39 MB | LaTeX thesis | HAS ISSUES -- see Section 3 |
| `docs/` | Dir | 368 KB | Documentation archives | OK |
| `models/` | Dir | 6.3 MB | Root-level model files | DUPLICATES -- see Section 7 |
| `outputs/` | Dir | 13 GB | Evaluation outputs | MASSIVE -- needs pruning |
| `research/` | Dir | 9.4 MB | Research notes + artifacts | OK |
| `.claude/` | Dir | 3.8 MB | Claude Code config | OK |
| `.carl/` | Dir | 22 MB | CARL MCP server | BLOATED for a config dir |
| `.paul/` | Dir | 37 KB | PAUL workflow state | OK |
| `.codex/` | Dir | 2.8 MB | Codex agent config | DELETABLE if not actively used |
| `.planning/` | Dir | 696 KB | GSD planning artifacts | OK |
| `.pytest_cache/` | Dir | 35 KB | Pytest cache | Should be gitignored |
| `.tmp_everything_claude_code/` | Dir | 38 MB | Temp tooling cache | SHOULD BE DELETED |

**Verdict:** The root is reasonably clean after the 2026-03-30 cleanup, but 3 hidden directories (`.tmp_everything_claude_code`, `.carl/carl-mcp`, `.codex`) add ~63 MB of tooling bloat. The real weight problem is `simulation_camera_scooter/` (6 GB) and `outputs/` (13 GB), both dominated by video files and model checkpoints.

---

## 2. `simulation_camera_scooter/` -- Main Package (6.0 GB)

### 2.1 Size Breakdown

| Subdir/File | Size | Contents | Verdict |
|-------------|------|----------|---------|
| `test_videos/` | 3.3 GB | 11 raw .MOV/.mp4 recordings | MOVE OUT -- not source code |
| `test_video_june_03_3.mp4` | 76 MB | Loose video in package root | MOVE to test_videos/ |
| `result/` | 763 MB | Evaluation result videos | MOVE to outputs/ |
| `models/` | 756 MB | 10 checkpoints + 2 model dirs | CONSOLIDATE with root models/ |
| `overnight_runs/` | 832 MB | Overnight eval data | MOVE to outputs/ |
| `annotation_frames/` | 131 MB | Fine-tuning frame extracts | OK but could live under data/ |
| `eval_runs/` | 109 MB | Evaluation run outputs | MOVE to outputs/ |
| `demo_outputs/` | 44 MB | Demo screenshots/videos | MOVE to outputs/ |
| `logs/` | 11 MB | Session CSV logs | OK |
| `scripts/` | 364 KB | 26 evaluation/training scripts | OK |
| `tests/` | 720 KB | 14 test files | OK |
| `__pycache__/` | 692 KB | Bytecode cache | Gitignored, fine |
| `yolov8n.pt` | 6.3 MB | YOLO weights (DUPLICATE) | REMOVE -- use models/ copy |
| `bev_calibration.npy` | 160 B | Calibration (DUPLICATE) | REMOVE -- use models/ copy |
| `bev_calibration_backup_20260312.npy` | 160 B | Dated backup of calibration | REMOVE or move to models/ |
| `bev_H.npy`, `bev_Hinv.npy` | 200 B each | Homography matrices | OK |
| `turn_analysis.png`, `turn_detail.png`, `turns_survey.png` | 3.2 MB | Loose analysis images | MOVE to outputs/ or docs/ |
| `path_planners/` | 88 KB | Empty (only `__pycache__/`) | DELETE |
| `.claude/` | nested | Nested Claude config | REDUNDANT with root .claude/ |

### 2.2 Misplaced Files in Package Root

| File | Issue | Action |
|------|-------|--------|
| `CALIBRATION_SOP.md` | Operational doc in code dir | Move to `docs/` |
| `MUST_READ_TURN_CONTAINMENT.md` | Design doc in code dir | Move to `docs/` |
| `PHASE_11_1_CHANGELOG.md` | Changelog in code dir | Move to `docs/phase11/` |
| `RUNTIME_RUNBOOK.md` | Operational doc in code dir | Move to `docs/` |
| `ReadME.tex` | 88 bytes, single CLI command | DELETE -- vestigial |
| `sample_route.csv` | Test data in code root | Move to `tests/fixtures/` |
| `cityscapes_iou_drivable-segformer-b0.json` | Eval results | Move to `outputs/evaluation/` |
| `rugd_iou_drivable-segformer-b0.json` | Eval results | Move to `outputs/evaluation/` |
| `turn_analysis.png`, `turn_detail.png`, `turns_survey.png` | Ad hoc analysis images | Move to `outputs/` or `docs/research/` |

### 2.3 Model Checkpoint Bloat

Inside `simulation_camera_scooter/models/`:

| Directory | Size | Status |
|-----------|------|--------|
| `checkpoint-500` through `checkpoint-4500` | 15 MB each (135 MB total) | INTERMEDIATE -- keep only best |
| `checkpoint-5000` | 89 KB | Appears corrupted/incomplete |
| `my-segformer-road/` | 143 MB | ACTIVE production model |
| `my-segformer-road_new/` | 485 MB | Newer model variant |

**Recommendation:** Keep `my-segformer-road/` and `my-segformer-road_new/`. Delete or archive intermediate checkpoints. The 9 intermediate checkpoints waste 135 MB for no runtime purpose.

---

## 3. `thesis/` -- LaTeX Thesis (39 MB)

### 3.1 Structure

```
thesis/
  main.tex              139 KB   THE thesis
  references.bib         21 KB   Bibliography
  figures/               12 MB   All thesis figures
    pipeline/                    Pipeline diagrams (3 files)
    resolution/                  Resolution comparison (2 files)
    scooter/                     Hardware photos (2 files)
    tables_generated.tex         MISPLACED -- .tex file in figures/
  tables/
    generated/
      training_progression.tex   Auto-generated LaTeX table
  tools/                         Figure generation scripts (22 files)
    __pycache__/                 SHOULD BE GITIGNORED
    paper_banana_contexts.md     AI image generation prompts
  ou_template_example/           OU format reference (1 MB PDF)
    README.md                    Explains it is reference only
  paper_src/                     Elsevier journal paper variant
    images/                      DUPLICATE figures from thesis/figures/
    cas-common.sty, cas-dc.cls   Elsevier LaTeX classes
  reference/                     Old drafts
    2025_Lkhaana_Manuscript_draft.pdf   7.6 MB
    elsevier_draft.md                   48 KB
```

### 3.2 Issues Found

**CRITICAL: `tables_generated.tex` is inside `figures/`** -- A LaTeX table file sitting in the figures directory. This should be in `tables/generated/`.

**DUPLICATE figures:** `thesis/paper_src/images/` contains 22 files that are exact copies from `thesis/figures/`. The paper_src variant has one extra file (`challenges_1.png`, 144 KB) and one extra scooter photo (`scooter-image2.jpeg`, 4.5 MB). Total duplication: ~4.2 MB.

**23 unreferenced figures in `thesis/figures/`:**

| Unreferenced Figure | Size | Likely Purpose |
|---------------------|------|----------------|
| `bev_transform_process.png` | 388 KB | BEV transform visualization (generated, not yet used) |
| `gps_intent_architecture.png` | 408 KB | GPS intent diagram (generated, not yet used) |
| `pipeline/pipeline_detailed.png` | 469 KB | Detailed pipeline (generated, not yet used) |
| `pipeline/pipeline_overview.png` | 382 KB | Overview pipeline (generated, not yet used) |
| `planner_comparison_arch.png` | 427 KB | Planner architecture (generated, not yet used) |
| `heading_analysis.png` | 161 KB | Heading analysis plot |
| `latency_violin.png` | 187 KB | Latency violin plot |
| `fps_comparison.png` | 138 KB | FPS comparison plot |
| `temporal_stability.png` | 265 KB | Temporal stability plot |
| `training_curves.png` | 197 KB | Training curves plot |
| `image.png` | 189 KB | GENERIC NAME -- unclear what this shows |
| `road_false.png` | 135 KB | Duplicate of road_sidewalk_problem.png (SAME size) |
| `road_sidewalk_problem.png` | 135 KB | Road classification failure case |
| `bev_clean.png` | 17 KB | Cleaned BEV mask |
| `bev_mask.png` | 6 KB | BEV mask |
| `cleaned_mask.png` | 4 KB | Cleaned mask |
| `branch_endpoints.png` | 13 KB | Skeleton branch endpoints |
| `planned_vs_skeleton_overlay.png` | 35 KB | Planner overlay comparison |
| `skeleton_0000.png` | 4 KB | Raw skeleton |
| `hand_annotated.jpg` | 37 KB | Hand annotation example |
| `test_frame.jpg` | 749 KB | Test input frame |
| `test_masks.jpg` | 44 KB | Test mask visualization |
| `resolution/latency_resolution.png` | 44 KB | Resolution vs latency |

**Note:** `road_false.png` and `road_sidewalk_problem.png` are identical in size (135,317 bytes) -- likely exact duplicates.

**`paper_banana_contexts.md`** in `tools/` is an AI image generation prompt file (15 KB). Useful reference but not a "tool" -- better placed in `docs/` or clearly labeled.

**`ou_template_example/`** has a clear README explaining it is format reference only. Acceptable to keep but could be moved to `reference/` for consolidation.

### 3.3 Figure Format Issues

| Concern | Details |
|---------|---------|
| Mixed formats | 5 JPEG files, 34 PNG files -- inconsistent |
| Oversized images | `planner_comparison_example.png` (2.9 MB), `seg_comparison_example.jpg` (2.3 MB), `scooter-image1.jpeg` (2.9 MB) -- these will make the PDF heavy |
| Generic names | `image.png` gives zero context about its content |
| Inconsistent naming | Some use underscores (`bev_clean.png`), some use hyphens (`scooter-image1.jpeg`), some are dates (`segformer_compare_20251105_154023.png`) |

### 3.4 Naming Convention Audit for Figures

Proposed consistent convention: `{chapter}_{description}.png` (all lowercase, underscores, PNG preferred).

Examples:
- `scooter-image1.jpeg` --> `ch1_scooter_platform.png`
- `segformer_compare_20251105_154023.png` --> `ch3_segformer_resolution_comparison.png`
- `image.png` --> identify content and rename

---

## 4. `docs/` -- Documentation (368 KB)

```
docs/
  phase11/                        4 files, 36 KB -- Phase 11 development logs
  research/                       5 files, 53 KB -- Research improvement docs
  simplification/                 3 files, 31 KB -- Simplification project docs
  thesis_improvement_plan/        6 files, 204 KB -- THIS audit series
```

**Verdict:** Clean, well-organized by topic. The 2026-03-30 cleanup that moved loose root markdowns here was effective.

**Suggestion:** Add `docs/operational/` for the 4 markdown files currently in `simulation_camera_scooter/` root (CALIBRATION_SOP, RUNTIME_RUNBOOK, MUST_READ_TURN_CONTAINMENT, PHASE_11_1_CHANGELOG).

---

## 5. `models/` -- Root-Level Models (6.3 MB)

| File | Size | Status |
|------|------|--------|
| `yolov8n.pt` | 6.3 MB | DUPLICATE of `simulation_camera_scooter/yolov8n.pt` |
| `bev_calibration.npy` | 160 B | DUPLICATE of `simulation_camera_scooter/bev_calibration.npy` |
| `bev_calibration_root_copy.npy` | 160 B | THIRD copy, explicitly named as copy |

**Verdict:** This directory was created during the 2026-03-30 cleanup to consolidate model files, but the originals in `simulation_camera_scooter/` were never removed. Result: 3 copies of `bev_calibration.npy` and 2 copies of `yolov8n.pt` across the project.

**Recommendation:** Decide on ONE canonical location. Either:
- (a) Keep models in `simulation_camera_scooter/` (where config.py references them) and delete `models/` at root
- (b) Move all models to root `models/`, update config.py paths, delete copies in package

Option (a) is simpler since `config.py` already points to relative paths within the package.

---

## 6. `outputs/` -- Evaluation Outputs (13 GB)

| Directory | Size | Description | Keep? |
|-----------|------|-------------|-------|
| `evaluation/` | 2.3 GB | Full evaluation runs | ARCHIVE |
| `path_planner_eval_new_model_all4_step3/` | 3.3 GB | Path planner eval (4 videos) | ARCHIVE |
| `path_planner_eval_new_model_vid018_020_best2_fullrate/` | 2.0 GB | Path planner eval (2 videos) | ARCHIVE |
| `test_videos/` | 1.6 GB | Copied test videos | LIKELY DUPLICATE of sim/test_videos |
| `replays/` | 2.1 GB | Session replays | ARCHIVE |
| `path_planner_eval_new_model_vid017_best2_fullrate/` | 863 MB | Path planner eval | ARCHIVE |
| `videos/` | 381 MB | Output videos | ARCHIVE |
| `comparisons/` | 136 MB | Side-by-side comparisons | KEEP for thesis reference |
| `runs/` | 118 MB | Individual runs | ARCHIVE |
| `training/` | 72 MB | Training outputs | KEEP (binary seg training) |
| `planner_comparison/` | 39 MB | Planner comparison data | KEEP for thesis |
| `profiling/` | 39 MB | Runtime profiling data | KEEP for thesis |
| `baseline/` | 32 MB | Baseline eval frames | KEEP for thesis |
| `improved/` | 32 MB | Improved eval frames | KEEP for thesis |
| `overnight_eval/` | 20 MB | Overnight eval results | KEEP |
| `overnight_report_20260324.md` | 7 KB | Loose report file | MOVE into overnight_eval/ |
| `smoke_eval_path_planners/` | 13 MB | Smoke test | ARCHIVE |
| `smoke_eval_path_planners_best2_vid017_fullrate/` | 1.9 MB | Smoke test | ARCHIVE |
| `phase11_predict_guard/` | 32 KB | Small eval | KEEP |
| `phase11_probe/` | 108 KB | Small eval | KEEP |

**Recommendation:** Archive or delete the 5 directories over 500 MB each. They total ~11.5 GB and are raw evaluation frame dumps that can be regenerated from test videos + scripts. Keep `comparisons/`, `planner_comparison/`, `profiling/`, `baseline/`, `improved/`, and `training/` -- these are needed for the thesis.

**Naming issue:** Directory names like `path_planner_eval_new_model_all4_step3` and `smoke_eval_path_planners_best2_vid017_fullrate` are long, ad-hoc experiment names without dates. A consistent `YYYY-MM-DD_description/` pattern would make chronological ordering possible.

---

## 7. `research/` -- Research Notes (9.4 MB)

```
research/
  00-current-pipeline-summary.md       4 KB
  01-current-system-failure-analysis.md 6 KB
  02-literature-review-segmentation.md  6 KB
  03-literature-review-bev.md           6 KB
  04-literature-review-pathing.md       5 KB
  05-candidate-selection-and-rationale.md 4 KB
  06-experiment-plan.md                 4 KB
  07-results-comparison.md              11 KB
  08-final-architecture-recommendation.md 5 KB
  99-work-log.md                        12 KB
  artifacts/
    images/   (6 comparison images, 2.3 MB)
    tables/   (9 CSV/JSON result files, 169 KB)
    videos/   (2 comparison videos, 7 MB)
```

**Verdict:** Excellent structure. Numbered files create clear reading order. Artifacts are cleanly separated. This directory is a model for how the rest of the project should be organized.

---

## 8. Tooling Directories

### 8.1 `.claude/` (3.8 MB)

Claude Code configuration. Contains agents, commands, hooks, GSD workflow engine, rules, and skills. Active and necessary.

### 8.2 `.carl/` (22 MB)

CARL dynamic rules MCP server. The `carl-mcp/` subdirectory (22 MB) is a full Node.js project cloned into the repo -- includes its own `.git/`, `node_modules/` equivalent, tests, docs, and README. This is the single largest tooling directory.

**Recommendation:** Check if `.carl/carl-mcp/` can be installed as a dependency rather than vendored. If it must stay, add it to `.gitignore` and install via setup script.

### 8.3 `.paul/` (37 KB)

PAUL workflow state. Minimal footprint, one completed phase (thesis figures). Active.

### 8.4 `.codex/` (2.8 MB)

OpenAI Codex agent configuration. Contains GSD workflow files mirrored from `.claude/`. If Codex is not actively used, this is dead weight.

**Recommendation:** If not using Codex agents, delete `.codex/`. If using, confirm it does not duplicate `.claude/` GSD config.

### 8.5 `.planning/` (696 KB)

GSD planning artifacts for the current thesis rewrite milestone. 5 phases, all with research, plans, summaries, and verification. Active and valuable.

### 8.6 `.tmp_everything_claude_code/` (38 MB)

Temporary tooling cache. Contains what appears to be a cloned framework (agents, commands, hooks, skills, tests). Has its own `.git/` directory.

**Recommendation:** DELETE immediately. This is a temporary directory that serves no runtime purpose. 38 MB of dead weight.

### 8.7 `.pytest_cache/` (35 KB)

Standard pytest cache. Should be gitignored (it is by pattern `__pycache__/` but `.pytest_cache/` is a different pattern).

---

## 9. Duplicate Files Across the Project

| File | Location 1 | Location 2 | Location 3 |
|------|-----------|-----------|-----------|
| `yolov8n.pt` (6.3 MB) | `simulation_camera_scooter/` | `models/` | -- |
| `bev_calibration.npy` (160 B) | `simulation_camera_scooter/` | `models/` | -- |
| `bev_calibration_root_copy.npy` | `models/` (explicit copy) | -- | -- |
| 22 image files | `thesis/figures/` | `thesis/paper_src/images/` | -- |
| `road_false.png` = `road_sidewalk_problem.png` | `thesis/figures/` (same dir, same size) | -- | -- |

**Total duplicate waste:** ~16.6 MB (6.3 MB yolov8n + 4.2 MB paper_src images + 6.0 MB within figures themselves including near-duplicates)

---

## 10. Naming Convention Issues

### 10.1 Inconsistent Casing Across Directories

| Convention | Examples | Count |
|------------|----------|-------|
| kebab-case | `thesis-improvement-plan`, `carl-mcp` | dirs |
| snake_case | `simulation_camera_scooter`, `test_videos` | dirs, files |
| UPPER_SNAKE | `CALIBRATION_SOP.md`, `CLAUDE.md` | docs |
| PascalCase | `ReadME.tex` | 1 file (typo) |

The project mixes snake_case (Python convention) with kebab-case (web convention). For a Python project, snake_case should be the standard.

### 10.2 Unclear or Generic Names

| Name | Issue | Better Name |
|------|-------|-------------|
| `image.png` | Completely generic | Identify content and rename |
| `result/` | In `simulation_camera_scooter/`, vague | `evaluation_videos/` or merge into `outputs/` |
| `ReadME.tex` | Odd casing, misleading extension | DELETE (it is one CLI command) |
| `test.md` | Was deleted (git shows `D test.md`) | Already removed |
| `path_planners/` | Empty except `__pycache__/` | DELETE |

### 10.3 Figures Naming

Current state: no prefix convention, mix of descriptive and cryptic names.

| Current | Issue |
|---------|-------|
| `image.png` | What image? |
| `segformer_compare_20251105_154023.png` | Timestamp in filename is unusual for a thesis figure |
| `scooter-image1.jpeg` | Hyphen + number instead of descriptive |
| `bev_clean.png` vs `cleaned_mask.png` | Same concept, inconsistent naming |

---

## 11. `simulation_camera_scooter/` Nested `.claude/` Directory

The `simulation_camera_scooter/.claude/` directory contains its own set of agents, commands, GSD manifest, hooks, and settings. This was likely created when GSD was initialized inside the package subdirectory.

**Issues:**
- Separate `settings.json` and `settings.local.json` may conflict with root `.claude/settings.json`
- Duplicate agent definitions
- The package directory is not a standalone repo -- it should not have its own `.claude/`

**Recommendation:** Remove `simulation_camera_scooter/.claude/` entirely. All Claude Code config should live at the project root.

---

## 12. `.gitignore` Gaps

Current `.gitignore`:
```
*.mp4
*.MOV
*.avi
*.mov
*.mkv
__pycache__/
*.pyc *.pyo *.pyd
simulation_camera_scooter/models/drivable-segformer-b0/
*.zip
*.tmp
/tmp/
```

**Missing entries that should be added:**

```gitignore
# Pytest cache
.pytest_cache/

# Tooling temp directories
.tmp_everything_claude_code/

# LaTeX build artifacts
thesis/*.aux
thesis/*.bbl
thesis/*.blg
thesis/*.log
thesis/*.out
thesis/*.toc
thesis/*.lof
thesis/*.lot
thesis/*.synctex.gz
thesis/*.fdb_latexmk
thesis/*.fls

# Python tools cache
thesis/tools/__pycache__/

# OS files
.DS_Store
Thumbs.db

# Large evaluation outputs (optional -- prevents accidental commits)
# outputs/evaluation/
# outputs/replays/
```

**Also notable:** `.gitignore` blocks `*.mp4` and `*.MOV` globally, but the project has `.MOV` files tracked in `simulation_camera_scooter/test_videos/` (3.3 GB). These are either tracked from before the gitignore was added, or the gitignore is not effective for already-tracked files.

---

## 13. Disk Space Summary

| Category | Size | % of Total |
|----------|------|-----------|
| `outputs/` (evaluation data) | 13 GB | 62% |
| `simulation_camera_scooter/` (code + data) | 6.0 GB | 29% |
| -- of which: test videos | 3.3 GB | 16% |
| -- of which: models/checkpoints | 756 MB | 4% |
| -- of which: overnight_runs | 832 MB | 4% |
| -- of which: actual source code | ~500 KB | <0.01% |
| Tooling (`.claude`, `.carl`, `.codex`, `.tmp_*`) | 67 MB | 0.3% |
| `thesis/` (LaTeX + figures) | 39 MB | 0.2% |
| `research/` | 9.4 MB | <0.1% |
| `docs/` | 368 KB | <0.01% |

**The actual source code and thesis comprise less than 1% of the project.** The rest is video, model weights, and evaluation frame dumps.

---

## 14. Proposed Clean Structure

### Phase A: Quick wins (no code changes, just moves + deletes)

```
DELETE:
  .tmp_everything_claude_code/           # 38 MB temp cache
  simulation_camera_scooter/path_planners/  # empty dir
  simulation_camera_scooter/ReadME.tex      # vestigial 88 bytes
  simulation_camera_scooter/.claude/        # redundant nested config
  models/bev_calibration_root_copy.npy      # explicit copy, unneeded
  thesis/figures/tables_generated.tex       # move, not delete (see below)

MOVE:
  thesis/figures/tables_generated.tex
    --> thesis/tables/generated/tables_generated.tex

  simulation_camera_scooter/CALIBRATION_SOP.md
    --> docs/operational/CALIBRATION_SOP.md
  simulation_camera_scooter/RUNTIME_RUNBOOK.md
    --> docs/operational/RUNTIME_RUNBOOK.md
  simulation_camera_scooter/MUST_READ_TURN_CONTAINMENT.md
    --> docs/operational/TURN_CONTAINMENT.md
  simulation_camera_scooter/PHASE_11_1_CHANGELOG.md
    --> docs/phase11/PHASE_11_1_CHANGELOG.md

  simulation_camera_scooter/test_video_june_03_3.mp4
    --> simulation_camera_scooter/test_videos/  (if keeping videos)

  outputs/overnight_report_20260324.md
    --> outputs/overnight_eval/overnight_report_20260324.md

DEDUPLICATE:
  models/yolov8n.pt                     # delete, keep sim/ copy
  models/bev_calibration.npy            # delete, keep sim/ copy
  (OR: delete sim/ copies and update config.py to point to models/)
```

### Phase B: Data cleanup (reclaim disk space)

```
ARCHIVE TO EXTERNAL STORAGE (11+ GB):
  outputs/evaluation/                   # 2.3 GB
  outputs/path_planner_eval_new_model_all4_step3/        # 3.3 GB
  outputs/path_planner_eval_new_model_vid018_020_best2/  # 2.0 GB
  outputs/path_planner_eval_new_model_vid017_best2/      # 863 MB
  outputs/replays/                      # 2.1 GB
  outputs/test_videos/                  # 1.6 GB (likely dupe of sim/)

PRUNE MODEL CHECKPOINTS:
  simulation_camera_scooter/models/checkpoint-{500..4500}/  # 135 MB
  simulation_camera_scooter/models/checkpoint-5000/         # 89 KB (corrupted?)
  (Keep only my-segformer-road/ and my-segformer-road_new/)

CONSIDER ARCHIVING:
  simulation_camera_scooter/overnight_runs/   # 832 MB
  simulation_camera_scooter/result/           # 763 MB
  simulation_camera_scooter/eval_runs/        # 109 MB
  simulation_camera_scooter/demo_outputs/     # 44 MB
```

### Phase C: Structural improvements (requires care)

```
PROPOSED DIRECTORY LAYOUT:
live_test_scooter_project/
  CLAUDE.md
  .gitignore
  .mcp.json

  simulation_camera_scooter/        # Python package (CODE ONLY after cleanup)
    config.py
    realtime_nav_core.py
    template_path_planner.py
    waypoint_turn_planner.py
    ... (all .py modules)
    tests/
    scripts/

  data/                             # All non-code data (new top-level dir)
    test_videos/                    # Raw recordings
    annotation_frames/              # Fine-tuning frames
    calibration/                    # bev_calibration.npy, meta.json, etc.
    intent_schedules/

  models/                           # Model weights only
    my-segformer-road/
    my-segformer-road_new/
    yolov8n.pt

  outputs/                          # Evaluation results (pruned)
    baseline/
    improved/
    comparisons/
    planner_comparison/
    profiling/
    training/
    overnight_eval/

  thesis/                           # LaTeX thesis
    main.tex
    references.bib
    figures/                        # Only figures referenced in main.tex
      pipeline/
      resolution/
      scooter/
    tables/
      generated/
    tools/
    reference/                      # Old drafts + OU template
      ou_template_example/
      2025_Lkhaana_Manuscript_draft.pdf
      elsevier_draft.md
    paper_src/                      # Elsevier variant (keep separate)

  docs/                             # All documentation
    phase11/
    research/
    simplification/
    operational/                    # SOPs and runbooks
    thesis_improvement_plan/        # This audit series

  research/                         # Research investigation notes

  .claude/                          # Tooling (hidden)
  .carl/
  .paul/
  .planning/
```

### Phase D: Thesis figures cleanup

```
1. RENAME generic figure:
   image.png --> identify and rename to {topic}_{description}.png

2. CONFIRM duplicates:
   road_false.png vs road_sidewalk_problem.png -- if identical, keep one

3. DECIDE on unreferenced figures:
   - 6 generated-but-unused diagrams (pipeline, GPS, planner arch): keep if
     planning to add to thesis, otherwise move to thesis/figures/unused/
   - 7 legacy/debug figures (test_frame, test_masks, skeleton_0000, etc.):
     move to thesis/figures/unused/
   - 6 quantitative plots (heading, latency, fps, temporal, training):
     these may be used in a future revision -- keep but flag

4. FORMAT STANDARDIZATION:
   - Convert remaining JPEGs to PNG for consistency (except scooter photos
     where JPEG compression is acceptable)
   - Compress oversized figures:
     planner_comparison_example.png (2.9 MB) -- resize or increase compression
     seg_comparison_example.jpg (2.3 MB) -- resize
     scooter-image1.jpeg (2.9 MB) -- resize to thesis column width
```

---

## 15. Priority Actions

### Immediate (< 5 minutes each)

1. Delete `.tmp_everything_claude_code/` -- 38 MB of dead cache
2. Delete `simulation_camera_scooter/path_planners/` -- empty
3. Delete `simulation_camera_scooter/ReadME.tex` -- vestigial
4. Move `thesis/figures/tables_generated.tex` to `thesis/tables/generated/`
5. Update `.gitignore` with missing patterns (`.pytest_cache/`, LaTeX artifacts, `.tmp_*`)

### Short-term (< 30 minutes)

6. Resolve `yolov8n.pt` and `bev_calibration.npy` duplication -- pick one location
7. Move 4 markdown docs from `simulation_camera_scooter/` to `docs/operational/`
8. Delete `models/bev_calibration_root_copy.npy`
9. Rename `thesis/figures/image.png` to something descriptive
10. Move `outputs/overnight_report_20260324.md` into `outputs/overnight_eval/`

### Medium-term (1-2 hours)

11. Archive 11+ GB of evaluation frame dumps from `outputs/`
12. Delete intermediate model checkpoints (checkpoint-500 through checkpoint-4500)
13. Separate data from code in `simulation_camera_scooter/` (move test_videos, annotation_frames, etc.)
14. Remove `simulation_camera_scooter/.claude/` nested config

### Before committee submission

15. Audit all thesis figures for correct referencing
16. Compress oversized figures to thesis-appropriate resolution
17. Clean `thesis/paper_src/` duplicate images
18. Verify `thesis/ou_template_example/` is explicitly marked as reference-only (already done via README)

---

## 16. Summary of Findings

| Category | Finding | Severity |
|----------|---------|----------|
| Disk bloat | 19 GB of video/eval data, only 500 KB of actual source | HIGH |
| Duplicate files | yolov8n.pt (x2), bev_calibration.npy (x3), figures (x22) | MEDIUM |
| Misplaced files | .tex in figures/, docs in code dir, loose report in outputs/ | MEDIUM |
| Dead directories | `.tmp_everything_claude_code/`, empty `path_planners/` | LOW |
| Naming inconsistency | Mixed kebab/snake, generic names, no figure prefix convention | LOW |
| Unreferenced figures | 23 of 39 figures (59%) not used in current thesis | MEDIUM |
| Gitignore gaps | Missing .pytest_cache, LaTeX artifacts, temp dirs | LOW |
| Nested tooling | `.claude/` inside `simulation_camera_scooter/` | LOW |
| Model checkpoint waste | 9 intermediate checkpoints (135 MB) with no runtime use | LOW |

The project's code and thesis are well-organized at the logical level. The main structural problem is that large binary data (videos, model checkpoints, evaluation frames) is mixed throughout the repository without clear boundaries or lifecycle management. Addressing the disk bloat and duplication issues would make the project dramatically more navigable.
