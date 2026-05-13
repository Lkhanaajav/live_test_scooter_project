# Overnight Changelog

## Code Files Changed

### realtime_nav_core.py
- `PathExtractorConfig`
  - Added fallback-related parameters for sparse-mask recovery and stability:
    - `fallback_enabled`
    - `fallback_hold_frames`
    - `fallback_row_step_px`
    - `fallback_min_row_pixels`
    - `fallback_min_width_m`
    - `fallback_min_forward_span_m`
    - `fallback_target_span_m`
    - `fallback_prev_blend`
    - `fallback_center_pull`
    - `fallback_low_conf_*`
    - `fallback_output_lateral_clip_m`
- `BEVPathExtractor.__init__`
  - Added temporal path memory state:
    - `prev_best_path_m`
    - `no_path_counter`
- `BEVPathExtractor._preprocess`
  - Added sparse-occupancy guard to avoid over-pruning thin masks.
- New helper methods
  - `_path_lateral_at_x`
  - `_extend_path_forward`
  - `_fallback_centerline`
  - `_hold_previous_path`
- `BEVPathExtractor.process`
  - Added hierarchical recovery logic:
    - graph candidate path first
    - fallback centerline when graph path/model fails
    - hold previous valid path for short dropout windows
  - Ensured selected path is always returned in candidate list for visualization.
  - Updated temporal state on success/failure for continuity.

## Behavior Changes
- Before: when BEV mask was weak/fragmented, extractor often returned no path at all (`has_path=0`).
- After: extractor degrades gracefully to fallback/hold path and maintains continuous output in low-quality frames.
- Added lateral guardrail on fallback outputs to reduce one-sided steering lock.

## Validation
- Unit + integration tests:
  - `python -m pytest tests -q` -> 58 passed
- Manual visual validation:
  - Baseline: `baseline/`
  - Experiment iterations: `exp1_fallback/`, `exp2_recenter/`, `exp3_ego_anchor/`, `exp4_lateral_clip/`
  - Selected best: `exp4_lateral_clip/`
