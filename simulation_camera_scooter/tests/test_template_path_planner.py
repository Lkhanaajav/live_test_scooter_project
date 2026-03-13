import numpy as np
import pandas as pd
from pathlib import Path

from template_path_planner import (
    CorridorConfig,
    TemplatePlannerConfig,
    approve_template_bank,
    corridor_from_mask,
    generate_template_bank,
    score_template_against_corridor,
)
from realtime_nav_core import BEVPathExtractor, PathExtractorConfig
from heading import apply_planner_speed_limit
from scripts.eval_template_planner import _collect_rendered_video, summarize_log


def test_corridor_from_mask_extracts_centered_bounds(straight_bev_mask):
    corridor = corridor_from_mask(straight_bev_mask, CorridorConfig(bev_forward_m=10.0, bev_lateral_m=10.0))
    assert corridor.valid_ratio > 0.90
    assert corridor.near_field_valid_count >= 1
    assert abs(float(np.mean(corridor.center_lateral_m[corridor.valid_mask]))) < 0.20
    assert 2.0 < corridor.mean_width_m < 3.5
    assert corridor.width_cv < 0.10


def test_corridor_from_mask_tracks_curving_centerline(curved_bev_mask):
    corridor = corridor_from_mask(curved_bev_mask, CorridorConfig(bev_forward_m=10.0, bev_lateral_m=10.0))
    centers = corridor.center_lateral_m[corridor.valid_mask]
    assert len(centers) >= 4
    assert abs(float(centers[-1] - centers[0])) > 0.6
    assert corridor.forward_span_m > 5.0


def test_fragmented_near_field_corridor_lowers_confidence(fragmented_near_field_bev_mask, straight_bev_mask):
    cfg = CorridorConfig(bev_forward_m=10.0, bev_lateral_m=10.0)
    clean = corridor_from_mask(straight_bev_mask, cfg)
    fragmented = corridor_from_mask(fragmented_near_field_bev_mask, cfg)
    assert fragmented.confidence < clean.confidence
    assert fragmented.near_field_valid_ratio < clean.near_field_valid_ratio


def test_false_side_pocket_does_not_dominate_center(false_pocket_bev_mask):
    corridor = corridor_from_mask(false_pocket_bev_mask, CorridorConfig(bev_forward_m=10.0, bev_lateral_m=10.0))
    near_mask = corridor.valid_mask & (corridor.rows_forward_m <= 1.0)
    assert np.any(near_mask)
    assert abs(float(np.mean(corridor.center_lateral_m[near_mask]))) < 0.50


def test_generate_template_bank_has_bounded_families():
    cfg = TemplatePlannerConfig(path_horizon_m=4.5, path_sample_ds_m=0.25)
    templates = generate_template_bank(cfg)
    families = {t.family for t in templates}
    assert len(templates) == 7
    assert "straight" in families
    assert "left" in families
    assert "right" in families
    for tmpl in templates:
        assert tmpl.points_m.shape[1] == 2
        assert abs(float(tmpl.points_m[0, 0])) < 1e-6
        assert abs(float(tmpl.points_m[0, 1])) < 1e-6
        assert tmpl.max_curvature_m_inv <= cfg.max_curvature_m_inv + 0.05


def test_straight_corridor_prefers_straight_template(straight_bev_mask):
    planner_cfg = TemplatePlannerConfig(
        bev_forward_m=10.0,
        bev_lateral_m=10.0,
        path_horizon_m=4.5,
        path_sample_ds_m=0.20,
        approval_threshold=0.90,
        corridor_cfg=CorridorConfig(
            bev_forward_m=10.0,
            bev_lateral_m=10.0,
            min_valid_ratio=0.40,
            min_near_field_valid_ratio=0.50,
        ),
    )
    corridor = corridor_from_mask(straight_bev_mask, planner_cfg.corridor_cfg)
    approval = approve_template_bank(corridor, planner_cfg)
    assert approval.approved
    assert approval.selected_template_family == "straight"
    assert approval.confidence > 0.55
    assert approval.is_low_confidence is False


def test_curving_corridor_prefers_turn_template(curved_bev_mask):
    planner_cfg = TemplatePlannerConfig(
        bev_forward_m=10.0,
        bev_lateral_m=10.0,
        path_horizon_m=4.5,
        path_sample_ds_m=0.20,
        corridor_cfg=CorridorConfig(bev_forward_m=10.0, bev_lateral_m=10.0),
    )
    corridor = corridor_from_mask(curved_bev_mask, planner_cfg.corridor_cfg)
    approval = approve_template_bank(corridor, planner_cfg)
    assert approval.approved
    assert approval.selected_template_family in {"left", "right"}
    assert approval.selected_template_family != "straight"


def test_obstacle_penalty_can_push_selection_off_center(straight_bev_mask):
    planner_cfg = TemplatePlannerConfig(
        bev_forward_m=10.0,
        bev_lateral_m=10.0,
        path_horizon_m=4.5,
        path_sample_ds_m=0.20,
        obstacle_penalty_weight=1.00,
        approval_margin=0.0,
        corridor_cfg=CorridorConfig(bev_forward_m=10.0, bev_lateral_m=10.0),
    )
    corridor = corridor_from_mask(straight_bev_mask, planner_cfg.corridor_cfg)
    templates = generate_template_bank(planner_cfg)
    straight = next(t for t in templates if t.family == "straight")
    right = next(t for t in templates if t.template_id == "right_near")
    obstacle_zone = (2.1, 0.3, 0.6)
    score_straight = score_template_against_corridor(
        straight,
        corridor,
        planner_cfg,
        obstacle_zones_m=[obstacle_zone],
    )
    score_right = score_template_against_corridor(
        right,
        corridor,
        planner_cfg,
        obstacle_zones_m=[obstacle_zone],
    )
    approval = approve_template_bank(corridor, planner_cfg, obstacle_zones_m=[obstacle_zone])
    assert score_right.total_score > score_straight.total_score
    assert approval.approved
    assert approval.selected_template_family == "right"


def test_continuity_bias_prefers_previous_family_when_scores_are_similar(wide_straight_bev_mask):
    planner_cfg = TemplatePlannerConfig(
        bev_forward_m=10.0,
        bev_lateral_m=10.0,
        path_horizon_m=4.5,
        path_sample_ds_m=0.20,
        continuity_weight=0.26,
        corridor_cfg=CorridorConfig(bev_forward_m=10.0, bev_lateral_m=10.0, max_width_m=5.5),
    )
    corridor = corridor_from_mask(wide_straight_bev_mask, planner_cfg.corridor_cfg)
    templates = generate_template_bank(planner_cfg)
    left_turn = next(t for t in templates if t.template_id == "left_mid")
    right_turn = next(t for t in templates if t.template_id == "right_mid")
    score_left = score_template_against_corridor(left_turn, corridor, planner_cfg, prev_path_m=left_turn.points_m)
    score_right = score_template_against_corridor(right_turn, corridor, planner_cfg, prev_path_m=left_turn.points_m)
    assert score_left.total_score > score_right.total_score


def test_high_margin_threshold_triggers_low_confidence(wide_straight_bev_mask):
    planner_cfg = TemplatePlannerConfig(
        bev_forward_m=10.0,
        bev_lateral_m=10.0,
        path_horizon_m=4.5,
        path_sample_ds_m=0.20,
        approval_margin=0.25,
        corridor_cfg=CorridorConfig(bev_forward_m=10.0, bev_lateral_m=10.0, max_width_m=5.5),
    )
    corridor = corridor_from_mask(wide_straight_bev_mask, planner_cfg.corridor_cfg)
    approval = approve_template_bank(corridor, planner_cfg)
    assert approval.approved is False
    assert approval.is_low_confidence is True
    assert approval.suggested_slowdown >= 0.35


def test_prev_family_can_reuse_path_when_margin_gate_is_weak(wide_straight_bev_mask):
    planner_cfg = TemplatePlannerConfig(
        bev_forward_m=10.0,
        bev_lateral_m=10.0,
        path_horizon_m=4.5,
        path_sample_ds_m=0.20,
        approval_margin=0.25,
        low_confidence_threshold=0.45,
        corridor_cfg=CorridorConfig(bev_forward_m=10.0, bev_lateral_m=10.0, max_width_m=5.5),
    )
    corridor = corridor_from_mask(wide_straight_bev_mask, planner_cfg.corridor_cfg)
    approval = approve_template_bank(
        corridor,
        planner_cfg,
        prev_template_family="straight",
    )
    assert approval.approved is False
    assert approval.reuse_selected_path is True
    assert approval.selected_template_family == "straight"
    assert approval.recommend_hold is False


def test_straight_preference_wins_startup_tie(wide_straight_bev_mask):
    planner_cfg = TemplatePlannerConfig(
        bev_forward_m=10.0,
        bev_lateral_m=10.0,
        path_horizon_m=4.5,
        path_sample_ds_m=0.20,
        approval_margin=0.25,
        straight_preference_margin=0.08,
        corridor_cfg=CorridorConfig(bev_forward_m=10.0, bev_lateral_m=10.0, max_width_m=5.5),
    )
    corridor = corridor_from_mask(wide_straight_bev_mask, planner_cfg.corridor_cfg)
    approval = approve_template_bank(corridor, planner_cfg)
    assert approval.selected_template_family == "straight"
    assert approval.selected_template_id == "straight_center"


def test_weak_corridor_support_recommends_hold(fragmented_near_field_bev_mask):
    planner_cfg = TemplatePlannerConfig(
        bev_forward_m=10.0,
        bev_lateral_m=10.0,
        path_horizon_m=4.5,
        path_sample_ds_m=0.20,
        approval_threshold=0.90,
        corridor_cfg=CorridorConfig(
            bev_forward_m=10.0,
            bev_lateral_m=10.0,
            min_valid_ratio=0.40,
            min_near_field_valid_ratio=0.50,
        ),
    )
    corridor = corridor_from_mask(fragmented_near_field_bev_mask, planner_cfg.corridor_cfg)
    approval = approve_template_bank(corridor, planner_cfg)
    assert approval.is_low_confidence is True
    assert approval.suggested_slowdown >= 0.35


def test_process_uses_template_path_when_approved(straight_bev_mask):
    cfg = PathExtractorConfig(
        template_planner_enabled=True,
        template_planner_cfg=TemplatePlannerConfig(
            bev_forward_m=10.0,
            bev_lateral_m=10.0,
            path_horizon_m=4.5,
            path_sample_ds_m=0.20,
            corridor_cfg=CorridorConfig(bev_forward_m=10.0, bev_lateral_m=10.0),
        ),
    )
    extractor = BEVPathExtractor(cfg)
    result = extractor.process(straight_bev_mask)
    assert result.has_path is True
    assert result.path_source == "template"
    assert result.path_model is not None
    assert result.selected_template_family == "straight"
    assert result.approval_confidence > 0.55
    assert result.best_idx == 0


def test_process_can_disable_template_planner_for_baseline(straight_bev_mask):
    cfg = PathExtractorConfig(
        min_sidewalk_width_m=0.3,
        branch_min_len_m=0.2,
        min_path_len_m=0.5,
        template_planner_enabled=False,
    )
    extractor = BEVPathExtractor(cfg)
    result = extractor.process(straight_bev_mask)
    assert result.path_source != "template"
    assert result.selected_template_family == ""
    assert result.best_path_m.ndim == 2


def test_process_propagates_low_confidence_template_diagnostics(fragmented_near_field_bev_mask):
    cfg = PathExtractorConfig(
        template_planner_enabled=True,
        template_planner_cfg=TemplatePlannerConfig(
            bev_forward_m=10.0,
            bev_lateral_m=10.0,
            path_horizon_m=4.5,
            path_sample_ds_m=0.20,
            corridor_cfg=CorridorConfig(bev_forward_m=10.0, bev_lateral_m=10.0),
        ),
    )
    extractor = BEVPathExtractor(cfg)
    result = extractor.process(fragmented_near_field_bev_mask)
    assert result.is_low_confidence is True
    assert result.suggested_slowdown >= 0.35
    assert result.selected_template_family != ""
    assert result.corridor_confidence >= 0.0


def test_apply_planner_speed_limit_slows_down_low_confidence_output():
    assert apply_planner_speed_limit(1.0, 0.0, False) == 1.0
    slowed = apply_planner_speed_limit(1.0, 0.8, True)
    assert 0.0 < slowed < 1.0


def test_eval_template_planner_summary_smoke(tmp_path):
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame(
        [
            {
                "heading_smoothed_deg": 1.0,
                "speed_smoothed_mps": 0.8,
                "has_path": 1,
                "planner_low_confidence": 0,
                "planner_slowdown": 0.1,
                "bev_mask_occ_ratio": 0.12,
                "path_source": "template",
                "selected_template_family": "straight",
            },
            {
                "heading_smoothed_deg": -4.0,
                "speed_smoothed_mps": 0.5,
                "has_path": 1,
                "planner_low_confidence": 1,
                "planner_slowdown": 0.6,
                "bev_mask_occ_ratio": 0.03,
                "path_source": "fallback_hold",
                "selected_template_family": "left",
            },
        ]
    ).to_csv(csv_path, index=False)
    summary = summarize_log(csv_path)
    assert summary["frames"] == 2
    assert summary["template_rate"] == 0.5
    assert summary["fallback_rate"] == 0.5
    assert summary["path_source_switches"] == 1


def test_eval_template_planner_collects_rendered_video_from_cwd(tmp_path, monkeypatch):
    workdir = tmp_path / "cwd"
    project_dir = tmp_path / "project"
    workdir.mkdir()
    project_dir.mkdir()
    src = workdir / "heading_demo_output.mp4"
    src.write_bytes(b"fake-video")
    dst = project_dir / "out.mp4"

    monkeypatch.chdir(workdir)
    moved = _collect_rendered_video((Path.cwd(), project_dir), dst)

    assert moved == dst
    assert dst.read_bytes() == b"fake-video"
    assert not src.exists()
