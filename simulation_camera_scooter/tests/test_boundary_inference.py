import numpy as np

from boundary_inference import BoundaryPathConfig, decode_boundary_prediction


def test_decode_boundary_prediction_returns_centerline_path():
    rows_y = np.array([1.0, 0.75, 0.5, 0.25, 0.0], dtype=np.float32)
    pred = {
        "left_x": np.full(5, 0.35, dtype=np.float32),
        "right_x": np.full(5, 0.65, dtype=np.float32),
        "valid_prob": np.full(5, 0.95, dtype=np.float32),
    }

    result = decode_boundary_prediction(
        pred,
        rows_y=rows_y,
        image_size=(160, 96),
        cfg=BoundaryPathConfig(bev_forward_m=8.0, bev_lateral_m=4.0, path_sample_ds_m=0.5),
    )

    assert result.has_path
    assert not result.is_low_confidence
    assert result.valid_row_count == 5
    assert result.forward_span_m >= 7.5
    assert result.confidence >= 0.9
    assert result.suggested_slowdown <= 0.1
    assert np.max(np.abs(result.path_m[:, 1])) < 1e-4
    assert abs(result.mean_width_m - 1.2) < 1e-3


def test_decode_boundary_prediction_flags_missing_near_field():
    rows_y = np.array([1.0, 0.75, 0.5, 0.25, 0.0], dtype=np.float32)
    pred = {
        "left_x": np.full(5, 0.30, dtype=np.float32),
        "right_x": np.full(5, 0.60, dtype=np.float32),
        "valid_prob": np.array([0.10, 0.20, 0.95, 0.95, 0.95], dtype=np.float32),
    }

    result = decode_boundary_prediction(
        pred,
        rows_y=rows_y,
        image_size=(160, 96),
        cfg=BoundaryPathConfig(bev_forward_m=8.0, bev_lateral_m=4.0, min_forward_span_m=3.0),
    )

    assert result.has_path
    assert result.is_low_confidence
    assert result.near_field_valid_count == 1
    assert result.confidence > 0.0
    assert result.suggested_slowdown >= 0.35


def test_decode_boundary_prediction_blends_previous_path_for_stability():
    rows_y = np.array([1.0, 0.75, 0.5, 0.25, 0.0], dtype=np.float32)
    pred = {
        "left_x": np.full(5, 0.55, dtype=np.float32),
        "right_x": np.full(5, 0.85, dtype=np.float32),
        "valid_prob": np.full(5, 0.95, dtype=np.float32),
    }
    prev_path_m = np.stack(
        [
            np.linspace(0.0, 8.0, 9, dtype=np.float32),
            np.zeros(9, dtype=np.float32),
        ],
        axis=1,
    )

    no_blend = decode_boundary_prediction(
        pred,
        rows_y=rows_y,
        image_size=(160, 96),
        cfg=BoundaryPathConfig(bev_forward_m=8.0, bev_lateral_m=4.0, previous_path_blend=0.0, path_sample_ds_m=0.5),
    )
    blended = decode_boundary_prediction(
        pred,
        rows_y=rows_y,
        image_size=(160, 96),
        cfg=BoundaryPathConfig(bev_forward_m=8.0, bev_lateral_m=4.0, previous_path_blend=0.5, path_sample_ds_m=0.5),
        prev_path_m=prev_path_m,
    )

    assert blended.has_path
    assert np.mean(np.abs(blended.path_m[:, 1])) < np.mean(np.abs(no_blend.path_m[:, 1]))
