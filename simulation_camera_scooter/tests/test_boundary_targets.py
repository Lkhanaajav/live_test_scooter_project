import numpy as np

from boundary_targets import build_boundary_record, extract_boundary_targets


def test_extract_boundary_targets_straight_corridor():
    mask = np.zeros((12, 20), dtype=np.uint8)
    mask[:, 5:15] = 255

    out = extract_boundary_targets(mask, row_step=3, min_width_px=3)

    assert out.valid.all()
    assert np.allclose(out.left_x, 5.0 / 19.0)
    assert np.allclose(out.right_x, 14.0 / 19.0)
    assert np.allclose(out.center_x, 9.5 / 19.0)
    assert np.all(out.width_px == 10.0)


def test_extract_boundary_targets_marks_invalid_rows():
    mask = np.zeros((10, 16), dtype=np.uint8)
    mask[6:, 4:12] = 255

    out = extract_boundary_targets(mask, row_step=2, min_width_px=3)

    assert out.valid[0]
    assert out.valid[1]
    assert not out.valid[-1]
    assert out.left_x[-1] == -1.0
    assert out.right_x[-1] == -1.0
    assert out.center_x[-1] == -1.0


def test_extract_boundary_targets_tracks_curving_centerline():
    mask = np.zeros((16, 24), dtype=np.uint8)
    for y in range(16):
        shift = int((15 - y) * 0.4)
        left = 6 + shift
        right = min(24, left + 8)
        mask[y, left:right] = 255

    out = extract_boundary_targets(mask, row_step=4, min_width_px=3)

    valid_centers = out.center_x[out.valid]
    assert len(valid_centers) >= 3
    assert valid_centers[-1] > valid_centers[0]


def test_build_boundary_record_reports_summary_fields():
    mask = np.zeros((8, 12), dtype=np.uint8)
    mask[:, 3:9] = 255

    rec = build_boundary_record(
        mask,
        name="sample",
        image_path="img.jpg",
        mask_path="mask.png",
        source="pseudo",
        row_step=2,
        min_width_px=2,
    )

    assert rec["name"] == "sample"
    assert rec["source"] == "pseudo"
    assert rec["num_rows"] == 4
    assert rec["num_valid_rows"] == 4
    assert rec["near_field_valid"] is True
    assert rec["mean_width_px"] == 6.0
    assert len(rec["targets"]["rows_y"]) == 4
