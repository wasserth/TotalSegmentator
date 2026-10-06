import nibabel as nib
import numpy as np
import pytest

from totalsegmentator.aorta_report.centerline import (
    Vertex,
    get_centerline,
    get_cumulative_arc_lengths,
    get_index_for_point,
    get_mid_of_points,
    get_next_point_by_distance,
    get_normal_for_cl_point,
    resample_cl_to_same_distance,
)
from totalsegmentator.aorta_report import centerline as centerline_module
from totalsegmentator.aorta_report.geometry import (
    draw_plane,
    get_plane_diameters,
)
from totalsegmentator.aorta_report.geometry_report import (
    get_2d_plane,
    get_aorta_section,
    get_max_diameter_of_centerline_section,
    keep_closest_blob,
)


def _centerline(points):
    return [Vertex(point) for point in points]


def test_centerline_arc_length_lookup_and_endpoints():
    centerline = _centerline([(0, 0, 0), (1, 0, 0), (3, 0, 0)])

    np.testing.assert_allclose(get_cumulative_arc_lengths(centerline), [0, 1, 3])
    np.testing.assert_allclose(
        get_cumulative_arc_lengths(centerline, (2, 1, 1)), [0, 2, 6]
    )
    assert get_index_for_point(centerline, (9, 9, 9)) is None
    assert get_index_for_point([], None) is None
    assert get_next_point_by_distance(centerline, 0, 3, (1, 1, 1)) is None
    assert get_next_point_by_distance(centerline, 2, -3, (1, 1, 1)) is None
    assert get_next_point_by_distance(centerline, 0, -1, (1, 1, 1)) is None
    assert get_mid_of_points(centerline, 0, 2) == 1
    assert get_mid_of_points(centerline, 1, 1) == 1
    assert get_mid_of_points(centerline, None, 1) is None


def test_even_thickness_mask_still_gets_a_centerline():
    mask = np.zeros((24, 24, 40), dtype=np.uint8)
    mask[8:16, 8:16, 4:36] = 1

    _, path = get_centerline(mask)

    assert len(path) > 10
    assert abs(path[0].point[2] - path[-1].point[2]) > 10


def test_centerline_uses_mask_axis_when_skeleton_is_empty(monkeypatch):
    monkeypatch.setattr(
        centerline_module,
        "skeletonize",
        lambda image: np.zeros(np.shape(image), dtype=bool),
    )
    mask = np.zeros((10, 10, 20), dtype=np.uint8)
    mask[4, 4, 2:18] = 1

    image, path = get_centerline(mask)

    assert len(path) >= 2
    assert image.sum() > 0
    assert tuple(np.rint(path[0].point)) in ((4, 4, 2), (4, 4, 17))


def test_empty_mask_cannot_produce_a_centerline():
    with pytest.raises(ValueError, match="Unable to create a centerline"):
        get_centerline(np.zeros((6, 6, 6), dtype=np.uint8))


def test_centerline_resampling_preserves_endpoints_and_handles_short_inputs():
    centerline = _centerline([(0, 0, 0), (0, 0, 10)])
    sampled = resample_cl_to_same_distance(centerline, np.eye(4), dist=3)

    np.testing.assert_allclose(sampled[0].point, centerline[0].point)
    np.testing.assert_allclose(sampled[-1].point, centerline[-1].point)
    assert len(resample_cl_to_same_distance(centerline[:1], np.eye(4))) == 1
    assert len(resample_cl_to_same_distance([], np.eye(4))) == 0


def test_draw_plane_and_diameters_handle_degenerate_masks():
    assert not draw_plane((5, 5, 5), (0, 0, 0), (2, 2, 2)).any()
    assert get_plane_diameters(np.zeros((5, 5), dtype=np.uint8)) == (
        0.0,
        0.0,
        None,
        None,
    )

    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 1:7] = 1
    diameter, perpendicular, major, perpendicular_points = get_plane_diameters(mask)
    assert diameter > 0
    assert perpendicular > 0
    assert major is not None
    assert perpendicular_points is not None


def test_keep_closest_blob_is_exact_and_deterministic():
    data = np.zeros((5, 3, 3), dtype=np.uint8)
    data[1, 1, 1] = 1
    data[3, 1, 1] = 1

    expected = np.zeros_like(data)
    expected[1, 1, 1] = 1
    for _ in range(3):
        np.testing.assert_array_equal(keep_closest_blob(data, (2, 1, 1)), expected)


def test_max_diameter_subsample_uses_original_centerline_indexes(monkeypatch):
    import totalsegmentator.aorta_report.geometry_report as geometry_report

    centerline = _centerline(
        [(2, 2, 1), (2, 2, 2), (3, 2, 3), (5, 2, 4), (5, 3, 5)]
    )
    image = nib.Nifti1Image(np.ones((8, 8, 8), dtype=np.uint8), np.eye(4))
    normals = []

    def intersection(_aorta, _point, normal):
        normals.append(normal)
        return np.zeros((8, 8, 8), dtype=np.uint8)

    monkeypatch.setattr(geometry_report, "get_plane_aorta_intersection", intersection)
    point = get_max_diameter_of_centerline_section(
        image, centerline, (1, 1, 1), subsample=2
    )

    assert point == tuple(centerline[0].point)
    expected = [get_normal_for_cl_point(centerline, idx) for idx in (0, 2, 4)]
    np.testing.assert_allclose(normals, expected)


def test_max_diameter_and_section_handle_short_or_offset_centerlines():
    image = nib.Nifti1Image(np.zeros((13, 13, 13), dtype=np.uint8), np.eye(4))
    assert get_max_diameter_of_centerline_section(image, [], (1, 1, 1)) is None
    assert get_max_diameter_of_centerline_section(
        image, [], (1, 1, 1), return_diameter=True
    ) == (None, np.float32(0))

    aorta = np.zeros((13, 13, 13), dtype=np.uint8)
    xx, yy = np.ogrid[:13, :13]
    aorta[(xx - 6) ** 2 + (yy - 6) ** 2 <= 4, 1:12] = 1
    image = nib.Nifti1Image(aorta, np.eye(4))
    # The centerline is deliberately outside the mask in x. Selection must
    # fall back to the nearest remaining component instead of label zero.
    centerline = _centerline([(9, 6, z) for z in range(1, 12)])
    section = get_aorta_section(image, 2, 8, centerline, crop_size=12).get_fdata()

    assert section.any()
    assert not section[:, :, :3].any()
    assert not section[:, :, 9:].any()


def test_get_2d_plane_accepts_an_empty_plane():
    affine = np.eye(4)
    empty = nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.uint8), affine)
    ct = nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.float32), affine)

    plane, rotated_ct, _, _, inverse = get_2d_plane(
        empty, ct, None, None, addon=(6, 6, 6)
    )

    assert plane.shape == rotated_ct.shape
    assert not plane.get_fdata().any()
    assert inverse.shape == (4, 4)
