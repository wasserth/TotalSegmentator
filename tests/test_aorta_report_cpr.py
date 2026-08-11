from types import SimpleNamespace

import nibabel as nib
import numpy as np

from totalsegmentator.aorta_report.cpr import (
    _build_sampling_grid,
    _centerline_tangents,
    _chunk_ranges,
    _parallel_transport_frames,
    _resample_centerline,
    cpr,
)
from totalsegmentator.aorta_report.cpr_animated import (
    _content_crop_bounds,
    _direction_to_render,
    _panel_with_position_line,
    _voxel_to_render,
)
from totalsegmentator.aorta_report.nifti import (
    _allocate_rgb_nifti_buffer,
    _save_rgb_nifti_buffer,
    _write_rgb_frame,
)


def test_resample_centerline_uses_physical_distance():
    affine = np.diag([2.0, 1.0, 1.0, 1.0])
    points = np.array([[0, 0, 0], [2, 0, 0]], dtype=float)

    sampled, positions, total_length = _resample_centerline(points, affine, 1.0)

    assert total_length == 4.0
    np.testing.assert_allclose(positions, np.arange(5))
    np.testing.assert_allclose(sampled[:, 0], np.arange(5) / 2)


def test_parallel_transport_frames_are_orthonormal_and_continuous():
    angles = np.linspace(0, np.pi / 2, 20)
    centerline = np.column_stack(
        [10 * np.cos(angles), 10 * np.sin(angles), angles]
    )
    tangents = _centerline_tangents(centerline)

    normals, binormals = _parallel_transport_frames(tangents)

    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(binormals, axis=1), 1.0, atol=1e-7)
    np.testing.assert_allclose(np.sum(tangents * normals, axis=1), 0.0, atol=1e-7)
    assert np.all(np.sum(normals[1:] * normals[:-1], axis=1) >= 0)


def test_sampling_grid_and_chunks_bound_working_set():
    grid_u, grid_v, spacing = _build_sampling_grid(30.0, 0.4)
    ranges, chunk_size = _chunk_ranges(100, grid_u.shape[0], target_points=10_000)

    assert grid_u.shape == grid_v.shape
    assert grid_u.shape[0] % 2 == 1
    assert spacing == 0.6
    assert chunk_size * grid_u.size <= 10_000
    assert list(ranges)[0] == 0


def test_cpr_preserves_entrypoint_shapes_metadata_and_extra_masks():
    shape = (17, 17, 17)
    affine = np.eye(4)
    ct_data = np.arange(np.prod(shape), dtype=np.int16).reshape(shape)
    mask_data = np.zeros(shape, dtype=np.uint8)
    mask_data[7:10, 7:10, 2:15] = 1
    ct_img = nib.Nifti1Image(ct_data, affine)
    mask_img = nib.Nifti1Image(mask_data, affine)
    centerline = [
        SimpleNamespace(point=np.array([8.0, 8.0, z]))
        for z in range(2, 15)
    ]

    ct_cpr, mask_cpr, extra_cpr, info = cpr(
        ct_img,
        mask_img,
        centerline,
        max_dia=10,
        extra_mask_imgs=[mask_img],
        return_info=True,
    )

    assert ct_cpr.shape == mask_cpr.shape == extra_cpr[0].shape
    assert ct_cpr.shape[2] == info["nr_slices"]
    assert ct_cpr.get_data_dtype() == np.dtype(np.int16)
    np.testing.assert_allclose(ct_cpr.affine, np.diag([1.0, 1.0, 1.0, 1.0]))
    np.testing.assert_array_equal(mask_cpr.get_fdata(), extra_cpr[0].get_fdata())


def test_animation_geometry_and_crop_helpers():
    point = np.array([2.0, 3.0, 4.0])
    direction = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(
        _voxel_to_render(point, (10, 20, 30)), [16.0, 4.0, 7.0]
    )
    np.testing.assert_array_equal(
        _direction_to_render(direction), [-2.0, 3.0, -1.0]
    )

    image = np.zeros((10, 12, 3), dtype=np.uint8)
    image[3:6, 4:8] = 10
    assert _content_crop_bounds(image, padding=1) == (2, 7, 3, 9)

    panel = np.zeros((8, 5, 3), dtype=np.uint8)
    lined = _panel_with_position_line(panel, 4)
    assert np.all(lined[3:6] == [255, 140, 0])
    assert not panel.any()


def test_rgb_frames_write_directly_to_compatible_nifti(tmp_path):
    buffer = _allocate_rgb_nifti_buffer(width=3, height=2, nr_frames=2)
    first = np.full((2, 3, 3), 10, dtype=np.uint8)
    second = np.full((2, 3, 3), 20, dtype=np.uint8)
    _write_rgb_frame(buffer, first, 0)
    _write_rgb_frame(buffer, second, 1)

    output_path = tmp_path / "animated.nii.gz"
    _save_rgb_nifti_buffer(buffer, output_path)
    output = nib.load(output_path)

    assert output.shape == (4, 3, 2)
    np.testing.assert_array_equal(np.diag(output.affine), [-1, -1, 1, 1])
