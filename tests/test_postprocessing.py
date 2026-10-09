import nibabel as nib
import numpy as np

from totalsegmentator.map_to_binary import class_map
from totalsegmentator.postprocessing import postprocess_vertebrae_pp
from totalsegmentator.postprocessing import refine_vertebrae_pp_with_body_mask
from totalsegmentator.postprocessing import remove_auxiliary_labels


def _add_block(data, z_start, z_stop, label):
    data[:, :, z_start:z_stop] = label


def test_vertebrae_pp_postprocessing_returns_unchanged_without_touching_labels():
    data = np.zeros((6, 6, 16), dtype=np.uint8)
    _add_block(data, 0, 4, 24)
    _add_block(data, 8, 12, 23)

    cleaned = postprocess_vertebrae_pp(data, class_map["vertebrae_pp"], dilation_mm=0)

    np.testing.assert_array_equal(cleaned, data)


def test_vertebrae_pp_postprocessing_dilates_separated_vertebrae():
    data = np.zeros((7, 7, 12), dtype=np.uint8)
    data[2:5, 2:5, 2:5] = 24
    data[2:5, 2:5, 8:10] = 23

    cleaned = postprocess_vertebrae_pp(data, class_map["vertebrae_pp"],
                                       voxel_spacing=(1, 1, 1), dilation_mm=1)

    assert cleaned[1, 3, 3] == 24
    assert cleaned[3, 3, 5] == 24
    assert cleaned[3, 3, 7] == 23
    assert np.all(cleaned[data == 24] == 24)
    assert np.all(cleaned[data == 23] == 23)


def test_vertebrae_pp_postprocessing_counts_from_bottom_and_removes_noise():
    data = np.zeros((6, 6, 24), dtype=np.uint8)
    _add_block(data, 0, 4, 24)
    data[:3, :, 0:4] = 23  # touching mixed label within the lowest body
    _add_block(data, 8, 12, 23)
    _add_block(data, 16, 20, 22)
    data[0, 0, 23] = 1  # small noisy component, removed by the 100 voxel threshold

    cleaned = postprocess_vertebrae_pp(data, class_map["vertebrae_pp"], dilation_mm=0)

    assert np.all(cleaned[:, :, 0:4] == 24)
    assert np.all(cleaned[:, :, 8:12] == 23)
    assert np.all(cleaned[:, :, 16:20] == 22)
    assert cleaned[0, 0, 23] == 0


def test_vertebrae_pp_postprocessing_min_size_is_mm3_not_voxels():
    data = np.zeros((6, 6, 8), dtype=np.uint8)
    _add_block(data, 0, 2, 24)  # 72 voxels, but 144 mm3 with voxel_volume=2
    data[:3, :, 0:2] = 23
    _add_block(data, 5, 7, 23)

    cleaned = postprocess_vertebrae_pp(data, class_map["vertebrae_pp"], voxel_volume=2,
                                       dilation_mm=0)

    assert np.all(cleaned[:, :, 0:2] == 24)
    assert np.all(cleaned[:, :, 5:7] == 23)


def test_vertebrae_pp_postprocessing_counts_from_top_when_c1_present_without_l5():
    data = np.zeros((6, 6, 24), dtype=np.uint8)
    _add_block(data, 20, 24, 1)
    data[:3, :, 20:24] = 2  # touching mixed label within the highest body
    _add_block(data, 12, 16, 2)
    _add_block(data, 4, 8, 3)

    cleaned = postprocess_vertebrae_pp(data, class_map["vertebrae_pp"], dilation_mm=0)

    assert np.all(cleaned[:, :, 20:24] == 1)
    assert np.all(cleaned[:, :, 12:16] == 2)
    assert np.all(cleaned[:, :, 4:8] == 3)


def test_refine_vertebrae_pp_with_body_mask_dilates_and_intersects_body():
    data = np.zeros((7, 7, 7), dtype=np.uint8)
    body = np.zeros_like(data)
    data[3, 3, 3] = 24
    body[2:5, 3, 3] = 1
    body[3, 2:5, 3] = 1
    body[3, 3, 2:5] = 1

    refined = refine_vertebrae_pp_with_body_mask(data, body, class_map["vertebrae_pp"],
                                                 voxel_spacing=(1, 1, 1), dilation_mm=2)

    assert refined[2, 3, 3] == 24
    assert refined[3, 4, 3] == 24
    assert refined[3, 3, 1] == 0  # dilated, but outside vertebrae_body


def test_remove_auxiliary_labels_strips_appendicular_bones_mr_aux_labels():
    data = np.zeros((4, 4, 4), dtype=np.uint8)
    data[0, 0, 0] = 7  # ulna — canonical, kept
    data[0, 0, 1] = 8  # radius — canonical, kept
    data[0, 0, 2] = 9  # humerus aux — removed
    data[0, 1, 0] = 10  # femur aux — removed
    data[0, 1, 1] = 11  # liver aux — removed
    data[0, 1, 2] = 12  # spleen aux — removed
    img = nib.Nifti1Image(data, np.eye(4))

    cleaned = remove_auxiliary_labels(img, "appendicular_bones_mr").get_fdata()

    assert cleaned[0, 0, 0] == 7
    assert cleaned[0, 0, 1] == 8
    assert cleaned[0, 0, 2] == 0
    assert cleaned[0, 1, 0] == 0
    assert cleaned[0, 1, 1] == 0
    assert cleaned[0, 1, 2] == 0


def test_remove_auxiliary_labels_keeps_appendicular_bones_ct_behaviour():
    data = np.zeros((4, 4, 4), dtype=np.uint8)
    data[0, 0, 0] = 5  # metatarsal — canonical, kept
    data[0, 0, 1] = 12  # humerus aux — removed
    data[0, 0, 2] = 15  # spleen aux — removed
    img = nib.Nifti1Image(data, np.eye(4))

    cleaned = remove_auxiliary_labels(img, "appendicular_bones").get_fdata()

    assert cleaned[0, 0, 0] == 5
    assert cleaned[0, 0, 1] == 0
    assert cleaned[0, 0, 2] == 0


def test_remove_auxiliary_labels_noop_for_tasks_without_aux_map():
    data = np.zeros((4, 4, 4), dtype=np.uint8)
    data[0, 0, 0] = 90
    img = nib.Nifti1Image(data, np.eye(4))

    cleaned = remove_auxiliary_labels(img, "total_mr").get_fdata()

    np.testing.assert_array_equal(cleaned, data)
