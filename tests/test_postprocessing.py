import numpy as np

from totalsegmentator.map_to_binary import class_map
from totalsegmentator.postprocessing import postprocess_vertebrae_pp


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
