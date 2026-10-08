import nibabel as nib
import numpy as np
import pytest

from totalsegmentator.nifti_ext_header import add_label_map_to_nifti, load_multilabel_nifti


@pytest.mark.parametrize(("labels", "expected"), [
    pytest.param((0, 1, 2), {1: "L1.0", 2: "L2.0"}, id="contiguous"),
    pytest.param((0, 5, 9), {5: "L5.0", 9: "L9.0"}, id="sparse"),
    pytest.param((5, 9), {5: "L5.0", 9: "L9.0"}, id="no_background"),
])
def test_inferred_label_map_preserves_voxel_ids(tmp_path, labels, expected):
    data = np.array(labels, dtype=np.uint8).reshape(1, 1, -1)
    image = nib.Nifti1Image(data, np.eye(4))
    output = tmp_path / "segmentation.nii.gz"

    nib.save(add_label_map_to_nifti(image, None), output)
    restored, label_map = load_multilabel_nifti(output)

    assert label_map == expected
    np.testing.assert_array_equal(np.asanyarray(restored.dataobj), data)


@pytest.mark.parametrize(("labels", "label_map", "expected"), [
    pytest.param((0, 5, 9), {5: "first", 9: "second"},
                 {5: "first", 9: "second"}, id="dictionary"),
    pytest.param((0, 1, 2), ["first", "second"],
                 {1: "first", 2: "second"}, id="list"),
])
def test_explicit_label_map(labels, label_map, expected):
    data = np.array(labels, dtype=np.uint8).reshape(1, 1, -1)
    image = nib.Nifti1Image(data, np.eye(4))

    restored, actual = load_multilabel_nifti(add_label_map_to_nifti(image, label_map))

    assert actual == expected
    np.testing.assert_array_equal(np.asanyarray(restored.dataobj), data)
