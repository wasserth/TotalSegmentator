from unittest import mock

import nibabel as nib
import numpy as np
import pytest

from totalsegmentator import python_api
from totalsegmentator.map_to_binary import class_map
from totalsegmentator.python_api import totalsegmentator, validate_roi_subset


class ReachedDownloadError(Exception):
    pass


@pytest.fixture
def run(tmp_path, monkeypatch):
    """Run totalsegmentator() without weights: record downloads and predictions instead of doing them."""
    monkeypatch.setenv("TOTALSEG_HOME_DIR", str(tmp_path))
    img = nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.int16), np.eye(4))
    download = mock.Mock(side_effect=ReachedDownloadError)
    predict = mock.Mock()

    def _run(**kwargs):
        with mock.patch.object(python_api, "download_pretrained_weights", download), \
             mock.patch("totalsegmentator.nnunet.nnUNet_predict_image", predict):
            return totalsegmentator(img, None, quiet=True, device="cpu", **kwargs)

    _run.download = download
    _run.predict = predict
    return _run


@pytest.mark.parametrize("task, roi_subset, unknown, suggestion", [
    ("total_mr", ["liver", "kidny_left"], "kidny_left", "kidney_left"),            # typo
    ("total_mr", ["lung_upper_lobe_left"], "lung_upper_lobe_left", "lung_left"),   # class of the CT task
    ("total", ["liver", "lung_upper_left"], "lung_upper_left", "lung_upper_lobe_left"),  # name from v1
])
def test_unknown_class_name_fails_before_download_and_prediction(run, task, roi_subset, unknown, suggestion):
    with pytest.raises(ValueError) as error:
        run(task=task, roi_subset=roi_subset)

    message = str(error.value)
    assert f"'{unknown}'" in message
    assert f"'{suggestion}'" in message
    assert "'liver'" not in message
    assert f"totalseg_info --classes -ta {task}" in message
    run.download.assert_not_called()
    run.predict.assert_not_called()


def test_roi_subset_robust_is_checked_too(run):
    with pytest.raises(ValueError, match="'kidny_left'"):
        run(task="total", roi_subset_robust=["kidny_left"])

    run.download.assert_not_called()


@pytest.mark.parametrize("task, roi_subset", [
    ("total", ["liver", "brain"]),
    ("total_mr", ["liver", "kidney_left", "vertebrae"]),
])
def test_valid_class_names_are_accepted(run, task, roi_subset):
    # Reaching the weights download means the names passed the check.
    with pytest.raises(ReachedDownloadError):
        run(task=task, roi_subset=roi_subset)


@pytest.mark.parametrize("task", ["total", "total_v3", "total_mr"])
def test_every_class_of_a_task_is_accepted(task):
    validate_roi_subset(list(class_map[task].values()), task)


def test_no_roi_subset_is_accepted():
    validate_roi_subset(None, "total")
