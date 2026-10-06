from unittest import mock

import nibabel as nib
import numpy as np

from totalsegmentator.nnunet import nnUNet_predict_image


class ReachedPredictionError(Exception):
    pass


def _predict_with_empty_crop(file_out):
    # An empty crop should return before any resampling or prediction, so
    # reaching change_spacing means the model would have run on the whole
    # uncropped image.
    img = nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.int16), np.eye(4))
    empty_crop = nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.uint8), np.eye(4))
    with mock.patch("totalsegmentator.nnunet.change_spacing", side_effect=ReachedPredictionError):
        return nnUNet_predict_image(img, file_out, 775, resample=[0.75, 0.75, 1.0],
                                    crop=empty_crop, task_name="head_glands_cavities",
                                    quiet=True, device="cpu")


def test_empty_crop_returns_empty_segmentation_with_output_path(tmp_path):
    file_out = tmp_path / "seg.nii.gz"

    seg_img, _, _ = _predict_with_empty_crop(file_out)

    assert not np.asarray(seg_img.get_fdata()).any()
    assert file_out.exists()


def test_empty_crop_returns_empty_segmentation_without_output_path():
    # Python API with output=None: used to fall through and run the model on
    # the whole image.
    seg_img, _, _ = _predict_with_empty_crop(None)

    assert not np.asarray(seg_img.get_fdata()).any()
