from pathlib import Path
from unittest import mock

import nibabel as nib
import numpy as np

from totalsegmentator.nnunet import nnUNet_predict_image


def _volume(shape=(24, 24, 24)):
    """A volume with many grey levels in [0, 1]."""
    return np.random.default_rng(0).random(shape)


def _image_passed_to_nnunet(img, crop=None):
    """Run nnUNet_predict_image up to the prediction and return the voxels nnU-Net would read."""
    seen = {}

    def fake_predict(dir_in, dir_out, *args, **kwargs):
        img_in = nib.load(Path(dir_in) / "s01_0000.nii.gz")
        seen["data"] = np.asarray(img_in.dataobj)
        empty_seg = nib.Nifti1Image(np.zeros(img_in.shape, dtype=np.uint8), img_in.affine)
        nib.save(empty_seg, Path(dir_out) / "s01.nii.gz")

    with mock.patch("totalsegmentator.nnunet.nnUNetv2_predict", side_effect=fake_predict):
        nnUNet_predict_image(img, None, 852, resample=3.0, crop=crop, task_name="total_mr",
                             trainer="nnUNetTrainer_2000epochs_NoMirroring", quiet=True, device="cpu")
    return seen["data"]


def test_float_image_with_small_range_keeps_its_contrast():
    # MR images are often float valued with a small range (normalised to [0, 1],
    # bias corrected, quantitative maps). An int32 cast turned them into zeros.
    img = nib.Nifti1Image(_volume().astype(np.float32), np.diag([1.5, 1.5, 1.5, 1]))

    data = _image_passed_to_nnunet(img)

    assert np.issubdtype(data.dtype, np.floating)
    assert len(np.unique(data)) > 100
    assert 0.4 < data.mean() < 0.6


def test_float_image_keeps_its_contrast_when_cropped():
    # roi_subset and the cropping tasks cast the image a second time while cropping.
    # The image is already at the model spacing here, so only the crop touches it.
    img = nib.Nifti1Image(_volume().astype(np.float32), np.diag([3.0, 3.0, 3.0, 1]))
    crop = nib.Nifti1Image(np.ones(img.shape, dtype=np.uint8), img.affine)

    data = _image_passed_to_nnunet(img, crop=crop)

    assert np.issubdtype(data.dtype, np.floating)
    assert len(np.unique(data)) > 100


def test_scaled_integer_image_keeps_its_contrast(tmp_path):
    # Stored as int16 with a scl_slope: integer on disk, fractional once loaded.
    img = nib.Nifti1Image(_volume(), np.diag([1.5, 1.5, 1.5, 1]))
    img.header.set_data_dtype(np.int16)
    nib.save(img, tmp_path / "scaled.nii.gz")
    img = nib.load(tmp_path / "scaled.nii.gz")
    assert np.issubdtype(img.get_data_dtype(), np.integer)

    data = _image_passed_to_nnunet(img)

    assert len(np.unique(data)) > 100


def test_integer_image_is_still_passed_as_int32():
    volume = np.round(_volume() * 1000).astype(np.int16)
    img = nib.Nifti1Image(volume, np.diag([1.5, 1.5, 1.5, 1]))

    assert _image_passed_to_nnunet(img).dtype == np.int32


def test_integer_valued_float_image_is_still_passed_as_int32():
    # e.g. a CT in Hounsfield units that was saved as float
    volume = np.round(_volume() * 1000 - 500).astype(np.float32)
    img = nib.Nifti1Image(volume, np.diag([1.5, 1.5, 1.5, 1]))

    assert _image_passed_to_nnunet(img).dtype == np.int32
