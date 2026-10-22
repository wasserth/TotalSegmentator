"""
Per-class output files hold the labels of the multilabel output, including postprocessing that is
applied to the label map after prediction, for any number of saving threads. test=1 makes
nnUNet_predict_image use tests/reference_files/example_seg.nii.gz as the prediction, so no model
weights are needed. Run from the repository root.
"""
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from totalsegmentator.map_to_binary import class_map
from totalsegmentator.nnunet import nnUNet_predict_image

CT = Path("tests/reference_files/example_ct_sm.nii.gz")
TOTAL_PARTS = [831, 832, 833, 834, 835]


def predict(out, multilabel, **kwargs):
    nnUNet_predict_image(CT, out, TOTAL_PARTS, task_name="total", multilabel_image=multilabel,
                         test=1, quiet=True, device="cpu", **kwargs)


def load(path):
    return np.asanyarray(nib.load(path).dataobj)


def first_half(img):
    mask = np.zeros(img.shape, dtype=np.uint8)
    mask[:, :, :img.shape[2] // 2] = 1
    return nib.Nifti1Image(mask, img.affine)


class SavePerClass(unittest.TestCase):

    def check_against_multilabel(self, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            predict(tmp / "ml.nii.gz", True, **kwargs)
            ml = load(tmp / "ml.nii.gz")
            self.assertGreater(len(np.unique(ml)), 10)
            for nr_threads_saving in (1, 6):
                out = tmp / f"classes_{nr_threads_saving}"
                predict(out, False, nr_threads_saving=nr_threads_saving, **kwargs)
                for k, name in class_map[kwargs.get("output_task_name", "total")].items():
                    np.testing.assert_array_equal(load(out / f"{name}.nii.gz"), (ml == k).astype(np.uint8),
                                                  err_msg=f"{name}, {nr_threads_saving} saving threads")

    def test_matches_multilabel(self):
        self.check_against_multilabel()

    def test_postprocessing_reaches_class_files(self):
        # remove_outside clears labels outside a (dilated) mask after prediction
        self.check_against_multilabel(remove_outside_mask=first_half(nib.load(CT)), remove_outside_dilation=3)

    def test_postprocessing_after_crop_reaches_class_files(self):
        # heartchambers_highres crops the image first. Undoing the crop gives a new array, so remove_outside
        # changes a copy of the prediction. A crop mask over the whole image keeps the shape of example_seg.
        img = nib.load(CT)
        self.check_against_multilabel(crop=nib.Nifti1Image(np.ones(img.shape, dtype=np.uint8), img.affine),
                                      crop_addon=[0, 0, 0], remove_outside_mask=first_half(img),
                                      remove_outside_dilation=3)

    def test_vertebrae_refinement_reaches_class_files(self):
        # vertebrae_pp_refined clips the labels to a vertebral body mask and returns a new array
        self.check_against_multilabel(vertebrae_body_mask=first_half(nib.load(CT)),
                                      output_task_name="vertebrae_pp_refined")

    def test_v1_order_only_changes_the_multilabel_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            predict(tmp / "v2", False)
            for nr_threads_saving in (1, 6):
                out = tmp / f"v1_{nr_threads_saving}"
                predict(out, False, v1_order=True, nr_threads_saving=nr_threads_saving)
                for name in class_map["total"].values():
                    np.testing.assert_array_equal(load(out / f"{name}.nii.gz"), load(tmp / "v2" / f"{name}.nii.gz"),
                                                  err_msg=f"{name}, {nr_threads_saving} saving threads")


if __name__ == "__main__":
    unittest.main()
