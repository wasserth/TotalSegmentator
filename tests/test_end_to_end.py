import os
import unittest
import pytest
import json
import subprocess

import nibabel as nib
import numpy as np
import pandas as pd


def dice_score(y_true, y_pred):
    if y_true.sum() == 0 and y_pred.sum() == 0:
        return 1.0
    intersect = np.sum(y_true * y_pred)
    denominator = np.sum(y_true) + np.sum(y_pred)
    f1 = (2 * intersect) / (denominator + 1e-6)
    return f1


def dice_score_multilabel(y_true, y_pred):
    """
    Calc dice for each class and then return the mean.
    """
    dice_scores = []
    for i in np.unique(y_true)[1:]:
        gt = y_true == i
        pred = y_pred == i
        dice_scores.append(dice_score(gt, pred))
    print(f"Dice scores per class: {dice_scores}")  # only gets printed if the test fails
    return np.mean(dice_scores)


class test_end_to_end(unittest.TestCase):

    def setUp(self):
        pass

    def test_prediction_multilabel(self):
        img_ref = nib.load("tests/reference_files/example_seg.nii.gz").get_fdata()
        img_new = nib.load("tests/unittest_prediction.nii.gz").get_fdata()
        # nr_of_diff_voxels = (img_ref != img_new).sum()
        # images_equal = nr_of_diff_voxels < 100
        dice = dice_score_multilabel(img_ref, img_new)
        images_equal = dice > 0.99
        self.assertTrue(images_equal, f"multilabel prediction not correct (dice: {dice:.6f})")

    def test_prediction_liver_roi_subset(self):
        img_ref = nib.load("tests/reference_files/example_seg_roi_subset.nii.gz").get_fdata()
        img_new = nib.load("tests/unittest_prediction_roi_subset.nii.gz").get_fdata()
        dice = dice_score_multilabel(img_ref, img_new)
        images_equal = dice > 0.99
        self.assertTrue(images_equal, f"roi subset prediction not correct (dice: {dice:.6f})")

    def test_preview(self):
        preview_exists = os.path.exists("tests/unittest_prediction_fast/preview_total.png")
        self.assertTrue(preview_exists, "Preview was not generated")

    def test_prediction_fast(self):
        for roi in ["liver", "vertebrae_L1"]:
            img_ref = nib.load(f"tests/reference_files/example_seg_fast/{roi}.nii.gz").get_fdata()
            img_new = nib.load(f"tests/unittest_prediction_fast/{roi}.nii.gz").get_fdata()
            dice = dice_score_multilabel(img_ref, img_new)
            images_equal = dice > 0.99
            self.assertTrue(images_equal, f"{roi} fast prediction not correct (dice: {dice:.6f})")

    def test_prediction_multilabel_fast(self):
        img_ref = nib.load("tests/reference_files/example_seg_fast.nii.gz").get_fdata()
        img_new = nib.load("tests/unittest_prediction_fast.nii.gz").get_fdata()
        dice = dice_score_multilabel(img_ref, img_new)
        images_equal = dice > 0.99
        self.assertTrue(images_equal, f"multilabel prediction fast not correct (dice: {dice:.6f})")

    def test_prediction_multilabel_fast_force_split(self):
        img_ref = nib.load("tests/reference_files/example_seg_fast_force_split.nii.gz").get_fdata()
        img_new = nib.load("tests/unittest_prediction_fast_force_split.nii.gz").get_fdata()
        dice = dice_score_multilabel(img_ref, img_new)
        images_equal = dice > 0.99
        self.assertTrue(images_equal, f"force_split prediction not correct (nr_of_diff_voxels: {dice:.6f})")

    def test_prediction_multilabel_fast_body_seg(self):
        img_ref = nib.load("tests/reference_files/example_seg_fast_body_seg.nii.gz").get_fdata()
        img_new = nib.load("tests/unittest_prediction_fast_body_seg.nii.gz").get_fdata()
        dice = dice_score_multilabel(img_ref, img_new)
        images_equal = dice > 0.99
        self.assertTrue(images_equal, f"body_seg prediction fast not correct (dice: {dice:.6f})")

    def test_lung_vessels(self):
        for roi in ["lung_trachea_bronchia", "lung_vessels"]:
            img_ref = nib.load(f"tests/reference_files/example_seg_lung_vessels/{roi}.nii.gz").get_fdata()
            img_new = nib.load(f"tests/unittest_prediction/{roi}.nii.gz").get_fdata()
            dice = dice_score(img_ref, img_new)
            images_equal = dice > 0.99
            self.assertTrue(images_equal, f"{roi} prediction not correct (dice: {dice:.6f})")

    def test_tissue_types_wo_license(self):
        no_output_file = not os.path.exists("tests/unittest_no_license.nii.gz")
        self.assertTrue(no_output_file, "A output file was generated even though no license was set.")

    def test_tissue_types_wrong_license(self):
        no_output_file = not os.path.exists("tests/unittest_wrong_license.nii.gz")
        self.assertTrue(no_output_file, "A output file was generated even though the license was wrong.")

    def test_tissue_types(self):
        for roi in ["subcutaneous_fat", "skeletal_muscle", "torso_fat"]:
            img_ref = nib.load(f"tests/reference_files/example_seg_tissue_types/{roi}.nii.gz").get_fdata()
            img_new = nib.load(f"tests/unittest_prediction/{roi}.nii.gz").get_fdata()
            dice = dice_score(img_ref, img_new)
            images_equal = dice > 0.99
            self.assertTrue(images_equal, f"{roi} prediction not correct (dice: {dice:.6f})")

    def test_appendicular_bones(self):
        for roi in ["patella", "phalanges_hand"]:
            img_ref = nib.load(f"tests/reference_files/example_seg_appendicular_bones/{roi}.nii.gz").get_fdata()
            img_new = nib.load(f"tests/unittest_prediction/{roi}.nii.gz").get_fdata()
            dice = dice_score(img_ref, img_new)
            images_equal = dice > 0.99
            self.assertTrue(images_equal, f"{roi} prediction not correct (dice: {dice:.6f})")

    def test_statistics(self):
        stats_ref = json.load(open("tests/reference_files/example_seg_fast/statistics.json"))
        stats_ref = pd.DataFrame(stats_ref)
        stats_new = json.load(open("tests/unittest_prediction_fast/statistics.json"))
        stats_new = pd.DataFrame(stats_new)
        stats_equal = np.allclose(stats_ref.loc["volume"].values, stats_new.loc["volume"].values,
                                  rtol=3e-2, atol=3e-2)
        self.assertTrue(stats_equal, "volume statistics are not correct")
        max_diff_intensity = np.abs(stats_ref.loc["intensity"].values - stats_new.loc["intensity"].values).max()
        stats_equal = max_diff_intensity < 2.0
        self.assertTrue(stats_equal, f"intensity statistics are not correct (max_diff: {max_diff_intensity:.5f})")

    # def test_radiomics(self):
    #     stats_ref = json.load(open("tests/reference_files/example_seg_fast/statistics_radiomics.json", "r"))
    #     stats_ref = pd.DataFrame(stats_ref)
    #     stats_ref = stats_ref.fillna(0)
    #     stats_new = json.load(open("tests/unittest_prediction_fast/statistics_radiomics.json", "r"))
    #     stats_new = pd.DataFrame(stats_new)
    #     stats_new = stats_new.fillna(0)
    #     # very big margin, but some of the radiomics features can change a lot if only a few voxels
    #     # of the segmentation change. So this test is only to check that radiomics ran successfully.
    #     stats_equal = np.allclose(stats_ref.values, stats_new.values, rtol=3e-1, atol=3e-1)
    #     self.assertTrue(stats_equal, "radiomics is not correct")

    def test_prediction_dicom(self):
        img_ref = nib.load("tests/reference_files/example_seg_dicom.nii.gz").get_fdata()
        img_new = nib.load("tests/unittest_prediction_dicom.nii.gz").get_fdata()
        dice = dice_score(img_ref, img_new)
        images_equal = dice > 0.99
        self.assertTrue(images_equal, f"Dicom prediction not correct (dice: {dice:.6f})")


if __name__ == '__main__':
    pytest.main(["-v", "tests/test_end_to_end.py::test_end_to_end::test_prediction_fast"])