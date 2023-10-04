import os
import unittest
import pytest
import json
import subprocess

import nibabel as nib
import numpy as np
import pandas as pd


class test_end_to_end(unittest.TestCase):

    def setUp(self):
        pass

    def test_prediction_multilabel(self):
        img_ref = nib.load("tests/reference_files/example_seg.nii.gz").get_fdata()
        img_new = nib.load("tests/unittest_prediction.nii.gz").get_fdata()
        nr_of_diff_voxels = (img_ref != img_new).sum()
        images_equal = nr_of_diff_voxels < 20
        self.assertTrue(images_equal, f"multilabel prediction not correct (nr_of_diff_voxels: {nr_of_diff_voxels})")

    def test_prediction_liver_roi_subset(self):
        img_ref = nib.load(f"tests/reference_files/example_seg_roi_subset.nii.gz").get_fdata()
        img_new = nib.load(f"tests/unittest_prediction_roi_subset.nii.gz").get_fdata()
        # prediction is not completely deterministic therefore allow for small differences
        nr_of_diff_voxels = (img_ref != img_new).sum()
        images_equal = nr_of_diff_voxels < 20
        self.assertTrue(images_equal, f"roi subset prediction not correct (nr_of_diff_voxels: {nr_of_diff_voxels})")

    def test_prediction_fast(self):
        for roi in ["liver", "vertebrae_L1"]:
            img_ref = nib.load(f"tests/reference_files/example_seg_fast/{roi}.nii.gz").get_fdata()
            img_new = nib.load(f"tests/unittest_prediction_fast/{roi}.nii.gz").get_fdata()
            # prediction is not completely deterministic therefore allow for small differences
            nr_of_diff_voxels = (img_ref != img_new).sum()
            images_equal = nr_of_diff_voxels < 20
            self.assertTrue(images_equal, f"{roi} fast prediction not correct (nr_of_diff_voxels: {nr_of_diff_voxels})")

    def test_preview(self):
        preview_exists = os.path.exists(f"tests/unittest_prediction_fast/preview_total.png")
        self.assertTrue(preview_exists, f"Preview was not generated")

    def test_prediction_multilabel_fast(self):
        img_ref = nib.load("tests/reference_files/example_seg_fast.nii.gz").get_fdata()
        img_new = nib.load("tests/unittest_prediction_fast.nii.gz").get_fdata()
        images_equal = np.array_equal(img_ref, img_new)
        self.assertTrue(images_equal, "multilabel prediction fast not correct")

    def test_prediction_multilabel_fast_force_split(self):
        img_ref = nib.load("tests/reference_files/example_seg_fast_force_split.nii.gz").get_fdata()
        img_new = nib.load("tests/unittest_prediction_fast_force_split.nii.gz").get_fdata()
        images_equal = np.array_equal(img_ref, img_new)
        self.assertTrue(images_equal, "force_split prediction not correct")

    def test_prediction_multilabel_fast_body_seg(self):
        img_ref = nib.load("tests/reference_files/example_seg_fast_body_seg.nii.gz").get_fdata()
        img_new = nib.load("tests/unittest_prediction_fast_body_seg.nii.gz").get_fdata()
        images_equal = np.array_equal(img_ref, img_new)
        self.assertTrue(images_equal, "body_seg prediction fast not correct")

    def test_lung_vessels(self):
        for roi in ["lung_trachea_bronchia", "lung_vessels"]:
            img_ref = nib.load(f"tests/reference_files/example_seg_lung_vessels/{roi}.nii.gz").get_fdata()
            img_new = nib.load(f"tests/unittest_prediction/{roi}.nii.gz").get_fdata()
            images_equal = np.array_equal(img_ref, img_new)
            self.assertTrue(images_equal, f"{roi} prediction not correct")

    def test_tissue_types_wo_license(self):
        no_output_file = not os.path.exists(f"tests/unittest_no_license.nii.gz")
        self.assertTrue(no_output_file, f"A output file was generated even though no license was set.")

    def test_tissue_types_wrong_license(self):
        no_output_file = not os.path.exists(f"tests/unittest_wrong_license.nii.gz")
        self.assertTrue(no_output_file, f"A output file was generated even though the license was wrong.")

    def test_tissue_types(self):
        for roi in ["subcutaneous_fat", "skeletal_muscle", "torso_fat"]:
            img_ref = nib.load(f"tests/reference_files/example_seg_tissue_types/{roi}.nii.gz").get_fdata()
            img_new = nib.load(f"tests/unittest_prediction/{roi}.nii.gz").get_fdata()
            images_equal = np.array_equal(img_ref, img_new)
            self.assertTrue(images_equal, f"{roi} prediction not correct")

    def test_appendicular_bones(self):
        for roi in ["patella", "phalanges_hand"]:
            img_ref = nib.load(f"tests/reference_files/example_seg_appendicular_bones/{roi}.nii.gz").get_fdata()
            img_new = nib.load(f"tests/unittest_prediction/{roi}.nii.gz").get_fdata()
            images_equal = np.array_equal(img_ref, img_new)
            self.assertTrue(images_equal, f"{roi} prediction not correct")

    def test_statistics(self):
        stats_ref = json.load(open("tests/reference_files/example_seg_fast/statistics.json", "r"))
        stats_ref = pd.DataFrame(stats_ref)
        stats_new = json.load(open("tests/unittest_prediction_fast/statistics.json", "r"))
        stats_new = pd.DataFrame(stats_new)
        stats_equal = np.allclose(stats_ref.loc["volume"].values, stats_new.loc["volume"].values,
                                  rtol=3e-2, atol=3e-2)
        self.assertTrue(stats_equal, "volume statistics are not correct")
        stats_equal = np.allclose(stats_ref.loc["intensity"].values, stats_new.loc["intensity"].values,
                                  rtol=3e-2, atol=3e-2)
        self.assertTrue(stats_equal, "intensity statistics are not correct")

    # def test_radiomics(self):
    #     stats_ref = json.load(open("tests/reference_files/example_seg_fast/statistics_radiomics.json", "r"))
    #     stats_ref = pd.DataFrame(stats_ref)
    #     stats_ref = stats_ref.fillna(0)
    #     stats_new = json.load(open("tests/unittest_prediction_fast/statistics_radiomics.json", "r"))
    #     stats_new = pd.DataFrame(stats_new)
    #     stats_new = stats_new.fillna(0)
    #     # very big margin, but some of the radiomics features can change a lot if only a few voxels
    #     # of the segmentation change. So this test is only to check that radiomics ran sucessfully.
    #     stats_equal = np.allclose(stats_ref.values, stats_new.values, rtol=3e-1, atol=3e-1)
    #     self.assertTrue(stats_equal, "radiomics is not correct")

    def test_prediction_dicom(self):
        img_ref = nib.load("tests/reference_files/example_seg_dicom.nii.gz").get_fdata()
        img_new = nib.load("tests/unittest_prediction_dicom.nii.gz").get_fdata()
        images_equal = np.array_equal(img_ref, img_new)
        self.assertTrue(images_equal, "Dicom prediction not correct")


if __name__ == '__main__':
    pytest.main(["-v", "tests/test_end_to_end.py::test_end_to_end::test_prediction_fast"])