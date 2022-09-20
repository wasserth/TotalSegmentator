import unittest
import json
import nibabel as nib
import numpy as np
import pandas as pd


class test_end_to_end(unittest.TestCase):

    def setUp(self):
        pass

    def test_prediction_liver(self):
        img_ref = nib.load("tests/reference_files/example_seg/liver.nii.gz").get_fdata()
        img_new = nib.load("tests/unittest_prediction/liver.nii.gz").get_fdata()
        images_equal = np.array_equal(img_ref, img_new)
        self.assertTrue(images_equal, "liver prediction not correct")

    def test_prediction_vertebrae(self):
        img_ref = nib.load("tests/reference_files/example_seg/vertebrae_L1.nii.gz").get_fdata()
        img_new = nib.load("tests/unittest_prediction/vertebrae_L1.nii.gz").get_fdata()
        images_equal = np.array_equal(img_ref, img_new)
        self.assertTrue(images_equal, "vertebrae prediction not correct")

    def test_prediction_liver_fast(self):
        img_ref = nib.load("tests/reference_files/example_seg_fast/liver.nii.gz").get_fdata()
        img_new = nib.load("tests/unittest_prediction_fast/liver.nii.gz").get_fdata()
        images_equal = np.array_equal(img_ref, img_new)
        self.assertTrue(images_equal, "liver fast prediction not correct")

    def test_prediction_vertebrae_fast(self):
        img_ref = nib.load("tests/reference_files/example_seg_fast/vertebrae_L1.nii.gz").get_fdata()
        img_new = nib.load("tests/unittest_prediction_fast/vertebrae_L1.nii.gz").get_fdata()
        images_equal = np.array_equal(img_ref, img_new)
        self.assertTrue(images_equal, "vertebrae fast prediction not correct")

    def test_prediction_multilabel(self):
        img_ref = nib.load("tests/reference_files/example_seg.nii.gz").get_fdata()
        img_new = nib.load("tests/unittest_prediction.nii.gz").get_fdata()
        images_equal = np.array_equal(img_ref, img_new)
        self.assertTrue(images_equal, "multilabel prediction not correct")

    def test_lung_vessels(self):
        img_ref = nib.load("tests/reference_files/example_seg/lung_vessels.nii.gz").get_fdata()
        img_new = nib.load("tests/unittest_prediction/lung_vessels.nii.gz").get_fdata()
        images_equal = np.array_equal(img_ref, img_new)
        self.assertTrue(images_equal, "lung_vessel prediction not correct")

    def test_statistics(self):
        stats_ref = json.load(open("tests/reference_files/example_seg_fast/statistics.json", "r"))
        stats_ref = pd.DataFrame(stats_ref)
        stats_new = json.load(open("tests/unittest_prediction_fast/statistics.json", "r"))
        stats_new = pd.DataFrame(stats_new)
        stats_equal = np.allclose(stats_ref.loc["volume"].values, stats_new.loc["volume"].values,
                                  rtol=3e-2, atol=3e-2)
        self.assertTrue(stats_equal, "volume statistics are not correct")
        stats_equal = np.allclose(stats_ref.loc["intensity"].values, stats_new.loc["intensity"].values,
                                  rtol=3e-3, atol=3e-3)
        self.assertTrue(stats_equal, "intensity statistics are not correct")

    def test_radiomics(self):
        stats_ref = json.load(open("tests/reference_files/example_seg_fast/statistics_radiomics.json", "r"))
        stats_ref = pd.DataFrame(stats_ref)
        stats_ref = stats_ref.fillna(0)
        stats_new = json.load(open("tests/unittest_prediction_fast/statistics_radiomics.json", "r"))
        stats_new = pd.DataFrame(stats_new)
        stats_new = stats_new.fillna(0)
        # very big margin, but some of the radiomics features can change a lot if only a few voxels
        # of the segmentation change. So this test is only to check that radiomics ran sucessfully.
        stats_equal = np.allclose(stats_ref.values, stats_new.values, rtol=3e-1, atol=3e-1)
        self.assertTrue(stats_equal, "radiomics is not correct")


if __name__ == '__main__':
    unittest.main()
