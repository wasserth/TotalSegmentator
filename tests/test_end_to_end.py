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

    def test_statistics(self):
        stats_ref = json.load(open("tests/reference_files/example_seg_fast/statistics.json", "r"))
        stats_new = json.load(open("tests/unittest_prediction_fast/statistics.json", "r"))
        stats_equal = stats_ref == stats_new
        self.assertTrue(stats_equal, "statistics are not correct")

    def test_radiomics(self):
        stats_ref = json.load(open("tests/reference_files/example_seg_fast/statistics_radiomics.json", "r"))
        stats_new = json.load(open("tests/unittest_prediction_fast/statistics_radiomics.json", "r"))
        stats_equal = stats_ref == stats_new
        self.assertTrue(stats_equal, "statistics are not correct")


if __name__ == '__main__':
    unittest.main()
