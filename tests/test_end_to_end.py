import unittest
import nibabel as nib
import numpy as np
import pandas as pd


class test_end_to_end(unittest.TestCase):

    def setUp(self):
        pass

    def test_prediction(self):
        img_ref = np.ones((10,10,10))
        img_new = np.ones((10,10,10))
        images_equal = np.array_equal(img_ref, img_new)
        self.assertTrue(images_equal, "prediction not correct")


if __name__ == '__main__':
    unittest.main()
