"""
remove_small_blobs_multilabel (one connected-components pass with cc3d) against the per-class
loop it replaced: remove_small_blobs on each roi's binary mask.
"""
import unittest

import numpy as np
from scipy import ndimage

from totalsegmentator.postprocessing import remove_small_blobs, remove_small_blobs_multilabel


def per_class(data, class_map, rois, interval):
    """The implementation before cc3d, verbatim in effect."""
    class_map_inv = {v: k for k, v in class_map.items()}
    for roi in rois:
        idx = class_map_inv[roi]
        data_roi = data == idx
        cleaned_roi = remove_small_blobs(data_roi, interval) > 0.5
        data[data_roi] = 0
        data[cleaned_roi] = idx
    return data


def speckled(shape, n_labels, seed):
    """Labels with blobs of every size: smoothed noise, argmaxed, plus scattered single voxels."""
    rng = np.random.default_rng(seed)
    field = ndimage.uniform_filter(rng.normal(size=(n_labels + 1, *shape)), size=(1, 5, 5, 5))
    data = field.argmax(0).astype(np.uint8)
    specks = rng.random(shape) < 0.002
    data[specks] = rng.integers(0, n_labels + 1, int(specks.sum()))
    return data


class RemoveSmallBlobs(unittest.TestCase):

    def check(self, data, class_map, rois, interval):
        want = per_class(data.copy(), class_map, rois, interval)
        got = remove_small_blobs_multilabel(data.copy(), class_map, rois, interval=interval, quiet=True)
        self.assertEqual(got.dtype, data.dtype)
        np.testing.assert_array_equal(got, want)
        return int((want != data).sum())

    def test_all_classes_lower_bound(self):
        class_map = {i: f"c{i}" for i in range(1, 13)}
        for seed in range(4):
            data = speckled((40, 45, 50), 12, seed)
            removed = self.check(data, class_map, list(class_map.values()), [30.5, 1e10])
            self.assertGreater(removed, 0)

    def test_a_subset_of_classes_and_an_upper_bound(self):
        """body uses one class; the interval's upper end removes big blobs too."""
        class_map = {i: f"c{i}" for i in range(1, 9)}
        data = speckled((30, 35, 40), 8, seed=7)
        self.check(data, class_map, ["c2", "c5"], [10, 1e10])
        self.check(data, class_map, ["c3"], [5, 400])
        self.check(data, class_map, list(class_map.values()), [0, 200])

    def test_integer_threshold_is_inclusive(self):
        """a blob of exactly interval[0] voxels is removed, one voxel more is kept"""
        class_map = {1: "a", 2: "b"}
        data = np.zeros((10, 10, 10), np.uint8)
        data[1, 1, 1:5] = 1                       # 4 voxels
        data[5, 5, 1:6] = 2                       # 5 voxels
        got = remove_small_blobs_multilabel(data.copy(), class_map, ["a", "b"], interval=[4, 1e10], quiet=True)
        self.assertEqual(int((got == 1).sum()), 0)
        self.assertEqual(int((got == 2).sum()), 5)
        self.check(data, class_map, ["a", "b"], [4, 1e10])

    def test_nothing_to_do(self):
        class_map = {1: "a", 2: "b"}
        data = np.zeros((5, 6, 7), np.uint8)
        np.testing.assert_array_equal(
            remove_small_blobs_multilabel(data.copy(), class_map, ["a"], interval=[3, 1e10], quiet=True), data)
        data[2, 2, 2] = 2                         # only a class that is not in rois
        self.check(data, class_map, ["a"], [3, 1e10])


    def test_other_dtypes(self):
        """float64 (a get_fdata() result) and uint16 (more than 255 classes) give the loop's result"""
        class_map = {i: f"c{i}" for i in range(1, 7)}
        data = speckled((25, 26, 27), 6, seed=11)
        for dtype in (np.float64, np.uint16, np.int16):
            self.check(data.astype(dtype), class_map, list(class_map.values()), [20, 1e10])
            self.check(data.astype(dtype), class_map, ["c2", "c5"], [8, 1e10])

    def test_empty_negative_and_huge_labels(self):
        """edge cases the per-class loop handled: an empty array, labels outside the rois below 0,
        and a large label value (the whole-image shortcut must not enumerate up to it)"""
        class_map = {1: "a", 2: "b"}
        empty = np.zeros((0, 4, 4), np.int32)
        self.assertEqual(remove_small_blobs_multilabel(empty, class_map, ["a", "b"], interval=[3, 1e10],
                                                       quiet=True).size, 0)
        data = speckled((20, 21, 22), 2, seed=3).astype(np.int32)
        data[data == 0] = -1                      # a label that is not in class_map
        self.check(data, class_map, ["a", "b"], [5, 1e10])
        big = data.copy()
        big[0, 0, 0] = 10 ** 9
        self.check(big, class_map, ["a", "b"], [5, 1e10])


if __name__ == "__main__":
    unittest.main()
