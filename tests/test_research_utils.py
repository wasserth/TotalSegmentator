import unittest

import pytest


class TestExtraMetrics(unittest.TestCase):
    """Needs numpy + nibabel (present in CI)."""

    def test_extra_metrics(self):
        np = pytest.importorskip("numpy")
        nib = pytest.importorskip("nibabel")
        from totalsegmentator.statistics import get_basic_statistics

        seg = np.zeros((10, 10, 10), dtype=np.uint8)
        seg[4:7, 4:7, 4:7] = 1  # spleen, interior (not touching border)
        ct = np.full((10, 10, 10), -50.0)
        ct[seg == 1] = 25.0
        img = nib.Nifti1Image(ct, np.eye(4))

        stats = get_basic_statistics(seg, img, None, quiet=True, task="total",
                                     roi_subset=["spleen"], extra_metrics=True)
        sp = stats["spleen"]
        self.assertEqual(sp["n_voxels"], 27)
        self.assertEqual(sp["bbox_vox"], [[4, 6], [4, 6], [4, 6]])
        self.assertEqual(sp["centroid_vox"], [5.0, 5.0, 5.0])
        self.assertEqual(sp["intensity_min"], 25.0)
        self.assertEqual(sp["intensity_max"], 25.0)

    def test_default_has_no_extra_metrics(self):
        np = pytest.importorskip("numpy")
        nib = pytest.importorskip("nibabel")
        from totalsegmentator.statistics import get_basic_statistics

        seg = np.zeros((10, 10, 10), dtype=np.uint8)
        seg[4:7, 4:7, 4:7] = 1
        ct = np.zeros((10, 10, 10))
        img = nib.Nifti1Image(ct, np.eye(4))
        stats = get_basic_statistics(seg, img, None, quiet=True, task="total", roi_subset=["spleen"])
        self.assertEqual(set(stats["spleen"]), {"volume", "intensity"})


if __name__ == "__main__":
    unittest.main()
