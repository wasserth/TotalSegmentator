import sys
import csv
import json
import tempfile
import subprocess
import unittest
from pathlib import Path

import pytest

from totalsegmentator.bin.totalseg_aggregate_stats import (
    flatten, rows_from_stats, collect_rows, ordered_columns, write_table,
)


class TestFlatten(unittest.TestCase):
    def test_scalar(self):
        self.assertEqual(flatten("volume", 12.0), {"volume": 12.0})

    def test_list(self):
        self.assertEqual(flatten("centroid_vox", [1.0, 2.0, 3.0]),
                         {"centroid_vox_0": 1.0, "centroid_vox_1": 2.0, "centroid_vox_2": 3.0})

    def test_nested_list(self):
        self.assertEqual(flatten("bbox_vox", [[4, 6], [4, 6]]),
                         {"bbox_vox_0_0": 4, "bbox_vox_0_1": 6, "bbox_vox_1_0": 4, "bbox_vox_1_1": 6})


class TestAggregate(unittest.TestCase):
    def _cohort(self, tmp):
        (tmp / "s1").mkdir()
        (tmp / "site2" / "s3").mkdir(parents=True)
        (tmp / "s1" / "statistics.json").write_text(json.dumps(
            {"spleen": {"volume": 27.0, "intensity": 20.0, "centroid_vox": [5.0, 5.0, 5.0]},
             "liver": {"volume": 0.0, "intensity": 0.0}}))
        (tmp / "site2" / "s3" / "statistics.json").write_text(json.dumps(
            {"spleen": {"volume": 19.0, "intensity": 18.0}}))

    def test_rows_from_stats(self):
        rows = rows_from_stats({"spleen": {"volume": 1.0, "centroid_vox": [1, 2, 3]}}, "subjA")
        self.assertEqual(rows, [{"subject": "subjA", "structure": "spleen",
                                 "volume": 1.0, "centroid_vox_0": 1, "centroid_vox_1": 2, "centroid_vox_2": 3}])

    def test_collect_and_columns(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            self._cohort(tmp)
            rows, n = collect_rows(tmp, "statistics.json", quiet=True)
            self.assertEqual(n, 2)
            self.assertEqual(len(rows), 3)  # 2 structures + 1 structure
            self.assertEqual({r["subject"] for r in rows}, {"s1", "site2/s3"})
            cols = ordered_columns(rows)
            self.assertEqual(cols[:4], ["subject", "structure", "volume", "intensity"])
            self.assertIn("centroid_vox_2", cols)

    def test_cli_csv(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            self._cohort(tmp)
            out = tmp / "agg.csv"
            r = subprocess.run([sys.executable, "-m", "totalsegmentator.bin.totalseg_aggregate_stats",
                                "-i", str(tmp), "-o", str(out), "-q"], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            with open(out) as f:
                reader = list(csv.DictReader(f))
            self.assertEqual(len(reader), 3)
            spleen_s1 = [row for row in reader if row["subject"] == "s1" and row["structure"] == "spleen"][0]
            self.assertEqual(spleen_s1["volume"], "27.0")
            self.assertEqual(spleen_s1["centroid_vox_0"], "5.0")

    def test_cli_empty_cohort_exits_2(self):
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run([sys.executable, "-m", "totalsegmentator.bin.totalseg_aggregate_stats",
                                "-i", d, "-o", str(Path(d) / "x.csv")], capture_output=True, text=True)
            self.assertEqual(r.returncode, 2)

    def test_parquet_roundtrip(self):
        pytest.importorskip("pyarrow")
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            self._cohort(tmp)
            out = tmp / "agg.parquet"
            rows, _ = collect_rows(tmp, "statistics.json", quiet=True)
            write_table(rows, ordered_columns(rows), out)
            table = pq.read_table(out)
            self.assertEqual(table.num_rows, 3)
            self.assertIn("volume", table.column_names)


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
