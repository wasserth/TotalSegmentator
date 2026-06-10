import tempfile
import unittest
from pathlib import Path

from totalsegmentator.bin.totalseg_batch import find_images, case_id, output_target


class TestBatchHelpers(unittest.TestCase):

    def test_case_id(self):
        self.assertEqual(case_id(Path("/x/ct.nii.gz")), "ct")
        self.assertEqual(case_id(Path("/x/subj01.nii")), "subj01")
        self.assertEqual(case_id(Path("/x/a.b.nii.gz")), "a.b")

    def test_output_target(self):
        out = Path("/out")
        self.assertEqual(output_target(out, "c1", ml=False), Path("/out/c1"))
        self.assertEqual(output_target(out, "c1", ml=True), Path("/out/c1/segmentation.nii.gz"))

    def test_find_images_default(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "a.nii.gz").write_text("x")
            (d / "b.nii").write_text("x")
            (d / "notes.txt").write_text("x")
            (d / "sub").mkdir()
            (d / "sub" / "deep.nii.gz").write_text("x")  # not picked up by default (not direct)
            names = [p.name for p in find_images(d)]
            self.assertEqual(names, ["a.nii.gz", "b.nii"])

    def test_find_images_pattern(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "s1").mkdir()
            (d / "s2").mkdir()
            (d / "s1" / "ct.nii.gz").write_text("x")
            (d / "s2" / "ct.nii.gz").write_text("x")
            rel = sorted(str(p.relative_to(d)) for p in find_images(d, pattern="*/ct.nii.gz"))
            self.assertEqual(rel, ["s1/ct.nii.gz", "s2/ct.nii.gz"])


if __name__ == "__main__":
    unittest.main()
