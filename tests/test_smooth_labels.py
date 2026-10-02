"""
Smooth label maps against the default pipeline, without model weights.

Synthetic logits stand in for three models on a triple-split model grid, each piece cropped by a
fake nnU-Net bounding box. The default path is reproduced step by step: argmax per piece, insert
into the uncropped piece, map through the part's label table, composite with
np.copyto(where != 0), reassemble the pieces, and upsample with change_spacing(order=0). The smooth
path with interp="nearest" has to give the same labels exactly; with "linear" it has to give
smooth ones.
"""
import unittest

import nibabel as nib
import numpy as np
import torch

from totalsegmentator.resampling import change_spacing
from totalsegmentator.smooth_labels import SmoothComposite

try:
    import labelfield  # noqa: F401  (an optional dependency: smooth labels need it)
    HAVE_LABELFIELD = True
except ImportError:
    HAVE_LABELFIELD = False
NEEDS_LABELFIELD = unittest.skipUnless(HAVE_LABELFIELD, "labelfield is not installed")

MODEL_XYZ = (23, 19, 70)            # canonical (nibabel) shape on the model grid
INPUT_XYZ = (41, 37, 131)           # and on the input grid
MARGIN = 5                          # the triple split's overlap (20 in nnunet.py; any value works)
LUTS = [[0, 1, 2, 3, 4], [0, 10, 11, 0, 13], [0, 20, 21]]   # part 2 has an unreported class (label 0)


def blobs(K, shape_zyx, seed):
    """Smooth random logits: K channels of blurred noise, so boundaries are soft and irregular."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(K, *shape_zyx))
    for axis in (1, 2, 3):
        for _ in range(3):
            x = (np.roll(x, 1, axis) + x + np.roll(x, -1, axis)) / 3
    return (x * 8).astype(np.float32)


def pieces(n_z):
    third = n_z // 3
    return {"s01": (0, third + MARGIN, (0, third)),
            "s02": (third + 1 - MARGIN, third * 2 + MARGIN, (third, third * 2)),
            "s03": (third * 2 + 1 - MARGIN, n_z, (third * 2, n_z))}, third


def crop_box(shape_zyx, seed):
    """A fake crop_to_nonzero box: trims a voxel or two off some sides."""
    rng = np.random.default_rng(seed)
    return [[int(rng.integers(0, 2)), int(n - rng.integers(0, 3))] for n in shape_zyx]


@NEEDS_LABELFIELD
class SmoothLabels(unittest.TestCase):

    def setUp(self):
        self.model_zyx = MODEL_XYZ[::-1]
        self.parts = [blobs(len(lut), self.model_zyx, seed=i) for i, lut in enumerate(LUTS)]
        self.pieces, self.third = pieces(MODEL_XYZ[2])
        self.boxes = {(i, name): crop_box((z1 - z0, *self.model_zyx[1:]), seed=10 * i + j)
                      for i in range(len(LUTS)) for j, (name, (z0, z1, _)) in enumerate(self.pieces.items())}

    def piece_logits(self, i, name):
        z0, z1, _ = self.pieces[name]
        box = self.boxes[(i, name)]
        full = self.parts[i][:, z0:z1]
        cropped = full[:, box[0][0]:box[0][1], box[1][0]:box[1][1], box[2][0]:box[2][1]]
        props = {"bbox_used_for_cropping": box,
                 "shape_after_cropping_and_before_resampling": cropped.shape[1:]}
        return full.shape[1:], cropped, props

    def default_pipeline(self):
        segmentations = {name: np.zeros((z1 - z0, *self.model_zyx[1:])[::-1], np.uint8)
                         for name, (z0, z1, _) in self.pieces.items()}
        for i, lut in enumerate(LUTS):
            for name in self.pieces:
                shape, cropped, props = self.piece_logits(i, name)
                box = props["bbox_used_for_cropping"]
                seg = np.zeros(shape, np.uint8)                                # nnU-Net's insert_crop
                seg[box[0][0]:box[0][1], box[1][0]:box[1][1], box[2][0]:box[2][1]] = cropped.argmax(0)
                mapped = np.asarray(lut, np.uint8)[seg.transpose(2, 1, 0)]
                np.copyto(segmentations[name], mapped, where=mapped != 0)
        third, margin = self.third, MARGIN
        combined = np.zeros(MODEL_XYZ, np.uint8)
        combined[:, :, :third] = segmentations["s01"][:, :, :-margin]
        combined[:, :, third:third * 2] = segmentations["s02"][:, :, margin - 1:-margin]
        combined[:, :, third * 2:] = segmentations["s03"][:, :, margin - 1:]
        img = nib.Nifti1Image(combined, np.diag([3.0, 3.0, 3.0, 1.0]))
        up = change_spacing(img, [1.5, 1.5, 1.5], INPUT_XYZ, order=0, dtype=np.uint8,
                            force_affine=np.diag([1.5, 1.5, 1.5, 1.0]))
        return np.asanyarray(up.dataobj)

    def smooth_pipeline(self, interp, device="cpu"):
        smooth = SmoothComposite(INPUT_XYZ, MODEL_XYZ, device, interp=interp)
        for i, lut in enumerate(LUTS):
            for name, (z0, _, keep) in self.pieces.items():
                _, cropped, props = self.piece_logits(i, name)
                smooth.paint(torch.from_numpy(cropped), props, lut=lut, z_offset=z0, keep=keep)
        return smooth.result()

    def test_nearest_is_the_default_output_exactly(self):
        want = self.default_pipeline()
        self.assertGreater(len(np.unique(want)), 6)
        np.testing.assert_array_equal(self.smooth_pipeline("nearest"), want)

    def test_nearest_in_small_slabs(self):
        """The logits go to the device a slab at a time; slab boundaries must not show."""
        want = self.default_pipeline()
        smooth = SmoothComposite(INPUT_XYZ, MODEL_XYZ, "cpu", interp="nearest", chunk_bytes=1)
        for i, lut in enumerate(LUTS):
            for name, (z0, _, keep) in self.pieces.items():
                _, cropped, props = self.piece_logits(i, name)
                smooth.paint(torch.from_numpy(cropped), props, lut=lut, z_offset=z0, keep=keep)
        np.testing.assert_array_equal(smooth.result(), want)

    def test_linear_is_smooth(self):
        blocky = self.default_pipeline()
        smooth = self.smooth_pipeline("linear")
        self.assertEqual(set(np.unique(smooth)) - {0}, set(np.unique(blocky)) - {0})
        differ = (smooth != blocky).mean()
        self.assertGreater(differ, 0.001)                      # boundaries moved off the model's voxels
        self.assertLess(differ, 0.2)                           # but it is the same segmentation

    @unittest.skipUnless(torch.backends.mps.is_available() or torch.cuda.is_available(), "no GPU")
    def test_gpu_matches_cpu_nearest(self):
        device = "cuda" if torch.cuda.is_available() else "mps"
        np.testing.assert_array_equal(self.smooth_pipeline("nearest", device), self.default_pipeline())



class ResolveSmoothLabels(unittest.TestCase):
    """What smooth_labels means for a call: "auto" is smooth wherever it applies, quietly nearest
    elsewhere; True raises where it cannot apply."""

    @NEEDS_LABELFIELD
    def test_auto_is_smooth_where_it_applies(self):
        from totalsegmentator.nnunet import resolve_smooth_labels
        self.assertEqual(resolve_smooth_labels("auto", resample=[1.5, 1.5, 1.5]), "linear")

    def test_auto_is_quietly_nearest_where_it_cannot_apply(self):
        from totalsegmentator.nnunet import resolve_smooth_labels
        for kw in ({"resample": None}, {"resample": [1.5] * 3, "save_lowres": True},
                   {"resample": [1.5] * 3, "higher_order_resampling_LEGACY": True},
                   {"resample": [1.5] * 3, "save_probabilities": "p.npz"}, {"resample": [1.5] * 3, "test": 1}):
            self.assertIs(resolve_smooth_labels("auto", **kw), False, kw)

    def test_auto_is_quietly_nearest_without_labelfield(self):
        import builtins
        import warnings
        from unittest import mock
        from totalsegmentator.nnunet import resolve_smooth_labels
        real = builtins.__import__

        def no_labelfield(name, *a, **k):
            if name == "labelfield" or name.startswith("labelfield."):
                raise ImportError("hidden for this test")
            return real(name, *a, **k)
        with mock.patch.object(builtins, "__import__", no_labelfield), warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertIs(resolve_smooth_labels("auto", resample=[1.5, 1.5, 1.5]), False)

    def test_explicit_true_raises_where_it_cannot_apply(self):
        from totalsegmentator.nnunet import resolve_smooth_labels
        self.assertEqual(resolve_smooth_labels(True, resample=[3.0] * 3), "linear")
        self.assertEqual(resolve_smooth_labels("nearest", resample=[3.0] * 3), "nearest")
        with self.assertRaises(ValueError):
            resolve_smooth_labels(True, resample=None)
        with self.assertRaises(ValueError):
            resolve_smooth_labels(True, resample=[1.5] * 3, save_lowres=True)

    def test_off(self):
        from totalsegmentator.nnunet import resolve_smooth_labels
        self.assertIs(resolve_smooth_labels(False, resample=[1.5] * 3), False)

    def test_the_cli_default_is_auto_and_the_flags_set_it(self):
        src = open("totalsegmentator/bin/TotalSegmentator.py").read()
        self.assertIn('parser.set_defaults(smooth_labels="auto")', src)
        self.assertIn('"--nearest_labels", action="store_const", dest="smooth_labels", const=False', src)


@NEEDS_LABELFIELD
class NoLeak(unittest.TestCase):
    """The first smooth call's arrays and models must not outlive it (labelfield < 0.1.3 kept a
    failed triton import's traceback, which reached every frame live at that import)."""

    def test_the_first_call_is_released(self):
        import subprocess
        import sys
        script = """
import gc, sys, weakref
sys.modules["triton"] = None
class Big: pass
def first_call():
    big = Big()
    import torch
    import totalsegmentator.smooth_labels as sl
    c = sl.SmoothComposite((8, 8, 8), (4, 4, 4), "cpu")
    c.paint(torch.zeros(2, 4, 4, 4), {"bbox_used_for_cropping": [[0, 4]] * 3,
            "shape_after_cropping_and_before_resampling": (4, 4, 4)})
    return weakref.ref(big)
r = first_call(); gc.collect(); print("freed" if r() is None else "leaked")
"""
        out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "freed", out.stderr)


if __name__ == "__main__":
    unittest.main()
