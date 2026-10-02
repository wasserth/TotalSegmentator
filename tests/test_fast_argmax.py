"""
_logits_to_segmentation (argmax in torch, in slabs) against nnU-Net's own export, which it replaces
for softmax models whose logits need no resampling.
"""
import unittest
from functools import partial
from types import SimpleNamespace

import numpy as np
import torch
from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape
from nnunetv2.utilities.label_handling.label_handling import LabelManager

from totalsegmentator.nnunet import _logits_to_segmentation
from totalsegmentator.nnunet_runtime_patches import convert_predicted_logits_to_segmentation_with_correct_shape


def fake_predictor(K, device="cpu"):
    labels = {"background": 0, **{f"c{i}": i for i in range(1, K)}}
    return SimpleNamespace(
        label_manager=LabelManager(labels, regions_class_order=None),
        plans_manager=SimpleNamespace(transpose_forward=[0, 1, 2], transpose_backward=[0, 1, 2]),
        configuration_manager=SimpleNamespace(
            spacing=[1.5, 1.5, 1.5],
            resampling_fn_probabilities=partial(resample_data_or_seg_to_shape, is_seg=False, order=1, order_z=0,
                                                force_separate_z=None)),
        device=torch.device(device))


class FastArgmax(unittest.TestCase):

    def check(self, K, full_shape, bbox, device="cpu", slab_bytes=1 << 30, dtype=torch.float16):
        rng = np.random.default_rng(K)
        cropped = tuple(b[1] - b[0] for b in bbox)
        logits = torch.from_numpy(rng.normal(size=(K, *cropped))).to(dtype)
        logits[:, :2] = 0                         # exact ties: both must take the first maximum
        props = {"bbox_used_for_cropping": [list(b) for b in bbox], "shape_before_cropping": full_shape,
                 "shape_after_cropping_and_before_resampling": cropped, "spacing": [1.5, 1.5, 1.5]}
        p = fake_predictor(K, device)
        want = convert_predicted_logits_to_segmentation_with_correct_shape(
            logits, p.plans_manager, p.configuration_manager, p.label_manager, props)
        got = _logits_to_segmentation(p, logits, props, slab_bytes=slab_bytes)
        self.assertEqual(got.dtype, np.asarray(want).dtype)
        np.testing.assert_array_equal(got, np.asarray(want))

    def test_matches_nnunet_export(self):
        self.check(6, (20, 24, 28), [(0, 20), (0, 24), (0, 28)])
        self.check(27, (20, 24, 28), [(2, 19), (1, 24), (0, 25)])            # a crop box
        self.check(27, (20, 24, 28), [(2, 19), (1, 24), (0, 25)], slab_bytes=1)  # one plane per slab
        self.check(5, (9, 10, 11), [(0, 9), (0, 10), (0, 11)], dtype=torch.float32)

    @unittest.skipUnless(torch.backends.mps.is_available() or torch.cuda.is_available(), "no GPU")
    def test_on_the_gpu(self):
        device = "cuda" if torch.cuda.is_available() else "mps"
        self.check(27, (20, 24, 28), [(2, 19), (1, 24), (0, 25)], device=device, slab_bytes=1 << 12)

    def test_declines_what_needs_resampling(self):
        p = fake_predictor(4)
        logits = torch.zeros((4, 5, 6, 7), dtype=torch.float16)
        props = {"bbox_used_for_cropping": [[0, 5], [0, 6], [0, 7]], "shape_before_cropping": (5, 6, 7),
                 "shape_after_cropping_and_before_resampling": (5, 6, 8)}
        self.assertIsNone(_logits_to_segmentation(p, logits, props))


if __name__ == "__main__":
    unittest.main()
