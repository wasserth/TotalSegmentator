"""
keep_models=True keeps initialized predictors between calls. What a cached predictor must not
change: the process-wide torch thread count its device calls for, and what clear_model_cache
releases.
"""
import multiprocessing
import unittest

import torch

from totalsegmentator import nnunet


class KeepModels(unittest.TestCase):

    def test_cpu_after_gpu_gets_all_threads(self):
        """The thread count is set for every prediction, cached model or not: a CPU call after a
        GPU call must not run on the GPU's single thread."""
        before = torch.get_num_threads()
        try:
            self.assertEqual(nnunet._configure_torch_device(torch.device("cpu")), torch.device("cpu"))
            self.assertEqual(torch.get_num_threads(), 1)          # as after any GPU build
            self.assertEqual(nnunet._configure_torch_device("cpu"), torch.device("cpu"))
            self.assertEqual(torch.get_num_threads(), multiprocessing.cpu_count())
        finally:
            torch.set_num_threads(before)

    def test_the_key_separates_what_shapes_a_predictor(self):
        k = nnunet._predictor_key(291, "3d_fullres", [0], "nnUNetTrainer", False, "nnUNetPlans", "cpu", 0.5)
        self.assertEqual(k, nnunet._predictor_key(291, "3d_fullres", (0,), "nnUNetTrainer", False,
                                                  "nnUNetPlans", "cpu", 0.5))
        for other in [(292, "3d_fullres", [0], "nnUNetTrainer", False, "nnUNetPlans", "cpu", 0.5),
                      (291, "3d_fullres", [0], "nnUNetTrainer", True, "nnUNetPlans", "cpu", 0.5),
                      (291, "3d_fullres", [0], "nnUNetTrainer", False, "nnUNetPlans", "mps", 0.5),
                      (291, "3d_fullres", [0], "nnUNetTrainer", False, "nnUNetPlans", "cpu", 0.8)]:
            self.assertNotEqual(k, nnunet._predictor_key(*other))

    def test_clear_model_cache_empties_it(self):
        nnunet._PREDICTOR_CACHE[("x",)] = object()
        from totalsegmentator.python_api import clear_model_cache
        clear_model_cache()
        self.assertEqual(nnunet._PREDICTOR_CACHE, {})


if __name__ == "__main__":
    unittest.main()
