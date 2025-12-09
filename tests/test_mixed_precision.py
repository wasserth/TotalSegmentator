import os
import pytest
import torch

def test_want_mixed_precision_flag_false_by_default():
    os.environ.pop("TOTALSEG_MIXED_PRECISION", None)
    import totalsegmentator.mixed_precision as mp
    assert mp.want_mixed_precision() is False

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_want_mixed_precision_flag_true_when_env_set_and_cuda():
    os.environ["TOTALSEG_MIXED_PRECISION"] = "1"
    import importlib
    mp = importlib.reload(__import__("totalsegmentator.mixed_precision", fromlist=["*"]))
    assert mp.want_mixed_precision() is True