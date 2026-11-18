import nibabel as nib
import numpy as np
import pytest
import torch
from scipy.ndimage import zoom
from torch import nn

from totalsegmentator.optimized_ct import model_grid, restore_labels, window_labels


def test_model_and_output_grid():
    rng = np.random.default_rng(14)
    data = rng.normal(size=(9, 6, 11)) * 100
    affine = np.diag([1.5, 1.0, 3.0, 1.0])
    affine[0, 3] = 42
    image = nib.Nifti1Image(data, affine)
    grid, new_affine = model_grid(image, [1.5] * 3, "cpu")
    expected = zoom(data, [1, 2 / 3, 2], order=1).astype(np.int32)
    np.testing.assert_array_equal(grid.numpy().T, expected)
    np.testing.assert_allclose(new_affine[:3, 3], affine[:3, 3])
    np.testing.assert_allclose(np.diag(new_affine)[:3], [1.5] * 3)
    labels = np.arange(grid.numel(), dtype=np.uint8).reshape(grid.shape)
    result = restore_labels(torch.from_numpy(labels), data.shape[::-1])
    np.testing.assert_array_equal(
        result.numpy(), zoom(labels, np.array(data.shape[::-1]) / grid.shape, order=0)
    )


def test_batched_depth_slabs_match_full_volume_logits():
    torch.manual_seed(19)
    patch = (4, 4, 4)
    image = torch.randn(9, 7, 8)
    net = nn.Conv3d(1, 3, 3, padding=1)
    weights = torch.rand(patch) + 0.1
    steps = [[0, 3, 5], [0, 3], [0, 2, 4]]
    with torch.no_grad():
        sums = torch.zeros((3, *image.shape))
        for z in steps[0]:
            for y in steps[1]:
                for x in steps[2]:
                    region = (slice(z, z + 4), slice(y, y + 4), slice(x, x + 4))
                    sums[(slice(None), *region)] += net(image[region][None, None])[0] * weights
        torch.testing.assert_close(
            window_labels(image, net, patch, steps, weights, 3), sums.argmax(0).byte()
        )


def test_split_merge_and_foreground_precedence(monkeypatch):
    from types import SimpleNamespace
    from totalsegmentator import optimized_ct

    session = optimized_ct.CTSession()
    cfg = SimpleNamespace(normalization_schemes=["CTNormalization"], spacing=[1.5] * 3)
    monkeypatch.setattr(session, "_load", lambda folder, device: (folder, None, cfg, 2))
    monkeypatch.setattr(optimized_ct, "predict_model", lambda chunk, net, *_: (chunk % net == 0).byte())
    ct = torch.arange(73 * 5 * 7).reshape(73, 5, 7)
    selected = [(2, [0, 11]), (3, [0, 17])]
    expected = torch.where(ct % 3 == 0, 17, torch.where(ct % 2 == 0, 11, 0)).byte()
    for split in (False, True):
        result = session.segment(ct, [1.5] * 3, selected, 0.8, split)
        torch.testing.assert_close(result, expected)


def test_unreleased_small_6mm_plan_is_rejected_before_download():
    from totalsegmentator.python_api import totalsegmentator

    image = nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.int16), np.eye(4))
    with pytest.raises(ValueError, match="6 mm release"):
        totalsegmentator(
            image, task="total_v3", fastest=True, model_size="small", device="cpu", optimized_ct=True
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_windows_match_stock_gaussian_accumulation():
    # Exercise overlapping tiles, FP16 rounding and the bound on live windows.
    from itertools import product

    torch.manual_seed(17)
    patch = (4, 4, 4)
    image = torch.randn(7, 6, 6, device="cuda")
    net = nn.Conv3d(1, 3, 3, padding=1).eval().cuda()
    starts = [[0, 2, 3], [0, 2], [0, 2]]
    weights = (torch.rand(patch, device="cuda") + 0.01).half()
    sums = torch.zeros((3, *image.shape), device="cuda", dtype=torch.float16)
    norm = torch.zeros_like(sums[0])
    with torch.inference_mode(), torch.autocast("cuda"):
        for z, y, x in product(*starts):
            region = (slice(z, z + 4), slice(y, y + 4), slice(x, x + 4))
            sums[(slice(None), *region)] += net(image[region][None, None].contiguous())[0] * weights
            norm[region] += weights
        expected = (sums / norm).argmax(0).byte()
        batches = []
        net.register_forward_pre_hook(lambda _, inputs: batches.append(inputs[0].shape[0]))
        labels = window_labels(image, net, patch, starts, weights, 3)
    torch.testing.assert_close(labels, expected)
    assert batches == [1] * 12


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_session_keeps_at_most_one_cuda_network_between_scans(monkeypatch):
    from types import SimpleNamespace
    from totalsegmentator import optimized_ct

    session = optimized_ct.CTSession()
    cfg = SimpleNamespace(normalization_schemes=["CTNormalization"], spacing=[1.5] * 3)
    nets = [nn.Conv3d(1, 1, 1) for _ in range(3)]
    monkeypatch.setattr(session, "_load", lambda folder, device: (nets[folder], None, cfg, 2))

    def predict(chunk, net, *_):
        assert [n for n in nets if n.weight.is_cuda] == [net]
        return torch.ones_like(chunk, dtype=torch.uint8)

    monkeypatch.setattr(optimized_ct, "predict_model", predict)
    ct = torch.zeros((2, 3, 4), device="cuda")
    session.segment(ct, 1.5, [(0, None)], 0.5, False)
    session.segment(ct, 1.5, [(1, None)], 0.5, False)
    assert nets[1].weight.is_cuda
    session.segment(ct, 1.5, [(1, [0, 1]), (2, [0, 2])], 0.5, False)
    assert all(not n.weight.is_cuda for n in nets)


@pytest.mark.parametrize("depth", [4, 6])
def test_nonfinite_logits_raise_before_argmax(depth):
    net = nn.Conv3d(1, 2, 1)
    with torch.no_grad():
        net.bias.fill_(float("inf"))
        with pytest.raises(RuntimeError, match="Non-finite"):
            window_labels(torch.zeros(depth, 4, 4), net, (4, 4, 4),
                          [[0] if depth == 4 else [0, 2], [0], [0]], torch.ones(4, 4, 4), 2)


def test_cuda_labels_use_stock_resize_when_cucim_is_available(monkeypatch):
    from unittest.mock import Mock
    from totalsegmentator import nnunet, optimized_ct, resampling

    image = nib.Nifti1Image(np.zeros((5, 5, 5), dtype=np.int16), np.eye(4))
    labels = torch.arange(27, dtype=torch.uint8).reshape(3, 3, 3)
    session = Mock()
    session.segment.return_value = labels
    monkeypatch.setattr(optimized_ct, "model_grid", lambda *_: (labels.float(), np.diag([2., 2., 2., 1.])))
    monkeypatch.setattr(nnunet, "get_output_folder", lambda *_: "model")
    for module in (nnunet, resampling):
        monkeypatch.setattr(module, "cupy_available", True)
        monkeypatch.setattr(module, "cucim_available", True)

    def resize(data, zoom, order, nr_cpus):
        from scipy.ndimage import zoom as scipy_zoom
        return scipy_zoom(data, zoom, order=order, mode="nearest", grid_mode=True)

    gpu_resize = Mock(side_effect=resize)
    monkeypatch.setattr(resampling, "resample_img_cucim", gpu_resize)
    result, *_ = nnunet.nnUNet_predict_image(image, None, 298, resample=2.0, device="cuda",
                                             optimized_ct=session, quiet=True)
    gpu_resize.assert_called_once()
    expected = resize(labels.numpy().T, 5 / 3, 0, 1)
    np.testing.assert_array_equal(result.dataobj, expected)
