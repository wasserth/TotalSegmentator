import numpy as np
import nibabel as nib
import pytest
import torch
from scipy import ndimage

import totalsegmentator.resampling as resampling
from totalsegmentator.resampling import change_spacing


def _blobs(shape, n_labels, seed=0):
    rng = np.random.default_rng(seed)
    field = ndimage.gaussian_filter(rng.random(shape), 2.0)
    seg = np.zeros(shape, dtype=np.uint8)
    for label in range(1, n_labels + 1):
        seg[field > np.quantile(field, 1 - 0.6 * label / n_labels)] = label
    return seg


def _record_dispatch(monkeypatch):
    """Which resampling implementation change_spacing actually reaches."""
    called = []
    for name in ("resample_img", "resample_img_torch", "resample_one_hot_crop"):
        original = getattr(resampling, name)

        def wrapper(*args, _name=name, _original=original, **kwargs):
            called.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(resampling, name, wrapper)
    return called


def test_torch_resample_does_not_intercept_one_hot_label_resampling(monkeypatch):
    """The flag is scoped to intensity data: crop_resample keeps its own dispatch."""
    seg = _blobs((20, 22, 18), 12)
    img = nib.Nifti1Image(seg, np.diag([1.5, 1.5, 1.5, 1]))
    called = _record_dispatch(monkeypatch)

    change_spacing(img, 1.0, target_shape=(30, 33, 27), order=1, dtype=np.uint8,
                   crop_resample=True, nr_cpus=1, torch_resample=True, device="cpu")

    assert called == ["resample_one_hot_crop"], called


def test_torch_resample_leaves_nearest_label_upsampling_on_scipy(monkeypatch):
    """order=0 is out of scope for the flag and keeps its own dispatch."""
    seg = _blobs((20, 22, 18), 12)
    img = nib.Nifti1Image(seg, np.diag([1.5, 1.5, 1.5, 1]))
    called = _record_dispatch(monkeypatch)

    change_spacing(img, 1.0, target_shape=(30, 33, 27), order=0, dtype=np.uint8,
                   nr_cpus=1, torch_resample=True, device="cpu")

    assert called == ["resample_img"], called


def test_torch_resample_still_handles_image_data(monkeypatch):
    """The other side of the gate: intensity data at order>0 is what the backend is for."""
    rng = np.random.default_rng(0)
    img = nib.Nifti1Image(rng.random((20, 22, 18)).astype(np.float32), np.diag([1.5, 1.5, 1.5, 1]))
    called = _record_dispatch(monkeypatch)

    change_spacing(img, 1.0, target_shape=(30, 33, 27), order=3, dtype=np.float32,
                   nr_cpus=1, torch_resample=True, device="cpu")

    assert called == ["resample_img_torch"], called


def test_one_hot_result_is_unchanged_by_the_torch_flag():
    """Outside its scope the flag is inert: same voxels, same affine."""
    seg = _blobs((20, 22, 18), 12)
    img = nib.Nifti1Image(seg, np.diag([1.5, 1.5, 1.5, 1]))
    kwargs = dict(target_shape=(30, 33, 27), order=1, dtype=np.uint8,
                  crop_resample=True, nr_cpus=1)
    stock = change_spacing(img, 1.0, **kwargs)
    flagged = change_spacing(img, 1.0, torch_resample=True, device="cpu", **kwargs)
    np.testing.assert_array_equal(np.asanyarray(stock.dataobj), np.asanyarray(flagged.dataobj))
    np.testing.assert_allclose(stock.affine, flagged.affine)


def _record_device(monkeypatch):
    """The device string change_spacing hands to the torch backend, without running it."""
    seen = {}

    def fake(data, new_shape, device="mps", order=3, anti_alias=False):
        seen["device"] = device
        return np.zeros(new_shape, dtype=np.float32)

    monkeypatch.setattr(resampling, "resample_img_torch", fake)
    return seen


def _intensity_image():
    rng = np.random.default_rng(0)
    return nib.Nifti1Image(rng.random((20, 22, 18)).astype(np.float32), np.diag([1.5, 1.5, 1.5, 1]))


def test_torch_resample_keeps_the_cuda_device_index(monkeypatch):
    """select_device() hands down a torch.device("cuda:N"), whose .type is only "cuda".
    Dropping the N resamples on whichever CUDA device happens to be current while inference
    runs on N - cross-device on a multi-GPU box, and on a shared cluster it reaches for a
    GPU this process was not allocated."""
    seen = _record_device(monkeypatch)

    change_spacing(_intensity_image(), 1.0, target_shape=(30, 33, 27), order=3,
                   dtype=np.float32, nr_cpus=1, torch_resample=True,
                   device=torch.device("cuda:1"))

    assert seen["device"] == "cuda:1"


@pytest.mark.parametrize("device, expected", [("mps", "mps"), ("cpu", "cpu"), ("gpu", "cpu")])
def test_torch_resample_normalizes_string_devices(monkeypatch, device, expected):
    """Plain strings pass through; anything the backend cannot name falls back to the CPU."""
    seen = _record_device(monkeypatch)

    change_spacing(_intensity_image(), 1.0, target_shape=(30, 33, 27), order=3,
                   dtype=np.float32, nr_cpus=1, torch_resample=True, device=device)

    assert seen["device"] == expected


@pytest.mark.parametrize("order", [1, 3])
@pytest.mark.parametrize("new_shape", [(30, 33, 27), (11, 9, 14)])
def test_resample_img_torch_reproduces_scipy_zoom(order, new_shape):
    """The backend applies scipy's own per-axis operators, so it is not an approximation
    of ndimage.zoom - it is ndimage.zoom, evaluated on torch."""
    rng = np.random.default_rng(0)
    vol = rng.random((20, 22, 18))

    got = resampling.resample_img_torch(vol, new_shape, device="cpu", order=order)
    zoom = tuple(n / o for n, o in zip(new_shape, vol.shape))
    want = ndimage.zoom(vol, zoom, mode="nearest", order=order)

    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-9)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an MPS device")
@pytest.mark.parametrize("order", [1, 3])
def test_resample_img_torch_matches_scipy_on_mps(order):
    """Same operators in float32 on the GPU: the residual is arithmetic, not sampling."""
    rng = np.random.default_rng(0)
    vol = rng.random((20, 22, 18))

    got = resampling.resample_img_torch(vol, (30, 33, 27), device="mps", order=order)
    want = ndimage.zoom(vol, (30 / 20, 33 / 22, 27 / 18), mode="nearest", order=order)

    np.testing.assert_allclose(got, want, rtol=0, atol=1e-5)


@pytest.mark.parametrize("order", [1, 3])
def test_change_spacing_agrees_with_the_scipy_path(order):
    """End to end through change_spacing: enabling the flag must not move the image."""
    rng = np.random.default_rng(0)
    img = nib.Nifti1Image(rng.random((20, 22, 18)) * 1000 - 500, np.diag([1.5, 1.5, 1.5, 1]))
    kwargs = dict(target_shape=(30, 33, 27), order=order, dtype=np.float32, nr_cpus=1)

    stock = change_spacing(img, 1.0, **kwargs)
    torched = change_spacing(img, 1.0, torch_resample=True, device="cpu", **kwargs)

    np.testing.assert_allclose(np.asanyarray(torched.dataobj), np.asanyarray(stock.dataobj),
                               rtol=0, atol=1e-4)
    np.testing.assert_allclose(torched.affine, stock.affine)


def test_resample_img_torch_needs_no_forked_nnunet():
    """The backend is numpy + scipy + torch. Requiring a fork of nnU-Net would make the
    flag unusable against a released nnunetv2 and untestable in CI."""
    import inspect

    source = inspect.getsource(resampling.resample_img_torch)
    assert "nnunetv2" not in source
