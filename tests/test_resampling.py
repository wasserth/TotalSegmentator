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
    """crop_resample keeps its own dispatch; torch is for intensity data."""
    seg = _blobs((20, 22, 18), 12)
    img = nib.Nifti1Image(seg, np.diag([1.5, 1.5, 1.5, 1]))
    called = _record_dispatch(monkeypatch)

    change_spacing(img, 1.0, target_shape=(30, 33, 27), order=1, dtype=np.uint8,
                   crop_resample=True, nr_cpus=1, device="cpu")

    assert called == ["resample_one_hot_crop"], called


def test_torch_resample_leaves_nearest_label_upsampling_on_scipy(monkeypatch):
    """order=0 is out of scope and keeps its own dispatch."""
    seg = _blobs((20, 22, 18), 12)
    img = nib.Nifti1Image(seg, np.diag([1.5, 1.5, 1.5, 1]))
    called = _record_dispatch(monkeypatch)

    change_spacing(img, 1.0, target_shape=(30, 33, 27), order=0, dtype=np.uint8,
                   nr_cpus=1, device="cpu")

    assert called == ["resample_img"], called


def test_torch_resample_handles_image_data(monkeypatch):
    """Intensity data at order>0 always uses the torch backend."""
    rng = np.random.default_rng(0)
    img = nib.Nifti1Image(rng.random((20, 22, 18)).astype(np.float32), np.diag([1.5, 1.5, 1.5, 1]))
    called = _record_dispatch(monkeypatch)

    change_spacing(img, 1.0, target_shape=(30, 33, 27), order=3, dtype=np.float32,
                   nr_cpus=1, device="cpu")

    assert called == ["resample_img_torch"], called


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
                   dtype=np.float32, nr_cpus=1, device=torch.device("cuda:1"))

    assert seen["device"] == "cuda:1"


@pytest.mark.parametrize("device, expected", [("mps", "mps"), ("cpu", "cpu"), ("gpu", "cpu")])
def test_torch_resample_normalizes_string_devices(monkeypatch, device, expected):
    """Plain strings pass through; anything the backend cannot name falls back to the CPU."""
    seen = _record_device(monkeypatch)

    change_spacing(_intensity_image(), 1.0, target_shape=(30, 33, 27), order=3,
                   dtype=np.float32, nr_cpus=1, device=device)

    assert seen["device"] == expected


@pytest.mark.parametrize("order", [1, 3])
def test_resample_img_torch_fortran_layout_matches_c_layout(order):
    """NIfTI data is F-contiguous. Forcing C-order on the host is a full-volume transpose
    and is not required: the same operators on F-order input must match C-order input."""
    rng = np.random.default_rng(0)
    vol_c = np.ascontiguousarray(rng.random((20, 22, 18)))
    vol_f = np.asfortranarray(vol_c)
    assert vol_c.flags.c_contiguous and vol_f.flags.f_contiguous
    new_shape = (30, 33, 27)

    got_c = resampling.resample_img_torch(vol_c, new_shape, device="cpu", order=order)
    got_f = resampling.resample_img_torch(vol_f, new_shape, device="cpu", order=order)

    np.testing.assert_array_equal(got_c, got_f)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("order", [1, 3])
def test_resample_img_torch_fortran_layout_matches_c_layout_cuda(order):
    rng = np.random.default_rng(0)
    vol_c = np.ascontiguousarray(rng.random((20, 22, 18), dtype=np.float32))
    vol_f = np.asfortranarray(vol_c)
    new_shape = (30, 33, 27)

    got_c = resampling.resample_img_torch(vol_c, new_shape, device="cuda:0", order=order)
    got_f = resampling.resample_img_torch(vol_f, new_shape, device="cuda:0", order=order)

    np.testing.assert_allclose(got_c, got_f, rtol=0, atol=0)


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
def test_change_spacing_agrees_with_ndimage_zoom(order):
    """End to end through change_spacing: torch path matches scipy.ndimage.zoom.

    Intensity data is loaded as float32, so the residual vs scipy's float64 zoom is
    arithmetic, not sampling. Tighter agreement is covered by
    test_resample_img_torch_reproduces_scipy_zoom (float64 on CPU).
    """
    rng = np.random.default_rng(0)
    vol = rng.random((20, 22, 18)) * 1000 - 500
    img = nib.Nifti1Image(vol, np.diag([1.5, 1.5, 1.5, 1]))
    new_shape = (30, 33, 27)

    got = change_spacing(img, 1.0, target_shape=new_shape, order=order,
                         dtype=np.float32, nr_cpus=1, device="cpu")
    want = ndimage.zoom(vol, tuple(n / o for n, o in zip(new_shape, vol.shape)),
                        mode="nearest", order=order).astype(np.float32)

    np.testing.assert_allclose(np.asanyarray(got.dataobj), want, rtol=0, atol=1e-3)
    np.testing.assert_allclose(got.affine, np.diag([1.0, 1.0, 1.0, 1]))


def test_resample_img_torch_needs_no_forked_nnunet():
    """The backend is numpy + scipy + torch. Requiring a fork of nnU-Net would make it
    unusable against a released nnunetv2 and untestable in CI."""
    import inspect

    source = inspect.getsource(resampling.resample_img_torch)
    assert "nnunetv2" not in source
