# pylint: disable=relative-beyond-top-level

import os
import time
import functools
import multiprocessing

import numpy as np
import nibabel as nib
from scipy import ndimage
from joblib import Parallel, delayed


def change_spacing_of_affine(affine, zoom=0.5):
    new_affine = np.copy(affine)
    for i in range(3):
        new_affine[i, i] /= zoom
    return new_affine


def resample_img(img, zoom=0.5, order=0, nr_cpus=-1):
    """
    img: [x,y,z,(t)]
    zoom: 0.5 will halfen the image resolution (make image smaller)

    Resize numpy image array to new size.

    Works for 2D and 3D and 4D images.
    """
    def _process_gradient(grad_idx):
        return ndimage.zoom(img[:, :, :, grad_idx], zoom, mode="nearest", order=order)

    dim = len(img.shape)

    # Add dimensions to make each input 4D
    if dim == 2:
        img = img[..., None, None]
    if dim == 3:
        img = img[..., None]

    nr_cpus = multiprocessing.cpu_count() if nr_cpus == -1 else nr_cpus
    img_sm = Parallel(n_jobs=nr_cpus)(delayed(_process_gradient)(grad_idx) for grad_idx in range(img.shape[3]))
    img_sm = np.array(img_sm).transpose(1, 2, 3, 0)  # grads channel was in front -> put to back
    # Remove added dimensions
    # img_sm = img_sm[:,:,:,0] if img_sm.shape[3] == 1 else img_sm  # remove channel dim if only 1 element
    if dim == 3:
        img_sm = img_sm[:,:,:,0]
    if dim == 2:
        img_sm = img_sm[:,:,0,0]
    return img_sm


@functools.lru_cache(maxsize=128)
def _zoom_axis_operator(n_in, n_out, order, mode, grid_mode=False):
    """The exact 1-D operator of ``scipy.ndimage.zoom`` along one axis, ``(n_out, n_in)`` float64.

    ``zoom`` is linear and separable, so zooming an identity matrix along one axis *is* the
    operator for that axis - spline prefilter, boundary mode and the corner-aligned
    coordinate convention included, for any order. scipy builds these (they are tiny and
    cached); torch applies them. Exact by construction: there is no sampling or boundary
    handling reimplemented here to get subtly wrong.

    ``grid_mode`` pins the convention: False is the voxel-corner point grid ``change_spacing``
    has always used; True gives the voxel-center (half-pixel) operator - ``skimage.resize``,
    and nnU-Net's own resampler. The convention lives in the probe, not in anything below it.
    """
    if n_in == n_out:
        return np.eye(n_in)
    probe = ndimage.zoom(np.eye(n_in, dtype=np.float64), (1.0, n_out / n_in),
                         order=order, mode=mode, grid_mode=grid_mode)
    if probe.shape != (n_in, n_out):
        raise RuntimeError(f"scipy zoom probe produced {probe.shape}, expected {(n_in, n_out)}")
    return np.ascontiguousarray(probe.T)


def resample_img_torch(data, new_shape, device="mps", order=3, grid_mode=False):
    """GPU resample of a 3D intensity array [x,y,z] to ``new_shape`` on MPS / CUDA / CPU.

    The torch analogue of ``resample_img``. It does not
    reimplement ``scipy.ndimage.zoom``; it applies scipy's own per-axis operators
    (:func:`_zoom_axis_operator`) as matmuls on ``device``, so the result matches
    ``ndimage.zoom(order=order, mode="nearest")`` to float precision on CPU and ~1e-6
    relative on MPS/CUDA. Needs nothing beyond numpy, scipy and torch.

    Intensity data only.

    NIfTI volumes are typically Fortran-contiguous. Do not ``ascontiguousarray`` them on
    the host first: that is a full F→C transpose of the volume. Torch accepts F-order
    arrays; ``movedim`` / matmul then run with the existing strides (or a cheap device
    copy), which is the same numeric result.
    """
    import torch

    arr = np.asarray(data)
    if any(stride < 0 for stride in arr.strides):  # a flipped view (reorientation); torch cannot wrap those
        arr = arr.copy()
    if arr.ndim != 3:
        raise ValueError(f"expected a 3-D volume; got shape {arr.shape}")
    new_shape = tuple(int(s) for s in new_shape)
    if len(new_shape) != 3:
        raise ValueError(f"new_shape must have 3 entries; got {new_shape}")

    dev = torch.device(device)
    # float64 is unsupported on MPS and unnecessary here: the operators are float32-exact
    # to ~1e-7 relative, far below the intensity quantization these volumes carry.
    work = torch.float64 if (dev.type == "cpu" and arr.dtype == np.float64) else torch.float32
    t = torch.as_tensor(arr, device=dev, dtype=work)
    for axis in range(3):
        n_in, n_out = int(t.shape[axis]), new_shape[axis]
        if n_in == n_out:
            continue
        w = torch.as_tensor(_zoom_axis_operator(n_in, n_out, int(order), "nearest", grid_mode),
                            device=dev, dtype=work)
        t = t.movedim(axis, -1)
        shape = t.shape
        t = (t.reshape(-1, shape[-1]) @ w.t()).reshape(*shape[:-1], w.shape[0]).movedim(-1, axis)
    out = t.cpu().numpy()
    del t
    return out


def change_spacing(img_in, new_spacing=1.25, target_shape=None, order=0, nr_cpus=1,
                   dtype=None, remove_negative=False, force_affine=None, device="cpu"):
    """
    Resample nifti image to the new spacing.

    3D intensity data (order > 0) is resampled with the torch backend on `device`.
    Labels (order 0) keep the scipy path.

    img_in: nifti image
    new_spacing: float or sequence of float
    target_shape: sequence of int (optional)
    order: resample order (optional)
    dtype: output datatype
    remove_negative: set all negative values to 0. Useful if resampling introduced negative values.
    force_affine: if you pass an affine then this will be used for the output image (useful if you have to make sure
                  that the resampled has identical affine to some other image. In this case also set target_shape.)

    Works for 2D and 3D and 4D images.

    If downsampling an image and then upsampling again to original resolution the resulting image can have
    a shape which is +-1 compared to original shape, because of rounding of the shape to int.
    To avoid this the exact output shape can be provided. Then new_spacing will be ignored and the exact
    spacing will be calculated which is needed to get to target_shape.
    In this case however the calculated spacing can be slightly different from the desired new_spacing. This will
    result in a slightly different affine. To avoid this the desired affine can be written by force with "force_affine".

    Note: Only works properly if affine is all 0 except for diagonal and offset (=no rotation and sheering)
    """
    # Nearest-neighbor interpolation with scipy does not need float64 input. Keeping label
    # maps in their native dtype avoids a full-volume conversion and cuts peak RAM.
    # Torch intensity resampling is float32 on GPU/MPS (and on CPU unless the input is
    # already float64), so load float32 and skip a 2x-larger float64 buffer.
    if order == 0:
        data = np.asanyarray(img_in.dataobj)
    else:
        data = img_in.get_fdata(dtype=np.float32)
    old_shape = np.array(data.shape)
    img_spacing = np.array(img_in.header.get_zooms())

    if len(img_spacing) == 4:
        img_spacing = img_spacing[:3]  # for 4D images only use spacing of first 3 dims

    if type(new_spacing) is float:
        new_spacing = [new_spacing,] * 3   # for 3D and 4D
    new_spacing = np.array(new_spacing)

    if len(old_shape) == 2:
        img_spacing = np.array(list(img_spacing) + [new_spacing[2],])

    if target_shape is not None:
        # Find the right zoom to exactly reach the target_shape.
        # We also have to adapt the spacing to this new zoom.
        zoom = np.array(target_shape) / old_shape
        new_spacing = img_spacing / zoom
    else:
        zoom = img_spacing / new_spacing

    if np.array_equal(img_spacing, new_spacing):
        # print("Input spacing is equal to new spacing. Return image without resampling.")
        return img_in

    # copy very important; otherwise new_affine changes will also be in old affine
    new_affine = np.copy(img_in.affine)

    # This is only correct if all off-diagonal elements are 0
    # new_affine[0, 0] = new_spacing[0] if img_in.affine[0, 0] > 0 else -new_spacing[0]
    # new_affine[1, 1] = new_spacing[1] if img_in.affine[1, 1] > 0 else -new_spacing[1]
    # new_affine[2, 2] = new_spacing[2] if img_in.affine[2, 2] > 0 else -new_spacing[2]

    # This is the proper solution
    # Scale each column vector by the zoom of this dimension
    new_affine = np.copy(img_in.affine)
    new_affine[:3, 0] = new_affine[:3, 0] / zoom[0]
    new_affine[:3, 1] = new_affine[:3, 1] / zoom[1]
    new_affine[:3, 2] = new_affine[:3, 2] / zoom[2]

    # Just for information: How to get spacing from affine with rotation:
    # Calc length of each column vector:
    # vecs = affine[:3, :3]
    # spacing = tuple(np.sqrt(np.sum(vecs ** 2, axis=0)))

    # Torch resampling backend (runs on `device`). Applies scipy's own zoom operators,
    # so results match the CPU scipy path. Works on CUDA, MPS, and CPU.
    #
    # Scope: image intensity data (order > 0). Labels stay on scipy.
    _use_torch = data.ndim == 3 and order != 0
    if _use_torch:
        # Keep the device INDEX. select_device() hands down a torch.device("cuda:N"), whose
        # .type is just "cuda" - resampling on that would land on whichever CUDA device is
        # current while inference runs on N: cross-device on a multi-GPU box, and on a
        # shared cluster it reaches for a GPU this process was not allocated.
        _dev = str(device)
        _kind = device.type if hasattr(device, "type") else _dev.split(":")[0]
        if _kind not in ("mps", "cuda", "cpu"):
            _dev = "cpu"
        if target_shape is not None:
            new_shape = tuple(int(s) for s in target_shape)
        else:
            new_shape = tuple(int(round(o * z)) for o, z in zip(old_shape, zoom))
        new_data = resample_img_torch(data, new_shape, device=_dev, order=order)
    else:
        new_data = resample_img(data, zoom=zoom, order=order, nr_cpus=nr_cpus)

    if remove_negative:
        new_data[new_data < 1e-4] = 0

    if dtype is not None:
        new_data = new_data.astype(dtype)

    if force_affine is not None:
        new_affine = force_affine

    return nib.Nifti1Image(new_data, new_affine)


# def resample_img_nifti(file_in, file_out, spacing, order, dtype):
#     img_in = nib.load(file_in)
#     img_out = change_spacing(img_in, spacing, order=order, dtype=dtype)
#     nib.save(img_out, file_out)


# if __name__ == "__main__":
#     args = sys.argv[1:]
#     file_in = Path(args[0])
#     file_out = Path(args[1])

#     # spacing in mm
#     x = float(args[2])
#     y = float(args[3])
#     z = float(args[4])

#     order = int(args[5])  # use 0 for binary masks and 3 for continuous images
#     dtype = str(args[6])
#     if dtype == "int16":
#         dtype = np.int16
#     elif dtype == "int32":
#         dtype = np.int32
#     elif dtype == "float32":
#         dtype = np.float32
#     elif dtype == "uint8":
#         dtype = np.uint8
#     else:
#         raise ValueError("dtype must be one of int16, int32, float32, or uint8")

#     resample_img_nifti(file_in, file_out, [x, y, z], order, dtype)
