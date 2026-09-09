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

    Faster than resample_img_nnunet.
    Resample_img_nnunet maybe slightly better quality on CT (but not sure).

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


def resample_one_hot_crop(seg, new_shape, order=1, nr_cpus=1):
    """
    Resample a multilabel segmentation by resampling cropped one-hot masks.

    This avoids interpolation artifacts between label ids, like nnU-Net's one-hot
    resampling, but only evaluates each label in and around its foreground bbox.
    """
    if len(seg.shape) != 3:
        raise ValueError("resample_one_hot_crop only supports 3D segmentations.")

    old_shape = np.array(seg.shape)
    new_shape = np.array(new_shape).astype(int)
    scale = new_shape / old_shape
    labels = np.unique(seg)
    labels = labels[labels != 0]
    resampled = np.zeros(new_shape, dtype=seg.dtype)

    if len(labels) == 0:
        return resampled

    nr_cpus = multiprocessing.cpu_count() if nr_cpus == -1 else nr_cpus
    margin = max(2, order + 1)

    def _resample_label(label):
        foreground = np.where(seg == label)
        bbox_start = np.array([axis.min() for axis in foreground])
        bbox_end = np.array([axis.max() + 1 for axis in foreground])

        crop_start = np.maximum(bbox_start - margin, 0)
        crop_end = np.minimum(bbox_end + margin, old_shape)
        crop_slices = tuple(slice(start, end) for start, end in zip(crop_start, crop_end))
        label_crop = (seg[crop_slices] == label).astype(np.float32)

        target_start = np.maximum(np.floor(crop_start * scale).astype(int), 0)
        target_end = np.minimum(np.ceil(crop_end * scale).astype(int), new_shape)
        target_shape = target_end - target_start

        if np.any(target_shape <= 0):
            return label, None, None

        coord = []
        for start, end, old_dim, new_dim, crop_offset in zip(
            target_start, target_end, old_shape, new_shape, crop_start
        ):
            # Match the global resize coordinate system on the cropped label mask.
            coord.append((np.arange(start, end) + 0.5) * old_dim / new_dim - 0.5 - crop_offset)

        coord_map = np.array(np.meshgrid(*coord, indexing="ij"))
        label_resampled = ndimage.map_coordinates(
            label_crop, coord_map, order=order, mode="nearest", cval=0
        )
        target_slices = tuple(slice(start, end) for start, end in zip(target_start, target_end))
        return label, target_slices, label_resampled >= 0.5

    resampled_labels = Parallel(n_jobs=nr_cpus, prefer="threads")(
        delayed(_resample_label)(label) for label in labels
    )

    for label, target_slices, label_mask in resampled_labels:
        if target_slices is not None:
            resampled[target_slices][label_mask] = label

    return resampled


@functools.lru_cache(maxsize=128)
def _zoom_axis_operator(n_in, n_out, order, mode):
    """The exact 1-D operator of ``scipy.ndimage.zoom`` along one axis, ``(n_out, n_in)`` float64.

    ``zoom`` is linear and separable, so zooming an identity matrix along one axis *is* the
    operator for that axis - spline prefilter, boundary mode and the corner-aligned
    coordinate convention included, for any order. scipy builds these (they are tiny and
    cached); torch applies them. Exact by construction: there is no sampling or boundary
    handling reimplemented here to get subtly wrong.

    ``grid_mode=False`` is what pins the convention: the voxel-corner point grid
    ``change_spacing`` has always used. The same probe with ``grid_mode=True`` gives the
    voxel-center (half-pixel) operator - ``skimage.resize``, and nnU-Net's own resampler.
    The convention lives in the probe, not in anything below it.
    """
    if n_in == n_out:
        return np.eye(n_in)
    probe = ndimage.zoom(np.eye(n_in, dtype=np.float64), (1.0, n_out / n_in),
                         order=order, mode=mode, grid_mode=False)
    if probe.shape != (n_in, n_out):
        raise RuntimeError(f"scipy zoom probe produced {probe.shape}, expected {(n_in, n_out)}")
    return np.ascontiguousarray(probe.T)


def resample_img_torch(data, new_shape, device="mps", order=3):
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
        w = torch.as_tensor(_zoom_axis_operator(n_in, n_out, int(order), "nearest"),
                            device=dev, dtype=work)
        t = t.movedim(axis, -1)
        shape = t.shape
        t = (t.reshape(-1, shape[-1]) @ w.t()).reshape(*shape[:-1], w.shape[0]).movedim(-1, axis)
    out = t.cpu().numpy()
    del t
    return out


def resample_img_nnunet(data, mask=None, original_spacing=1.0, target_spacing=2.0,
                       order_data=3, order_seg=0):
    """
    Args:
        data: [x,y,z]
        mask: [x,y,z]
        original_spacing:
        target_spacing:

    Zoom = original_spacing / target_spacing
    (1 / 2 will reduce size by 50%)

    Returns:
        [x,y,z], [x,y,z]
    """
    from .resample_nnunet import resample_patient

    if type(original_spacing) is float:
        original_spacing = [original_spacing,] * 3
    original_spacing = np.array(original_spacing)

    if type(target_spacing) is float:
        target_spacing = [target_spacing,] * 3
    target_spacing = np.array(target_spacing)

    if data is not None:
        data = data.transpose((2, 0, 1))  # z is in front for nnUnet
        data = data[None, ...]  # [1,z,x,y], nnunet requires a channel dimension
    if mask is not None:
        mask = mask.transpose((2, 0, 1))
        mask = mask[None, ...]

    def move_last_elem_to_front(l):
        return np.array([l[2], l[0], l[1]])

    # if anisotropy too big, then will resample z axis separately with order=0
    original_spacing = move_last_elem_to_front(original_spacing)
    target_spacing = move_last_elem_to_front(target_spacing)
    data_res, mask_res = resample_patient(data, mask, original_spacing, target_spacing, 
                                          force_separate_z=None, order_data=order_data, order_seg=order_seg)

    if data is not None:
        data_res = data_res[0,...] # remove channel dimension
        data_res = data_res.transpose((1, 2, 0)) # Move z to back
    if mask is not None:
        mask_res = mask_res[0,...]
        mask_res = mask_res.transpose((1, 2, 0))
    return data_res, mask_res


def change_spacing(img_in, new_spacing=1.25, target_shape=None, order=0, nr_cpus=1,
                   nnunet_resample=False, crop_resample=False, dtype=None, remove_negative=False,
                   force_affine=None, device="cpu"):
    """
    Resample nifti image to the new spacing.

    3D intensity data (order > 0) is resampled with the torch backend on `device`.
    Labels (order 0, nnunet_resample, crop_resample) keep the scipy / one-hot paths.

    img_in: nifti image
    new_spacing: float or sequence of float
    target_shape: sequence of int (optional)
    order: resample order (optional)
    nnunet_resample: nnunet resampling will use order=0 sampling for z if very anisotropic. Sometimes results
                     in a little bit less blurry results
    crop_resample: one-hot resample each label only within its foreground bbox. Faster than nnunet_resample
                   for sparse multilabel segmentations.
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
    _one_hot = nnunet_resample or crop_resample  # both mean "resample labels, not intensities"
    if order == 0 and not _one_hot:
        data = np.asanyarray(img_in.dataobj)
    elif order != 0 and not _one_hot:
        data = img_in.get_fdata(dtype=np.float32)
    else:
        data = img_in.get_fdata()
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

    if nnunet_resample and crop_resample:
        raise ValueError("Only one of nnunet_resample and crop_resample can be enabled.")

    # Torch resampling backend (runs on `device`). Applies scipy's own zoom operators,
    # so results match the CPU scipy path. Works on CUDA, MPS, and CPU.
    #
    # Scope: forward image intensity data (order > 0). Labels stay on scipy / one-hot.
    _use_torch = data.ndim == 3 and order != 0 and not _one_hot
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
    elif nnunet_resample:
        # new_data, _ = resample_img_nnunet(data, None, img_spacing, new_spacing, order_data=order, order_seg=order)
        _, new_data = resample_img_nnunet(None, data, img_spacing, new_spacing, order_data=order, order_seg=order)
    elif crop_resample:
        output_shape = np.round(old_shape * zoom).astype(int)
        new_data = resample_one_hot_crop(data, output_shape, order=order, nr_cpus=nr_cpus)
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
