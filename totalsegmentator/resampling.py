# pylint: disable=relative-beyond-top-level

import os
import time
import importlib

import numpy as np
import nibabel as nib
from scipy import ndimage
import psutil
from joblib import Parallel, delayed

cupy_available = importlib.util.find_spec("cupy") is not None
cucim_available = importlib.util.find_spec("cucim") is not None


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
        return ndimage.zoom(img[:, :, :, grad_idx], zoom, order=order)

    dim = len(img.shape)

    # Add dimensions to make each input 4D
    if dim == 2:
        img = img[..., None, None]
    if dim == 3:
        img = img[..., None]

    nr_cpus = psutil.cpu_count() if nr_cpus == -1 else nr_cpus
    img_sm = Parallel(n_jobs=nr_cpus)(delayed(_process_gradient)(grad_idx) for grad_idx in range(img.shape[3]))
    img_sm = np.array(img_sm).transpose(1, 2, 3, 0)  # grads channel was in front -> put to back
    # Remove added dimensions
    # img_sm = img_sm[:,:,:,0] if img_sm.shape[3] == 1 else img_sm  # remove channel dim if only 1 element
    if dim == 3:
        img_sm = img_sm[:,:,:,0]
    if dim == 2:
        img_sm = img_sm[:,:,0,0]
    return img_sm


def resample_img_cucim(img, zoom=0.5, order=0, nr_cpus=-1):
    """
    Completely speedup of resampling compare to non-gpu version not as big, because much time is lost in
    loading the file and then in copying to the GPU.

    For small image no significant speedup.
    For large images reducing resampling time by over 50%.

    On our slurm gpu cluster it is actually slower with cucim than without it.
    """
    import cupy as cp
    from cucim.skimage.transform import resize

    img = cp.asarray(img)  # slow
    new_shape = (np.array(img.shape) * zoom).round().astype(np.int32)
    resampled_img = resize(img, output_shape=new_shape, order=order, mode="edge", anti_aliasing=False)  # very fast
    resampled_img = cp.asnumpy(resampled_img)  # Alternative: img_arr = cp.float32(resampled_img.get())   # very fast
    return resampled_img


def resample_img_nnunet(data, mask=None, original_spacing=1.0, target_spacing=2.0):
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
    data_res, mask_res = resample_patient(data, mask, original_spacing, target_spacing, force_separate_z=None)

    data_res = data_res[0,...] # remove channel dimension
    data_res = data_res.transpose((1, 2, 0)) # Move z to back
    if mask is not None:
        mask_res = mask_res[0,...]
        mask_res = mask_res.transpose((1, 2, 0))
    return data_res, mask_res


def change_spacing(img_in, new_spacing=1.25, target_shape=None, order=0, nr_cpus=1,
                   nnunet_resample=False, dtype=None, remove_negative=False, force_affine=None):
    """
    Resample nifti image to the new spacing (uses resample_img() internally).

    img_in: nifti image
    new_spacing: float or sequence of float
    target_shape: sequence of int (optional)
    order: resample order (optional)
    nnunet_resample: nnunet resampling will use order=0 sampling for z if very anisotropic. Sometimes results
                     in a little bit less blurry results
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
    data = img_in.get_fdata()  # quite slow
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

    if nnunet_resample:
        new_data, _ = resample_img_nnunet(data, None, img_spacing, new_spacing)
    else:
        if cupy_available and cucim_available:
            new_data = resample_img_cucim(data, zoom=zoom, order=order, nr_cpus=nr_cpus)  # gpu resampling
        else:
            new_data = resample_img(data, zoom=zoom, order=order, nr_cpus=nr_cpus)  # cpu resampling

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
