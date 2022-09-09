import time
from pathlib import Path
import numpy as np
import nibabel as nib

from scipy.ndimage import binary_dilation


def remove_outside_of_mask(seg_path, mask_path):
    """
    Remove all segmentations outside of mask.

    seg_path: path to nifti file
    mask_path: path to nifti file
    """
    seg_img = nib.load(seg_path)
    seg = seg_img.get_fdata()
    mask = nib.load(mask_path).get_fdata()
    mask = binary_dilation(mask, iterations=1)
    seg[mask == 0] = 0
    nib.save(nib.Nifti1Image(seg.astype(np.uint8), seg_img.affine), seg_path)
