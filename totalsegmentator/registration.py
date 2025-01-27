import time

import nibabel as nib
import numpy as np
# pip install antspyx
import ants

from totalsegmentator.resampling import change_spacing


def calc_transform(moving_image_nib, fixed_image_nib, transform_type="Rigid", aff_metric="mattes", 
                   resample=None, verbose=False):
    """
    transform_type: Rigid | Affine | SyN | see ants docs:
    
    https://antspy.readthedocs.io/en/latest/registration.html
    
    For elastic registration (SyN) maybe other return value is needed ??
    
    """
    if resample is not None:
        moving_image_nib = change_spacing(moving_image_nib, resample, dtype=np.uint8, order=0)
        fixed_image_nib = change_spacing(fixed_image_nib, resample, dtype=np.uint8, order=0)

    moving_image_data = moving_image_nib.get_fdata()
    fixed_image_data = fixed_image_nib.get_fdata()

    moving_image = ants.from_numpy(moving_image_data)
    fixed_image = ants.from_numpy(fixed_image_data)

    # fixed_image.plot(overlay=moving_image, title='Before Registration')
    mytx = ants.registration(fixed=fixed_image , 
                             moving=moving_image, 
                             type_of_transform=transform_type,
                             aff_metric=aff_metric,
                             verbose=verbose)
    
    if verbose:
        print(mytx)
    return mytx['fwdtransforms']


def apply_transform(moving_image_nib, fixed_image_nib, transform_list, resample=None, 
                    dtype=np.int16, order=1, interp="linear"):
    """
    Info: have to use the same resampling as in calc_transform (but did not try different ways yet)
    
    For multilabel important to set interp to "genericLabel"
    """
    if resample is not None:
        orig_spacing = fixed_image_nib.header.get_zooms()
        moving_image_nib = change_spacing(moving_image_nib, resample, dtype=dtype, order=order)
        fixed_image_nib = change_spacing(fixed_image_nib, resample, dtype=dtype, order=order)
        
    moving_image_data = moving_image_nib.get_fdata()
    fixed_image_data = fixed_image_nib.get_fdata()
    
    moving_image = ants.from_numpy(moving_image_data)
    fixed_image = ants.from_numpy(fixed_image_data)
    
    transformed_image = ants.apply_transforms(fixed=fixed_image, moving=moving_image, transformlist=transform_list,
                                              interpolator=interp)
    transformed_image = nib.Nifti1Image(transformed_image.numpy().astype(dtype), fixed_image_nib.affine)
    
    if resample is not None:
        transformed_image = change_spacing(transformed_image, orig_spacing, dtype=dtype, order=order)
    
    return transformed_image
