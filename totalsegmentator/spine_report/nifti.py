import os
from pathlib import Path

import numpy as np
import nibabel as nib
from PIL import Image


def rgb_array_to_structured(a: np.ndarray) -> np.ndarray:
    """ Converts a numpy array a with RGB channels in the last dimension to structured dtype removing the last dimension """
    # NOTE: use `from numpy.lib import recfunctions as rfn; rfn.structured_to_unstructured(report.get_data())` or similar to revert
    rgb_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])  # structured dtype
    return a.copy().view(dtype=rgb_dtype).reshape(a.shape[:-1])


def combine_as_nifti(tmp_dir, logger, ref_img):
    
    fns = [tmp_dir / "spine_report_frontpage.png"]

    # Reverse: frontpage at the top in nora
    # natural order: frontpage at the top in SECTRA PACS Viewer -> keep this order
    # fns = fns[::-1]

    # Get largest width and height
    max_width, max_height = 0, 0
    for fn in fns:
        width, height = Image.open(fn).size
        # logger.info(f"w x h: {width} x {height}")
        max_width = max(max_width, width)
        max_height = max(max_height, height)
    width, height = max_width+1, max_height+1

    logger.info(f"width: {width}, height: {height}")
    nii_img = np.zeros((width, height, 1, 3), dtype="uint8")  # [x, y, z, rgb]

    for idx, fn in enumerate(fns):
        img = Image.open(fn)
        img.load()  # load into memory  (otherwise lazy loading)
        img = np.asarray(img.convert("RGB"))
        img = img.transpose((1,0,2))[::-1,::-1,:]
        img_x, img_y, _ = img.shape
        # nii_img[:img_x, :img_y, idx, :] = img  # aligns to bottom
        nii_img[:img_x, -img_y:, idx, :] = img  # aligns to top

    nii_struct = rgb_array_to_structured(nii_img)

    # Infos around image orientation:
    #  (https://nipy.org/nibabel/nifti_images.html#the-sform-affine)
    #
    # sform_code:  (code | label | meaning)
    # 0 | unknown | sform not defined
    # 1 | scanner | RAS+ in scanner coordinates
    # 2 | aligned | RAS+ aligned to some other scan
    # 3 | talairach | RAS+ in Talairach atlas space
    # 4 | mni | RAS+ in MNI atlas space
    # 
    # Header contains 3 affines: sform, qform, fallback
    # Chooses sform (if not "unknown"), otherwise qform (if not "unknown"), otherwise fallback
    #
    # If setting affine=np.eye(4):
    #  The sform and qform codes will be initialised to 2 (aligned) and 0 (unknown)
    
    print(f"nii_struct.shape: {nii_struct.shape}")

    # Adapt affine to make it work in Sectra PACS Viewer:
    #   flip x- and y-axis in array and affine
    new_affine = np.eye(4)
    new_affine[0,0] = -1
    new_affine[1,1] = -1
    nii_struct = nii_struct[::-1,::-1,:]  # mirror along x- and y-axis
    img_out = nib.Nifti1Image(nii_struct, new_affine)
    # Set sform_code to scanner (instead of aligned which is default)
    # -> this does not change anything in Sectra Pacs -> not needed
    # img_out.set_sform(None, code=1)  # If you also want to clear out the sform matrix (not required but can be a good practice)
    # img_out.header['sform_code'] = 1

    # Sectra PACS Viewer settings maybe similar to Nora Viewing Mode "memory aligned?"

    return img_out
