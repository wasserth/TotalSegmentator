import os
import sys
import time
import shutil
import zipfile
from pathlib import Path
import subprocess
import platform
from tqdm import tqdm

import numpy as np
import nibabel as nib

from totalsegmentator.libs import get_config_dir


def download_dcm2niix():
    import urllib.request
    print("  Downloading dcm2niix...")

    if platform.system() == "Windows":
        url = "https://github.com/rordenlab/dcm2niix/releases/latest/download/dcm2niix_win.zip"
    elif platform.system() == "Darwin":  # Mac
        if platform.machine().startswith("arm") or platform.machine().startswith("aarch"):  # arm
            url = "https://github.com/rordenlab/dcm2niix/releases/latest/download/dcm2niix_mac_arm.pkg"
        else:  # intel
            url = "https://github.com/rordenlab/dcm2niix/releases/latest/download/dcm2niix_mac.zip"
    elif platform.system() == "Linux":
        url = "https://github.com/rordenlab/dcm2niix/releases/latest/download/dcm2niix_lnx.zip"
    else:
        raise ValueError("Unknown operating system. Can not download the right version of dcm2niix.")

    config_dir = get_config_dir()

    urllib.request.urlretrieve(url, config_dir / "dcm2niix.zip")
    with zipfile.ZipFile(config_dir / "dcm2niix.zip", 'r') as zip_ref:
        zip_ref.extractall(config_dir)

    # Give execution permission to the script
    os.chmod(config_dir / "dcm2niix", 0o755)

    # Clean up
    os.remove(config_dir / "dcm2niix.zip")
    os.remove(config_dir / "dcm2niibatch")


def dcm_to_nifti(input_path, output_path, verbose=False):
    """
    input_path: a directory of dicom slices
    output_path: a nifti file path
    """
    verbose_str = "" if verbose else "> /dev/null"

    config_dir = get_config_dir()
    dcm2niix = config_dir / "dcm2niix"

    if not dcm2niix.exists():
        download_dcm2niix()

    subprocess.call(f"{dcm2niix} -o {output_path.parent} -z y -f {output_path.name[:-7]} {input_path} {verbose_str}", shell=True)

    nii_files = list(output_path.parent.glob("*.nii.gz"))
    if len(nii_files) > 1:
        print("WARNING: Dicom to nifti resulted in several nifti files. Only using first one.")
        print([f.name for f in nii_files])
        for nii_file in nii_files[1:]:
            os.remove(nii_file)
        # todo: have to rename first file to not contain any counter which is automatically added by dcm2niix

    os.remove(str(output_path)[:-7] + ".json")


def save_mask_as_rtstruct(img_data, selected_classes, dcm_reference_file, output_path):
    """
    dcm_reference_file: a directory with dcm slices ??
    """
    from rt_utils import RTStructBuilder
    import logging
    logging.basicConfig(level=logging.WARNING)  # avoid messages from rt_utils

    # create new RT Struct - requires original DICOM
    rtstruct = RTStructBuilder.create_new(dicom_series_path=dcm_reference_file)

    # add mask to RT Struct
    for class_idx, class_name in tqdm(selected_classes.items()):
        binary_img = img_data == class_idx
        if binary_img.sum() > 0:  # only save none-empty images

            # rotate nii to match DICOM orientation
            binary_img = np.rot90(binary_img, 1, (0, 1))  # rotate segmentation in-plane

            # add segmentation to RT Struct
            rtstruct.add_roi(
                mask=binary_img,  # has to be a binary numpy array
                name=class_name
            )

    rtstruct.save(str(output_path))
