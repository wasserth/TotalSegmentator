import os
import contextlib
import sys
import shutil
import zipfile
from pathlib import Path

import requests
import numpy as np
import nibabel as nib

from totalsegmentator.map_to_binary import class_map


"""
Helpers to suppress stdout prints from nnunet
https://stackoverflow.com/questions/2828953/silence-the-stdout-of-a-function-in-python-without-trashing-sys-stdout-and-resto
"""
class DummyFile(object):
    def write(self, x): pass

@contextlib.contextmanager
def nostdout(verbose=False):
    if not verbose:
        save_stdout = sys.stdout
        sys.stdout = DummyFile()
        yield
        sys.stdout = save_stdout
    else:
        yield


def download_pretrained_weights(task_id):

    config_dir = Path.home() / ".totalsegmentator/nnunet/results/nnUNet/3d_fullres"
    config_dir.mkdir(exist_ok=True, parents=True)

    old_weights = [
        "Task223_my_test"
    ]

    if task_id == 251:
        weights_path = config_dir / "Task251_TotalSegmentator_part1_organs_1139subj"
        WEIGHTS_URL = "https://zenodo.org/record/6802342/files/Task251_TotalSegmentator_part1_organs_1139subj.zip?download=1"
    elif task_id == 252:
        weights_path = config_dir / "Task252_TotalSegmentator_part2_vertebrae_1139subj"
        WEIGHTS_URL = "https://zenodo.org/record/6802358/files/Task252_TotalSegmentator_part2_vertebrae_1139subj.zip?download=1"
    elif task_id == 253:
        weights_path = config_dir / "Task253_TotalSegmentator_part3_cardiac_1139subj"
        WEIGHTS_URL = "https://zenodo.org/record/6802360/files/Task253_TotalSegmentator_part3_cardiac_1139subj.zip?download=1"
    elif task_id == 254:
        weights_path = config_dir / "Task254_TotalSegmentator_part4_muscles_1139subj"
        WEIGHTS_URL = "https://zenodo.org/record/6802366/files/Task254_TotalSegmentator_part4_muscles_1139subj.zip?download=1"
    elif task_id == 255:
        weights_path = config_dir / "Task255_TotalSegmentator_part5_ribs_1139subj"
        WEIGHTS_URL = "https://zenodo.org/record/6802452/files/Task255_TotalSegmentator_part5_ribs_1139subj.zip?download=1"
    elif task_id == 256:
        weights_path = config_dir / "Task256_TotalSegmentator_3mm_1139subj"
        WEIGHTS_URL = "https://zenodo.org/record/6802052/files/Task256_TotalSegmentator_3mm_1139subj.zip?download=1"

    for old_weight in old_weights:
        if (config_dir / old_weight).exists():
            shutil.rmtree(config_dir / old_weight)

    if WEIGHTS_URL is not None and not weights_path.exists():
        print(f"Downloading pretrained weights for Task {task_id} (~230MB) ...")

        data = requests.get(WEIGHTS_URL).content
        with open(config_dir / "tmp_download_file.zip", "wb") as weight_file:
            weight_file.write(data)

        with zipfile.ZipFile(config_dir / "tmp_download_file.zip", 'r') as zip_f:
            zip_f.extractall(config_dir)
            print(config_dir)

        # delete tmp file
        (config_dir / "tmp_download_file.zip").unlink()


def setup_nnunet():
    # check if environment variable totalsegmentator_config is set
    if "TOTALSEG_WEIGHTS_PATH" in os.environ:
        weights_dir = os.environ["TOTALSEG_WEIGHTS_PATH"]
    else:
        config_dir = Path.home() / ".totalsegmentator"
        (config_dir / "nnunet/results/nnUNet/3d_fullres").mkdir(exist_ok=True, parents=True)
        weights_dir = config_dir / "nnunet/results"

    # This variables will only be active during the python script execution. Therefore
    # do not have to unset them in the end.
    os.environ["nnUNet_raw_data_base"] = str(weights_dir)  # not needed, just needs to be an existing directory
    os.environ["nnUNet_preprocessed"] = str(weights_dir)  # not needed, just needs to be an existing directory
    os.environ["RESULTS_FOLDER"] = str(weights_dir)


def combine_masks_to_multilabel_file(masks_dir, multilabel_file):
    """
    Generate one multilabel nifti file from a directory of single binary masks of each class.
    This multilabel file is needed to train a nnU-Net.

    masks_dir: path to directory containing all the masks for one subject
    multilabel_file: path of the output file (a nifti file)
    """
    masks_dir = Path(masks_dir)
    ref_img = nib.load(masks_dir / "liver.nii.gz")
    masks = class_map.values()
    img_out = np.zeros(ref_img.shape).astype(np.uint8)

    for idx, mask in enumerate(masks):
        if os.path.exists(f"{masks_dir}/{mask}.nii.gz"):
            img = nib.load(f"{masks_dir}/{mask}.nii.gz").get_fdata()
        else:
            print(f"Mask {mask} is missing. Filling with zeros.")
            img = np.zeros(ref_img.shape)
        img_out[img > 0.5] = idx+1

    nib.save(nib.Nifti1Image(img_out, ref_img.affine), multilabel_file)
