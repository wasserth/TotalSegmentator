import io
import os
import contextlib
import sys
import time
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


# def download_url(url, save_path, chunk_size=128):
#     r = requests.get(url, stream=True)
#     with open(save_path, 'wb') as f:
#         for chunk in r.iter_content(chunk_size=chunk_size):
#             f.write(chunk)


def download_url_and_unpack(url, config_dir):
    import http.client
    # helps to solve incomplete read erros
    # https://stackoverflow.com/questions/37816596/restrict-request-to-only-ask-for-http-1-0-to-prevent-chunking-error
    http.client.HTTPConnection._http_vsn = 10
    http.client.HTTPConnection._http_vsn_str = 'HTTP/1.0'

    tempfile = config_dir / "tmp_download_file.zip"

    try:
        st = time.time()
        with open(tempfile, 'wb') as f:
            # session = requests.Session()  # making it slower
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=8192 * 16):
                    # If you have chunk encoded response uncomment if
                    # and set chunk_size parameter to None.
                    # if chunk:
                    f.write(chunk)

        print("Download finished. Extracting...")
        # call(['unzip', '-o', '-d', network_training_output_dir, tempfile])
        with zipfile.ZipFile(config_dir / "tmp_download_file.zip", 'r') as zip_f:
            zip_f.extractall(config_dir)
        print(f"  downloaded in {time.time()-st:.2f}s")
    except Exception as e:
        raise e
    finally:
        if tempfile.exists():
            os.remove(tempfile)


def download_pretrained_weights(task_id):

    config_dir = Path.home() / ".totalsegmentator/nnunet/results/nnUNet"
    (config_dir / "3d_fullres").mkdir(exist_ok=True, parents=True)
    (config_dir / "2d").mkdir(exist_ok=True, parents=True)

    old_weights = [
        "Task223_my_test"
    ]

    if task_id == 251:
        config_dir = config_dir / "3d_fullres"
        weights_path = config_dir / "Task251_TotalSegmentator_part1_organs_1139subj"
        WEIGHTS_URL = "https://zenodo.org/record/6802342/files/Task251_TotalSegmentator_part1_organs_1139subj.zip?download=1"
    elif task_id == 252:
        config_dir = config_dir / "3d_fullres"
        weights_path = config_dir / "Task252_TotalSegmentator_part2_vertebrae_1139subj"
        WEIGHTS_URL = "https://zenodo.org/record/6802358/files/Task252_TotalSegmentator_part2_vertebrae_1139subj.zip?download=1"
    elif task_id == 253:
        config_dir = config_dir / "3d_fullres"
        weights_path = config_dir / "Task253_TotalSegmentator_part3_cardiac_1139subj"
        WEIGHTS_URL = "https://zenodo.org/record/6802360/files/Task253_TotalSegmentator_part3_cardiac_1139subj.zip?download=1"
    elif task_id == 254:
        config_dir = config_dir / "3d_fullres"
        weights_path = config_dir / "Task254_TotalSegmentator_part4_muscles_1139subj"
        WEIGHTS_URL = "https://zenodo.org/record/6802366/files/Task254_TotalSegmentator_part4_muscles_1139subj.zip?download=1"
    elif task_id == 255:
        config_dir = config_dir / "3d_fullres"
        weights_path = config_dir / "Task255_TotalSegmentator_part5_ribs_1139subj"
        WEIGHTS_URL = "https://zenodo.org/record/6802452/files/Task255_TotalSegmentator_part5_ribs_1139subj.zip?download=1"
    elif task_id == 256:
        config_dir = config_dir / "3d_fullres"
        weights_path = config_dir / "Task256_TotalSegmentator_3mm_1139subj"
        WEIGHTS_URL = "https://zenodo.org/record/6802052/files/Task256_TotalSegmentator_3mm_1139subj.zip?download=1"
    elif task_id == 258:
        config_dir = config_dir / "3d_fullres"
        weights_path = config_dir / "Task258_lung_vessels_248subj"
        WEIGHTS_URL = "https://zenodo.org/record/7064718/files/Task258_lung_vessels_248subj.zip?download=1"
    elif task_id == 200:
        config_dir = config_dir / "3d_fullres"
        weights_path = config_dir / "Task200_covid_challenge"
        WEIGHTS_URL = "TODO"
    # elif task_id == 152:
    #     config_dir = config_dir / "2d"
    #     weights_path = config_dir / "Task152_icbbig_TN"
    #     WEIGHTS_URL = "TODO"
    elif task_id == 150:
        config_dir = config_dir / "3d_fullres"
        weights_path = config_dir / "Task150_icb_v0"
        WEIGHTS_URL = "https://zenodo.org/record/7079161/files/Task150_icb_v0.zip?download=1"

    for old_weight in old_weights:
        if (config_dir / old_weight).exists():
            shutil.rmtree(config_dir / old_weight)

    if WEIGHTS_URL is not None and not weights_path.exists():
        print(f"Downloading pretrained weights for Task {task_id} (~230MB) ...")

        # r = requests.get(WEIGHTS_URL)
        # with zipfile.ZipFile(io.BytesIO(r.content)) as zip_f:
        #     zip_f.extractall(config_dir)
        #     print(f"Saving to: {config_dir}")

        # download_url(WEIGHTS_URL, config_dir / "tmp_download_file.zip")
        # with zipfile.ZipFile(config_dir / "tmp_download_file.zip", 'r') as zip_f:
        #     zip_f.extractall(config_dir)
        #     print(config_dir)
        # delete tmp file
        # (config_dir / "tmp_download_file.zip").unlink()

        download_url_and_unpack(WEIGHTS_URL, config_dir)


def setup_nnunet():
    # check if environment variable totalsegmentator_config is set
    if "TOTALSEG_WEIGHTS_PATH" in os.environ:
        weights_dir = os.environ["TOTALSEG_WEIGHTS_PATH"]
    else:
        config_dir = Path.home() / ".totalsegmentator"
        (config_dir / "nnunet/results/nnUNet/3d_fullres").mkdir(exist_ok=True, parents=True)
        (config_dir / "nnunet/results/nnUNet/2d").mkdir(exist_ok=True, parents=True)
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
    masks = class_map["total"].values()
    img_out = np.zeros(ref_img.shape).astype(np.uint8)

    for idx, mask in enumerate(masks):
        if os.path.exists(f"{masks_dir}/{mask}.nii.gz"):
            img = nib.load(f"{masks_dir}/{mask}.nii.gz").get_fdata()
        else:
            print(f"Mask {mask} is missing. Filling with zeros.")
            img = np.zeros(ref_img.shape)
        img_out[img > 0.5] = idx+1

    nib.save(nib.Nifti1Image(img_out, ref_img.affine), multilabel_file)


def combine_masks(mask_dir, output, class_type):
    """
    Combine classes to masks

    mask_dir: directory of totalsegmetator masks
    output: output path
    class_type: ribs | vertebrae | vertebrae_ribs | lung | heart
    """
    if class_type == "ribs":
        masks = list(class_map_5_parts["class_map_part_ribs"].values())
    elif class_type == "vertebrae":
        masks = list(class_map_5_parts["class_map_part_vertebrae"].values())
    elif class_type == "vertebrae_ribs":
        masks = list(class_map_5_parts["class_map_part_vertebrae"].values()) + list(class_map_5_parts["class_map_part_ribs"].values())
    elif class_type == "lung":
        masks = ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
                 "lung_middle_lobe_right", "lung_lower_lobe_right"]
    elif class_type == "lung_left":
        masks = ["lung_upper_lobe_left", "lung_lower_lobe_left"]
    elif class_type == "lung_right":
        masks = ["lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"]
    elif class_type == "heart":
        masks = ["heart_myocardium", "heart_atrium_left", "heart_ventricle_left",
                 "heart_atrium_right", "heart_ventricle_right"]

    ref_img = None
    for mask in masks:
        if (mask_dir / f"{mask}.nii.gz").exists():
            ref_img = nib.load(mask_dir / f"{masks[0]}.nii.gz")
        else:
            raise ValueError(f"Could not find {mask_dir / mask}.nii.gz. Did you run TotalSegmentator successfully?")

    combined = np.zeros(ref_img.shape, dtype=np.uint8)
    for idx, mask in enumerate(masks):
        if (mask_dir / f"{mask}.nii.gz").exists():
            img = nib.load(mask_dir / f"{mask}.nii.gz").get_fdata()
            combined[img > 0.5] = 1

    nib.save(nib.Nifti1Image(combined, ref_img.affine), output)


def compress_nifti(file_in, file_out, dtype=np.int32):
    img = nib.load(file_in)
    data = img.get_fdata()
    new_image = nib.Nifti1Image(data.astype(dtype), img.affine)
    nib.save(new_image, file_out)
