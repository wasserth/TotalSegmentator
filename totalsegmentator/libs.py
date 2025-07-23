import io
import os
import contextlib
import sys
import random
import json
import time
import string
import shutil
import zipfile
from pathlib import Path

from tqdm import tqdm
import requests
import numpy as np
import nibabel as nib

from totalsegmentator.map_to_binary import class_map, class_map_5_parts, commercial_models
from totalsegmentator.config import get_totalseg_dir, get_weights_dir, is_valid_license, has_valid_license, has_valid_license_offline, get_version

"""
Helpers to suppress stdout prints from nnunet
https://stackoverflow.com/questions/2828953/silence-the-stdout-of-a-function-in-python-without-trashing-sys-stdout-and-resto
"""
class DummyFile:
    def write(self, x): pass
    def flush(self): pass

@contextlib.contextmanager
def nostdout(verbose=False):
    if not verbose:
        save_stdout = sys.stdout
        sys.stdout = DummyFile()
        yield
        sys.stdout = save_stdout
    else:
        yield


def download_model_with_license_and_unpack(task_name: str, config_dir: str):
    # Get License Number
    totalseg_dir = get_totalseg_dir()
    totalseg_config_file = totalseg_dir / "config.json"
    if totalseg_config_file.exists():
        with open(totalseg_config_file) as f:
            config = json.load(f)
        license_number = config["license_number"]
    else:
        print(f"ERROR: Could not find config file: {totalseg_config_file}")
        return False

    tempfile = config_dir / "tmp_download_file.zip"
    url = "http://backend.totalsegmentator.com:80/"

    # Download
    try:
        st = time.time()
        r = requests.post(url + "download_weights",
                          json={"license_number": license_number,
                                "task": task_name,
                                "version": get_version()},
                          timeout=300,
                          stream=True)
        r.raise_for_status()  # Raise an exception for HTTP errors (4xx, 5xx)

        if r.ok:
            with open(tempfile, "wb") as f:
                # without progress bar
                # f.write(r.content)

                total_size = int(r.headers.get('content-length', 0))
                progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading")
                for chunk in r.iter_content(chunk_size=8192 * 16):
                    progress_bar.update(len(chunk))
                    f.write(chunk)
                progress_bar.close()

            print("Download finished. Extracting...")
            with zipfile.ZipFile(config_dir / "tmp_download_file.zip", 'r') as zip_f:
                zip_f.extractall(config_dir)
            # print(f"  downloaded in {time.time()-st:.2f}s")
        else:
            if r.json()['status'] == "invalid_license":
                print(f"ERROR: Invalid license number ({license_number}). Please check your license number or contact support.")
                sys.exit(1)

    except Exception as e:
        raise e
    finally:
        if tempfile.exists():
            os.remove(tempfile)


def download_url_and_unpack(url, config_dir):

    # Not needed anymore since downloading from github assets (actually results in an error)
    # if "TOTALSEG_DISABLE_HTTP1" in os.environ and os.environ["TOTALSEG_DISABLE_HTTP1"]:
    #     print("Disabling HTTP/1.0")
    # else:
    #     import http.client
    #     # helps to solve incomplete read errors
    #     # https://stackoverflow.com/questions/37816596/restrict-request-to-only-ask-for-http-1-0-to-prevent-chunking-error
    #     http.client.HTTPConnection._http_vsn = 10
    #     http.client.HTTPConnection._http_vsn_str = 'HTTP/1.0'

    tempfile = config_dir / "tmp_download_file.zip"

    try:
        st = time.time()
        with open(tempfile, 'wb') as f:
            # session = requests.Session()  # making it slower

            with requests.get(url, stream=True) as r:
                r.raise_for_status()

                # With progress bar
                total_size = int(r.headers.get('content-length', 0))
                progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading")
                for chunk in r.iter_content(chunk_size=8192 * 16):
                    progress_bar.update(len(chunk))
                    f.write(chunk)
                progress_bar.close()

        print("Download finished. Extracting...")
        # call(['unzip', '-o', '-d', network_training_output_dir, tempfile])
        with zipfile.ZipFile(config_dir / "tmp_download_file.zip", 'r') as zip_f:
            zip_f.extractall(config_dir)
        # print(f"  downloaded in {time.time()-st:.2f}s")
    except Exception as e:
        raise e
    finally:
        if tempfile.exists():
            os.remove(tempfile)


def old_model_cleanup(config_dir: str):
    old_weights = [
        "nnUNet/3d_fullres/Task251_TotalSegmentator_part1_organs_1139subj",
        "nnUNet/3d_fullres/Task252_TotalSegmentator_part2_vertebrae_1139subj",
        "nnUNet/3d_fullres/Task253_TotalSegmentator_part3_cardiac_1139subj",
        "nnUNet/3d_fullres/Task254_TotalSegmentator_part4_muscles_1139subj",
        "nnUNet/3d_fullres/Task255_TotalSegmentator_part5_ribs_1139subj",
        "nnUNet/3d_fullres/Task256_TotalSegmentator_3mm_1139subj",
        "nnUNet/3d_fullres/Task258_lung_vessels_248subj",
        "nnUNet/3d_fullres/Task200_covid_challenge",
        "nnUNet/3d_fullres/Task201_covid",
        "nnUNet/3d_fullres/Task150_icb_v0",
        "nnUNet/3d_fullres/Task260_hip_implant_71subj",
        "nnUNet/3d_fullres/Task269_Body_extrem_6mm_1200subj",
        "nnUNet/3d_fullres/Task503_cardiac_motion",
        "nnUNet/3d_fullres/Task273_Body_extrem_1259subj",
        "nnUNet/3d_fullres/Task315_thoraxCT",
        "nnUNet/3d_fullres/Task008_HepaticVessel",
        "nnUNet/3d_fullres/Task417_heart_mixed_317subj",
        "nnUNet/3d_fullres/Task278_TotalSegmentator_part6_bones_1259subj",
        "nnUNet/3d_fullres/Task435_Heart_vessels_118subj",
        # "Dataset297_TotalSegmentator_total_3mm_1559subj",  # for >= v2.0.4
        "Dataset297_TotalSegmentator_total_3mm_1559subj_v204",  # for >= v2.0.5
        # "Dataset298_TotalSegmentator_total_6mm_1559subj",  # for >= v2.0.5
        "Dataset302_vertebrae_body_1559subj"
    ]

    for old_weight in old_weights:
        if (config_dir / old_weight).exists():
            shutil.rmtree(config_dir / old_weight)


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


def combine_masks(mask_dir, class_type):
    """
    Combine classes to masks

    mask_dir: directory of totalsegmetator masks
    class_type: ribs | vertebrae | vertebrae_ribs | lung | heart

    returns: nibabel image
    """
    rib_classes = [f"rib_left_{idx}" for idx in range(1, 13)] + [f"rib_right_{idx}" for idx in range(1, 13)]  # + ["sternum",]
    if class_type == "ribs":
        masks = rib_classes
    elif class_type == "vertebrae":
        masks = list(class_map_5_parts["class_map_part_vertebrae"].values())
    elif class_type == "vertebrae_ribs":
        masks = list(class_map_5_parts["class_map_part_vertebrae"].values()) + rib_classes
    elif class_type == "lung":
        masks = ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
                 "lung_middle_lobe_right", "lung_lower_lobe_right"]
    elif class_type == "lung_left":
        masks = ["lung_upper_lobe_left", "lung_lower_lobe_left"]
    elif class_type == "lung_right":
        masks = ["lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"]
    elif class_type == "pelvis":
        masks = ["femur_left", "femur_right", "hip_left", "hip_right"]
    elif class_type == "body":
        masks = ["body_trunc", "body_extremities"]

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

    return nib.Nifti1Image(combined, ref_img.affine)


def compress_nifti(file_in, file_out, dtype=np.int32, force_3d=True):
    img = nib.load(file_in)
    data = img.get_fdata()
    if force_3d and len(data.shape) > 3:
        print("Info: Input image contains more than 3 dimensions. Only keeping first 3 dimensions.")
        data = data[:,:,:,0]
    new_image = nib.Nifti1Image(data.astype(dtype), img.affine)
    nib.save(new_image, file_out)


def check_if_shape_and_affine_identical(img_1, img_2):

    max_diff = np.abs(img_1.affine - img_2.affine).max()
    if max_diff > 1e-5:
        print("Affine in:")
        print(img_1.affine)
        print("Affine out:")
        print(img_2.affine)
        print("Diff:")
        print(np.abs(img_1.affine-img_2.affine))
        print("WARNING: Output affine not equal to input affine. This should not happen.")

    if img_1.shape != img_2.shape:
        print("Shape in:")
        print(img_1.shape)
        print("Shape out:")
        print(img_2.shape)
        print("WARNING: Output shape not equal to input shape. This should not happen.")


def reorder_multilabel_like_v1(data, label_map_v2, label_map_v1):
    """
    Reorder a multilabel image from v2 to v1
    """
    label_map_v2_inv = {v: k for k, v in label_map_v2.items()}
    data_out = np.zeros(data.shape, dtype=np.uint8)
    for label_id, label_name in label_map_v1.items():
        if label_name in label_map_v2_inv:
            data_out[data == label_map_v2_inv[label_name]] = label_id
        # heart chambers are not in v2 anymore. The results seg will be empty for these classes
    return data_out
