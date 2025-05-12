import sys
import os
from pathlib import Path
import time
import textwrap
from typing import Union
import tempfile

import numpy as np
import nibabel as nib
from nibabel.nifti1 import Nifti1Image
import torch
from totalsegmentator.statistics import get_basic_statistics, get_radiomics_features_for_entire_dir
from totalsegmentator.libs import download_pretrained_weights
from totalsegmentator.config import setup_nnunet, setup_totalseg, increase_prediction_counter
from totalsegmentator.config import send_usage_stats, set_license_number, has_valid_license_offline
from totalsegmentator.config import get_config_key, set_config_key
from totalsegmentator.map_to_binary import class_map
from totalsegmentator.map_to_total import map_to_total
import re


def validate_device_type_api(value):
    valid_strings = ["gpu", "cpu", "mps"]
    if value in valid_strings:
        return value

    # Check if the value matches the pattern "gpu:X" where X is an integer
    pattern = r"^gpu:(\d+)$"
    match = re.match(pattern, value)
    if match:
        device_id = int(match.group(1))
        return value

    raise ValueError(
        f"Invalid device type: '{value}'. Must be 'gpu', 'cpu', 'mps', or 'gpu:X' where X is an integer representing the GPU device ID.")


def convert_device_to_cuda(device):
    if device in ["cpu", "mps", "gpu"]:
        return device
    else:  # gpu:X
        return f"cuda:{device.split(':')[1]}"


def select_device(device):
    device = convert_device_to_cuda(device)

    # available devices: gpu | cpu | mps | gpu:1, gpu:2, etc.
    if device == "gpu": 
        device = "cuda"
    if device.startswith("cuda"): 
        if device == "cuda": device = "cuda:0"
        if not torch.cuda.is_available():
            print("No GPU detected. Running on CPU. This can be very slow. The '--fast' or the `--roi_subset` option can help to reduce runtime.")
            device = "cpu"
        else:
            device_id = int(device[5:])
            if device_id < torch.cuda.device_count():
                device = torch.device(device)
            else:
                print("Invalid GPU config, running on the CPU")
                device = "cpu"
    return device


def show_license_info():
    status, message = has_valid_license_offline()
    if status == "missing_license":
        # textwarp needed to remove the indentation of the multiline string
        print(textwrap.dedent("""\
              In contrast to the other tasks this task is not openly available.
              It requires a license. For non-commercial usage a free license can be
              acquired here:
              https://backend.totalsegmentator.com/license-academic/

              For commercial usage contact: jakob.wasserthal@usb.ch
              """))
        sys.exit(1)
    elif status == "invalid_license":
        print(message)
        sys.exit(1)
    elif status == "missing_config_file":
        print(message)
        sys.exit(1)


def totalsegmentator(input: Union[str, Path, Nifti1Image], output: Union[str, Path, None]=None, ml=False, nr_thr_resamp=1, nr_thr_saving=6,
                     fast=False, nora_tag="None", preview=False, task="total", roi_subset=None,
                     statistics=False, radiomics=False, crop_path=None, body_seg=False,
                     force_split=False, output_type="nifti", quiet=False, verbose=False, test=0,
                     skip_saving=False, device="gpu", license_number=None,
                     statistics_exclude_masks_at_border=True, no_derived_masks=False,
                     v1_order=False, fastest=False, roi_subset_robust=None, stats_aggregation="mean",
                     remove_small_blobs=False, statistics_normalized_intensities=False, 
                     robust_crop=False, higher_order_resampling=False, save_probabilities=None):
    """
    Run TotalSegmentator from within python.

    For explanation of the arguments see description of command line
    arguments in bin/TotalSegmentator.

    Return: multilabel Nifti1Image
    """
    if not isinstance(input, Nifti1Image):
        input = Path(input)

    if output is not None:
        output = Path(output)
    else:
        if radiomics:
            raise ValueError("Output path is required for radiomics.")

    nora_tag = "None" if nora_tag is None else nora_tag

    # Store initial torch settings
    initial_cudnn_benchmark = torch.backends.cudnn.benchmark
    initial_num_threads = torch.get_num_threads()

    validate_device_type_api(device)
    device = select_device(device)
    if verbose: print(f"Using Device: {device}")
    
    if output_type == "dicom":
        try:
            from rt_utils import RTStructBuilder
        except ImportError:
            raise ImportError("rt_utils is required for output_type='dicom'. Please install it with 'pip install rt_utils'.")

    if not quiet:
        print("\nIf you use this tool please cite: https://pubs.rsna.org/doi/10.1148/ryai.230024\n")

    setup_nnunet()
    setup_totalseg()
    if license_number is not None:
        set_license_number(license_number)

    if not get_config_key("statistics_disclaimer_shown"):  # Evaluates to True is variable not set (None) or set to False
        print("TotalSegmentator sends anonymous usage statistics. If you want to disable it check the documentation.")
        set_config_key("statistics_disclaimer_shown", True)

    from totalsegmentator.nnunet import nnUNet_predict_image  # this has to be after setting new env vars

    crop_addon = [3, 3, 3]  # default value
    cascade = None
    
    if task == "total":
        if fast:
            task_id = 297
            resample = 3.0
            trainer = "nnUNetTrainer_4000epochs_NoMirroring"
            # trainer = "nnUNetTrainerNoMirroring"
            crop = None
            if not quiet: print("Using 'fast' option: resampling to lower resolution (3mm)")
        elif fastest:
            task_id = 298
            resample = 6.0
            trainer = "nnUNetTrainer_4000epochs_NoMirroring"
            crop = None
            if not quiet: print("Using 'fastest' option: resampling to lower resolution (6mm)")
        else:
            task_id = [291, 292, 293, 294, 295]
            resample = 1.5
            trainer = "nnUNetTrainerNoMirroring"
            crop = None
        model = "3d_fullres"
        folds = [0]
    # todo: add to download and preview
    elif task == "total_highres_test":
        # task_id = 955
        task_id = 956
        # resample = [0.75, 0.75, 1.0]
        resample = [0.78125, 0.78125, 1.0]
        trainer = "nnUNetTrainer_DASegOrd0_NoMirroring"
        crop_addon = [30, 30, 30]
        crop = ["liver", "spleen", "colon", "small_bowel", "stomach", "lung_upper_lobe_left", "lung_upper_lobe_right", "aorta"] # abdomen_thorax
        # model = "3d_fullres_high"
        # model = "3d_fullres_high_bigPS"
        model = "3d_fullres"
        cascade = True
        folds = [0]
    elif task == "total_mr":
        if fast:
            task_id = 852
            resample = 3.0
            trainer = "nnUNetTrainer_2000epochs_NoMirroring"
            # trainer = "nnUNetTrainerNoMirroring"
            crop = None
            if not quiet: print("Using 'fast' option: resampling to lower resolution (3mm)")
        elif fastest:
            task_id = 853
            resample = 6.0
            trainer = "nnUNetTrainer_2000epochs_NoMirroring"
            crop = None
            if not quiet: print("Using 'fastest' option: resampling to lower resolution (6mm)")
        else:
            task_id = [850, 851]
            resample = 1.5
            trainer = "nnUNetTrainer_2000epochs_NoMirroring"
            crop = None
        model = "3d_fullres"
        folds = [0]
    elif task == "lung_vessels":
        task_id = 258
        resample = None
        trainer = "nnUNetTrainer"
        crop = ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
                "lung_middle_lobe_right", "lung_lower_lobe_right"]
        # if ml: raise ValueError("task lung_vessels does not work with option --ml, because of postprocessing.")
        if fast: raise ValueError("task lung_vessels does not work with option --fast")
        model = "3d_fullres"
        folds = [0]
    # elif task == "covid":
    #     task_id = 201
    #     resample = None
    #     trainer = "nnUNetTrainer"
    #     crop = ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
    #             "lung_middle_lobe_right", "lung_lower_lobe_right"]
    #     model = "3d_fullres"
    #     folds = [0]
    #     print("WARNING: The COVID model finds many types of lung opacity not only COVID. Use with care!")
    #     if fast: raise ValueError("task covid does not work with option --fast")
    elif task == "cerebral_bleed":
        task_id = 150
        resample = None
        trainer = "nnUNetTrainer"
        crop = ["brain"]
        model = "3d_fullres"
        folds = [0]
        if fast: raise ValueError("task cerebral_bleed does not work with option --fast")
    elif task == "hip_implant":
        task_id = 260
        resample = None
        trainer = "nnUNetTrainer"
        crop = ["femur_left", "femur_right", "hip_left", "hip_right"]
        model = "3d_fullres"
        folds = [0]
        if fast: raise ValueError("task hip_implant does not work with option --fast")
    elif task == "body":
        if fast:
            task_id = 300
            resample = 6.0
            trainer = "nnUNetTrainer"
            crop = None
            model = "3d_fullres"
            folds = [0]
            if not quiet: print("Using 'fast' option: resampling to lower resolution (6mm)")
        else:
            task_id = 299
            resample = 1.5
            trainer = "nnUNetTrainer"
            crop = None
            model = "3d_fullres"
            folds = [0]
    elif task == "body_mr":
        if fast:
            task_id = 598  # todo: train
            resample = 6.0
            trainer = "nnUNetTrainer_DASegOrd0"
            crop = None
            model = "3d_fullres"
            folds = [0]
            if not quiet: print("Using 'fast' option: resampling to lower resolution (6mm)")
        else:
            task_id = 597
            resample = 1.5
            trainer = "nnUNetTrainer_DASegOrd0"
            crop = None
            model = "3d_fullres"
            folds = [0]
    elif task == "vertebrae_mr":
        task_id = 756
        resample = 1.5
        trainer = "nnUNetTrainer_DASegOrd0_NoMirroring"
        crop = None
        model = "3d_fullres"
        folds = [0]
    elif task == "pleural_pericard_effusion":
        task_id = 315
        resample = None
        trainer = "nnUNetTrainer"
        crop = ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
                "lung_middle_lobe_right", "lung_lower_lobe_right"]
        crop_addon = [50, 50, 50]
        model = "3d_fullres"
        folds = None
        if fast: raise ValueError("task pleural_pericard_effusion does not work with option --fast")
    elif task == "liver_vessels":
        task_id = 8
        resample = None
        trainer = "nnUNetTrainer"
        crop = ["liver"]
        crop_addon = [20, 20, 20]
        model = "3d_fullres"
        folds = [0]
        if fast: raise ValueError("task liver_vessels does not work with option --fast")
    elif task == "head_glands_cavities":
        task_id = 775
        resample = [0.75, 0.75, 1.0]
        trainer = "nnUNetTrainer_DASegOrd0_NoMirroring"
        crop = ["skull"]
        crop_addon = [10, 10, 10]
        model = "3d_fullres_high"
        folds = [0]
        if fast: raise ValueError("task head_glands_cavities does not work with option --fast")
    elif task == "headneck_bones_vessels":
        task_id = 776
        resample = [0.75, 0.75, 1.0]
        trainer = "nnUNetTrainer_DASegOrd0_NoMirroring"
        # crop = ["skull", "clavicula_left", "clavicula_right", "vertebrae_C5", "vertebrae_T1", "vertebrae_T4"]
        # crop_addon = [10, 10, 10]
        crop = ["clavicula_left", "clavicula_right", "vertebrae_C1", "vertebrae_C5", "vertebrae_T1", "vertebrae_T4"]
        crop_addon = [40, 40, 40]
        model = "3d_fullres_high"
        folds = [0]
        if fast: raise ValueError("task headneck_bones_vessels does not work with option --fast")
    elif task == "head_muscles":
        task_id = 777
        resample = [0.75, 0.75, 1.0]
        trainer = "nnUNetTrainer_DASegOrd0_NoMirroring"
        crop = ["skull"]
        crop_addon = [10, 10, 10]
        model = "3d_fullres_high"
        folds = [0]
        if fast: raise ValueError("task head_muscles does not work with option --fast")
    elif task == "headneck_muscles":
        task_id = [778, 779]
        resample = [0.75, 0.75, 1.0]
        trainer = "nnUNetTrainer_DASegOrd0_NoMirroring"
        # crop = ["skull", "clavicula_left", "clavicula_right", "vertebrae_C5", "vertebrae_T1", "vertebrae_T4"]
        # crop_addon = [10, 10, 10]
        crop = ["clavicula_left", "clavicula_right", "vertebrae_C1", "vertebrae_C5", "vertebrae_T1", "vertebrae_T4"]
        crop_addon = [40, 40, 40]
        model = "3d_fullres_high"
        folds = [0]
        if fast: raise ValueError("task headneck_muscles does not work with option --fast")
    elif task == "oculomotor_muscles":
        task_id = 351
        resample = [0.47251562774181366, 0.47251562774181366, 0.8500002026557922]
        trainer = "nnUNetTrainer_DASegOrd0_NoMirroring"
        crop = ["skull"]
        crop_addon = [20, 20, 20]
        model = "3d_fullres"
        folds = [0]
        if fast: raise ValueError("task oculomotor_muscles does not work with option --fast")
    elif task == "lung_nodules":
        task_id = 913
        resample = [1.5, 1.5, 1.5]
        trainer = "nnUNetTrainer_MOSAIC_1k_QuarterLR_NoMirroring"
        crop = ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
                "lung_middle_lobe_right", "lung_lower_lobe_right"]
        crop_addon = [10, 10, 10]
        model = "3d_fullres"
        folds = [0]
        if fast: raise ValueError("task lung_nodules does not work with option --fast")
    elif task == "kidney_cysts":
        task_id = 789
        resample = [1.5, 1.5, 1.5]
        trainer = "nnUNetTrainer_DASegOrd0_NoMirroring"
        crop = ["kidney_left", "kidney_right", "liver", "spleen", "colon"]
        crop_addon = [10, 10, 10]
        model = "3d_fullres"
        folds = [0]
        if fast: raise ValueError("task kidney_cysts does not work with option --fast")
    elif task == "breasts":
        task_id = 527
        resample = [1.5, 1.5, 1.5]
        trainer = "nnUNetTrainer_DASegOrd0_NoMirroring"
        crop = None
        model = "3d_fullres"
        folds = [0]
        if fast: raise ValueError("task breasts does not work with option --fast")
    elif task == "ventricle_parts":
        task_id = 552
        resample = [1.0, 0.4345703125, 0.4384765625]
        trainer = "nnUNetTrainerNoMirroring"
        crop = ["brain"]
        crop_addon = [0, 0, 0]
        model = "3d_fullres"
        folds = [0]
        if fast: raise ValueError("task ventricle_parts does not work with option --fast")
    elif task == "liver_segments":
        task_id = 570
        resample = [1.5, 0.8046879768371582, 0.8046879768371582]
        trainer = "nnUNetTrainerNoMirroring"
        crop = ["liver"]
        crop_addon = [10, 10, 10]
        model = "3d_fullres"
        folds = [0]
        if fast: raise ValueError("task liver_segments does not work with option --fast")
    elif task == "liver_segments_mr":
        task_id = 576
        resample = [3.0, 1.1875, 1.1250001788139343]
        trainer = "nnUNetTrainer_DASegOrd0_NoMirroring"
        crop = ["liver"]
        crop_addon = [10, 10, 10]
        model = "3d_fullres"
        folds = [0]
        if fast: raise ValueError("task liver_segments_mr does not work with option --fast")
    elif task == "craniofacial_structures":
        task_id = 115
        resample = [0.5, 0.5, 0.5]
        trainer = "nnUNetTrainer_DASegOrd0_NoMirroring"
        crop = ["skull"]
        crop_addon = [20, 20, 20]
        model = "3d_fullres"
        folds = [0]
        if fast: raise ValueError("task craniofacial_structures does not work with option --fast")

        
    # Commercial models
    elif task == "vertebrae_body":
        task_id = 305
        resample = 1.5
        trainer = "nnUNetTrainer_DASegOrd0"
        crop = None
        model = "3d_fullres"
        folds = [0]
        if fast: raise ValueError("task vertebrae_body does not work with option --fast")
        show_license_info()
    elif task == "heartchambers_highres":
        task_id = 301
        resample = None
        trainer = "nnUNetTrainer"
        crop = ["heart"]
        crop_addon = [5, 5, 5]
        model = "3d_fullres"
        folds = [0]
        if fast: raise ValueError("task heartchambers_highres does not work with option --fast")
        show_license_info()
    elif task == "appendicular_bones":
        task_id = 304
        resample = 1.5
        trainer = "nnUNetTrainerNoMirroring"
        crop = None
        model = "3d_fullres"
        folds = [0]
        if fast: raise ValueError("task appendicular_bones does not work with option --fast")
        show_license_info()
    elif task == "appendicular_bones_mr":
        task_id = 855
        resample = 1.5
        trainer = "nnUNetTrainer_2000epochs_NoMirroring"
        crop = None
        model = "3d_fullres"
        folds = [0]
        if fast: raise ValueError("task appendicular_bones_mr does not work with option --fast")
        show_license_info()
    elif task == "tissue_types":
        task_id = 481
        resample = 1.5
        trainer = "nnUNetTrainer"
        crop = None
        model = "3d_fullres"
        folds = [0]
        if fast: raise ValueError("task tissue_types does not work with option --fast")
        show_license_info()
    elif task == "tissue_types_mr":
        task_id = 925
        resample = 1.5
        trainer = "nnUNetTrainer_DASegOrd0_NoMirroring"
        crop = None
        model = "3d_fullres"
        folds = [0]
        if fast: raise ValueError("task tissue_types_mr does not work with option --fast")
        show_license_info()
    elif task == "tissue_4_types":
        task_id = 485
        resample = 1.5
        trainer = "nnUNetTrainer"
        crop = None
        model = "3d_fullres"
        folds = [0]
        if fast: raise ValueError("task tissue_4_types does not work with option --fast")
        show_license_info()
    elif task == "face":
        task_id = 303
        resample = 1.5
        trainer = "nnUNetTrainerNoMirroring"
        crop = None
        model = "3d_fullres"
        folds = [0]
        if fast: raise ValueError("task face does not work with option --fast")
        show_license_info()
    elif task == "face_mr":
        task_id = 856
        resample = 1.5
        trainer = "nnUNetTrainer_2000epochs_NoMirroring"
        crop = None
        model = "3d_fullres"
        folds = [0]
        if fast: raise ValueError("task face_mr does not work with option --fast")
        show_license_info()
    elif task == "brain_structures":
        task_id = 409
        resample = [0.5, 0.5, 1.0]
        trainer = "nnUNetTrainer_DASegOrd0"
        crop = ["brain"]
        crop_addon = [10, 10, 10]
        model = "3d_fullres_high"
        folds = [0]
        if fast: raise ValueError("task brain_structures does not work with option --fast")
        show_license_info()
    elif task == "thigh_shoulder_muscles":
        task_id = 857  # at the moment only one mixed model for CT and MR; when annotated all CT samples -> train separate CT model
        resample = 1.5
        trainer = "nnUNetTrainer_2000epochs_NoMirroring"
        crop = None
        model = "3d_fullres"
        folds = [0]
        if fast: raise ValueError("task thigh_shoulder_muscles does not work with option --fast")
        show_license_info()
    elif task == "thigh_shoulder_muscles_mr":
        task_id = 857
        resample = 1.5
        trainer = "nnUNetTrainer_2000epochs_NoMirroring"
        crop = None
        model = "3d_fullres"
        folds = [0]
        if fast: raise ValueError("task thigh_shoulder_muscles_mr does not work with option --fast")
        show_license_info()
    elif task == "coronary_arteries":
        task_id = 507
        resample = [0.7, 0.7, 0.7]
        trainer = "nnUNetTrainer_DASegOrd0_NoMirroring"
        crop = ["heart"]
        crop_addon = [20, 20, 20]
        model = "3d_fullres_high"
        folds = [0]
        if fast: raise ValueError("task coronary_arteries does not work with option --fast")
        show_license_info()
    elif task == "aortic_sinuses":
        task_id = 920
        resample = [0.7, 0.7, 0.7]
        trainer = "nnUNetTrainer_DASegOrd0_NoMirroring"
        crop = ["heart"]
        crop_addon = [0, 0, 0]
        model = "3d_fullres_high"
        folds = [0]
        if fast: raise ValueError("task aortic_sinuses does not work with option --fast")
        show_license_info()

    elif task == "test":
        task_id = [517]
        resample = None
        trainer = "nnUNetTrainerV2"
        crop = "body"
        model = "3d_fullres"
        folds = [0]

    crop_path = output if crop_path is None else crop_path

    if isinstance(input, Nifti1Image) or input.suffix == ".nii" or input.suffixes == [".nii", ".gz"]:
        img_type = "nifti"
    else:
        img_type = "dicom"

    # fast statistics are calculated on the downsampled image
    if statistics and fast:
        statistics_fast = True
        statistics = False
    else:
        statistics_fast = False

    if type(task_id) is list:
        for tid in task_id:
            download_pretrained_weights(tid)
    else:
        download_pretrained_weights(task_id)

    # For MR always run 3mm model for roi_subset, because 6mm too bad results
    #  (runtime for 3mm still very good for MR)
    if task.endswith("_mr") and roi_subset is not None:
        roi_subset_robust = roi_subset
        robust_rs = True

    if roi_subset_robust is not None:
        roi_subset = roi_subset_robust
        robust_rs = True
    else:
        robust_rs = False

    if roi_subset is not None and type(roi_subset) is not list:
        raise ValueError("roi_subset must be a list of strings")
    if roi_subset is not None and not task.startswith("total"):
        raise ValueError("roi_subset only works with task 'total' or 'total_mr'")

    if task.endswith("_mr"):
        if body_seg:
            body_seg = False
            print("INFO: For MR models the argument '--body_seg' is not supported and will be ignored.")

    # Generate rough organ segmentation (6mm) for speed up if crop or roi_subset is used
    # (for "fast" on GPU it makes no big difference, but on CPU it can help even for "fast")
    if crop is not None or roi_subset is not None or cascade:

        body_seg = False  # can not be used together with body_seg
        st = time.time()
        if not quiet: print("Generating rough segmentation for cropping...")
        if robust_rs or robust_crop:
            print("  (Using more robust (but slower) 3mm model for cropping.)")
            crop_model_task = 852 if task.endswith("_mr") else 297
            crop_spacing = 3.0
        else:
            # For MR always run 3mm model for cropping, because 6mm too bad results
            #  (runtime for 3mm still very good for MR)
            if task.endswith("_mr"):
                crop_model_task = 852
                crop_spacing = 3.0
            else:
                crop_model_task = 298
                crop_spacing = 6.0
        crop_task = "total_mr" if task.endswith("_mr") else "total"
        crop_trainer = "nnUNetTrainer_2000epochs_NoMirroring" if task.endswith("_mr") else "nnUNetTrainer_4000epochs_NoMirroring"
        download_pretrained_weights(crop_model_task)
        
        organ_seg, _, _ = nnUNet_predict_image(input, None, crop_model_task, model="3d_fullres", folds=[0],
                            trainer=crop_trainer, tta=False, multilabel_image=True, resample=crop_spacing,
                            crop=None, crop_path=None, task_name=crop_task, nora_tag="None", preview=False,
                            save_binary=False, nr_threads_resampling=nr_thr_resamp, nr_threads_saving=1,
                            crop_addon=crop_addon, output_type=output_type, statistics=False,
                            quiet=quiet, verbose=verbose, test=0, skip_saving=False, device=device)
        class_map_inv = {v: k for k, v in class_map[crop_task].items()}
        crop_mask = np.zeros(organ_seg.shape, dtype=np.uint8)
        organ_seg_data = organ_seg.get_fdata()
        # roi_subset_crop = [map_to_total[roi] if roi in map_to_total else roi for roi in roi_subset]
        roi_subset_crop = crop if crop is not None else roi_subset
        for roi in roi_subset_crop:
            crop_mask[organ_seg_data == class_map_inv[roi]] = 1
        crop_mask = nib.Nifti1Image(crop_mask, organ_seg.affine)
        crop_addon = [20,20,20]
        crop = crop_mask
        cascade = crop_mask if cascade else None
        if verbose: print(f"Rough organ segmentation generated in {time.time()-st:.2f}s")

    # Generate rough body segmentation (6mm) (speedup for big images; not useful in combination with --fast option)
    if crop is None and body_seg:
        download_pretrained_weights(300)
        st = time.time()
        if not quiet: print("Generating rough body segmentation...")
        body_seg, _, _ = nnUNet_predict_image(input, None, 300, model="3d_fullres", folds=[0],
                            trainer="nnUNetTrainer", tta=False, multilabel_image=True, resample=6.0,
                            crop=None, crop_path=None, task_name="body", nora_tag="None", preview=False,
                            save_binary=True, nr_threads_resampling=nr_thr_resamp, nr_threads_saving=1,
                            crop_addon=crop_addon, output_type=output_type, statistics=False,
                            quiet=quiet, verbose=verbose, test=0, skip_saving=False, device=device)
        crop = body_seg
        if verbose: print(f"Rough body segmentation generated in {time.time()-st:.2f}s")

    folds = [0]  # None
    seg_img, ct_img, stats = nnUNet_predict_image(input, output, task_id, model=model, folds=folds,
                            trainer=trainer, tta=False, multilabel_image=ml, resample=resample,
                            crop=crop, crop_path=crop_path, task_name=task, nora_tag=nora_tag, preview=preview,
                            nr_threads_resampling=nr_thr_resamp, nr_threads_saving=nr_thr_saving,
                            force_split=force_split, crop_addon=crop_addon, roi_subset=roi_subset,
                            output_type=output_type, statistics=statistics_fast,
                            quiet=quiet, verbose=verbose, test=test, skip_saving=skip_saving, device=device,
                            exclude_masks_at_border=statistics_exclude_masks_at_border,
                            no_derived_masks=no_derived_masks, v1_order=v1_order,
                            stats_aggregation=stats_aggregation, remove_small_blobs=remove_small_blobs,
                            normalized_intensities=statistics_normalized_intensities, 
                            nnunet_resampling=higher_order_resampling, save_probabilities=save_probabilities,
                            cascade=cascade)
    seg = seg_img.get_fdata().astype(np.uint8)

    try:
        # this can result in error if running multiple processes in parallel because all try to write the same file.
        # Trying to fix with lock from portalocker did not work. Network drive seems to not support this locking.
        config = increase_prediction_counter()
        send_usage_stats(config, {"task": task, "fast": fast, "preview": preview,
                                "multilabel": ml, "roi_subset": roi_subset,
                                "statistics": statistics, "radiomics": radiomics})
    except Exception as e:
        # print(f"Error while sending usage stats: {e}")
        pass

    if statistics:
        if not quiet: print("Calculating statistics...")
        st = time.time()
        if output is not None:
            stats_dir = output.parent if ml else output
            stats_file = stats_dir / "statistics.json"
        else:
            stats_file = None
        stats = get_basic_statistics(seg, ct_img, stats_file, 
                                     quiet, task, statistics_exclude_masks_at_border,
                                     roi_subset, 
                                     metric=stats_aggregation,
                                     normalized_intensities=statistics_normalized_intensities)
        # get_radiomics_features_for_entire_dir(input, output, output / "statistics_radiomics.json")
        if not quiet: print(f"  calculated in {time.time()-st:.2f}s")

    if radiomics:
        if ml:
            raise ValueError("Radiomics not supported for multilabel segmentation. Use without --ml option.")
        if img_type == "dicom":
            raise ValueError("Radiomics not supported for DICOM input. Use nifti input.")
        if not quiet: print("Calculating radiomics...")
        st = time.time()
        stats_dir = output.parent if ml else output
        with tempfile.TemporaryDirectory(prefix="radiomics_tmp_") as tmp_folder:
            if isinstance(input, Nifti1Image):
                input_path = tmp_folder / "ct.nii.gz"
                nib.save(input, input_path)
            else:
                input_path = input
            get_radiomics_features_for_entire_dir(input_path, output, stats_dir / "statistics_radiomics.json")
            if not quiet: print(f"  calculated in {time.time()-st:.2f}s")

    # Restore initial torch settings
    torch.backends.cudnn.benchmark = initial_cudnn_benchmark
    torch.set_num_threads(initial_num_threads)

    if statistics or statistics_fast:
        return seg_img, stats
    else:
        return seg_img
