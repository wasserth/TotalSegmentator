import sys
import os
from pathlib import Path
import time
import textwrap

import numpy as np
import nibabel as nib
import torch

from totalsegmentator.statistics import get_basic_statistics, get_radiomics_features_for_entire_dir
from totalsegmentator.libs import download_pretrained_weights
from totalsegmentator.config import setup_nnunet, setup_totalseg, increase_prediction_counter
from totalsegmentator.config import send_usage_stats, set_license_number, has_valid_license_offline
from totalsegmentator.config import get_config_key, set_config_key
from totalsegmentator.map_to_binary import class_map
from totalsegmentator.map_to_total import map_to_total


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
        sys.exit(0)
    elif status == "invalid_license":
        print(message)
        sys.exit(0)
    elif status == "missing_config_file":
        print(message)
        sys.exit(0)


def totalsegmentator(input, output, ml=False, nr_thr_resamp=1, nr_thr_saving=6,
                     fast=False, nora_tag="None", preview=False, task="total", roi_subset=None,
                     statistics=False, radiomics=False, crop_path=None, body_seg=False,
                     force_split=False, output_type="nifti", quiet=False, verbose=False, test=0,
                     skip_saving=False, device="gpu", license_number=None,
                     statistics_exclude_masks_at_border=True, no_derived_masks=False,
                     v1_order=False):
    """
    Run TotalSegmentator from within python. 

    For explanation of the arguments see description of command line 
    arguments in bin/TotalSegmentator.
    """
    input = Path(input)
    output = Path(output)

    nora_tag = "None" if nora_tag is None else nora_tag

    if not quiet: 
        print("\nIf you use this tool please cite: https://pubs.rsna.org/doi/10.1148/ryai.230024\n")

    # available devices: gpu | cpu | mps
    if device == "gpu": device = "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        print("No GPU detected. Running on CPU. This can be very slow. The '--fast' or the `--roi_subset` option can help to reduce runtime.")
        device = "cpu"

    setup_nnunet()
    setup_totalseg()
    if license_number is not None:
        set_license_number(license_number)

    if not get_config_key("statistics_disclaimer_shown"):  # Evaluates to True is variable not set (None) or set to False
        print("TotalSegmentator sends anonymous usage statistics. If you want to disable it check the documentation.")
        set_config_key("statistics_disclaimer_shown", True)

    from totalsegmentator.nnunet import nnUNet_predict_image  # this has to be after setting new env vars

    crop_addon = [3, 3, 3]  # default value

    if task == "total":
        if fast:
            task_id = 297
            resample = 3.0
            trainer = "nnUNetTrainer_4000epochs_NoMirroring"
            # trainer = "nnUNetTrainerNoMirroring"
            crop = None
            if not quiet: print("Using 'fast' option: resampling to lower resolution (3mm)")
        else:
            task_id = [291, 292, 293, 294, 295]
            resample = 1.5
            trainer = "nnUNetTrainerNoMirroring"
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
    elif task == "coronary_arteries":
        task_id = 503
        resample = None
        trainer = "nnUNetTrainer"
        crop = ["heart"]
        model = "3d_fullres"
        folds = [0]
        print("WARNING: The coronary artery model does not work very robustly. Use with care!")
        if fast: raise ValueError("task coronary_arteries does not work with option --fast")
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
        folds = None
        if fast: raise ValueError("task liver_vessels does not work with option --fast")
    elif task == "vertebrae_body":
        task_id = 302
        resample = 1.5
        trainer = "nnUNetTrainer"
        crop = None
        model = "3d_fullres"
        folds = [0]
        if fast: raise ValueError("task vertebrae_body does not work with option --fast")

    # Commercial models
    elif task == "heartchambers_highres":
        task_id = 301
        resample = None
        trainer = "nnUNetTrainer"
        crop = ["heart"]
        crop_addon = [5, 5, 5]
        model = "3d_fullres"
        folds = None
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
    elif task == "tissue_types":
        task_id = 481
        resample = 1.5
        trainer = "nnUNetTrainer"
        crop = None
        model = "3d_fullres"
        folds = [0]
        if fast: raise ValueError("task tissue_types does not work with option --fast")
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
    elif task == "test":
        task_id = [517]
        resample = None
        trainer = "nnUNetTrainerV2"
        crop = "body"
        model = "3d_fullres"
        folds = [0]

    crop_path = output if crop_path is None else crop_path

    img_type = "nifti" if str(input).endswith(".nii") or str(input).endswith(".nii.gz") else "dicom"

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

    if roi_subset is not None and type(roi_subset) is not list:
        raise ValueError("roi_subset must be a list of strings")
    if roi_subset is not None and task != "total":
        raise ValueError("roi_subset only works with task 'total'")

    # Generate rough organ segmentation (6mm) for speed up if crop or roi_subset is used
    if crop is not None or roi_subset is not None:
        
        body_seg = False  # can not be used together with body_seg
        download_pretrained_weights(298)
        st = time.time()
        if not quiet: print("Generating rough body segmentation...")
        organ_seg, _ = nnUNet_predict_image(input, None, 298, model="3d_fullres", folds=[0],
                            trainer="nnUNetTrainer_4000epochs_NoMirroring", tta=False, multilabel_image=True, resample=6.0,
                            crop=None, crop_path=None, task_name="total", nora_tag="None", preview=False, 
                            save_binary=False, nr_threads_resampling=nr_thr_resamp, nr_threads_saving=1, 
                            crop_addon=crop_addon, output_type=output_type, statistics=False,
                            quiet=quiet, verbose=verbose, test=0, skip_saving=False, device=device)
        class_map_inv = {v: k for k, v in class_map["total"].items()}
        crop_mask = np.zeros(organ_seg.shape, dtype=np.uint8)
        organ_seg_data = organ_seg.get_fdata()
        # roi_subset_crop = [map_to_total[roi] if roi in map_to_total else roi for roi in roi_subset]        
        roi_subset_crop = crop if crop is not None else roi_subset
        for roi in roi_subset_crop:
            crop_mask[organ_seg_data == class_map_inv[roi]] = 1
        crop_mask = nib.Nifti1Image(crop_mask, organ_seg.affine)
        crop_addon = [20,20,20]
        crop = crop_mask
        if verbose: print(f"Rough organ segmentation generated in {time.time()-st:.2f}s")

    # Generate rough body segmentation (6mm) (speedup for big images; not useful in combination with --fast option)
    if crop is None and body_seg:
        download_pretrained_weights(300)
        st = time.time()
        if not quiet: print("Generating rough body segmentation...")
        body_seg, _ = nnUNet_predict_image(input, None, 300, model="3d_fullres", folds=[0],
                            trainer="nnUNetTrainer", tta=False, multilabel_image=True, resample=6.0,
                            crop=None, crop_path=None, task_name="body", nora_tag="None", preview=False, 
                            save_binary=True, nr_threads_resampling=nr_thr_resamp, nr_threads_saving=1, 
                            crop_addon=crop_addon, output_type=output_type, statistics=False,
                            quiet=quiet, verbose=verbose, test=0, skip_saving=False, device=device)
        crop = body_seg
        if verbose: print(f"Rough body segmentation generated in {time.time()-st:.2f}s")

    folds = [0]  # None
    seg_img, ct_img = nnUNet_predict_image(input, output, task_id, model=model, folds=folds,
                         trainer=trainer, tta=False, multilabel_image=ml, resample=resample,
                         crop=crop, crop_path=crop_path, task_name=task, nora_tag=nora_tag, preview=preview, 
                         nr_threads_resampling=nr_thr_resamp, nr_threads_saving=nr_thr_saving, 
                         force_split=force_split, crop_addon=crop_addon, roi_subset=roi_subset,
                         output_type=output_type, statistics=statistics_fast, 
                         quiet=quiet, verbose=verbose, test=test, skip_saving=skip_saving, device=device,
                         exclude_masks_at_border=statistics_exclude_masks_at_border,
                         no_derived_masks=no_derived_masks, v1_order=v1_order)
    seg = seg_img.get_fdata().astype(np.uint8)

    config = increase_prediction_counter()
    send_usage_stats(config, {"task": task, "fast": fast, "preview": preview,
                              "multilabel": ml, "roi_subset": roi_subset, 
                              "statistics": statistics, "radiomics": radiomics})

    if statistics:
        if not quiet: print("Calculating statistics...")
        st = time.time()
        stats_dir = output.parent if ml else output
        get_basic_statistics(seg, ct_img, stats_dir / "statistics.json", quiet, task, statistics_exclude_masks_at_border)
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
        get_radiomics_features_for_entire_dir(input, output, stats_dir / "statistics_radiomics.json")
        if not quiet: print(f"  calculated in {time.time()-st:.2f}s")

    return seg_img
