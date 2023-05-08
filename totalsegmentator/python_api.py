import sys
import os
from pathlib import Path
import time

import numpy as np
import nibabel as nib
import torch

from totalsegmentator.statistics import get_basic_statistics_for_entire_dir, get_radiomics_features_for_entire_dir
from totalsegmentator.libs import download_pretrained_weights
from totalsegmentator.config import setup_nnunet, setup_totalseg, increase_prediction_counter, send_usage_stats


def totalsegmentator(input, output, ml=False, nr_thr_resamp=1, nr_thr_saving=6,
                     fast=False, nora_tag="None", preview=False, task="total", roi_subset=None,
                     statistics=False, radiomics=False, crop_path=None, body_seg=False,
                     force_split=False, output_type="nifti", quiet=False, verbose=False, test=0):
    """
    Run TotalSegmentator from within python. 

    For explanation of the arguments see description of command line 
    arguments in bin/TotalSegmentator.
    """
    input = Path(input)
    output = Path(output)

    nora_tag = "None" if nora_tag is None else nora_tag

    if not quiet: 
        print("\nIf you use this tool please cite: https://doi.org/10.48550/arXiv.2208.05868\n")

    if not torch.cuda.is_available():
        print("No GPU detected. Running on CPU. This can be very slow. The '--fast' option can help to some extend.")

    setup_nnunet()

    from totalsegmentator.nnunet import nnUNet_predict_image  # this has to be after setting new env vars

    crop_addon = [3, 3, 3]  # default value

    if task == "total":
        if fast:
            task_id = 256
            resample = 3.0
            trainer = "nnUNetTrainerV2_ep8000_nomirror"
            crop = None
            if not quiet: print("Using 'fast' option: resampling to lower resolution (3mm)")
        else:
            task_id = [251, 252, 253, 254, 255]
            resample = 1.5
            trainer = "nnUNetTrainerV2_ep4000_nomirror"
            crop = None
        model = "3d_fullres"
        folds = [0]
    elif task == "lung_vessels":
        task_id = 258
        resample = None
        trainer = "nnUNetTrainerV2"
        crop = "lung"
        if ml: raise ValueError("task lung_vessels does not work with option --ml, because of postprocessing.")
        if fast: raise ValueError("task lung_vessels does not work with option --fast")
        model = "3d_fullres"
        folds = [0]
    elif task == "covid":
        task_id = 201
        resample = None
        trainer = "nnUNetTrainerV2"
        crop = "lung"
        model = "3d_fullres"
        folds = [0]
        print("WARNING: The COVID model finds many types of lung opacity not only COVID. Use with care!")
        if fast: raise ValueError("task covid does not work with option --fast")
    elif task == "cerebral_bleed":
        task_id = 150
        resample = None
        trainer = "nnUNetTrainerV2"
        crop = "brain"
        model = "3d_fullres"
        folds = [0]
        if fast: raise ValueError("task cerebral_bleed does not work with option --fast")
    elif task == "hip_implant":
        task_id = 260
        resample = None
        trainer = "nnUNetTrainerV2"
        crop = "pelvis"
        model = "3d_fullres"
        folds = [0]
        if fast: raise ValueError("task hip_implant does not work with option --fast")
    elif task == "coronary_arteries":
        task_id = 503
        resample = None
        trainer = "nnUNetTrainerV2"
        crop = "heart"
        model = "3d_fullres"
        folds = [0]
        print("WARNING: The coronary artery model does not work very robustly. Use with care!")
        if fast: raise ValueError("task coronary_arteries does not work with option --fast")
    elif task == "body":
        if fast:
            task_id = 269
            resample = 6.0
            trainer = "nnUNetTrainerV2"
            crop = None
            model = "3d_fullres"
            folds = [0]
            if not quiet: print("Using 'fast' option: resampling to lower resolution (6mm)")
        else:
            task_id = 273
            resample = 1.5
            trainer = "nnUNetTrainerV2"
            crop = None
            model = "3d_fullres"
            folds = [0]
        if ml: raise ValueError("task body does not work with option --ml, because of postprocessing.")
    elif task == "pleural_pericard_effusion":
        task_id = 315
        resample = None
        trainer = "nnUNetTrainerV2"
        crop = "lung"
        crop_addon = [50, 50, 50]
        model = "3d_fullres"
        folds = None
        if fast: raise ValueError("task pleural_pericard_effusion does not work with option --fast")
    elif task == "liver_vessels":
        task_id = 8
        resample = None
        trainer = "nnUNetTrainerV2"
        crop = "liver"
        crop_addon = [20, 20, 20]
        model = "3d_fullres"
        folds = None
        if fast: raise ValueError("task liver_vessels does not work with option --fast")
    elif task == "heartchambers_test":
        task_id = 417
        resample = None
        trainer = "nnUNetTrainerV2"
        crop = "heart"
        crop_addon = [5, 5, 5]
        model = "3d_lowres"
        folds = None
        if fast: raise ValueError("task heartchambers_test does not work with option --fast")
    elif task == "bones_tissue_test":
        task_id = 278
        resample = 1.5
        trainer = "nnUNetTrainerV2_ep4000_nomirror"
        crop = None
        model = "3d_fullres"
        folds = [0]
        if fast: raise ValueError("task bones_tissue_test does not work with option --fast")
    elif task == "aortic_branches_test":
        task_id = 435
        resample = 1.5
        trainer = "nnUNetTrainerV2_nomirror"
        crop = None
        model = "3d_fullres"
        folds = [0]
        if fast: raise ValueError("task aortic_branches_test does not work with option --fast")
    elif task in ["bones_extremities", "tissue_types", "heartchambers_highres",
                       "head", "aortic_branches"]:
        print("\nThis model is only available upon purchase of a license (free licenses available for " +
              "academic projects). \nContact jakob.wasserthal@usb.ch if you are interested.\n")
        sys.exit()
    elif task == "test":
        task_id = [517]
        resample = None
        trainer = "nnUNetTrainerV2"
        crop = "body"
        model = "3d_fullres"
        folds = [0]

    crop_path = output if crop_path is None else crop_path

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

    # Generate rough body segmentation (speedup for big images; not useful in combination with --fast option)
    if crop is None and body_seg:
        download_pretrained_weights(269)
        st = time.time()
        if not quiet: print("Generating rough body segmentation...")
        body_seg = nnUNet_predict_image(input, None, 269, model="3d_fullres", folds=[0],
                            trainer="nnUNetTrainerV2", tta=False, multilabel_image=True, resample=6.0,
                            crop=None, crop_path=None, task_name="body", nora_tag="None", preview=False, 
                            save_binary=True, nr_threads_resampling=nr_thr_resamp, nr_threads_saving=1, 
                            crop_addon=crop_addon, output_type=output_type, statistics=False,
                            quiet=quiet, verbose=verbose, test=0)
        crop = body_seg
        if verbose: print(f"Rough body segmentation generated in {time.time()-st:.2f}s")

    folds = [0]  # None
    seg_img = nnUNet_predict_image(input, output, task_id, model=model, folds=folds,
                         trainer=trainer, tta=False, multilabel_image=ml, resample=resample,
                         crop=crop, crop_path=crop_path, task_name=task, nora_tag=nora_tag, preview=preview, 
                         nr_threads_resampling=nr_thr_resamp, nr_threads_saving=nr_thr_saving, 
                         force_split=force_split, crop_addon=crop_addon, roi_subset=roi_subset,
                         output_type=output_type, statistics=statistics_fast, 
                         quiet=quiet, verbose=verbose, test=test)
    seg = seg_img.get_fdata().astype(np.uint8)

    config = setup_totalseg()
    increase_prediction_counter()
    send_usage_stats(config, {"task": task, "fast": fast, "preview": preview,
                              "multilabel": ml, "roi_subset": roi_subset, 
                              "statistics": statistics, "radiomics": radiomics})

    if statistics:
        if not quiet: print("Calculating statistics...")
        st = time.time()
        stats_dir = output.parent if ml else output
        get_basic_statistics_for_entire_dir(seg, input, stats_dir / "statistics.json", quiet)
        # get_radiomics_features_for_entire_dir(input, output, output / "statistics_radiomics.json")
        if not quiet: print(f"  calculated in {time.time()-st:.2f}s")

    if radiomics:
        if ml:
            raise ValueError("Radiomics not supported for multilabel segmentation. Use without --ml option.")
        if not quiet: print("Calculating radiomics...")  
        st = time.time()
        stats_dir = output.parent if ml else output
        get_radiomics_features_for_entire_dir(input, output, stats_dir / "statistics_radiomics.json")
        if not quiet: print(f"  calculated in {time.time()-st:.2f}s")

    return seg_img
