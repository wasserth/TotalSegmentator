import re
import time
import tempfile
from pathlib import Path
from typing import Union

import numpy as np
import nibabel as nib
from nibabel.nifti1 import Nifti1Image
import torch
from totalsegmentator.statistics import (
    get_basic_statistics,
    get_radiomics_features_for_entire_dir,
)
from totalsegmentator.config import setup_nnunet, setup_totalseg, increase_prediction_counter
from totalsegmentator.config import send_usage_stats, set_license_number
from totalsegmentator.config import get_config_key, set_config_key
from totalsegmentator.map_to_binary import class_map
from totalsegmentator.task import get_task, Task


def validate_device_type_api(value: str) -> str:
    """
    Validate device type.

    Args:
        value (str): Device type: 'gpu', 'cpu', 'mps', or 'gpu:X' where X is an integer representing the GPU device ID.

    Returns:
        str: Device type.

    Raises:
        ValueError: If the device type is invalid.
    """
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


def convert_device_to_cuda(device: str) -> str:
    """
    Convert device to CUDA format.

    Args:
        device (str): Device type: 'gpu', 'cpu', 'mps', or 'gpu:X' where X is an
            integer representing the GPU device ID.

    Returns:
        str: Device type in CUDA format.
    """
    if device in ["cpu", "mps", "gpu"]:
        return device
    else:  # gpu:X
        return f"cuda:{device.split(':')[1]}"


def select_device(device: str) -> torch.device:
    """
    Select device.

    Args:
        device (str): Device type: 'gpu', 'cpu', 'mps', or 'gpu:X' where X is an integer representing the GPU device ID.

    Returns:
        torch.device: Device type.
    """
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


def totalsegmentator(
    input: Union[str, Path, Nifti1Image],
    output: Union[str, Path, None] = None,
    ml=False,
    nr_thr_resamp=1,
    nr_thr_saving=6,
    fast=False,
    nora_tag="None",
    preview=False,
    task="total",
    roi_subset=None,
    statistics=False,
    radiomics=False,
    crop_path=None,
    body_seg=False,
    force_split=False,
    output_type="nifti",
    quiet=False,
    verbose=False,
    test=0,
    skip_saving=False,
    device="gpu",
    license_number=None,
    statistics_exclude_masks_at_border=True,
    no_derived_masks=False,
    v1_order=False,
    fastest=False,
    roi_subset_robust=None,
    stats_aggregation="mean",
    remove_small_blobs=False,
    statistics_normalized_intensities=False,
    robust_crop=False,
    higher_order_resampling=False,
    save_probabilities=None,
):
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

    task: Task = get_task(task, fast, fastest, quiet)

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

    # For MR always run 3mm model for roi_subset, because 6mm too bad results
    #  (runtime for 3mm still very good for MR)
    if task.task.endswith("_mr") and roi_subset is not None:
        roi_subset_robust = roi_subset
        robust_rs = True

    if roi_subset_robust is not None:
        roi_subset = roi_subset_robust
        robust_rs = True
    else:
        robust_rs = False

    if roi_subset is not None and not isinstance(roi_subset, list):
        raise ValueError("roi_subset must be a list of strings")
    if roi_subset is not None and not task.task.startswith("total"):
        raise ValueError("roi_subset only works with task 'total' or 'total_mr'")

    if task.task.endswith("_mr"):
        if body_seg:
            body_seg = False
            print("INFO: For MR models the argument '--body_seg' is not supported and will be ignored.")

    # Generate rough organ segmentation (6mm) for speed up if crop or roi_subset is used
    # (for "fast" on GPU it makes no big difference, but on CPU it can help even for "fast")
    if task.crop is not None or roi_subset is not None or task.cascade:

        body_seg = False  # can not be used together with body_seg
        st = time.time()
        if not quiet: print("Generating rough segmentation for cropping...")
        if robust_rs or robust_crop:
            print("  (Using more robust (but slower) 3mm model for cropping.)")
            if task.task.endswith("_mr"):
                crop_task = get_task("total_mr", True, False)
            else:
                crop_task = get_task("total", True, False)
        else:
            # For MR always run 3mm model for cropping, because 6mm too bad results
            #  (runtime for 3mm still very good for MR)
            if task.task.endswith("_mr"):
                crop_task = get_task("total_mr", True, False)
            else:
                crop_task = get_task("total", True, False)
        if task.crop is not None and (
            "body_trunc" in task.crop or "body_extremities" in task.crop
        ):
            crop_task = get_task("body", True, False)

        organ_seg, _, _ = nnUNet_predict_image(
            input,
            None,
            crop_task.task_id,
            model="3d_fullres",
            folds=[0],
            trainer=crop_task.trainer,
            tta=False,
            multilabel_image=True,
            crop=None,
            crop_path=None,
            task_name=crop_task.task,
            nora_tag="None",
            preview=False,
            save_binary=False,
            nr_threads_resampling=nr_thr_resamp,
            nr_threads_saving=1,
            crop_addon=task.crop_addon,
            output_type=output_type,
            statistics=False,
            quiet=quiet,
            verbose=verbose,
            test=0,
            skip_saving=False,
            device=device,
        )
        class_map_inv = {v: k for k, v in class_map[crop_task.task].items()}
        crop_mask = np.zeros(organ_seg.shape, dtype=np.uint8)
        organ_seg_data = organ_seg.get_fdata()
        # roi_subset_crop = [map_to_total[roi] if roi in map_to_total else roi for roi in roi_subset]
        roi_subset_crop = task.crop if task.crop is not None else roi_subset
        for roi in roi_subset_crop:
            crop_mask[organ_seg_data == class_map_inv[roi]] = 1
        crop_mask = nib.Nifti1Image(crop_mask, organ_seg.affine)
        task.crop_addon = [20, 20, 20]
        task.crop = crop_mask
        task.cascade = crop_mask if task.cascade else None
        if verbose: print(f"Rough organ segmentation generated in {time.time()-st:.2f}s")

    # Generate rough body segmentation (6mm) (speedup for big images; not useful in combination with --fast option)
    if task.crop is None and body_seg:
        crop_task = get_task("body", True, False)
        st = time.time()
        if not quiet: print("Generating rough body segmentation...")
        body_seg, _, _ = nnUNet_predict_image(
            input,
            None,
            crop_task.task_id,
            model="3d_fullres",
            folds=[0],
            trainer="nnUNetTrainer",
            tta=False,
            multilabel_image=True,
            resample=6.0,
            crop=None,
            crop_path=None,
            task_name=crop_task.task,
            nora_tag="None",
            preview=False,
            save_binary=True,
            nr_threads_resampling=nr_thr_resamp,
            nr_threads_saving=1,
            crop_addon=task.crop_addon,
            output_type=output_type,
            statistics=False,
            quiet=quiet,
            verbose=verbose,
            test=0,
            skip_saving=False,
            device=device,
        )
        task.crop = body_seg
        if verbose: print(f"Rough body segmentation generated in {time.time()-st:.2f}s")

    seg_img, ct_img, stats = nnUNet_predict_image(
        input,
        output,
        task.task_id,
        model=task.model,
        folds=task.folds,
        trainer=task.trainer,
        tta=False,
        multilabel_image=ml,
        resample=task.resample,
        crop=task.crop,
        crop_path=crop_path,
        task_name=task.task,
        nora_tag=nora_tag,
        preview=preview,
        nr_threads_resampling=nr_thr_resamp,
        nr_threads_saving=nr_thr_saving,
        force_split=force_split,
        crop_addon=task.crop_addon,
        roi_subset=roi_subset,
        output_type=output_type,
        statistics=statistics_fast,
        quiet=quiet,
        verbose=verbose,
        test=test,
        skip_saving=skip_saving,
        device=device,
        exclude_masks_at_border=statistics_exclude_masks_at_border,
        no_derived_masks=no_derived_masks,
        v1_order=v1_order,
        stats_aggregation=stats_aggregation,
        remove_small_blobs=remove_small_blobs,
        normalized_intensities=statistics_normalized_intensities,
        nnunet_resampling=higher_order_resampling,
        save_probabilities=save_probabilities,
        cascade=task.cascade,
    )
    seg = seg_img.get_fdata().astype(np.uint8)

    try:
        # this can result in error if running multiple processes in parallel because all try to write the same file.
        # Trying to fix with lock from portalocker did not work. Network drive seems to not support this locking.
        config = increase_prediction_counter()
        send_usage_stats(
            config,
            {
                "task": task.task,
                "fast": fast,
                "preview": preview,
                "multilabel": ml,
                "roi_subset": roi_subset,
                "statistics": statistics,
                "radiomics": radiomics,
            },
        )
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
        stats = get_basic_statistics(
            seg,
            ct_img,
            stats_file,
            quiet,
            task,
            statistics_exclude_masks_at_border,
            roi_subset,
            metric=stats_aggregation,
            normalized_intensities=statistics_normalized_intensities,
        )
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
            get_radiomics_features_for_entire_dir(
                input_path, output, stats_dir / "statistics_radiomics.json"
            )
            if not quiet: print(f"  calculated in {time.time()-st:.2f}s")

    # Restore initial torch settings
    torch.backends.cudnn.benchmark = initial_cudnn_benchmark
    torch.set_num_threads(initial_num_threads)

    if statistics or statistics_fast:
        return seg_img, stats
    else:
        return seg_img
