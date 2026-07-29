import sys
import os
import json
import importlib.metadata
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
from totalsegmentator.map_tasks_config import DEFAULT_CONFIG, TASK_CONFIGS
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


def convert_device_to_string(device):
    if hasattr(device, 'type'):  # torch.device object
        if device.type == "cuda":
            return "gpu"
        else:
            return device.type


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


def device_to_str(device):
    """Normalize a device value (torch.device or string) to 'gpu'/'cpu'/'mps' for reporting."""
    if hasattr(device, "type"):  # torch.device object
        return "gpu" if device.type == "cuda" else device.type
    return str(device)


def build_run_report(input, output, task, device, fast, fastest, ml, output_type,
                     roi_subset, runtime_seconds, save_lowres=False):
    """Assemble a machine-readable manifest describing a completed run.

    Pure function (no side effects): captures software versions, the resolved
    device, the run options, the classes produced (filtered by roi_subset when
    set) and the files written to the output directory. Used by the
    `--report` CLI option so that automation can verify and chain runs without
    parsing stdout.
    """
    from totalsegmentator.registry import task_modality, get_task_classes, requires_license, package_version

    try:
        nnunet_version = importlib.metadata.version("nnunetv2")
    except importlib.metadata.PackageNotFoundError:
        nnunet_version = None

    classes = get_task_classes(task)
    if roi_subset:
        classes = {idx: name for idx, name in classes.items() if name in roi_subset}

    output_files = []
    if output is not None and Path(output).is_dir():
        output_files = sorted(p.name for p in Path(output).glob("*.nii.gz"))

    return {
        "totalsegmentator_version": package_version(),
        "nnunetv2_version": nnunet_version,
        "torch_version": torch.__version__,
        "task": task,
        "modality": task_modality(task),
        "license_required": requires_license(task),
        "device": device_to_str(device),
        "fast": fast,
        "fastest": fastest,
        "save_lowres": save_lowres,
        "multilabel": ml,
        "output_type": output_type,
        "roi_subset": roi_subset,
        "input": "Nifti1Image" if isinstance(input, Nifti1Image) else str(input),
        "output": None if output is None else str(output),
        "num_classes": len(classes),
        "classes": {str(idx): name for idx, name in classes.items()},
        "runtime_seconds": round(runtime_seconds, 2),
        "output_files": output_files,
    }


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


def get_task_config(task, fast=False, fastest=False, quiet=False, robust_crop=None, plans=None, model_size=None):
    """Retrieves and builds task configurations dynamically based on parameters."""
    if task not in TASK_CONFIGS:
        raise ValueError(f"Unknown task: {task}")

    # Build base config using defaults
    config = DEFAULT_CONFIG.copy()
    raw_task = TASK_CONFIGS[task]

    # Explicit overrides for defaults from pass-through arguments
    if robust_crop is not None:
        config["robust_crop"] = robust_crop

    if plans is not None:
        config["plans"] = plans

    # Update config values for sub-mode tasks (fast / fastest / default)
    if "sub_modes" in raw_task:
        mode = "fast" if fast else ("fastest" if fastest else "default")
        if mode not in raw_task["sub_modes"]:
            # Fall back to default if 'fastest' requested but only 'fast' defined (e.g., body tasks)
            mode = "fast" if fast or fastest else "default"

        sub_config = raw_task["sub_modes"][mode]
        config.update(sub_config)

        if not quiet and mode in ["fast", "fastest"]:
            resample_mm = int(sub_config["resample"])
            print(f"Using '{mode}' option: resampling to lower resolution ({resample_mm}mm)")
    else:
        # Update config values for standard tasks
        config.update({k: v for k, v in raw_task.items() if k not in ("disallow_fast", "commercial", "info_msg")})

        if raw_task.get("disallow_fast") and fast:
            raise ValueError(f"task {task} does not work with option --fast")

    # Handle special case task-specific logic and options
    if task == "total_v3" and model_size == "small":
        config["plans"] = "nnUNetResEncUNetLPlans_8"

    if "info_msg" in raw_task and not quiet:
        print(raw_task["info_msg"])

    if raw_task.get("commercial"):
        show_license_info()

    return config

def totalsegmentator(input: Union[str, Path, Nifti1Image], output: Union[str, Path, None]=None, ml=False, nr_thr_resamp=1, nr_thr_saving=6,
                     fast=False, nora_tag="None", preview=False, task="total", roi_subset=None,
                     statistics: Union[bool, str, Path]=False, radiomics=False, crop_path=None, body_seg=False,
                     force_split=False, output_type="nifti", quiet=False, verbose=False, test=0,
                     skip_saving=False, device="gpu", license_number=None,
                     statistics_exclude_masks_at_border=True, no_derived_masks=False,
                     v1_order=False, fastest=False, roi_subset_robust=None, stats_aggregation="mean",
                     remove_small_blobs=False, statistics_normalized_intensities=False,
                     robust_crop=False, higher_order_resampling_LEGACY=False, higher_order_resampling=False,
                     save_probabilities=None,
                     debug=False, report=None, statistics_extra=False, save_lowres=False, resampling_order=1,
                     plans="nnUNetPlans", model_size="big"):
    """
    Run TotalSegmentator from within python.

    For explanation of the arguments see description of command line
    arguments in bin/TotalSegmentator.

    Return: multilabel Nifti1Image
    """
    run_start = time.time()
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
    
    if output_type == "dicom_rtstruct":
        try:
            from rt_utils import RTStructBuilder
        except ImportError:
            raise ImportError("rt_utils is required for output_type='dicom_rtstruct'. Please install it with 'pip install rt_utils'.")
    
    if output_type == "dicom_seg":
        try:
            import highdicom
        except ImportError:
            raise ImportError("highdicom is required for output_type='dicom_seg'. Please install it with 'pip install highdicom'.")

    if save_lowres and not (fast or fastest):
        raise ValueError("save_lowres only works together with fast or fastest mode.")

    output_types = [output_type] if isinstance(output_type, str) else list(output_type)
    if save_lowres and any(out_type in ["dicom_rtstruct", "dicom_seg"] for out_type in output_types):
        raise ValueError("save_lowres only supports nifti output.")

    if model_size not in ["big", "small"]:
        raise ValueError("model_size must be 'big' or 'small'.")
    if model_size == "small" and task != "total_v3":
        raise ValueError("model_size='small' is currently only supported for task 'total_v3'.")

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

    # Unpack the retrieved config directly into local variable names
    task_config = get_task_config(task, fast=fast, fastest=fastest, plans=plans, robust_crop=robust_crop, quiet=quiet, model_size=model_size)

    crop_model = task_config.get("crop_model")
    crop_addon = task_config.get("crop_addon")
    cascade = task_config.get("cascade")
    remove_outside = task_config.get("remove_outside")
    remove_outside_dilation = task_config.get("remove_outside_dilation")
    remove_mask = task_config.get("remove_mask")
    modality = task_config.get("modality")

    task_id = task_config.get("task_id")
    resample = task_config.get("resample")
    trainer = task_config.get("trainer")
    crop = task_config.get("crop")
    model = task_config.get("model")
    folds = task_config.get("folds")
    plans = task_config.get("plans")
    robust_crop = task_config.get("robust_crop")

    if crop_path is None:
        crop_path = output.parent if output is not None else None
    else:
        crop_path = crop_path

    if isinstance(input, Nifti1Image) or input.suffix == ".nii" or input.suffixes == [".nii", ".gz"]:
        img_type = "nifti"
    else:
        img_type = "dicom"

    # fast statistics are calculated on the downsampled image
    if statistics and fast:
        statistics_fast = statistics  # preserve path if provided
        statistics = False
    else:
        statistics_fast = False

    if type(task_id) is list:
        for tid in task_id:
            download_pretrained_weights(tid)
    else:
        download_pretrained_weights(task_id)
    if task == "vertebrae_pp_refined":
        download_pretrained_weights(305)

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

    if higher_order_resampling:
        resample = None
        save_lowres = False

    if save_lowres and (crop is not None or roi_subset is not None or cascade or body_seg):
        raise ValueError("save_lowres is not supported together with cropping, roi_subset, body_seg, or cascade.")

    # Generate rough organ segmentation (6mm) for speed up if crop or roi_subset is used
    # (for "fast" on GPU it makes no big difference, but on CPU it can help even for "fast")
    if crop is not None or roi_subset is not None or cascade:

        body_seg = False  # can not be used together with body_seg
        st = time.time()
        if not quiet: print("Generating rough segmentation for cropping...")

        if crop_model is None:  # use default "total" model for cropping
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
            if task.endswith("_mr") or modality == "mr":
                crop_task = "total_mr"
            else:
                crop_task = "total"
            crop_trainer = "nnUNetTrainer_2000epochs_NoMirroring" if task.endswith("_mr") else "nnUNetTrainer_4000epochs_NoMirroring"
            if crop is not None and ("body_trunc" in crop or "body_extremities" in crop):
                crop_model_task = 300
                crop_spacing = 6.0
                crop_trainer = "nnUNetTrainer"
                crop_task = "body"
            download_pretrained_weights(crop_model_task)
            
            organ_seg, _, _ = nnUNet_predict_image(input, None, crop_model_task, model="3d_fullres", folds=[0],
                                trainer=crop_trainer, tta=False, multilabel_image=True, resample=crop_spacing,
                                crop=None, crop_path=None, task_name=crop_task, nora_tag="None", preview=False,
                                save_binary=False, nr_threads_resampling=nr_thr_resamp, nr_threads_saving=1,
                                crop_addon=None, output_type=output_type, statistics=False,
                                quiet=quiet, verbose=verbose, test=0, skip_saving=False, device=device,
                                debug=debug, resampling_order=resampling_order)
            class_map_inv = {v: k for k, v in class_map[crop_task].items()}
            
        else:
            # If crop_model is specified, run totalsegmentator for the crop model
            organ_seg = totalsegmentator(input, None, task=crop_model, nr_thr_resamp=nr_thr_resamp, 
                                         device=convert_device_to_string(device), quiet=quiet, verbose=verbose,
                                         resampling_order=resampling_order)
            class_map_inv = {v: k for k, v in class_map[crop_model].items()}

        crop_mask = np.zeros(organ_seg.shape, dtype=np.uint8)
        organ_seg_data = organ_seg.get_fdata()
        # roi_subset_crop = [map_to_total[roi] if roi in map_to_total else roi for roi in roi_subset]
        roi_subset_crop = crop if crop is not None else roi_subset
        for roi in roi_subset_crop:
            crop_mask[organ_seg_data == class_map_inv[roi]] = 1
        crop_mask = nib.Nifti1Image(crop_mask, organ_seg.affine)
        crop_addon = [20,20,20] if crop_model is None else crop_addon  # default to 20,20,20 for roi_subset
        crop = crop_mask
        cascade = crop_mask if cascade else None

        if remove_outside is not None:
            remove_mask = np.zeros(organ_seg.shape, dtype=np.uint8)
            for roi in remove_outside:
                remove_mask[organ_seg_data == class_map_inv[roi]] = 1
            remove_mask = nib.Nifti1Image(remove_mask, organ_seg.affine)

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
                            quiet=quiet, verbose=verbose, test=0, skip_saving=False, device=device,
                            debug=debug, resampling_order=resampling_order)
        crop = body_seg
        if verbose: print(f"Rough body segmentation generated in {time.time()-st:.2f}s")

    vertebrae_body_mask = None
    if task == "vertebrae_pp_refined":
        st = time.time()
        if not quiet: print("Generating vertebrae body mask for refinement...")
        vertebrae_body_mask, _, _ = nnUNet_predict_image(input, None, 305, model="3d_fullres", folds=[0],
                            trainer="nnUNetTrainer_DASegOrd0", tta=False, multilabel_image=True, resample=1.5,
                            crop=None, crop_path=None, task_name="vertebrae_body", nora_tag="None", preview=False,
                            save_binary=False, nr_threads_resampling=nr_thr_resamp, nr_threads_saving=1,
                            crop_addon=crop_addon, output_type="nifti", statistics=False,
                            quiet=quiet, verbose=verbose, test=0, skip_saving=False, device=device,
                            debug=debug, resampling_order=resampling_order,
                            use_cropped_logits_resampling=higher_order_resampling)
        if verbose: print(f"Vertebrae body mask generated in {time.time()-st:.2f}s")

    prediction_task = "vertebrae_pp" if task == "vertebrae_pp_refined" else task
    seg_img, ct_img, stats = nnUNet_predict_image(input, output, task_id, model=model, folds=folds,
                            trainer=trainer, tta=False, multilabel_image=ml, resample=resample,
                            crop=crop, crop_path=crop_path, task_name=prediction_task, nora_tag=nora_tag, preview=preview,
                            nr_threads_resampling=nr_thr_resamp, nr_threads_saving=nr_thr_saving,
                            force_split=force_split, crop_addon=crop_addon, roi_subset=roi_subset,
                            output_type=output_type, statistics=statistics_fast,
                            quiet=quiet, verbose=verbose, test=test, skip_saving=skip_saving, device=device,
                            exclude_masks_at_border=statistics_exclude_masks_at_border,
                            no_derived_masks=no_derived_masks, v1_order=v1_order,
                            stats_aggregation=stats_aggregation, remove_small_blobs=remove_small_blobs,
                            normalized_intensities=statistics_normalized_intensities,
                            higher_order_resampling_LEGACY=higher_order_resampling_LEGACY,
                            save_probabilities=save_probabilities,
                            cascade=cascade, remove_outside_mask=remove_mask, remove_outside_dilation=remove_outside_dilation,
                            debug=debug, save_lowres=save_lowres and (fast or fastest),
                            resampling_order=resampling_order, plans=plans,
                            vertebrae_body_mask=vertebrae_body_mask, output_task_name=task,
                            use_cropped_logits_resampling=higher_order_resampling)
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
        # Check if statistics is a custom path (string or Path) rather than just True
        if isinstance(statistics, (str, Path)):
            stats_file = Path(statistics).absolute()
        elif output is not None:
            # For DICOM output types, output is always a file path, so use parent directory
            if output_type in ["dicom_seg", "dicom_rtstruct"]:
                stats_dir = output.parent
            else:
                stats_dir = output.parent if ml else output
            stats_file = stats_dir / "statistics.json"
        else:
            stats_file = None
        stats = get_basic_statistics(seg, ct_img, stats_file,
                                     quiet, task, statistics_exclude_masks_at_border,
                                     roi_subset,
                                     metric=stats_aggregation,
                                     normalized_intensities=statistics_normalized_intensities,
                                     extra_metrics=statistics_extra)
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
                input_path = Path(tmp_folder) / "ct.nii.gz"
                nib.save(input, input_path)
            else:
                input_path = input
            get_radiomics_features_for_entire_dir(input_path, output, stats_dir / "statistics_radiomics.json")
            if not quiet: print(f"  calculated in {time.time()-st:.2f}s")

    if report is not None:
        report_data = build_run_report(input, output, task, device, fast, fastest, ml,
                                       output_type, roi_subset, time.time() - run_start,
                                       save_lowres=save_lowres and (fast or fastest))
        report_path = Path(report).absolute()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=2)
        if not quiet: print(f"Run report written to {report_path}")

    # Restore initial torch settings
    torch.backends.cudnn.benchmark = initial_cudnn_benchmark
    torch.set_num_threads(initial_num_threads)

    if statistics or statistics_fast:
        return seg_img, stats
    else:
        return seg_img

