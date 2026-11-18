import json
import importlib.metadata
from pathlib import Path
import time
from typing import Union
import tempfile
import re

import nibabel as nib
from nibabel.nifti1 import Nifti1Image
import torch
from totalsegmentator.statistics import get_radiomics_features_for_entire_dir
from totalsegmentator.config import increase_prediction_counter, send_usage_stats, get_config_key, set_config_key
from totalsegmentator.predictor import TotalSegmentatorPredictor, nifti_cases, output_file


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


def device_to_str(device):
    """Normalize a device value (torch.device or string) to 'gpu'/'cpu'/'mps' for reporting."""
    if hasattr(device, "type"):  # torch.device object
        return "gpu" if device.type == "cuda" else device.type
    return str(device)


def build_run_report(input, output, task, device, fast, fastest, ml, output_type,
                     roi_subset, runtime_seconds):
    """Assemble a machine-readable manifest describing a completed run.

    Pure function (no side effects): captures software versions, the resolved
    device, the run options, the classes produced (filtered by roi_subset when
    set) and the NIfTI output filenames. Used by the
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
        # relative paths: a folder of images without --ml writes <name>/<class>.nii.gz
        output_files = sorted(p.relative_to(output).as_posix() for p in Path(output).rglob("*.nii.gz"))
    elif output is not None and output_type == "nifti" and Path(output).is_file():
        output_files = [Path(output).name]

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


def totalsegmentator(input: Union[str, Path, Nifti1Image], output: Union[str, Path, None]=None, *, ml=False,
                     nr_thr_saving=6, fast=False, nora_tag="None", preview: Union[bool, str, Path]=False, task="total", roi_subset=None,
                     statistics: Union[bool, str, Path]=False, radiomics=False, output_type="nifti", quiet=False,
                     verbose=False, skip_saving=False, device="gpu", license_number=None,
                     statistics_exclude_masks_at_border=True, no_derived_masks=False, v1_order=False, fastest=False,
                     roi_subset_robust=None, stats_aggregation="mean", remove_small_blobs=False,
                     statistics_normalized_intensities=False, robust_crop=False, debug=False, report=None,
                     statistics_extra=False, model_size="big"):
    """
    Run TotalSegmentator from within python.

    For explanation of the arguments see description of command line
    arguments in bin/TotalSegmentator. All of them but input and output are
    keyword arguments (some arguments of earlier versions are gone, so
    positions would silently mean something else).

    input can also be a folder of NIfTI images: all of them are segmented in
    one run (the models are loaded once), and output is a folder that gets
    <case>.nii.gz per image with ml, else a folder <case>/ per image.

    Return: multilabel Nifti1Image (with statistics: (image, statistics)); a
    list of them for a folder (None per image when output is given).

    Every call loads the models of the task. To segment images from Python in
    several calls, keep a totalsegmentator.predictor.TotalSegmentatorPredictor.
    """
    run_start = time.time()
    if not isinstance(input, Nifti1Image):
        input = Path(input)
    # a folder that holds NIfTI images is a batch of cases; any other folder is one DICOM series
    cases = nifti_cases(input) if isinstance(input, Path) and input.is_dir() else None
    if output is not None:
        output = Path(output)
        if cases is not None and output.name.endswith((".nii", ".nii.gz")):
            raise ValueError("For a folder of images, the output is a folder (one result per image), not a file.")
    elif radiomics:
        raise ValueError("Output path is required for radiomics.")
    if radiomics and ml:
        raise ValueError("Radiomics not supported for multilabel segmentation. Use without --ml option.")
    if radiomics and cases is None and not (isinstance(input, Nifti1Image) or str(input).endswith((".nii", ".nii.gz"))):
        raise ValueError("Radiomics not supported for DICOM input. Use nifti input.")
    output_types = [output_type] if isinstance(output_type, str) else list(output_type)
    if cases is None:
        output = output_file(output, ml, output_types)      # the file the predictor writes, for the run report
    if "dicom_rtstruct" in output_types:
        try:
            from rt_utils import RTStructBuilder
        except ImportError:
            raise ImportError("rt_utils is required for output_type='dicom_rtstruct'. Please install it with 'pip install rt_utils'.")
    if "dicom_seg" in output_types:
        try:
            import highdicom
        except ImportError:
            raise ImportError("highdicom is required for output_type='dicom_seg'. Please install it with 'pip install highdicom'.")

    validate_device_type_api(device)
    device = select_device(device)
    if verbose: print(f"Using Device: {device}")
    if not quiet:
        print("\nIf you use this tool please cite: https://pubs.rsna.org/doi/10.1148/ryai.230024\n")

    if roi_subset_robust is not None:  # the class subset, located with the 3mm instead of the 6mm model
        roi_subset, robust_crop = roi_subset_robust, True
    try:
        with TotalSegmentatorPredictor(task, fast=fast, fastest=fastest, device=device, quiet=quiet, verbose=verbose,
                                       robust_crop=robust_crop, model_size=model_size,
                                       license_number=license_number) as predictor:
            if not get_config_key("statistics_disclaimer_shown"):  # True if not set (None) or set to False
                print("TotalSegmentator sends anonymous usage statistics. If you want to disable it check the documentation.")
                set_config_key("statistics_disclaimer_shown", True)
            if cases is not None:
                if not quiet: print(f"Found {len(cases)} images in {input}")
                images = [file for _, file in cases]
                outputs = None if output is None else [output / (f"{name}.nii.gz" if ml else name) for name, _ in cases]
            else:
                images, outputs = input, output
            result = predictor.predict(images, outputs, ml=ml, roi_subset=roi_subset, output_type=output_type,
                                       statistics=statistics, statistics_extra=statistics_extra,
                                       stats_aggregation=stats_aggregation,
                                       statistics_exclude_masks_at_border=statistics_exclude_masks_at_border,
                                       statistics_normalized_intensities=statistics_normalized_intensities,
                                       preview=preview, remove_small_blobs=remove_small_blobs, v1_order=v1_order,
                                       no_derived_masks=no_derived_masks, nr_thr_saving=nr_thr_saving,
                                       skip_saving=skip_saving, nora_tag="None" if nora_tag is None else nora_tag)
    except Exception:
        if debug:
            print(f"Error during prediction for input: {input}, task: {task}")
        raise
    try:
        # this can result in error if running multiple processes in parallel because all try to write the same file.
        config = increase_prediction_counter()
        send_usage_stats(config, {"task": task, "fast": fast, "preview": bool(preview),
                                  "multilabel": ml, "roi_subset": roi_subset,
                                  "statistics": statistics, "radiomics": radiomics})
    except Exception:
        pass

    if radiomics:
        if not quiet: print("Calculating radiomics...")
        with tempfile.TemporaryDirectory(prefix="radiomics_tmp_") as tmp_folder:
            input_path = input
            if isinstance(input, Nifti1Image):
                input_path = Path(tmp_folder) / "ct.nii.gz"
                nib.save(input, input_path)
            if cases is None:
                get_radiomics_features_for_entire_dir(input_path, output, output / "statistics_radiomics.json")
            else:
                for name, file in cases:
                    get_radiomics_features_for_entire_dir(file, output / name, output / name / "statistics_radiomics.json")

    if report is not None:
        report_data = build_run_report(input, output, task, device, fast, fastest, ml, output_type, roi_subset,
                                       time.time() - run_start)
        report_path = Path(report).absolute()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=2)
        if not quiet: print(f"Run report written to {report_path}")

    return result
