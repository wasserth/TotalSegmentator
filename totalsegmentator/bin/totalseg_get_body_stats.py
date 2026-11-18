#!/usr/bin/env python
from pathlib import Path
import time
import argparse
import json
import tempfile

import nibabel as nib
import numpy as np

from totalsegmentator.cnn import (
    predict_all_body_stats_with_cnn,
)
from totalsegmentator.contrast_phase import pi_time_to_phase
from totalsegmentator.predictor import show_license_info
from totalsegmentator.config import (
    send_usage_stats_application,
    set_license_number,
    setup_totalseg,
)
from totalsegmentator.serialization_utils import filestream_to_nifti
from totalsegmentator.dicom_io import dcm_to_nifti

"""
More details: 
resources/body_stats_prediction.md
"""


def add_contrast_phase(result: dict) -> dict:
    """Map predicted pi_time to contrast_phase (phase string + probability)."""
    if "pi_time" not in result:
        return result
    phase, probability = pi_time_to_phase(result["pi_time"]["value"])
    contrast_phase = {
        "value": phase,
        "unit": None,
        "probability": probability,
    }
    inserted = {}
    for key, value in result.items():
        inserted[key] = value
        if key == "pi_time":
            inserted["contrast_phase"] = contrast_phase
    return inserted


def get_body_stats(img, modality: str, f_type: str = "niigz", model_file: Path = None,
                   quiet: bool = False, device: str = "cpu",
                   fold: int | str = "all", license_number: str = None,
                   only_weight: bool = False, skip_canonical: bool = False,
                   test_time_augmentation: bool = False, debug: bool = False):
    """
    Predict body weight, body size, age and sex based on a CT or MR scan.
    Also calculates BMI and body surface area based on the predicted values.

    This is a generator that yields progress dicts (like evans_index).

    Args:
        img: file path (Path) to nifti/dicom | nib.Nifti1Image | filestream
        modality: str, "ct" or "mr"
        f_type: "niigz" or "nii" or "dicom"
        model_file: Path, optional - CNN experiment directory
        quiet: bool, optional
        device: str, optional
        fold: int | "all", optional - default "all" (ensemble all 5 folds). Use 0-4 for a single fold.
        license_number: str, optional
        only_weight: bool, optional - if True, predict only body weight and skip all
            other targets and derived measures
        skip_canonical: bool, optional - if True, do not convert the input image to
            closest canonical orientation
        test_time_augmentation: bool, optional - if True, run the CNN model on
            flipped images along each axis, with and without canonical conversion,
            and ensemble predictions via mean
        debug: bool, optional - if True, print additional debugging information
    """
    if not quiet:
        print("\nIf you use this tool please cite: https://arxiv.org/abs/2608.29348\n")

    setup_totalseg()
    if license_number is not None:
        set_license_number(license_number)
    show_license_info()

    yield {"id": 1, "progress": 2, "status": "Loading data"}

    if isinstance(img, Path) and f_type != "dicom":
        img = nib.load(img)
    elif isinstance(img, nib.Nifti1Image):
        pass
    elif f_type == "dicom":
        print("Converting dicom to nifti...")
        with tempfile.TemporaryDirectory(prefix="totalseg_tmp_") as tmp_folder:
            ct_tmp_path = Path(tmp_folder) / "ct.nii.gz"
            dcm_to_nifti(img, ct_tmp_path, tmp_dir=Path(tmp_folder), verbose=True)
            img = nib.load(ct_tmp_path)
            img = nib.Nifti1Image(np.asanyarray(img.dataobj), img.affine, img.header)
    elif f_type == "niigz":
        img = filestream_to_nifti(img, gzipped=True)
    else:
        img = filestream_to_nifti(img, gzipped=False)

    yield {
        "id": 2,
        "progress": 75,
        "status": "Predicting body stats with CNN",
    }
    result = predict_all_body_stats_with_cnn(
        img, modality=modality, model_dir=model_file, fold=fold,
        device=device, skip_canonical=skip_canonical,
        test_time_augmentation=test_time_augmentation, debug=debug
    )
    if only_weight:
        result = {"weight": result["weight"]}

    if not only_weight:
        weight_kg = result["weight"]["value"]
        height_cm = result["size"]["value"]
        height_m = height_cm / 100.0
        # BMI = weight(kg) / height(m)^2
        bmi = weight_kg / (height_m ** 2)
        result["bmi"] = {"value": round(bmi, 2), "unit": "kg/m^2"}
        # Body Surface Area (Mosteller formula): BSA = sqrt(height(cm) x weight(kg) / 3600)
        bsa = float(np.sqrt((height_cm * weight_kg) / 3600))
        result["bsa"] = {"value": round(bsa, 2), "unit": "m^2"}

    result = add_contrast_phase(result)
    yield {"id": 3, "progress": 100, "status": "Done", "result": result}


def print_body_stats_result(result: dict):
    """Pretty-print body stats results as aligned key/value lines."""
    label_width = max(len(str(key)) for key in result) + 1
    for key, entry in result.items():
        value = entry["value"]
        unit = entry.get("unit")
        value_str = f"{value} {unit}" if unit else str(value)
        if key == "contrast_phase" and "probability" in entry:
            value_str += f"  (probability: {entry['probability']})"
        if key == "noise" and "noise_outlier" in entry:
            value_str += f"  (outlier: {entry['noise_outlier']})"
        print(f"{key + ':':<{label_width}}  {value_str}")


def main():
    """
    Predicts the body weight and size based on a CT or MR scan. Results are a lot better if the field of view 
    is large (e.g. entire abdomen and thorax).
    """
    parser = argparse.ArgumentParser(description="Get body stats.",
                                     epilog="Written by Jakob Wasserthal. If you use this tool please cite https://arxiv.org/abs/2608.29348")

    parser.add_argument("-i", metavar="filepath", dest="input_file",
                        help="path to CT/MR file (.nii.gz, .nii, or .zip for DICOM)",
                        type=lambda p: Path(p).absolute(), required=True)

    parser.add_argument("-o", metavar="filepath", dest="output_file",
                        help="path to output json file",
                        type=lambda p: Path(p).absolute(), required=False, default=None)
    
    parser.add_argument("-mf", metavar="filepath", dest="model_file",
                        help="path to CNN experiment directory",
                        type=lambda p: Path(p).absolute(), required=False, default=None)

    parser.add_argument("-m", metavar="modality", dest="modality", type=str, choices=["ct", "mr"], required=True,
                        help="Imaging modality: 'ct' or 'mr'")

    parser.add_argument("--only_weight", action="store_true", default=False,
                        help="Predict only body weight and skip size, age, sex, BMI, and BSA.")

    parser.add_argument("-sc", "--skip_canonical", action="store_true", default=False,
                        help="Do not convert the input image to closest canonical orientation.")

    parser.add_argument("-tta", "--test_time_augmentation", action="store_true", default=False,
                        help="Run six augmented predictions: input flipped along "
                             "each image axis, with and without closest-canonical conversion, and "
                             "ensemble them via mean.")

    parser.add_argument("--debug", action="store_true", default=False,
                        help="Print debugging information, including CNN input tensor shape.")

    parser.add_argument("-d",'--device', type=str, default="cpu",
                        help="Device type: 'gpu', 'cpu', 'mps', or 'gpu:X' where X is an integer representing the GPU device ID.")
    
    parser.add_argument("-f", "--fold", type=str, default="all",
                        help="Fold number (0-4) to use for prediction, or 'all' to ensemble all 5 folds. Default: all.")
    
    parser.add_argument("-q", dest="quiet", action="store_true",
                        help="Print no output to stdout", default=False)

    parser.add_argument("-l", "--license_number", type=str, default=None,
                        help="License number for the body stats models.")

    args = parser.parse_args()

    if str(args.input_file).endswith(".nii.gz"):
        f_type = "niigz"
    elif str(args.input_file).endswith(".zip"):
        f_type = "dicom"
    else:
        f_type = "nii"

    st = time.time()
    res_gen = get_body_stats(args.input_file, args.modality, f_type=f_type,
                             model_file=args.model_file, quiet=args.quiet, device=args.device,
                             fold=args.fold,
                             license_number=args.license_number,
                             only_weight=args.only_weight,
                             skip_canonical=args.skip_canonical,
                             test_time_augmentation=args.test_time_augmentation,
                             debug=args.debug)

    for r in res_gen:
        if not args.quiet:
            print(r['status'])
            if r["progress"] == 100:
                print(f"took {time.time() - st:.1f}s")
                print()
        if r["progress"] == 100:
            final_result = r

    res = final_result["result"]

    if not args.quiet:
        print_body_stats_result(res)

    if args.output_file is not None:
        with open(args.output_file, "w") as f:
            f.write(json.dumps(res, indent=4))

    send_usage_stats_application("get_body_stats")


if __name__ == "__main__":
    main()
