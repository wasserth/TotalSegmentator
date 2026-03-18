#!/usr/bin/env python
import sys
from pathlib import Path
import time
import argparse
import json
import pickle
import tempfile
import subprocess
from pprint import pprint
import importlib.resources
import importlib.metadata

import nibabel as nib
import numpy as np

from totalsegmentator.python_api import totalsegmentator
from totalsegmentator.config import send_usage_stats_application
from totalsegmentator.serialization_utils import filestream_to_nifti, nib_load_eager
from totalsegmentator.dicom_io import dcm_to_nifti

"""
Additional requirements for this script:
xgboost
"""

def pi_time_to_phase(pi_time: float) -> str:
    """
    Convert the pi time to a phase and get a probability for the value.

    native: 0-10
    arterial_early: 10-30
    arterial_late:  30-60
    portal_venous:  60-100
    delayed: 100+

    returns: phase, probability
    """
    if pi_time < 5:
        return "native", 1.0
    elif pi_time < 10:
        return "native", 0.7
    elif pi_time < 20:
        return "arterial_early", 0.7
    elif pi_time < 30:
        return "arterial_early", 1.0
    elif pi_time < 50:
        return "arterial_late", 1.0
    elif pi_time < 60:
        return "arterial_late", 0.7  # in previous version: "portal_venous"
    elif pi_time < 70:
        return "portal_venous", 1.0
    elif pi_time < 90:
        return "portal_venous", 1.0
    elif pi_time < 100:
        return "portal_venous", 0.7
    else:
        return "portal_venous", 0.3
        # return "delayed", 0.7  # not enough good training data for this


def run_models_shell(ct_img, device="gpu", quiet=True):
    """Run TotalSegmentator models via subprocess instead of python_api.
    Required if calling from e.g. streamlit where python_api does not work properly.
    Yields (seg_img, stats) for each model. The headneck model is only run if the
    caller advances the generator past the first yield.
    """
    quiet_flag = "--quiet" if quiet else ""

    with tempfile.TemporaryDirectory(prefix="totalseg_phase_") as tmp_folder:
        tmp_dir = Path(tmp_folder)
        ct_img_path = tmp_dir / "ct.nii.gz"
        nib.save(ct_img, ct_img_path)

        stats_total_path = tmp_dir / "stats_total.json"
        seg_total_path = tmp_dir / "seg_total.nii.gz"
        subprocess.call(
            f"TotalSegmentator -i {ct_img_path} -o {seg_total_path} --ml --fast"
            f" -s {stats_total_path} --sa median -sii -nr 1 -ns 1 -d {device} {quiet_flag}",
            shell=True)
        seg_img = nib_load_eager(seg_total_path)
        with open(stats_total_path) as f:
            stats = json.load(f)
        yield seg_img, stats

        stats_hn_path = tmp_dir / "stats_hn.json"
        seg_hn_path = tmp_dir / "seg_hn.nii.gz"
        subprocess.call(
            f"TotalSegmentator -i {ct_img_path} -o {seg_hn_path} --ml"
            f" -t headneck_bones_vessels -s {stats_hn_path} --sa median -sii -nr 1 -ns 1 -d {device} {quiet_flag}",
            shell=True)
        seg_img_hn = nib_load_eager(seg_hn_path)
        with open(stats_hn_path) as f:
            stats_hn = json.load(f)
        yield seg_img_hn, stats_hn


def get_ct_contrast_phase(ct_img, f_type: str = "niigz", model_file: Path = None,
                          quiet: bool = False, device: str = "gpu", existing_stats: dict = None,
                          call_via_subprocess: bool = False):
    """
    Predict the contrast phase of a CT scan.

    This is a generator that yields progress dicts.

    Args:
        ct_img: file path (Path) to nifti/dicom | nib.Nifti1Image | filestream
        f_type: "niigz" or "nii" or "dicom"
        model_file: Path, optional
        quiet: bool, optional
        device: str, optional
        existing_stats: dict, optional
        call_via_subprocess: bool, optional - if True, run TotalSegmentator via subprocess
    """
    yield {"id": 1, "progress": 2, "status": "Loading data"}

    if isinstance(ct_img, Path) and f_type != "dicom":
        ct_img = nib.load(ct_img)
    elif isinstance(ct_img, nib.Nifti1Image):
        pass
    elif f_type == "dicom":
        print("Converting dicom to nifti...")
        with tempfile.TemporaryDirectory(prefix="totalseg_tmp_") as tmp_folder:
            ct_tmp_path = Path(tmp_folder) / "ct.nii.gz"
            dcm_to_nifti(ct_img, ct_tmp_path, tmp_dir=Path(tmp_folder), verbose=True)
            ct_img = nib.load(ct_tmp_path)
            ct_img = nib.Nifti1Image(np.asanyarray(ct_img.dataobj), ct_img.affine, ct_img.header)
    elif f_type == "niigz":
        ct_img = filestream_to_nifti(ct_img, gzipped=True)
    else:
        ct_img = filestream_to_nifti(ct_img, gzipped=False)

    organs = ["liver", "pancreas", "urinary_bladder", "gallbladder",
              "heart", "aorta", "inferior_vena_cava", "portal_vein_and_splenic_vein",
              "iliac_vena_left", "iliac_vena_right", "iliac_artery_left", "iliac_artery_right",
              "pulmonary_vein", "brain", "colon", "small_bowel"]
    
    organs_hn = ["internal_carotid_artery_right", "internal_carotid_artery_left",
                 "internal_jugular_vein_right", "internal_jugular_vein_left"]

    if call_via_subprocess:
        model_gen = run_models_shell(ct_img, device=device, quiet=quiet)

        if existing_stats is None:
            yield {"id": 2, "progress": 10, "status": "Running TotalSegmentator model"}
            _, stats = next(model_gen)
        else:
            stats = existing_stats

        if stats["brain"]["volume"] > 100:
            yield {"id": 3, "progress": 50, "status": "Running headneck model"}
            _, stats_hn = next(model_gen)
        else:
            stats_hn = {organ: {"intensity": 0.0} for organ in organs_hn}
    else:
        st = time.time()
        if existing_stats is None:
            yield {"id": 2, "progress": 10, "status": "Running TotalSegmentator model"}
            seg_img, stats = totalsegmentator(ct_img, None, ml=True, fast=True, statistics=True, 
                                            roi_subset=None, statistics_exclude_masks_at_border=False,
                                            quiet=True, stats_aggregation="median", device=device)
        else:
            stats = existing_stats
        if not quiet:
            print(f"  took: {time.time()-st:.2f}s")
        
        if stats["brain"]["volume"] > 100:
            yield {"id": 3, "progress": 50, "status": "Running headneck model"}
            st = time.time()
            seg_img_hn, stats_hn = totalsegmentator(ct_img, None, ml=True, fast=False, statistics=True, 
                                                    task="headneck_bones_vessels",
                                                    roi_subset=None, statistics_exclude_masks_at_border=False,
                                                    quiet=True, stats_aggregation="median")
            if not quiet:
                print(f"  took: {time.time()-st:.2f}s")
        else:
            stats_hn = {organ: {"intensity": 0.0} for organ in organs_hn}

    yield {"id": 4, "progress": 85, "status": "Predicting phase"}
    features = []
    for organ in organs:
        features.append(stats[organ]["intensity"])
    for organ in organs_hn:
        features.append(stats_hn[organ]["intensity"])

    if model_file is None:
        classifier_path = str(importlib.resources.files('totalsegmentator') / 'resources/contrast_phase_classifiers_2024_07_19.pkl')
    else: 
        classifier_path = model_file
    clfs = pickle.load(open(classifier_path, "rb"))

    preds = []
    for fold, clf in clfs.items():
        preds.append(clf.predict([features])[0])
    preds = np.array(preds)
    pi_time = round(float(np.mean(preds)), 2)
    pi_time_std = round(float(np.std(preds)), 4)

    phase, probability = pi_time_to_phase(pi_time)

    result = {"pi_time": pi_time, 
              "phase": phase, 
              "probability": probability, 
              "pi_time_min": round(float(preds.min()), 2), 
              "pi_time_max": round(float(preds.max()), 2),
              "stddev": pi_time_std
              }

    yield {"id": 5, "progress": 100, "status": "Done", "result": result}


def main():
    """
    Predicts the contrast phase of a CT scan. Specifically this script will predict the
    pi (post injection) time (in seconds) of a CT scan based on the intensity of different regions
    in the image. 
    """
    parser = argparse.ArgumentParser(description="Get CT contrast phase.",
                                     epilog="Written by Jakob Wasserthal. If you use this tool please cite https://pubs.rsna.org/doi/10.1148/ryai.230024")

    parser.add_argument("-i", metavar="filepath", dest="input_file",
                        help="path to CT file (.nii.gz, .nii, or .zip for DICOM)",
                        type=lambda p: Path(p).absolute(), required=True)

    parser.add_argument("-o", metavar="filepath", dest="output_file",
                        help="path to output json file",
                        type=lambda p: Path(p).absolute(), required=True)
    
    parser.add_argument("-m", metavar="filepath", dest="model_file",
                        help="path to classifier model",
                        type=lambda p: Path(p).absolute(), required=False, default=None)
    
    parser.add_argument("-s", metavar="filepath", dest="existing_stats",
                        help="path to existing statistics json file. The script will not run TotalSegmentator but use the existing statistics.",
                        type=lambda p: Path(p).absolute(), required=False, default=None)

    parser.add_argument("-d",'--device', type=str, default="gpu",
                        help="Device type: 'gpu', 'cpu', 'mps', or 'gpu:X' where X is an integer representing the GPU device ID.")
    
    parser.add_argument("-q", dest="quiet", action="store_true",
                        help="Print no output to stdout", default=False)

    parser.add_argument("--call_via_subprocess", action="store_true", default=False,
                        help="Run TotalSegmentator models via subprocess instead of python_api. "
                             "Slightly slower but required in some environments (e.g. streamlit).")

    args = parser.parse_args()

    if str(args.input_file).endswith(".nii.gz"):
        f_type = "niigz"
    elif str(args.input_file).endswith(".zip"):
        f_type = "dicom"
    else:
        f_type = "nii"

    existing_stats = json.load(open(args.existing_stats)) if args.existing_stats is not None else None

    res_gen = get_ct_contrast_phase(args.input_file, f_type=f_type, model_file=args.model_file,
                                    quiet=args.quiet, device=args.device,
                                    existing_stats=existing_stats,
                                    call_via_subprocess=args.call_via_subprocess)

    for r in res_gen:
        if not args.quiet:
            print(r['status'])
        if r["progress"] == 100:
            final_result = r

    res = final_result["result"]

    if not args.quiet:
        print("Result:")
        pprint(res)

    with open(args.output_file, "w") as f:
        f.write(json.dumps(res, indent=4))

    send_usage_stats_application("get_phase")


if __name__ == "__main__":
    main()
