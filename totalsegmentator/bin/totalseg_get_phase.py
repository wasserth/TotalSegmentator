#!/usr/bin/env python
import sys
from pathlib import Path
import argparse
import json
import pickle
from pprint import pprint

import nibabel as nib
import numpy as np

from totalsegmentator.python_api import totalsegmentator


"""
Additional requirements for this script:
xgboost
"""

def pi_time_to_phase(pi_time: float) -> str:
    """
    Convert the pi time to a phase and get a probability for the value.

    native: 0-10
    arterial_early: 10-30
    arterial_late:  30-50
    portal_venous:  50-100
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
        return "portal_venous", 0.7
    elif pi_time < 90:
        return "portal_venous", 1.0
    elif pi_time < 100:
        return "portal_venous", 0.7
    else:
        return "delayed", 0.7


def get_ct_contrast_phase(ct_img: nib.Nifti1Image):

    organs = ["liver", "spleen", "kidney_left", "kidney_right", "pancreas", "urinary_bladder", "gallbladder",
              "heart", "aorta", "inferior_vena_cava", "portal_vein_and_splenic_vein",
              "iliac_vena_left", "iliac_vena_right", "iliac_artery_left", "iliac_artery_right",
              "pulmonary_vein"]

    seg_img, stats = totalsegmentator(ct_img, None, ml=True, fast=True, statistics=True, 
                                      roi_subset=None, statistics_exclude_masks_at_border=False,
                                      quiet=True)

    features = []
    for organ in organs:
        features.append(stats[organ]["intensity"])

    # weights from longitudinalliver dataset
    classifier_path = Path(__file__).parents[2] / "resources" / "contrast_phase_classifiers.pkl"
    # classifier_path = "/mnt/nvme/data/phase_classification/classifiers.pkl"
    clfs = pickle.load(open(classifier_path, "rb"))

    # ensemble across folds
    preds = []
    for fold, clf in clfs.items():
        preds.append(clf.predict([features])[0])
    preds = np.array(preds)
    pi_time = round(float(np.mean(preds)), 2)
    pi_time_std = round(float(np.std(preds)), 4)

    # print("Ensemble res:")
    # print(preds)
    # print(f"mean: {pi_time} +/- {pi_time_std}")
    # print(f"mean: {pi_time} [{preds.min():.1f}-{preds.max():.1f}]")
    phase, probability = pi_time_to_phase(pi_time)

    return {"pi_time": pi_time, "phase": phase, "probability": probability, 
            "pi_time_min": round(float(preds.min()), 2), "pi_time_max": round(float(preds.max()), 2)}


def main():
    """
    Predicts the contrast phase of a CT scan. Specifically this script will predict the
    pi (post injection) time (in seconds) of a CT scan based on the intensity of different regions
    in the image. 
    """
    parser = argparse.ArgumentParser(description="Get CT contrast phase.",
                                     epilog="Written by Jakob Wasserthal. If you use this tool please cite https://pubs.rsna.org/doi/10.1148/ryai.230024")

    parser.add_argument("-i", metavar="filepath", dest="input_file",
                        help="path to CT file",
                        type=lambda p: Path(p).absolute(), required=True)

    parser.add_argument("-o", metavar="filepath", dest="output_file",
                        help="path to output json file",
                        type=lambda p: Path(p).absolute(), required=True)

    args = parser.parse_args()

    res = get_ct_contrast_phase(nib.load(args.input_file))

    print("Result:")
    pprint(res)

    with open(args.output_file, "w") as f:
        f.write(json.dumps(res, indent=4))


if __name__ == "__main__":
    main()
