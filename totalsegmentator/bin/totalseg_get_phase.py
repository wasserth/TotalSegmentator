#!/usr/bin/env python
import sys
from pathlib import Path
import time
import argparse
import json
import pickle
from pprint import pprint
import pkg_resources

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
        return "portal_venous", 0.3
        # return "delayed", 0.7  # not enough good training data for this


def get_ct_contrast_phase(ct_img: nib.Nifti1Image, model_file: Path = None):

    organs = ["liver", "pancreas", "urinary_bladder", "gallbladder",
              "heart", "aorta", "inferior_vena_cava", "portal_vein_and_splenic_vein",
              "iliac_vena_left", "iliac_vena_right", "iliac_artery_left", "iliac_artery_right",
              "pulmonary_vein", "brain", "colon", "small_bowel"]
    
    organs_hn = ["internal_carotid_artery_right", "internal_carotid_artery_left",
                 "internal_jugular_vein_right", "internal_jugular_vein_left"]

    st = time.time()
    seg_img, stats = totalsegmentator(ct_img, None, ml=True, fast=True, statistics=True, 
                                      roi_subset=None, statistics_exclude_masks_at_border=False,
                                      quiet=True, stats_aggregation="median")
    # print(f"ts took: {time.time()-st:.2f}s")
    
    if stats["brain"]["volume"] > 100:
        # print("Brain in image, therefore also running headneck model.")
        st = time.time()
        seg_img_hn, stats_hn = totalsegmentator(ct_img, None, ml=True, fast=False, statistics=True, 
                                                task="headneck_bones_vessels",
                                                roi_subset=None, statistics_exclude_masks_at_border=False,
                                                quiet=True, stats_aggregation="median")
        # print(f"hn took: {time.time()-st:.2f}s")
    else:
        stats_hn = {organ: {"intensity": 0.0} for organ in organs_hn}

    features = []
    for organ in organs:
        features.append(stats[organ]["intensity"])
    for organ in organs_hn:
        features.append(stats_hn[organ]["intensity"])

    if model_file is None:
        # classifier_path = Path(__file__).parents[2] / "resources" / "contrast_phase_classifiers_2024_07_19.pkl"
        classifier_path = pkg_resources.resource_filename('totalsegmentator', 'resources/contrast_phase_classifiers_2024_07_19.pkl')
    else: 
        # manually set model file
        classifier_path = model_file
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

    return {"pi_time": pi_time, 
            "phase": phase, 
            "probability": probability, 
            "pi_time_min": round(float(preds.min()), 2), 
            "pi_time_max": round(float(preds.max()), 2),
            "stddev": pi_time_std  # measure of uncertainty
            }


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
    
    parser.add_argument("-m", metavar="filepath", dest="model_file",
                        help="path to classifier model",
                        type=lambda p: Path(p).absolute(), required=False, default=None)
    
    parser.add_argument("-q", dest="quiet", action="store_true",
                        help="Print no output to stdout", default=False)

    args = parser.parse_args()

    res = get_ct_contrast_phase(nib.load(args.input_file), args.model_file)

    if not args.quiet:
        print("Result:")
        pprint(res)

    with open(args.output_file, "w") as f:
        f.write(json.dumps(res, indent=4))


if __name__ == "__main__":
    main()
