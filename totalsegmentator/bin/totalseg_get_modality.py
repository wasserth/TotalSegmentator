#!/usr/bin/env python
import sys
from pathlib import Path
import time
import argparse
import json
import pickle
from pprint import pprint
import importlib.resources
import importlib.metadata
import xgboost as xgb

import nibabel as nib
import numpy as np

from totalsegmentator.python_api import totalsegmentator
from totalsegmentator.config import send_usage_stats_application

"""
Additional requirements for this script:
xgboost
"""

def get_features(nifti_img):
    data = nifti_img.get_fdata()

    mean = np.mean(data)
    std = np.std(data)
    min = np.min(data)
    max = np.max(data)
    
    return [mean, std, min, max]


# only use image level intensity features. Faster and high accuracy if image intensities not normalized (original HU values)
def get_modality(img: nib.Nifti1Image):
    """
    Predict modality
    
    returns: 
        prediction: "ct" | "mr"
        probability: float
    """
    st = time.time()
    features = get_features(img)  # 5s for big ct image
    # print(f"features took: {time.time() - st:.2f}s")

    classifier_path = str(importlib.resources.files('totalsegmentator') / 'resources/modality_classifiers_2025_02_24.json')
    clfs = {}
    for fold in range(5):  # assuming 5 folds
        clf = xgb.XGBClassifier()
        clf.load_model(f"{classifier_path}.{fold}")
        clfs[fold] = clf

    # ensemble across folds
    preds = []
    for fold, clf in clfs.items():
        preds.append(clf.predict([features])[0])
    preds = np.array(preds)
    preds = np.mean(preds)
    prediction_str = "ct" if preds < 0.5 else "mr"
    probability = 1 - preds if preds < 0.5 else preds
    return {"modality": prediction_str, 
            "probability": float(probability)}


# use normalized intensities only within rois; slower but also works if HU values are normalized
def get_modality_from_rois(img: nib.Nifti1Image, device: str = "gpu"):
    """
    Predict modality
    
    returns: 
        prediction: "ct" | "mr"
        probability: float
    """
    st = time.time()

    organs = ["brain", "esophagus", "colon", "spinal_cord", 
              "scapula_left", "scapula_right", 
              "femur_left", "femur_right", "hip_left", "hip_right", 
              "gluteus_maximus_left", "gluteus_maximus_right", 
              "autochthon_left", "autochthon_right", 
              "iliopsoas_left", "iliopsoas_right"]

    seg_img, stats = totalsegmentator(img, None, ml=True, fast=True, statistics=True, task="total_mr",
                                      roi_subset=None, statistics_exclude_masks_at_border=False,
                                      quiet=True, stats_aggregation="median", statistics_normalized_intensities=True,
                                      device=device)

    features = []
    for organ in organs:
        features.append(stats[organ]["intensity"])
    # print(f"TS took: {time.time() - st:.2f}s")

    classifier_path = str(importlib.resources.files('totalsegmentator') / 'resources/modality_classifiers_normalized_2025_02_24.json')
    clfs = {}
    for fold in range(5):  # assuming 5 folds
        clf = xgb.XGBClassifier()
        clf.load_model(f"{classifier_path}.{fold}")
        clfs[fold] = clf

    # ensemble across folds
    preds = []
    for fold, clf in clfs.items():
        preds.append(clf.predict([features])[0])
    preds = np.array(preds)
    preds = np.mean(preds)
    prediction_str = "ct" if preds < 0.5 else "mr"
    probability = 1 - preds if preds < 0.5 else preds
    return {"modality": prediction_str, 
            "probability": float(probability)}


def main():
    """
    Predicts the modality of a MR/CT scan. Looks at the image intensities.
    """
    parser = argparse.ArgumentParser(description="Get modality.",
                                     epilog="Written by Jakob Wasserthal. If you use this tool please cite https://pubs.rsna.org/doi/10.1148/ryai.230024")

    parser.add_argument("-i", metavar="filepath", dest="input_file",
                        help="path to CT/MR file",
                        type=lambda p: Path(p).absolute(), required=True)

    parser.add_argument("-o", metavar="filepath", dest="output_file",
                        help="path to output json file",
                        type=lambda p: Path(p).absolute(), required=True)

    parser.add_argument("-d",'--device', type=str, default="gpu",
                        help="Device type: 'gpu', 'cpu', 'mps', or 'gpu:X' where X is an integer representing the GPU device ID.")
        
    parser.add_argument("-q", dest="quiet", action="store_true",
                        help="Print no output to stdout", default=False)

    # Use this option if want to get modality of a image which has been normalized (does not contain original HU values anymore)
    parser.add_argument("-n", dest="normalized_intensities", action="store_true",
                        help="Use normalized intensities within rois for prediction", default=False)

    args = parser.parse_args()

    if args.normalized_intensities:
        res = get_modality_from_rois(nib.load(args.input_file), args.device)
    else:
        res = get_modality(nib.load(args.input_file))

    if not args.quiet:
        print("Result:")
        pprint(res)

    with open(args.output_file, "w") as f:
        f.write(json.dumps(res, indent=4))

    send_usage_stats_application("get_modality")


if __name__ == "__main__":
    main()
