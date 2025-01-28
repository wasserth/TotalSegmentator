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

    classifier_path = pkg_resources.resource_filename('totalsegmentator', 'resources/modality_classifiers_2024_10_04.pkl')
    clfs = pickle.load(open(classifier_path, "rb"))

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
    
    parser.add_argument("-q", dest="quiet", action="store_true",
                        help="Print no output to stdout", default=False)

    args = parser.parse_args()

    res = get_modality(nib.load(args.input_file))

    if not args.quiet:
        print("Result:")
        pprint(res)

    with open(args.output_file, "w") as f:
        f.write(json.dumps(res, indent=4))

    send_usage_stats_application("get_modality")


if __name__ == "__main__":
    main()
