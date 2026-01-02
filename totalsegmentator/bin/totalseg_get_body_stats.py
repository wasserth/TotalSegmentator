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

import nibabel as nib
import numpy as np
import xgboost as xgb

from totalsegmentator.python_api import totalsegmentator
from totalsegmentator.config import get_totalseg_dir, send_usage_stats_application
from totalsegmentator.nifti_ext_header import load_multilabel_nifti

"""
Additional requirements for this script:
xgboost

More details: 
resources/body_stats_prediction.md

Info for me:
Training script is in: predict_body_size/body_size_training.py (not public)
"""


def load_models(classifier_path, target):
    clfs = {}
    for fold in range(5):  # assuming 5 folds
        if target == "sex":
            clf = xgb.XGBClassifier(device="cpu")
        else:
            clf = xgb.XGBRegressor(device="cpu")
        clf.load_model(f"{classifier_path}.{fold}")
        clfs[fold] = clf
    return clfs


def combine_lung_lobes(stats):
    """Combine 5 lung lobes into left and right lung ROIs for CT data."""
    lobe_names_right = ['lung_upper_lobe_right', 'lung_middle_lobe_right', 'lung_lower_lobe_right']
    lobe_names_left = ['lung_upper_lobe_left', 'lung_lower_lobe_left']
    
    # Combine volumes by summing
    right_total_vol = sum([stats[lobe]["volume"] for lobe in lobe_names_right])
    left_total_vol = sum([stats[lobe]["volume"] for lobe in lobe_names_left])
    
    # Combine intensities by taking weighted average (weighted by volume)
    right_weighted_sum = sum(stats[lobe]["intensity"] * stats[lobe]["volume"] for lobe in lobe_names_right if stats[lobe]["volume"] > 0)
    right_vol_for_intensity = sum(stats[lobe]["volume"] for lobe in lobe_names_right if stats[lobe]["volume"] > 0)
    
    left_weighted_sum = sum(stats[lobe]["intensity"] * stats[lobe]["volume"] for lobe in lobe_names_left if stats[lobe]["volume"] > 0)
    left_vol_for_intensity = sum(stats[lobe]["volume"] for lobe in lobe_names_left if stats[lobe]["volume"] > 0)
    
    # Create combined lung entries
    stats["lung_right"] = {}
    stats["lung_right"]["volume"] = right_total_vol
    stats["lung_right"]["intensity"] = right_weighted_sum / right_vol_for_intensity if right_vol_for_intensity > 0 else 0.0

    stats["lung_left"] = {}
    stats["lung_left"]["volume"] = left_total_vol
    stats["lung_left"]["intensity"] = left_weighted_sum / left_vol_for_intensity if left_vol_for_intensity > 0 else 0.0
    
    return stats


def get_tissue_types_slices(ct_img, vertebrae_img, tissue_types_img, vertebrae, tissue_types):
    """
    At each vertebrae get the centroid and at this z-level cut one slice through all classes
    of the segmentation file. Calc stats (volume, intensity) for each class.
    (When calculating volume, assume that the slice is 1mm thick. Then the slices can be
    compared across images with different slice thicknesses)
    
    Args:
        ct_img: nib.Nifti1Image - CT image
        vertebrae_img: nib.Nifti1Image - Vertebrae segmentation image
        tissue_types_img: nib.Nifti1Image - Tissue types segmentation image
        vertebrae: list - List of vertebra names (e.g., ["vertebrae_C1", "vertebrae_C2", ...])
        tissue_types: list - List of tissue type names (e.g., ["subcutaneous_fat", "torso_fat", "skeletal_muscle"])
    
    Returns:
        dict - Dictionary with ROI names as keys and stats as values
    """
    ct_data = ct_img.get_fdata()
    
    # Load label maps from segmentation images
    _, vertebrae_map = load_multilabel_nifti(vertebrae_img)
    _, tissue_types_map = load_multilabel_nifti(tissue_types_img)
    
    # Reverse maps: name -> id
    vertebrae_map_inv = {v: k for k, v in vertebrae_map.items()}
    tissue_types_map_inv = {v: k for k, v in tissue_types_map.items()}
    
    vertebrae_data = vertebrae_img.get_fdata()
    tissue_types_data = tissue_types_img.get_fdata()
    
    spacing = ct_img.header.get_zooms()
    slice_vol = spacing[0] * spacing[1] * 1.0  # assume 1mm thickness
    
    stats_tissue_slices = {}
    for vert_name in vertebrae:      
        vert_id = vertebrae_map_inv[vert_name]
        vert_mask = vertebrae_data == vert_id
        
        if vert_mask.sum() == 0:
            for seg_name in tissue_types:
                stats_tissue_slices[f"{seg_name}_{vert_name}"] = {
                    "volume": 0.0,
                    "intensity": 0.0
                }
            continue
            
        # get centroid z-coordinate
        z_coords = np.where(vert_mask)[2]
        z_centroid = int(np.mean(z_coords))
        
        # extract slice at z_centroid
        ct_slice = ct_data[:, :, z_centroid]
        seg_slice = tissue_types_data[:, :, z_centroid]
        
        for seg_name in tissue_types:
            seg_id = tissue_types_map_inv[seg_name]
            seg_mask = seg_slice == seg_id
            
            if seg_mask.sum() == 0:
                stats_tissue_slices[f"{seg_name}_{vert_name}"] = {
                    "volume": 0.0,
                    "intensity": 0.0
                }
            else:
                volume = (seg_mask.sum() * slice_vol).round(3)
                intensity_mean = ct_slice[seg_mask].mean().round(5)
                stats_tissue_slices[f"{seg_name}_{vert_name}"] = {
                    "volume": volume,
                    "intensity": intensity_mean
                }
    
    return stats_tissue_slices


def get_body_stats(img: nib.Nifti1Image, modality: str, model_file: Path = None, 
                   quiet: bool = False, device: str = "gpu", 
                   existing_stats: dict = None, existing_seg_img: nib.Nifti1Image = None):
    """
    Predict body weight, body size, age and sex based on a CT or MR scan.
    Also calculates BMI and body surface area based on the predicted values.

    Args:
        img: nib.Nifti1Image
        modality: str, "ct" or "mr"
        model_file: Path, optional
        quiet: bool, optional
        device: str, optional
        existing_stats: dict, optional
        existing_seg_img: nib.Nifti1Image, optional
    """

    organs = ['gluteus_maximus_left', 'hip_right', 
              'spinal_cord', 'heart', 'spleen', 'hip_left', 'clavicula_left', 'scapula_left', 
              'gluteus_maximus_right', 'gallbladder', 'humerus_right', 
              'gluteus_minimus_right', 'autochthon_left', 'gluteus_minimus_left', 
              'scapula_right', 'femur_right', 'pancreas', 'prostate', 'aorta', 'liver', 
              'iliopsoas_left', 'clavicula_right', 
              'brain', 'gluteus_medius_left', 'humerus_left', 
              'gluteus_medius_right', 'kidney_left', 
              'femur_left', 'kidney_right', 'autochthon_right', 'iliopsoas_right', 'lung_left', 'lung_right']

    vertebrae = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 
                  'T9', 'T10', 'T11', 'T12', 'L1', 'L2', 'L3', 'L4', 'L5']
    vertebrae = [f"vertebrae_{roi}" for roi in vertebrae]

    tissue_types = ['subcutaneous_fat', 'torso_fat', 'skeletal_muscle']

    tissue_types_slices = [f"{tissue}_{vertebra}" for tissue in tissue_types for vertebra in vertebrae]

    st = time.time()
    if existing_stats is None:
        if not quiet:
            print("Running TotalSegmentator...")
        seg_img, stats = totalsegmentator(img, None, ml=True, fast=True, statistics=True, 
                                          roi_subset=None, statistics_exclude_masks_at_border=True,
                                          quiet=True, stats_aggregation="median", device=device)
    else:
        if not quiet:
            print("Using existing statistics...")
        stats = existing_stats
        seg_img = existing_seg_img

    if not quiet:
        print(f"  took: {time.time()-st:.2f}s")
    
    if modality == "ct":
        stats = combine_lung_lobes(stats)

    # Check FOV requirement: at least one of the key organs must be present
    fov_check_passed = (
        ("liver" in stats and stats["liver"]["volume"] > 10) or
        ("colon" in stats and stats["colon"]["volume"] > 10) or
        ("lung_left" in stats and stats["lung_left"]["volume"] > 10) or
        ("lung_right" in stats and stats["lung_right"]["volume"] > 10) or
        ("hip_left" in stats and stats["hip_left"]["volume"] > 10) or
        ("hip_right" in stats and stats["hip_right"]["volume"] > 10)
    )
    
    if not fov_check_passed:
        print(f"ERROR: Field of view too small for proper prediction. Stopping.")
        print(f"       At least one of these must be present: liver, colon, lung, hip")
        return None

    if not quiet:
        print("Running tissue types model...")
    st = time.time()
    seg_img_tissue, stats_tissue = totalsegmentator(img, None, ml=True, fast=False, statistics=True, 
                                            task="tissue_types",
                                            roi_subset=None, statistics_exclude_masks_at_border=True,
                                            quiet=True, stats_aggregation="median")
    if not quiet:
        print(f"  took: {time.time()-st:.2f}s")
    
    stats_tissue_slices = get_tissue_types_slices(img, seg_img, seg_img_tissue, vertebrae, tissue_types)

    if not quiet:
        print("Preparing features...")
    features = []
    features += [stats[roi]["volume"] for roi in organs + vertebrae]
    features += [stats_tissue_slices[roi]["volume"] for roi in tissue_types_slices]
    features += [stats[roi]["intensity"] for roi in organs + vertebrae]
    features += [stats_tissue_slices[roi]["intensity"] for roi in tissue_types_slices]

    # DEBUG: features have same order as in training
    # features_names = []
    # features_names += [roi + "_volume" for roi in organs + vertebrae]
    # features_names += [roi + "_volume" for roi in tissue_types_slices]
    # features_names += [roi + "_intensity" for roi in organs + vertebrae]
    # features_names += [roi + "_intensity" for roi in tissue_types_slices]
    # pprint(features_names)

    result = {}
    for target in ["weight", "size", "age", "sex"]:
        if not quiet:
            print(f"Predicting {target}...")
        if model_file is None:
            # classifier_path = str(importlib.resources.files('totalsegmentator') / f'resources/{target}_{modality}_classifiers_2025_12_19.json')
            classifier_path = get_totalseg_dir() / f'models/{target}_{modality}_classifiers_2025_12_19.json'
        else: 
            # manually set model file
            classifier_path = model_file
            
        # TODO: download models if not present
        clfs = load_models(classifier_path, target)

        # ensemble across folds
        if target == "sex":
            # For classification: use predict_proba and ensemble the probabilities
            probs = []
            for fold, clf in clfs.items():
                prob = clf.predict_proba([features])[0]  # returns [prob_class_0, prob_class_1]
                probs.append(prob[1])  # probability of class 1 (male)
            probs = np.array(probs)
            mean_prob = float(np.mean(probs))
            # Final prediction: threshold at 0.5, map to M/F (1=male, 0=female)
            pred_binary = int(mean_prob >= 0.5)
            pred = "M" if pred_binary == 1 else "F"
            # Invert probability for female to show confidence in the predicted class
            reported_prob = mean_prob if pred == "M" else 1.0 - mean_prob
            pred_std = round(float(np.std(probs)), 4)
            
            result[target] = {"value": pred,
                              "probability": round(reported_prob, 4),
                              "stddev": pred_std,
                              "unit": None
                              }
        else:
            # For regression: use predict and ensemble predictions
            preds = []
            for fold, clf in clfs.items():
                preds.append(clf.predict([features])[0])  # very fast
            preds = np.array(preds)
            pred = round(float(np.mean(preds)), 2)
            pred_std = round(float(np.std(preds)), 4)

            # print("Ensemble res:")
            # print(preds)
            # print(f"mean: {pred} +/- {pred_std}")
            # print(f"mean: {pred} [{preds.min():.1f}-{preds.max():.1f}]")

            result[target] = {"value": pred, 
                              "min": round(float(preds.min()), 2), 
                              "max": round(float(preds.max()), 2),
                              "stddev": pred_std,  # measure of uncertainty
                              "unit": "kg" if target == "weight" else "cm" if target == "size" else None
                              }
    
    # Calculate BMI and Body Surface Area based on predicted values
    weight_kg = result["weight"]["value"]
    height_cm = result["size"]["value"]
    height_m = height_cm / 100.0
    
    # BMI = weight(kg) / height(m)^2
    bmi = weight_kg / (height_m ** 2)
    result["bmi"] = {"value": round(bmi, 2), 
                     "unit": "kg/m^2"}
    
    # Body Surface Area (Mosteller formula): BSA = sqrt(height(cm) x weight(kg) / 3600)
    bsa = float(np.sqrt((height_cm * weight_kg) / 3600))
    result["bsa"] = {"value": round(bsa, 2), 
                     "unit": "m^2"}
    
    return result


def main():
    """
    Predicts the body weight and size based on a CT or MR scan. Results are a lot better if the field of view 
    is large (e.g. entire abdomen and thorax).
    """
    parser = argparse.ArgumentParser(description="Get body stats.",
                                     epilog="Written by Jakob Wasserthal. If you use this tool please cite https://pubs.rsna.org/doi/10.1148/ryai.230024")

    parser.add_argument("-i", metavar="filepath", dest="input_file",
                        help="path to CT/MR file",
                        type=lambda p: Path(p).absolute(), required=True)

    parser.add_argument("-o", metavar="filepath", dest="output_file",
                        help="path to output json file",
                        type=lambda p: Path(p).absolute(), required=False, default=None)
    
    parser.add_argument("-mf", metavar="filepath", dest="model_file",
                        help="path to classifier model",
                        type=lambda p: Path(p).absolute(), required=False, default=None)

    parser.add_argument("-m", metavar="modality", dest="modality", type=str, choices=["ct", "mr"], required=True,
                        help="Imaging modality: 'ct' or 'mr'")

    parser.add_argument("-s", metavar="filepath", dest="existing_stats",
                        help="path to existing statistics json file. The script will not run TotalSegmentator but use the existing statistics.",
                        type=lambda p: Path(p).absolute(), required=False, default=None)

    parser.add_argument("-d",'--device', type=str, default="gpu",
                        help="Device type: 'gpu', 'cpu', 'mps', or 'gpu:X' where X is an integer representing the GPU device ID.")
    
    parser.add_argument("-q", dest="quiet", action="store_true",
                        help="Print no output to stdout", default=False)

    args = parser.parse_args()

    existing_stats = json.load(open(args.existing_stats)) if args.existing_stats is not None else None

    res = get_body_stats(nib.load(args.input_file), args.modality, args.model_file, args.quiet, args.device, existing_stats)

    if res is None:
        # FOV check failed, processing was skipped
        return

    if not args.quiet:
        print("Result:")
        pprint(res)

    if args.output_file is not None:
        with open(args.output_file, "w") as f:
            f.write(json.dumps(res, indent=4))

    send_usage_stats_application("get_body_stats")


if __name__ == "__main__":
    main()
