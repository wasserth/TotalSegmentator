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
import xgboost as xgb

from totalsegmentator.cnn import (
    predict_body_stats_with_cnn,
)
from totalsegmentator.python_api import totalsegmentator
from totalsegmentator.config import get_totalseg_dir, get_weights_dir, send_usage_stats_application
from totalsegmentator.nifti_ext_header import load_multilabel_nifti
from totalsegmentator.libs import download_pretrained_weights
from totalsegmentator.serialization_utils import filestream_to_nifti, nib_load_eager
from totalsegmentator.dicom_io import dcm_to_nifti

"""
Additional requirements for this script:
xgboost

More details: 
resources/body_stats_prediction.md

Info for me:
Training script is in: predict_body_size/body_size_training.py (not public)
"""


def check_body_stats_models_exist():
    """Check if all body stats models exist."""
    models_dir = get_weights_dir() / "body_stats_models_2026_03_24"
    
    for modality in ["ct", "mr"]:
        for target in ["weight", "size", "age", "sex"]:
            base_path = models_dir / f"{target}_{modality}_classifiers_2026_03_24.json"
            # Check if all 5 folds exist
            for fold_idx in range(5):
                model_file = Path(f"{base_path}.{fold_idx}")
                # For future models do
                # model_file = Path(f"{base_path}_fold{fold_idx}.json")
                if not model_file.exists():
                    return False
    return True


def load_models(classifier_path, target, fold=None):
    clfs = {}
    # Determine which folds to load
    fold_indices = [fold] if fold is not None else range(5)
    
    # Load the specified fold(s)
    for fold_idx in fold_indices:
        if target == "sex":
            clf = xgb.XGBClassifier(device="cpu")
        else:
            clf = xgb.XGBRegressor(device="cpu")
        clf.load_model(f"{classifier_path}.{fold_idx}")
        # For future models do
        # clf.load_model(f"{classifier_path}_fold{fold_idx}.json")
        clfs[fold_idx] = clf
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


def run_models_shell(ct_img, modality, device="gpu", quiet=True, license_number=None):
    """Run TotalSegmentator models via subprocess instead of python_api.
    Required if calling from e.g. streamlit where python_api does not work properly.
    """
    quiet_flag = "--quiet" if quiet else ""
    task = "total" if modality == "ct" else "total_mr"
    task_tissue = "tissue_types" if modality == "ct" else "tissue_types_mr"

    with tempfile.TemporaryDirectory(prefix="totalseg_body_stats_") as tmp_folder:
        tmp_dir = Path(tmp_folder)
        ct_img_path = tmp_dir / "ct.nii.gz"
        nib.save(ct_img, ct_img_path)
        stats_total_path = tmp_dir / "stats_total.json"
        seg_total_path = tmp_dir / "seg_total.nii.gz"

        subprocess.call(
            f"TotalSegmentator -i {ct_img_path} -o {seg_total_path} --ml --fast"
            f" -ta {task} -s {stats_total_path} --sa median -nr 1 -ns 1 -d {device} {quiet_flag}",
            shell=True)
        seg_img = nib_load_eager(seg_total_path)
        with open(stats_total_path) as f:
            stats = json.load(f)
        yield "total", seg_img, stats

        vertebrae_seg_img = None
        if modality == "mr":
            stats_vert_path = tmp_dir / "stats_vertebrae.json"
            seg_vert_path = tmp_dir / "seg_vertebrae.nii.gz"
            subprocess.call(
                f"TotalSegmentator -i {ct_img_path} -o {seg_vert_path} --ml"
                f" -ta vertebrae_mr -s {stats_vert_path} --sa median -nr 1 -ns 1 -d {device} {quiet_flag}",
                shell=True)
            vertebrae_seg_img = nib_load_eager(seg_vert_path)
            with open(stats_vert_path) as f:
                vertebrae_stats = json.load(f)
            yield "vertebrae_mr", vertebrae_seg_img, vertebrae_stats

        stats_tissue_path = tmp_dir / "stats_tissue.json"
        seg_tissue_path = tmp_dir / "seg_tissue.nii.gz"
        license_flag = f"-l {license_number}" if license_number else ""
        subprocess.call(
            f"TotalSegmentator -i {ct_img_path} -o {seg_tissue_path} --ml"
            f" -ta {task_tissue} -s {stats_tissue_path} --sa median -nr 1 -ns 1 -d {device}"
            f" {quiet_flag} {license_flag}",
            shell=True)
        seg_img_tissue = nib_load_eager(seg_tissue_path)
        yield "tissue_types", seg_img_tissue, None

        task_body = "body" if modality == "ct" else "body_mr"
        seg_body_path = tmp_dir / "seg_body.nii.gz"
        subprocess.call(
            f"TotalSegmentator -i {ct_img_path} -o {seg_body_path} --ml"
            f" -ta {task_body} -nr 1 -ns 1 -d {device} {quiet_flag}",
            shell=True)
        seg_img_body = nib_load_eager(seg_body_path)
        yield "body", seg_img_body, None


def touches_border_2d(mask_2d):
    """Check if a 2D mask touches the image border (first/last 2 pixels along x or y)."""
    if np.any(mask_2d[:2, :]) or np.any(mask_2d[-2:, :]):
        return True
    if np.any(mask_2d[:, :2]) or np.any(mask_2d[:, -2:]):
        return True
    return False


def get_tissue_types_slices(ct_img, vertebrae_img, tissue_types_img, body_img, vertebrae, tissue_types, use_border=False):
    """
    At each vertebrae get the centroid and at this z-level cut one slice through all classes
    of the segmentation file. Calc stats (volume, intensity) for each class.
    (When calculating volume, assume that the slice is 1mm thick. Then the slices can be
    compared across images with different slice thicknesses)
    
    Args:
        ct_img: nib.Nifti1Image - CT image
        vertebrae_img: nib.Nifti1Image - Vertebrae segmentation image
        tissue_types_img: nib.Nifti1Image - Tissue types segmentation image
        body_img: nib.Nifti1Image - Body segmentation image
        vertebrae: list - List of vertebra names (e.g., ["vertebrae_C1", "vertebrae_C2", ...])
        tissue_types: list - List of tissue type names (e.g., ["subcutaneous_fat", "torso_fat", "skeletal_muscle"])
        use_border: bool - If False, set stats to 0 for slices where the mask touches the image border.
    
    Returns:
        dict - Dictionary with ROI names as keys and stats as values
    """
    ct_data = ct_img.get_fdata()
    
    # Load label maps from segmentation images
    _, vertebrae_map = load_multilabel_nifti(vertebrae_img)
    _, tissue_types_map = load_multilabel_nifti(tissue_types_img)
    _, body_map = load_multilabel_nifti(body_img)
    
    # Reverse maps: name -> id
    vertebrae_map_inv = {v: k for k, v in vertebrae_map.items()}
    tissue_types_map_inv = {v: k for k, v in tissue_types_map.items()}
    body_map_inv = {v: k for k, v in body_map.items()}
    
    vertebrae_data = vertebrae_img.get_fdata()
    tissue_types_data = tissue_types_img.get_fdata()
    body_data = body_img.get_fdata()
    body_trunc = body_data == body_map_inv["body_trunc"]
    
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
        seg_slice = tissue_types_data[:, :, z_centroid].copy()
        body_trunc_slice = body_trunc[:, :, z_centroid]
        seg_slice[~body_trunc_slice] = 0
        
        for seg_name in tissue_types:
            seg_id = tissue_types_map_inv[seg_name]
            seg_mask = seg_slice == seg_id
            
            if seg_mask.sum() == 0 or (not use_border and touches_border_2d(seg_mask)):
                if not use_border and seg_mask.sum() > 0 and touches_border_2d(seg_mask):
                    print(f"Skipping {seg_name}_{vert_name} because it touches the image border.")
                stats_tissue_slices[f"{seg_name}_{vert_name}"] = {
                    "volume": 0.0,
                    "intensity": 0.0
                }
            else:
                volume = (seg_mask.sum() * slice_vol).round(3)
                intensity_median = np.median(ct_slice[seg_mask]).round(5)
                stats_tissue_slices[f"{seg_name}_{vert_name}"] = {
                    "volume": volume,
                    "intensity": intensity_median
                }
    
    return stats_tissue_slices


def get_body_stats(img, modality: str, f_type: str = "niigz", model_file: Path = None,
                   quiet: bool = False, device: str = "gpu", 
                   existing_stats: dict = None, existing_seg_img: nib.Nifti1Image = None,
                   fold: int = None, license_number: str = None, use_border: bool = False,
                   call_via_subprocess: bool = False, model_type: str = "xgboost",
                   only_weight: bool = False):
    """
    Predict body weight, body size, age and sex based on a CT or MR scan.
    Also calculates BMI and body surface area based on the predicted values.

    This is a generator that yields progress dicts (like evans_index).

    Args:
        img: file path (Path) to nifti/dicom | nib.Nifti1Image | filestream
        modality: str, "ct" or "mr"
        f_type: "niigz" or "nii" or "dicom"
        model_file: Path, optional
        quiet: bool, optional
        device: str, optional
        existing_stats: dict, optional
        existing_seg_img: nib.Nifti1Image, optional
        fold: int, optional - if set, only this fold is used; if None, ensemble of all folds is used
        license_number: str, optional
        use_border: bool, optional
        call_via_subprocess: bool, optional - if True, run TotalSegmentator via subprocess
        model_type: str, optional - "xgboost" for the existing feature-based model,
            "cnn" to use the 5-fold CNN ensemble for weight prediction
        only_weight: bool, optional - if True, predict only body weight and skip all
            other targets and derived measures
    """
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

    if model_type == "cnn" and modality != "mr":
        raise ValueError("The CNN body-stats models currently only support MR images.")

    needs_default_xgboost_models = model_type == "xgboost" and model_file is None
    if needs_default_xgboost_models and not check_body_stats_models_exist():
        download_pretrained_weights("body_stats")

    img = nib.as_closest_canonical(img)  # important to cut tissue slices along correct axis

    if model_type == "cnn":
        result = {}
        targets = ["weight"] if only_weight else ["weight", "size", "age", "sex"]
        target_progress = {
            "weight": 35,
            "size": 55,
            "age": 75,
            "sex": 90,
        }
        for target in targets:
            yield {
                "id": 2,
                "progress": target_progress[target],
                "status": f"Predicting {target} with CNN ensemble",
            }
            result[target] = predict_body_stats_with_cnn(
                img, target=target, model_dir=model_file, fold=fold, device=device
            )

        if not only_weight:
            weight_kg = result["weight"]["value"]
            height_cm = result["size"]["value"]
            height_m = height_cm / 100.0
            bmi = weight_kg / (height_m ** 2)
            result["bmi"] = {"value": round(bmi, 2), "unit": "kg/m^2"}
            bsa = float(np.sqrt((height_cm * weight_kg) / 3600))
            result["bsa"] = {"value": round(bsa, 2), "unit": "m^2"}

        yield {"id": 3, "progress": 100, "status": "Done", "result": result}
        return

    st = time.time()
    vertebrae_seg_img = None

    if call_via_subprocess:
        model_gen = run_models_shell(img, modality, device=device, quiet=quiet,
                                     license_number=license_number)

        if existing_stats is None:
            yield {"id": 2, "progress": 5, "status": "Running TotalSegmentator model"}
            _, seg_img, stats = next(model_gen)
        else:
            stats = existing_stats
            seg_img = existing_seg_img

        if modality == "mr":
            yield {"id": 3, "progress": 30, "status": "Running vertebrae_mr model"}
            _, vertebrae_seg_img, vertebrae_stats = next(model_gen)
            stats.update(vertebrae_stats)

        yield {"id": 4, "progress": 55, "status": "Running tissue types model"}
        _, seg_img_tissue, _ = next(model_gen)
        yield {"id": 5, "progress": 65, "status": "Running body model"}
        _, seg_img_body, _ = next(model_gen)
    else:
        if existing_stats is None:
            yield {"id": 2, "progress": 5, "status": "Running TotalSegmentator model"}
            task = "total" if modality == "ct" else "total_mr"
            seg_img, stats = totalsegmentator(img, None, ml=True, fast=True, statistics=True, 
                                              task=task,
                                              roi_subset=None, statistics_exclude_masks_at_border=True,
                                              quiet=True, stats_aggregation="median", device=device,
                                              nr_thr_resamp=1, nr_thr_saving=1)
        else:
            stats = existing_stats
            seg_img = existing_seg_img
        
        if not quiet:
            print(f"  took: {time.time()-st:.2f}s")

        if modality == "mr":
            yield {"id": 3, "progress": 30, "status": "Running vertebrae_mr model"}
            st = time.time()
            vertebrae_seg_img, vertebrae_stats = totalsegmentator(
                                                    img, None, ml=True, fast=False, statistics=True,
                                                    task="vertebrae_mr",
                                                    roi_subset=None, statistics_exclude_masks_at_border=True,
                                                    quiet=True, stats_aggregation="median", device=device,
                                                    nr_thr_resamp=1, nr_thr_saving=1)
            stats.update(vertebrae_stats)
            if not quiet:
                print(f"  took: {time.time()-st:.2f}s")

        yield {"id": 4, "progress": 55, "status": "Running tissue types model"}
        st = time.time()
        task_tissue = "tissue_types" if modality == "ct" else "tissue_types_mr"
        seg_img_tissue, stats_tissue = totalsegmentator(img, None, ml=True, fast=False, statistics=True, 
                                                task=task_tissue,
                                                roi_subset=None, statistics_exclude_masks_at_border=True,
                                                quiet=True, stats_aggregation="median",
                                                nr_thr_resamp=1, nr_thr_saving=1,
                                                license_number=license_number)
        if not quiet:
            print(f"  took: {time.time()-st:.2f}s")

        yield {"id": 5, "progress": 65, "status": "Running body model"}
        st = time.time()
        task_body = "body" if modality == "ct" else "body_mr"
        seg_img_body = totalsegmentator(img, None, ml=True, fast=False, statistics=False,
                                        task=task_body,
                                        roi_subset=None,
                                        quiet=True,
                                        nr_thr_resamp=1, nr_thr_saving=1,
                                        device=device)
        if not quiet:
            print(f"  took: {time.time()-st:.2f}s")

    if modality == "ct":
        stats = combine_lung_lobes(stats)

    yield {"id": 6, "progress": 75, "status": "Computing tissue slices"}
    vertebrae_img_for_slices = vertebrae_seg_img if vertebrae_seg_img is not None else seg_img
    stats_tissue_slices = get_tissue_types_slices(
        img, vertebrae_img_for_slices, seg_img_tissue, seg_img_body, vertebrae, tissue_types, use_border=use_border
    )

    yield {"id": 7, "progress": 85, "status": "Preparing features and predicting"}
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
    # print("DEBUG: Feature names and values:")
    # for name, value in zip(features_names, features):
    #     print(f"  {name}: {value}")

    result = {}
    targets = ["weight"] if only_weight else ["weight", "size", "age", "sex"]
    for target in targets:
    # for target in ["weight"]:
        if not quiet:
            print(f"Predicting {target}...")
        if model_file is None:
            classifier_path = get_weights_dir() / f'body_stats_models_2026_03_24/{target}_{modality}_classifiers_2026_03_24.json'
        else: 
            classifier_path = model_file
            
        clfs = load_models(classifier_path, target, fold)

        # ensemble across folds
        if target == "sex":
            # For classification: use predict_proba and ensemble the probabilities
            probs = []
            for fold_idx, clf in clfs.items():
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
            for fold_idx, clf in clfs.items():
                preds.append(clf.predict([features])[0])  # very fast
            preds = np.array(preds)
            pred = round(float(np.mean(preds)), 2)
            pred_std = round(float(np.std(preds)), 4)

            result[target] = {"value": pred, 
                              "min": round(float(preds.min()), 2), 
                              "max": round(float(preds.max()), 2),
                              "stddev": pred_std,  # measure of uncertainty
                              "unit": "kg" if target == "weight" else "cm" if target == "size" else "years" if target == "age" else None
                              }
    
    if not only_weight:
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
    
    yield {"id": 8, "progress": 100, "status": "Done", "result": result}


def main():
    """
    Predicts the body weight and size based on a CT or MR scan. Results are a lot better if the field of view 
    is large (e.g. entire abdomen and thorax).
    """
    parser = argparse.ArgumentParser(description="Get body stats.",
                                     epilog="Written by Jakob Wasserthal. If you use this tool please cite https://pubs.rsna.org/doi/10.1148/ryai.230024")

    parser.add_argument("-i", metavar="filepath", dest="input_file",
                        help="path to CT/MR file (.nii.gz, .nii, or .zip for DICOM)",
                        type=lambda p: Path(p).absolute(), required=True)

    parser.add_argument("-o", metavar="filepath", dest="output_file",
                        help="path to output json file",
                        type=lambda p: Path(p).absolute(), required=False, default=None)
    
    parser.add_argument("-mf", metavar="filepath", dest="model_file",
                        help="path to classifier model base path (xgboost) or experiment directory (cnn)",
                        type=lambda p: Path(p).absolute(), required=False, default=None)

    parser.add_argument("-m", metavar="modality", dest="modality", type=str, choices=["ct", "mr"], required=True,
                        help="Imaging modality: 'ct' or 'mr'")

    parser.add_argument("-mt", "--model_type", type=str, choices=["xgboost", "cnn"], default="xgboost",
                        help="Prediction backend: 'xgboost' or 'cnn'.")

    parser.add_argument("--only_weight", action="store_true", default=False,
                        help="Predict only body weight and skip size, age, sex, BMI, and BSA.")

    # This does not work, because fast total model anyways has to run to get the vertebrae segmentation
    # needed to get the tissue slices. Only works if also providing existing seg image (can be passed
    # as arg to get_body_stats function). But also important to run TotalSegmentator with identical settings,
    # otherwise significant change in model performance.
    # parser.add_argument("-s", metavar="filepath", dest="existing_stats",
    #                     help="path to existing statistics json file. The script will not run TotalSegmentator but use the existing statistics.",
    #                     type=lambda p: Path(p).absolute(), required=False, default=None)

    parser.add_argument("-d",'--device', type=str, default="gpu",
                        help="Device type: 'gpu', 'cpu', 'mps', or 'gpu:X' where X is an integer representing the GPU device ID.")
    
    parser.add_argument("-f", "--fold", type=int, default=None,
                        help="Fold number (0-4) to use for prediction. If not set, ensemble of all folds is used.")
    
    parser.add_argument("-q", dest="quiet", action="store_true",
                        help="Print no output to stdout", default=False)

    parser.add_argument("-l", "--license_number", type=str, default=None,
                        help="License number for tasks that require a license (e.g. tissue_types).")

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

    existing_stats = None

    res_gen = get_body_stats(args.input_file, args.modality, f_type=f_type,
                             model_file=args.model_file, quiet=args.quiet, device=args.device,
                             existing_stats=existing_stats, fold=args.fold,
                             license_number=args.license_number,
                             call_via_subprocess=args.call_via_subprocess,
                             model_type=args.model_type,
                             only_weight=args.only_weight)

    for r in res_gen:
        if not args.quiet:
            print(r['status'])
        if r["progress"] == 100:
            final_result = r

    res = final_result["result"]

    if not args.quiet:
        print("Result:")
        pprint(res)

    if args.output_file is not None:
        with open(args.output_file, "w") as f:
            f.write(json.dumps(res, indent=4))

    send_usage_stats_application("get_body_stats")


if __name__ == "__main__":
    main()
