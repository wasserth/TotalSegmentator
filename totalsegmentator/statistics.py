import pathlib
from pathlib import Path
import json
from functools import partial
import time

import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
from p_tqdm import p_map
import numpy.ma as ma

from totalsegmentator.map_to_binary import class_map


def get_radiomics_features(seg_file, img_file="ct.nii.gz"):
    # import SimpleITK as sitk
    import radiomics
    from radiomics import featureextractor

    standard_features = ['shape_Elongation', 'shape_Flatness', 'shape_LeastAxisLength', 'shape_MajorAxisLength', 'shape_Maximum2DDiameterColumn', 'shape_Maximum2DDiameterRow', 'shape_Maximum2DDiameterSlice', 'shape_Maximum3DDiameter', 'shape_MeshVolume', 'shape_MinorAxisLength', 'shape_Sphericity', 'shape_SurfaceArea', 'shape_SurfaceVolumeRatio', 'shape_VoxelVolume', 'firstorder_10Percentile', 'firstorder_90Percentile', 'firstorder_Energy', 'firstorder_Entropy', 'firstorder_InterquartileRange', 'firstorder_Kurtosis', 'firstorder_Maximum', 'firstorder_MeanAbsoluteDeviation', 'firstorder_Mean', 'firstorder_Median', 'firstorder_Minimum', 'firstorder_Range', 'firstorder_RobustMeanAbsoluteDeviation', 'firstorder_RootMeanSquared', 'firstorder_Skewness', 'firstorder_TotalEnergy', 'firstorder_Uniformity', 'firstorder_Variance', 'glcm_Autocorrelation', 'glcm_ClusterProminence', 'glcm_ClusterShade', 'glcm_ClusterTendency', 'glcm_Contrast', 'glcm_Correlation', 'glcm_DifferenceAverage', 'glcm_DifferenceEntropy', 'glcm_DifferenceVariance', 'glcm_Id', 'glcm_Idm', 'glcm_Idmn', 'glcm_Idn', 'glcm_Imc1', 'glcm_Imc2', 'glcm_InverseVariance', 'glcm_JointAverage', 'glcm_JointEnergy', 'glcm_JointEntropy', 'glcm_MCC', 'glcm_MaximumProbability', 'glcm_SumAverage', 'glcm_SumEntropy', 'glcm_SumSquares', 'gldm_DependenceEntropy', 'gldm_DependenceNonUniformity', 'gldm_DependenceNonUniformityNormalized', 'gldm_DependenceVariance', 'gldm_GrayLevelNonUniformity', 'gldm_GrayLevelVariance', 'gldm_HighGrayLevelEmphasis', 'gldm_LargeDependenceEmphasis', 'gldm_LargeDependenceHighGrayLevelEmphasis', 'gldm_LargeDependenceLowGrayLevelEmphasis', 'gldm_LowGrayLevelEmphasis', 'gldm_SmallDependenceEmphasis', 'gldm_SmallDependenceHighGrayLevelEmphasis', 'gldm_SmallDependenceLowGrayLevelEmphasis', 'glrlm_GrayLevelNonUniformity', 'glrlm_GrayLevelNonUniformityNormalized', 'glrlm_GrayLevelVariance', 'glrlm_HighGrayLevelRunEmphasis', 'glrlm_LongRunEmphasis', 'glrlm_LongRunHighGrayLevelEmphasis', 'glrlm_LongRunLowGrayLevelEmphasis', 'glrlm_LowGrayLevelRunEmphasis', 'glrlm_RunEntropy', 'glrlm_RunLengthNonUniformity', 'glrlm_RunLengthNonUniformityNormalized', 'glrlm_RunPercentage', 'glrlm_RunVariance', 'glrlm_ShortRunEmphasis', 'glrlm_ShortRunHighGrayLevelEmphasis', 'glrlm_ShortRunLowGrayLevelEmphasis', 'glszm_GrayLevelNonUniformity', 'glszm_GrayLevelNonUniformityNormalized', 'glszm_GrayLevelVariance', 'glszm_HighGrayLevelZoneEmphasis', 'glszm_LargeAreaEmphasis', 'glszm_LargeAreaHighGrayLevelEmphasis', 'glszm_LargeAreaLowGrayLevelEmphasis', 'glszm_LowGrayLevelZoneEmphasis', 'glszm_SizeZoneNonUniformity', 'glszm_SizeZoneNonUniformityNormalized', 'glszm_SmallAreaEmphasis', 'glszm_SmallAreaHighGrayLevelEmphasis', 'glszm_SmallAreaLowGrayLevelEmphasis', 'glszm_ZoneEntropy', 'glszm_ZonePercentage', 'glszm_ZoneVariance', 'ngtdm_Busyness', 'ngtdm_Coarseness', 'ngtdm_Complexity', 'ngtdm_Contrast', 'ngtdm_Strength']
    
    try:
        if len(np.unique(nib.load(seg_file).get_fdata())) > 1:
            settings = {}
            # settings["binWidth"] = 25
            # settings["resampledPixelSpacing"] = None  # [3,3,3] is an example for defining resampling (voxels with size 3x3x3mm)
            settings["resampledPixelSpacing"] = [3,3,3]
            # settings["interpolator"] = sitk.sitkBSpline
            settings["geometryTolerance"] = 1e-3  # default: 1e-6
            settings["featureClass"] = ["shape"]
            extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
            # Only use subset of features
            extractor.disableAllFeatures()
            extractor.enableFeatureClassByName("shape")
            extractor.enableFeatureClassByName("firstorder")
            features = extractor.execute(str(img_file), str(seg_file))
            
            features = {k.replace("original_", ""): v for k, v in features.items() if k.startswith("original_")}
        else:
            print("WARNING: Entire mask is 0 or 1. Setting all features to 0")
            features = {feat: 0 for feat in standard_features}
    except Exception as e:
        print(f"WARNING: radiomics raised an exception (settings all features to 0): {e}")
        features = {feat: 0 for feat in standard_features}

    # only keep subset of features
    # meaningful_features = ['shape_Elongation', 'shape_Flatness', 'shape_LeastAxisLength']
    # features = {k: v for k, v in features.items() if k in meaningful_features}  

    features = {k: round(float(v), 4) for k, v in features.items()}  # round to 4 decimals and cast to python float
    
    return seg_file.name.split(".")[0], features


def get_radiomics_features_for_entire_dir(ct_file:Path, mask_dir:Path, file_out:Path):
    masks = sorted(list(mask_dir.glob("*.nii.gz")))
    stats = p_map(partial(get_radiomics_features, img_file=ct_file),
                    masks, num_cpus=1, disable=False)
    stats = {mask_name: stats for mask_name, stats in stats}
    with open(file_out, "w") as f:
        json.dump(stats, f, indent=4)


# def is_in_first_or_last_slice(mask):
#     """
#     Check if a mask is in the first or last slice of the image.
#     Then we do not calc any statistics for it because the mask
#     is incomplete.
#     """
#     # in first or last slice segmentation of bad. Therefore they one slice before and after.
#     first_slice = mask[:, :, 1]
#     last_slice = mask[:, :, -2]
#     return (first_slice.sum() > 0) or (last_slice.sum() > 0)


def touches_border(mask):
    """
    Check if mask touches any of the borders. Then we do not calc any statistics for it because the mask
    is incomplete.
    Do not check last slice by one previous, because segmentation on last slice often bad.
    """
    if np.any(mask[1, :, :]) or np.any(mask[-2, :, :]):
        return True
    if np.any(mask[:, 1, :]) or np.any(mask[:, -2, :]):
        return True
    if np.any(mask[:, :, 1]) or np.any(mask[:, :, -2]):
        return True
    return False


def get_basic_statistics(seg: np.array, ct_file, file_out: Path, quiet: bool = False,
                         task: str = "total", exclude_masks_at_border: bool = True):
    """
    ct_file: path to a ct_file or a nifti file object
    """
    ct_img = nib.load(ct_file) if type(ct_file) == pathlib.PosixPath else ct_file
    ct = ct_img.get_fdata()
    spacing = ct_img.header.get_zooms()
    vox_vol = spacing[0] * spacing[1] * spacing[2]
    stats = {}
    for k, mask_name in tqdm(class_map[task].items(), disable=quiet):
        stats[mask_name] = {}
        # data = nib.load(mask).get_fdata()  # loading: 0.6s
        data = seg == k  # 0.18s
        if touches_border(data) and exclude_masks_at_border:
            # print(f"WARNING: {mask_name} touches border. Skipping.")
            stats[mask_name]["volume"] = 0.0
            stats[mask_name]["intensity"] = 0.0
        else:
            stats[mask_name]["volume"] = data.sum() * vox_vol  # vol in mm3; 0.2s
            roi_mask = (data > 0).astype(np.uint8)  # 0.16s
            # stats[mask_name]["intensity"] = ct[roi_mask > 0].mean().round(2) if roi_mask.sum() > 0 else 0.0  # 3.0s
            stats[mask_name]["intensity"] = np.average(ct, weights=roi_mask).round(2) if roi_mask.sum() > 0 else 0.0  # 0.9s
    
    # For nora json is good
    # For other people csv might be better -> not really because here only for one subject each -> use json
    with open(file_out, "w") as f:
        json.dump(stats, f, indent=4)
    