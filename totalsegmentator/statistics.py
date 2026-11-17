import pathlib
from pathlib import Path
import json
from functools import partial
from typing import Union

import numpy as np
from scipy import ndimage
import nibabel as nib
from nibabel.nifti1 import Nifti1Image
from tqdm import tqdm

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
    stats = [get_radiomics_features(mask, ct_file) for mask in masks]
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


def _empty_extra_metrics(target):
    target["n_voxels"] = 0
    target["intensity_std"] = 0.0
    target["intensity_min"] = 0.0
    target["intensity_max"] = 0.0
    target["centroid_vox"] = None
    target["bbox_vox"] = None


def _add_extra_metrics(target, data, ct, offset=(0, 0, 0)):
    """Add morphometric + intensity-distribution metrics for one structure.

    centroid_vox and bbox_vox are in voxel/index coordinates (numpy axes i,j,k);
    bbox_vox is [[i_min,i_max],[j_min,j_max],[k_min,k_max]] with inclusive maxima.
    data and ct can be a crop of the image that starts at index `offset`.
    """
    idx = np.argwhere(data) + np.asarray(offset)
    if idx.shape[0] == 0:
        _empty_extra_metrics(target)
        return
    vals = ct[data]
    target["n_voxels"] = int(idx.shape[0])
    target["intensity_std"] = float(np.std(vals, dtype=np.float64).round(5))
    target["intensity_min"] = round(float(vals.min()), 5)
    target["intensity_max"] = round(float(vals.max()), 5)
    target["centroid_vox"] = [round(float(c), 2) for c in idx.mean(axis=0)]
    target["bbox_vox"] = [[int(idx[:, a].min()), int(idx[:, a].max())] for a in range(idx.shape[1])]


def get_basic_statistics(seg: np.array,
                         ct_file: Union[Path, Nifti1Image],
                         file_out: Union[Path, None]=None,
                         quiet: bool=False,
                         task: str="total",
                         exclude_masks_at_border: bool=True,
                         roi_subset: list=None,
                         metric: str="mean",
                         normalized_intensities: bool=False,
                         extra_metrics: bool=False):
    """
    ct_file: path to a ct_file or a nifti file object

    extra_metrics: in addition to volume and intensity, also compute
        n_voxels, intensity_std/min/max and the morphometric centroid_vox and
        bbox_vox (both in voxel/index coordinates). Off by default to keep the
        default runtime unchanged.
    """
    ct_img = nib.load(ct_file) if isinstance(ct_file, (str, Path)) else ct_file
    # float32: no truncation of fractional (MR) intensities, no int16 overflow above 32767; caching="unchanged": do not
    # leave a copy in the image
    ct = ct_img.get_fdata(dtype=np.float32, caching="unchanged")
    spacing = ct_img.header.get_zooms()
    vox_vol = spacing[0] * spacing[1] * spacing[2]
    
    if normalized_intensities:
        ct = (ct - ct.min()) / (ct.max() - ct.min())

    class_map_stats = class_map[task]
    if roi_subset is not None:
        class_map_stats = {k: v for k, v in class_map_stats.items() if v in roi_subset}

    # One pass over the volume for every class: voxel counts and intensity sums per label (bincount, in slabs to
    # bound the int64 copy bincount makes), and the labels in the 3-voxel border shell (a mask there is incomplete).
    # Same values as one full-volume pass per class, up to float summation order.
    seg = np.asarray(seg)
    if not np.issubdtype(seg.dtype, np.integer):
        seg = seg.astype(np.int32)
    n = max(int(seg.max()) if seg.size else 0, max(class_map_stats.keys(), default=0)) + 1
    # in the arrays' own memory layout where they share one (NIfTI's): no transposing copies
    order = "F" if seg.flags.f_contiguous and ct.flags.f_contiguous else "C"
    flat_seg, flat_ct = seg.ravel(order=order), ct.ravel(order=order)
    counts, sums = np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.float64)
    for start in range(0, flat_seg.size, 1 << 22):
        s = flat_seg[start:start + (1 << 22)]
        counts += np.bincount(s, minlength=n)
        sums += np.bincount(s, weights=flat_ct[start:start + (1 << 22)], minlength=n)
    on_border = np.zeros(n, dtype=bool)
    for axis in range(3):
        for border in (slice(None, 3), slice(-3, None)):
            index = [slice(None)] * 3
            index[axis] = border
            on_border[np.unique(seg[tuple(index)])] = True
    # the per-class work that is left (median, extra metrics) only looks at the class's bounding box
    boxes = ndimage.find_objects(seg, max_label=n - 1) if metric == "median" or extra_metrics else None

    stats = {}
    for k, mask_name in tqdm(class_map_stats.items(), disable=quiet):
        stats[mask_name] = {}
        if on_border[k] and exclude_masks_at_border:
            # print(f"WARNING: {mask_name} touches border. Skipping.")
            stats[mask_name]["volume"] = 0.0
            stats[mask_name]["intensity"] = 0.0
            if extra_metrics:
                _empty_extra_metrics(stats[mask_name])
        else:
            stats[mask_name]["volume"] = counts[k] * vox_vol  # vol in mm3
            if metric == "mean":
                stats[mask_name]["intensity"] = float((sums[k] / counts[k]).round(5)) if counts[k] > 0 else 0.0
            elif metric == "median":
                stats[mask_name]["intensity"] = round(float(np.median(ct[boxes[k - 1]][seg[boxes[k - 1]] == k])), 5) \
                    if counts[k] > 0 else 0.0
            if extra_metrics:
                if counts[k] > 0:
                    box = boxes[k - 1]
                    _add_extra_metrics(stats[mask_name], seg[box] == k, ct[box], [s.start for s in box])
                else:
                    _empty_extra_metrics(stats[mask_name])

    if file_out is not None:
        with open(file_out, "w") as f:
            json.dump(stats, f, indent=4)

    return stats
