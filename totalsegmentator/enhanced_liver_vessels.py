"""
Enhanced liver vessel post-processing utilities with robust fallbacks.

This module now:
- Accepts a missing or unreadable liver mask and falls back to a full-volume mask.
- Handles empty vessel predictions gracefully.
- Produces metadata even when fallback logic is used.
"""

import json
import numpy as np
import nibabel as nib
from pathlib import Path
import datetime
import warnings


class EnhancedLiverVesselProcessor:
    def __init__(self, contrast_phase=None, min_component_size=40):
        self.contrast_phase = contrast_phase
        self.min_component_size = min_component_size

    def enhance_vessel_connectivity(self, vessel_mask, liver_mask):
        # Restrict vessels to liver region (if liver_mask is not empty)
        if liver_mask.sum() > 0:
            vessel_mask = vessel_mask * liver_mask

        # Remove very small components (try scipy if available)
        if vessel_mask.sum() == 0:
            return vessel_mask

        try:
            from scipy import ndimage
            labeled, ncomp = ndimage.label(vessel_mask)
            if ncomp > 0:
                sizes = ndimage.sum(vessel_mask, labeled, range(1, ncomp + 1))
                for idx, sz in enumerate(sizes, start=1):
                    if sz < self.min_component_size:
                        vessel_mask[labeled == idx] = 0
        except Exception:
            # No scipy -> skip refinement
            pass
        return vessel_mask

    def optimize_for_contrast_phase(self, ct_data, vessel_mask):
        # Placeholder for future intensity / phase–specific adjustments.
        return vessel_mask


def _load_nifti_safely(path: Path, purpose: str):
    try:
        img = nib.load(str(path))
        return img
    except FileNotFoundError:
        warnings.warn(f"[EnhancedLiver] {purpose} file not found at {path}. Using fallback.")
    except Exception as e:
        warnings.warn(f"[EnhancedLiver] Failed loading {purpose} ({path}): {e}. Using fallback.")
    return None


def _create_full_mask_like(img):
    data = img.get_fdata()
    return np.ones(data.shape, dtype=np.uint8)


def process_enhanced_liver_vessels(
    input_path,
    output_path,
    liver_mask_path=None,
    device="auto",
    allow_fallback_full_liver=True,
    min_component_size=40
):
    """
    Run liver_vessels task + light enhancement and save output + metadata.

    Parameters
    ----------
    input_path : str or Path
        Path to CT NIfTI.
    output_path : str or Path
        Output NIfTI path for enhanced vessel mask.
    liver_mask_path : str or Path or None
        Optional precomputed liver mask. If missing/unreadable and allow_fallback_full_liver=True,
        a full-volume mask is used.
    device : str
        Device spec forwarded to underlying totalsegmentator calls.
    allow_fallback_full_liver : bool
        If True, absence of a liver mask will not crash; we use full volume.
    min_component_size : int
        Minimum size (voxels) for connected vessel components to keep.
    """
    from totalsegmentator.python_api import totalsegmentator

    input_path = Path(input_path)
    output_path = Path(output_path)

    ct_img = nib.load(str(input_path))
    ct_data = ct_img.get_fdata()

    # Liver mask handling
    liver_mask = None
    liver_mask_source = None
    if liver_mask_path is not None:
        liver_mask_img = _load_nifti_safely(Path(liver_mask_path), "liver mask")
        if liver_mask_img is not None:
            liver_mask = (liver_mask_img.get_fdata() > 0).astype(np.uint8)
            liver_mask_source = "provided"
    if liver_mask is None:
        if allow_fallback_full_liver:
            liver_mask = _create_full_mask_like(ct_img)
            liver_mask_source = "fallback_full_volume"
            warnings.warn("[EnhancedLiver] Using full-volume mask as fallback (not ideal for real inference).")
        else:
            raise FileNotFoundError("Liver mask missing and fallback disabled.")

    # Vessel prediction
    try:
        vessels_res_img = totalsegmentator(
            str(input_path),
            None,
            task="liver_vessels",
            ml=True,
            device=device,
            robust_crop=True
        )
        vessel_mask_raw = (vessels_res_img.get_fdata() == 1).astype(np.uint8)
        vessel_prediction_ok = True
    except Exception as e:
        warnings.warn(f"[EnhancedLiver] liver_vessels task failed: {e}. Creating empty mask.")
        vessel_mask_raw = np.zeros(ct_data.shape, dtype=np.uint8)
        vessel_prediction_ok = False

    processor = EnhancedLiverVesselProcessor(min_component_size=min_component_size)
    vessel_mask_enh = processor.enhance_vessel_connectivity(vessel_mask_raw, liver_mask)
    vessel_mask_enh = processor.optimize_for_contrast_phase(ct_data, vessel_mask_enh)

    # Save enhanced vessel mask
    out_img = nib.Nifti1Image(vessel_mask_enh, ct_img.affine, ct_img.header)
    nib.save(out_img, str(output_path))

    voxel_volume = float(np.prod(ct_img.header.get_zooms()))
    metadata = {
        "input": str(input_path),
        "output": str(output_path),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "contrast_phase": processor.contrast_phase,
        "voxels": int(vessel_mask_enh.sum()),
        "voxel_volume_mm3": voxel_volume,
        "volume_mm3": float(vessel_mask_enh.sum() * voxel_volume),
        "enhancement_steps": ["liver_intersection", "small_component_removal_if_scipy"],
        "liver_mask_source": liver_mask_source,
        "vessel_prediction_ok": vessel_prediction_ok,
        "min_component_size": processor.min_component_size,
    }

    meta_path = output_path.with_name(output_path.name.replace(".nii.gz", "_metadata.json"))
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return out_img, metadata