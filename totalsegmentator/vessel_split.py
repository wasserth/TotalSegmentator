"""
Heuristic splitting of intrahepatic vessels into portal vein branches vs hepatic veins.

Outputs in output_dir:
  portal_vein_branches.nii.gz
  hepatic_veins.nii.gz
  liver_vessels_labeled.nii.gz  (1=portal, 2=hepatic)
  (optional) liver_vessels_skeleton_labeled.nii.gz

QC dict returned (and can be recorded in metadata):
  portal_voxels, hepatic_voxels, total_vessel_voxels,
  portal_fraction, hepatic_fraction,
  portal_seed_voxels, hepatic_seed_voxels, use_skeleton
"""

from __future__ import annotations
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy import ndimage
import warnings

try:
    from skimage.morphology import skeletonize_3d
    _HAVE_SKIMAGE = True
except Exception:
    _HAVE_SKIMAGE = False


def _load_mask(path: Path, binary=True):
    if path is None or not path.exists():
        return None, None
    img = nib.load(str(path))
    data = img.get_fdata()
    if binary:
        data = (data > 0).astype(np.uint8)
    return img, data


def _multi_source_geodesic(vessel_mask: np.ndarray,
                           portal_seed: np.ndarray,
                           hepatic_seed: np.ndarray) -> np.ndarray:
    shape = vessel_mask.shape
    labels = np.zeros(shape, dtype=np.uint8)

    frontier_portal = np.argwhere(portal_seed)
    frontier_hepatic = np.argwhere(hepatic_seed)

    if frontier_portal.size == 0 and frontier_hepatic.size == 0:
        return labels
    if frontier_portal.size == 0:
        labels[vessel_mask > 0] = 2
        return labels
    if frontier_hepatic.size == 0:
        labels[vessel_mask > 0] = 1
        return labels

    INF = 10**9
    d_portal = np.full(shape, INF, dtype=np.int32)
    d_hepatic = np.full(shape, INF, dtype=np.int32)
    for p in frontier_portal:
        d_portal[tuple(p)] = 0
    for h in frontier_hepatic:
        d_hepatic[tuple(h)] = 0

    neigh = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
    from collections import deque
    q_portal = deque([tuple(p) for p in frontier_portal])
    q_hepatic = deque([tuple(h) for h in frontier_hepatic])

    while q_portal:
        x,y,z = q_portal.popleft()
        base = d_portal[x,y,z] + 1
        for dx,dy,dz in neigh:
            nx,ny,nz = x+dx, y+dy, z+dz
            if 0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]:
                if vessel_mask[nx,ny,nz] and d_portal[nx,ny,nz] > base:
                    d_portal[nx,ny,nz] = base
                    q_portal.append((nx,ny,nz))
    while q_hepatic:
        x,y,z = q_hepatic.popleft()
        base = d_hepatic[x,y,z] + 1
        for dx,dy,dz in neigh:
            nx,ny,nz = x+dx, y+dy, z+dz
            if 0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]:
                if vessel_mask[nx,ny,nz] and d_hepatic[nx,ny,nz] > base:
                    d_hepatic[nx,ny,nz] = base
                    q_hepatic.append((nx,ny,nz))

    inside = np.where(vessel_mask > 0)
    for x,y,z in zip(*inside):
        dp = d_portal[x,y,z]
        dh = d_hepatic[x,y,z]
        if dp == INF and dh == INF:
            continue
        if dp < dh:
            labels[x,y,z] = 1
        elif dh < dp:
            labels[x,y,z] = 2
        else:
            labels[x,y,z] = 1  # tie-breaker

    return labels


def split_portal_hepatic(
    liver_vessels_path: Path,
    liver_path: Path | None,
    portal_trunk_path: Path | None,
    ivc_path: Path | None,
    output_dir: Path,
    use_skeleton: bool = True,
    portal_dilate_mm: float = 3.0,
    ivc_distance_mm: float = 5.0,
    min_component_voxels: int = 20
):
    img_lv, vessels = _load_mask(liver_vessels_path)
    if img_lv is None:
        raise FileNotFoundError(f"Liver vessels file not found: {liver_vessels_path}")
    if vessels.sum() == 0:
        warnings.warn("Liver vessel mask empty; skipping split.")
        return {
            "portal_voxels": 0,
            "hepatic_voxels": 0,
            "total_vessel_voxels": 0,
            "portal_fraction": 0.0,
            "hepatic_fraction": 0.0,
            "portal_seed_voxels": 0,
            "hepatic_seed_voxels": 0,
            "use_skeleton": False
        }

    _, liver = _load_mask(liver_path) if liver_path and liver_path.exists() else (None, None)
    if liver is None:
        # fallback full volume union (not ideal but allows code path)
        liver = np.ones_like(vessels, dtype=np.uint8)

    zooms = img_lv.header.get_zooms()[:3]
    voxel_avg = float(sum(zooms) / 3.0)

    # Portal seeds
    portal_seed = np.zeros_like(vessels, dtype=np.uint8)
    if portal_trunk_path and portal_trunk_path.exists():
        _, portal_trunk = _load_mask(portal_trunk_path)
        if portal_trunk is not None:
            portal_seed = (portal_trunk & liver & vessels).astype(np.uint8)
            if portal_seed.sum() == 0:
                dist = ndimage.distance_transform_edt(1 - portal_trunk)
                rad = portal_dilate_mm / voxel_avg
                portal_seed = ((dist <= rad) & liver & vessels).astype(np.uint8)

    # Hepatic seeds
    hepatic_seed = np.zeros_like(vessels, dtype=np.uint8)
    if ivc_path and ivc_path.exists():
        _, ivc = _load_mask(ivc_path)
        if ivc is not None:
            inv = 1 - ivc
            dist_ivc = ndimage.distance_transform_edt(inv, sampling=zooms)
            hepatic_seed = ((dist_ivc <= ivc_distance_mm) & vessels & liver).astype(np.uint8)

    # Fallback seeds
    if portal_seed.sum() == 0 and vessels.sum() > 0:
        coords = np.argwhere(vessels)
        if coords.size:
            z_cut = np.percentile(coords[:,2], 40)
            subset = coords[coords[:,2] <= z_cut]
            if subset.size:
                center = subset.mean(axis=0)
                dist = np.sqrt(((coords - center)**2).sum(1))
                sel = coords[dist < np.percentile(dist, 15)]
                portal_seed[tuple(sel.T)] = 1
    if hepatic_seed.sum() == 0 and vessels.sum() > 0:
        coords = np.argwhere(vessels)
        if coords.size:
            z_cut = np.percentile(coords[:,2], 70)
            subset = coords[coords[:,2] >= z_cut]
            if subset.size:
                center = subset.mean(axis=0)
                dist = np.sqrt(((coords - center)**2).sum(1))
                sel = coords[dist < np.percentile(dist, 20)]
                hepatic_seed[tuple(sel.T)] = 1

    # Skeleton path classification
    use_skel = bool(use_skeleton and _HAVE_SKIMAGE and vessels.sum() > 0)
    if use_skel:
        skel = skeletonize_3d(vessels.astype(bool)).astype(np.uint8)
        portal_seed = (portal_seed & skel) if portal_seed.sum() else portal_seed
        hepatic_seed = (hepatic_seed & skel) if hepatic_seed.sum() else hepatic_seed
        labels_skel = _multi_source_geodesic(skel, portal_seed, hepatic_seed)
        if (labels_skel > 0).any():
            # map each vessel voxel to nearest labeled skeleton voxel
            label_mask = labels_skel > 0
            dist, inds = ndimage.distance_transform_edt(1 - label_mask, return_indices=True)
            skx, sky, skz = inds
            full_labels = labels_skel[skx, sky, skz]
        else:
            full_labels = _multi_source_geodesic(vessels, portal_seed, hepatic_seed)
    else:
        full_labels = _multi_source_geodesic(vessels, portal_seed, hepatic_seed)

    # Remove tiny components per class
    def prune(labelmap, cls):
        mask = (labelmap == cls)
        if mask.sum() == 0:
            return
        lab, n = ndimage.label(mask)
        sizes = ndimage.sum(mask, lab, range(1, n+1))
        for idx, sz in enumerate(sizes, start=1):
            if sz < min_component_voxels:
                labelmap[lab == idx] = 0

    prune(full_labels, 1)
    prune(full_labels, 2)

    portal_mask = (full_labels == 1).astype(np.uint8)
    hepatic_mask = (full_labels == 2).astype(np.uint8)

    # Save
    nib.save(nib.Nifti1Image(portal_mask, img_lv.affine, img_lv.header),
             str(output_dir / "portal_vein_branches.nii.gz"))
    nib.save(nib.Nifti1Image(hepatic_mask, img_lv.affine, img_lv.header),
             str(output_dir / "hepatic_veins.nii.gz"))
    nib.save(nib.Nifti1Image(full_labels, img_lv.affine, img_lv.header),
             str(output_dir / "liver_vessels_labeled.nii.gz"))

    if use_skel and 'skel' in locals():
        skel_labels = np.zeros_like(skel, dtype=np.uint8)
        skel_labels[skel > 0] = full_labels[skel > 0]
        nib.save(nib.Nifti1Image(skel_labels, img_lv.affine, img_lv.header),
                 str(output_dir / "liver_vessels_skeleton_labeled.nii.gz"))

    total_v = vessels.sum()
    qc = {
        "portal_voxels": int(portal_mask.sum()),
        "hepatic_voxels": int(hepatic_mask.sum()),
        "total_vessel_voxels": int(total_v),
        "portal_fraction": float(portal_mask.sum() / total_v) if total_v else 0.0,
        "hepatic_fraction": float(hepatic_mask.sum() / total_v) if total_v else 0.0,
        "portal_seed_voxels": int(portal_seed.sum()),
        "hepatic_seed_voxels": int(hepatic_seed.sum()),
        "use_skeleton": bool(use_skel)
    }
    return qc