#!/usr/bin/env python3
"""
totalseg_dataset_build

Build a 2D PNG dataset (images + masks) from a CT/MR volume and its segmentation.

Features
- Accepts input as NIfTI file or DICOM directory (auto-converted to NIfTI)
- Accepts segmentation as multi-label NIfTI (with embedded label map) or a directory of per-class NIfTI masks
- Outputs 8-bit PNGs for images and masks (per-class folders and/or multi-label)
- Cross-platform (Mac and Windows)

Examples
  # From NIfTI image + TotalSegmentator outputs
  totalseg_dataset_build -i ct.nii.gz -s segmentations/ -o dataset --mode both

  # From DICOM study and a multi-label mask
  totalseg_dataset_build -i dicom_study/ -s seg_ml.nii.gz -o dataset --window abdomen
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Iterable, Tuple, Optional

import numpy as np

try:
    import nibabel as nib
except Exception as e:
    nib = None  # type: ignore
    _nib_err = e

try:
    from PIL import Image
except Exception as e:
    Image = None  # type: ignore
    _pillow_err = e

try:
    from totalsegmentator.nifti_ext_header import load_multilabel_nifti
except Exception:
    load_multilabel_nifti = None  # type: ignore

try:
    from totalsegmentator.dicom_io import dcm_to_nifti
except Exception:
    dcm_to_nifti = None  # type: ignore


WINDOW_PRESETS = {
    "auto": None,              # percentile-based
    "soft": (40.0, 400.0),
    "abdomen": (50.0, 350.0),
    "bone": (500.0, 2500.0),
    "lung": (-600.0, 1500.0),
    "brain": (40.0, 80.0),
}


def _check_deps():
    if nib is None:
        raise RuntimeError(f"nibabel not available: {_nib_err}")
    if Image is None:
        raise RuntimeError(f"Pillow not available: {_pillow_err}")


def _as_nifti(path: Path) -> Path:
    """Return a NIfTI path, converting from DICOM dir if needed (temp file)."""
    if path.is_file() and (path.suffix in {".nii", ".gz"} or path.name.endswith(".nii.gz")):
        return path
    if path.is_dir():
        if dcm_to_nifti is None:
            raise RuntimeError("Cannot convert DICOM to NIfTI (dicom2nifti not available)")
        tmpdir = Path(tempfile.mkdtemp(prefix="ts_ds_"))
        out = tmpdir / "input_from_dicom.nii.gz"
        dcm_to_nifti(str(path), str(out), tmp_dir=tmpdir)
        return out
    raise ValueError(f"Unsupported input path (expect NIfTI file or DICOM directory): {path}")


def _load_nifti(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    img = nib.load(str(path))
    data = img.get_fdata()
    return data, img.affine


def _slice_axis(data: np.ndarray, axis: str) -> Iterable[np.ndarray]:
    ax = {"axial": 2, "coronal": 1, "sagittal": 0}[axis]
    n = data.shape[ax]
    for i in range(n):
        if ax == 2:
            yield data[:, :, i]
        elif ax == 1:
            yield data[:, i, :]
        else:
            yield data[i, :, :]


def _window_volume(data: np.ndarray, window: str, wl: Optional[float], ww: Optional[float], per_slice: bool) -> Iterable[np.ndarray]:
    if per_slice:
        for sl in _slice_axis(data, axis="axial"):
            yield _window_array(sl, window, wl, ww)
    else:
        yield from _window_array_volume(data, window, wl, ww)


def _window_array(arr: np.ndarray, window: str, wl: Optional[float], ww: Optional[float]) -> np.ndarray:
    arr = arr.astype(np.float32)
    if window == "auto":
        lo, hi = np.percentile(arr, [0.5, 99.5])
    elif window == "custom" and wl is not None and ww is not None:
        lo, hi = wl - ww / 2.0, wl + ww / 2.0
    else:
        if window in WINDOW_PRESETS and WINDOW_PRESETS[window] is not None:
            _wl, _ww = WINDOW_PRESETS[window]
            lo, hi = _wl - _ww / 2.0, _wl + _ww / 2.0
        else:
            lo, hi = np.percentile(arr, [0.5, 99.5])
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / max(hi - lo, 1e-6)
    arr = (arr * 255.0).round().astype(np.uint8)
    return arr


def _window_array_volume(data: np.ndarray, window: str, wl: Optional[float], ww: Optional[float]) -> Iterable[np.ndarray]:
    arr = data.astype(np.float32)
    if window == "auto":
        lo, hi = np.percentile(arr, [0.5, 99.5])
    elif window == "custom" and wl is not None and ww is not None:
        lo, hi = wl - ww / 2.0, wl + ww / 2.0
    else:
        if window in WINDOW_PRESETS and WINDOW_PRESETS[window] is not None:
            _wl, _ww = WINDOW_PRESETS[window]
            lo, hi = _wl - _ww / 2.0, _wl + _ww / 2.0
        else:
            lo, hi = np.percentile(arr, [0.5, 99.5])
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / max(hi - lo, 1e-6)
    arr = (arr * 255.0).round().astype(np.uint8)
    # yield axial slices for consistency
    for sl in _slice_axis(arr, axis="axial"):
        yield sl


def _save_png(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(str(path))


def _collect_seg_from_dir(seg_dir: Path) -> Dict[str, np.ndarray]:
    vol_map: Dict[str, np.ndarray] = {}
    for p in sorted(seg_dir.glob("*.nii*")):
        try:
            vol, _ = _load_nifti(p)
            # Treat any non-zero as foreground
            vol_map[p.stem.replace(".nii", "")] = (vol > 0.5).astype(np.uint8)
        except Exception:
            continue
    if not vol_map:
        raise RuntimeError(f"No NIfTI masks found in {seg_dir}")
    return vol_map


def _collect_seg_from_multilabel(seg_ml_path: Path) -> Tuple[Dict[str, np.ndarray], Dict[int, str]]:
    if load_multilabel_nifti is None:
        raise RuntimeError("nifti_ext_header not available to parse label map")
    img, label_map = load_multilabel_nifti(seg_ml_path)
    data = img.get_fdata()
    vol_map: Dict[str, np.ndarray] = {}
    for label_id, label_name in label_map.items():
        if label_id == 0:
            continue
        mask = (data == label_id).astype(np.uint8)
        if mask.any():
            vol_map[str(label_name)] = mask
    return vol_map, {int(k): v for k, v in label_map.items()}


def _match_shape_or_raise(ref: np.ndarray, tgt: np.ndarray, name: str):
    if ref.shape != tgt.shape:
        raise ValueError(f"Shape mismatch for {name}: {tgt.shape} vs image {ref.shape}")


def build_dataset(
    input_path: Path,
    seg_path: Path,
    output_dir: Path,
    mode: str = "both",
    axis: str = "axial",
    window: str = "auto",
    wl: Optional[float] = None,
    ww: Optional[float] = None,
    per_slice_norm: bool = False,
    skip_empty: bool = False,
    case_id: Optional[str] = None,
    class_filter: Optional[List[str]] = None,
) -> Dict[str, object]:
    _check_deps()

    nii_path = _as_nifti(input_path)
    vol, _ = _load_nifti(nii_path)

    seg_vols: Dict[str, np.ndarray] = {}
    label_map: Optional[Dict[int, str]] = None

    if seg_path.is_dir():
        seg_vols = _collect_seg_from_dir(seg_path)
    else:
        seg_vols, label_map = _collect_seg_from_multilabel(seg_path)

    if class_filter:
        seg_vols = {k: v for k, v in seg_vols.items() if k in class_filter}

    # Validate shapes
    for name, m in seg_vols.items():
        _match_shape_or_raise(vol, m, name)

    # Prepare outputs
    img_dir = output_dir / "images"
    multilabel_dir = output_dir / "masks_multilabel"
    masks_dir = output_dir / "masks"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine image windows for all slices
    axial_slices = list(_slice_axis(vol, axis))

    if per_slice_norm:
        # compute per-slice windowing
        img_slices = [_window_array(sl, window, wl, ww) for sl in axial_slices]
    else:
        # compute per-volume windowing for consistent intensity across slices
        img_slices = list(_window_array_volume(vol, window, wl, ww))

    # Prepare multi-label if requested
    have_multilabel = (mode in {"both", "multilabel_png"}) and (label_map is not None)
    if have_multilabel:
        ml_data, _ = _load_nifti(seg_path)

    written = []
    classes = sorted(seg_vols.keys())
    case = case_id or nii_path.stem.replace(".nii", "")

    for idx, (img_slice) in enumerate(img_slices):
        # derive matching seg index
        if idx >= vol.shape[2]:  # axial only for now
            break
        # Skip empty slices if requested
        if skip_empty and not any(m[..., idx].any() for m in seg_vols.values()):
            continue

        img_out = img_dir / f"{case}_{idx:04d}.png"
        _save_png(img_slice, img_out)

        row = {
            "slice_id": f"{case}_{idx:04d}",
            "image": str(img_out.relative_to(output_dir)),
        }

        if mode in {"both", "per_class_png"}:
            for cname, vol_mask in seg_vols.items():
                sl = vol_mask[..., idx] if axis == "axial" else None  # axial only
                if sl is None:
                    raise NotImplementedError("Only axial mode currently implemented for mask export")
                out = masks_dir / cname / f"{case}_{idx:04d}.png"
                _save_png((sl > 0).astype(np.uint8) * 255, out)

        if have_multilabel:
            ml_sl = ml_data[..., idx]
            out = multilabel_dir / f"{case}_{idx:04d}.png"
            # store directly as uint16 if labels exceed 255, else uint8
            if ml_sl.max() > 255:
                arr = ml_sl.astype(np.uint16)
                Image.fromarray(arr, mode="I;16").save(str(out))
            else:
                _save_png(ml_sl.astype(np.uint8), out)
            row["mask_multi"] = str(out.relative_to(output_dir))

        written.append(row)

    # Write manifest
    manifest_csv = output_dir / "manifest.csv"
    with open(manifest_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in written for k in r.keys()}))
        w.writeheader()
        w.writerows(written)

    # Save class list and optional label map
    classes_json = output_dir / "classes.json"
    with open(classes_json, "w") as f:
        json.dump({"classes": classes}, f, indent=2)

    if label_map is not None:
        with open(output_dir / "label_map.json", "w") as f:
            json.dump(label_map, f, indent=2)

    return {
        "images": len(written),
        "classes": classes,
        "manifest": str(manifest_csv),
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build PNG dataset (images + masks) from volume + segmentation")
    p.add_argument("-i", "--input", required=True, type=Path, help="Input NIfTI file or DICOM directory")
    p.add_argument("-s", "--seg", required=True, type=Path,
                   help="Segmentation: multi-label NIfTI or directory of per-class NIfTI masks")
    p.add_argument("-o", "--output", required=True, type=Path, help="Output dataset directory")
    p.add_argument("--mode", choices=["per_class_png", "multilabel_png", "both"], default="both",
                   help="Which mask outputs to generate")
    p.add_argument("--axis", choices=["axial"], default="axial", help="Slice axis (axial only for now)")
    p.add_argument("--window", choices=["auto", "custom", "soft", "abdomen", "bone", "lung", "brain"], default="auto",
                   help="Image window preset or auto")
    p.add_argument("--wl", type=float, default=None, help="Window level for custom window")
    p.add_argument("--ww", type=float, default=None, help="Window width for custom window")
    p.add_argument("--per-slice-norm", action="store_true",
                   help="Normalize intensities per slice instead of per volume")
    p.add_argument("--skip-empty", action="store_true", help="Skip slices with no foreground mask")
    p.add_argument("--case-id", type=str, default=None, help="Optional case identifier for filenames")
    p.add_argument("--class-filter", nargs="*", default=None, help="Subset of class names to include")
    return p


def main():
    args = build_argparser().parse_args()
    try:
        summary = build_dataset(
            input_path=args.input,
            seg_path=args.seg,
            output_dir=args.output,
            mode=args.mode,
            axis=args.axis,
            window=args.window,
            wl=args.wl,
            ww=args.ww,
            per_slice_norm=args.per_slice_norm,
            skip_empty=args.skip_empty,
            case_id=args.case_id,
            class_filter=args.class_filter,
        )
        print(json.dumps(summary, indent=2))
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()

