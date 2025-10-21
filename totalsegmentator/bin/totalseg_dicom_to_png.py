#!/usr/bin/env python3
"""
totalseg_dicom_to_png

Cross-platform DICOM → PNG converter with sensible windowing.

Highlights
- Works on single-file or directory inputs (recursively with --recursive)
- Uses VOI LUT when present; otherwise supports presets or auto percentile windowing
- Outputs 8-bit PNGs compatible with Mac and Windows

Examples
  totalseg_dicom_to_png -i /path/to/dicom_study -o out/pngs
  totalseg_dicom_to_png -i series/ -o out/pngs --window abdomen
  totalseg_dicom_to_png -i image.dcm -o out/ --window custom --wl 40 --ww 400
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple, Optional

import numpy as np

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut, apply_modality_lut
except Exception as e:
    pydicom = None  # type: ignore
    _pydicom_err = e

try:
    from PIL import Image
except Exception as e:
    Image = None  # type: ignore
    _pillow_err = e


WINDOW_PRESETS = {
    # name: (WL, WW)
    "soft": (40.0, 400.0),        # generic soft tissue
    "abdomen": (50.0, 350.0),     # typical abdomen window
    "bone": (500.0, 2500.0),
    "lung": (-600.0, 1500.0),
    "brain": (40.0, 80.0),
}


def _check_deps():
    if pydicom is None:
        raise RuntimeError(f"pydicom not available: {_pydicom_err}")
    if Image is None:
        raise RuntimeError(f"Pillow not available: {_pillow_err}")


def _safe_spacing(ds) -> Tuple[float, float, float]:
    px = py = 1.0
    ps = getattr(ds, "PixelSpacing", None)
    if ps is not None and len(ps) >= 2:
        try:
            px, py = float(ps[0]), float(ps[1])
        except Exception:
            px = py = 1.0
    pz = getattr(ds, "SpacingBetweenSlices", None)
    if pz is not None:
        try:
            pz = float(pz)
        except Exception:
            pz = None
    if pz is None:
        try:
            pz = float(getattr(ds, "SliceThickness"))
        except Exception:
            pz = px
    return float(px), float(py), float(pz)


def _is_dicom_file(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            pre = f.read(132)
            return pre[128:132] == b"DICM"
    except Exception:
        return False


def _iter_dicom_files(inp: Path, recursive: bool = False) -> Iterable[Path]:
    if inp.is_file():
        yield inp
        return
    if inp.is_dir():
        if recursive:
            for p in inp.rglob("*"):
                if p.is_file():
                    yield p
        else:
            for p in inp.iterdir():
                if p.is_file():
                    yield p


def _read_num_frames(ds) -> int:
    try:
        return int(getattr(ds, "NumberOfFrames", 1))
    except Exception:
        return 1


def _to_float32(arr: np.ndarray, ds) -> np.ndarray:
    # Prefer Modality LUT if available; else slope/intercept
    try:
        return apply_modality_lut(arr, ds).astype(np.float32)
    except Exception:
        arr = arr.astype(np.float32)
        slope = getattr(ds, "RescaleSlope", 1.0)
        intercept = getattr(ds, "RescaleIntercept", 0.0)
        try:
            arr = arr * float(slope) + float(intercept)
        except Exception:
            pass
        return arr


def _window_image(
    arr: np.ndarray,
    window: str,
    wl: Optional[float] = None,
    ww: Optional[float] = None,
    use_voi_lut: bool = True,
) -> np.ndarray:
    if use_voi_lut:
        # apply_voi_lut falls back gracefully when VOI not present
        try:
            arr = apply_voi_lut(arr, None)
        except Exception:
            pass

    if window == "auto":
        lo, hi = np.percentile(arr, [0.5, 99.5])
    elif window == "custom" and wl is not None and ww is not None:
        lo, hi = wl - ww / 2.0, wl + ww / 2.0
    else:
        if window in WINDOW_PRESETS:
            _wl, _ww = WINDOW_PRESETS[window]
            lo, hi = _wl - _ww / 2.0, _wl + _ww / 2.0
        else:
            # fallback to auto
            lo, hi = np.percentile(arr, [0.5, 99.5])

    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / max(hi - lo, 1e-6)
    arr = (arr * 255.0).round().astype(np.uint8)
    return arr


def _save_image(array: np.ndarray, out_path: Path, fmt: str = "png", jpeg_quality: int = 95) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(array, mode="L")
    if fmt.lower() in ("jpg", "jpeg"):
        img.save(str(out_path), quality=jpeg_quality)
    else:
        img.save(str(out_path))


def _compose_name(ds, idx: int, series_fallback: int = 0) -> str:
    # Try to make a stable, human-readable name
    series_uid = getattr(ds, "SeriesInstanceUID", None)
    series_no = getattr(ds, "SeriesNumber", None)
    inst_no = getattr(ds, "InstanceNumber", None)
    # Fallbacks
    series_tag = f"s{series_no}" if series_no is not None else (f"S{series_fallback}" if series_uid is None else f"S")
    instance_tag = f"i{inst_no:04d}" if isinstance(inst_no, (int, float)) else f"i{idx:04d}"
    return f"{series_tag}_{instance_tag}"


def convert_dicom_to_png(
    input_path: Path,
    output_dir: Path,
    recursive: bool = False,
    window: str = "auto",
    wl: Optional[float] = None,
    ww: Optional[float] = None,
    prefix: Optional[str] = None,
    fmt: str = "png",
    jpeg_quality: int = 95,
    multi_views: bool = False,
    iso_mm: Optional[float] = None,
    interp: int = 1,
    save_axial: bool = True,
    save_coronal: bool = True,
    save_sagittal: bool = True,
    flip_ax_lr: bool = False,
    flip_ax_ud: bool = False,
    flip_cor_lr: bool = False,
    flip_cor_ud: bool = False,
    flip_sag_lr: bool = False,
    flip_sag_ud: bool = False,
    write_metadata: bool = False,
) -> int:
    _check_deps()

    if not multi_views:
        output_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        series_count = 0
        for f in _iter_dicom_files(input_path, recursive=recursive):
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=False, force=True)
            except Exception:
                continue

            frames = _read_num_frames(ds)
            if frames <= 1:
                arr = ds.pixel_array  # type: ignore[attr-defined]
                arr = _to_float32(arr, ds)
                png = _window_image(arr, window=window, wl=wl, ww=ww)
                name = _compose_name(ds, idx=0, series_fallback=series_count)
                if prefix:
                    name = f"{prefix}_{name}"
                ext = ".jpg" if fmt.lower() in ("jpg", "jpeg") else ".png"
                _save_image(png, output_dir / f"{name}{ext}", fmt=fmt, jpeg_quality=jpeg_quality)
                count += 1
            else:
                vol = ds.pixel_array
                if vol.ndim == 3 and vol.shape[0] == frames:
                    iterable = [vol[i] for i in range(frames)]
                elif vol.ndim == 3 and vol.shape[-1] == frames:
                    iterable = [vol[..., i] for i in range(frames)]
                else:
                    iterable = [vol]
                for i, sl in enumerate(iterable):
                    sl = _to_float32(sl, ds)
                    png = _window_image(sl, window=window, wl=wl, ww=ww)
                    name = _compose_name(ds, idx=i, series_fallback=series_count)
                    if prefix:
                        name = f"{prefix}_{name}"
                    ext = ".jpg" if fmt.lower() in ("jpg", "jpeg") else ".png"
                    _save_image(png, output_dir / f"{name}{ext}", fmt=fmt, jpeg_quality=jpeg_quality)
                    count += 1
            series_count += 1
        return count

    # 3-plane export path
    if not input_path.is_dir():
        raise ValueError("--multi-views requires --input to be a DICOM directory")
    try:
        from scipy.ndimage import zoom as ndi_zoom
    except Exception as e:
        raise RuntimeError(f"scipy is required for --multi-views: {e}")

    # group by series and pick the largest
    series_map = {}
    iterator = input_path.rglob("*") if recursive else input_path.iterdir()
    for p in iterator:
        if p.is_file() and _is_dicom_file(p):
            try:
                ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
                key = (getattr(ds, "StudyInstanceUID", None), getattr(ds, "SeriesInstanceUID", None))
                if key[1]:
                    series_map.setdefault(key, []).append(p)
            except Exception:
                pass
    if not series_map:
        raise RuntimeError(f"No DICOM series found under: {input_path}")
    key = max(series_map, key=lambda k: len(series_map[k]))
    files = series_map[key]

    # sort slices by z or instance number
    metas = []
    for p in files:
        ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
        inst = int(getattr(ds, "InstanceNumber", 0))
        ipp = getattr(ds, "ImagePositionPatient", None)
        z = float(ipp[2]) if (ipp and len(ipp) == 3) else None
        metas.append((p, z, inst))
    if any(m[1] is not None for m in metas):
        metas.sort(key=lambda x: (x[1] is None, x[1]))
    else:
        metas.sort(key=lambda x: x[2])
    files = [m[0] for m in metas]

    # load volume (Z,Y,X)
    ds0 = pydicom.dcmread(str(files[0]), force=True)
    px, py, pz = _safe_spacing(ds0)
    Y, X = int(ds0.Rows), int(ds0.Columns)
    Z = len(files)
    vol = np.zeros((Z, Y, X), dtype=np.float32)
    for i, p in enumerate(files):
        ds = pydicom.dcmread(str(p), force=True)
        arr = ds.pixel_array
        arr = _to_float32(arr, ds)
        if window == "auto" and hasattr(ds, "VOILUTSequence"):
            try:
                arr = apply_voi_lut(arr, ds).astype(np.float32)
            except Exception:
                pass
        vol[i] = arr

    # isotropic resampling
    iso = min(px, py, pz) if iso_mm is None else float(iso_mm)
    fz, fy, fx = pz / iso, py / iso, px / iso
    newZ = max(1, int(round(Z * fz)))
    newY = max(1, int(round(Y * fy)))
    newX = max(1, int(round(X * fx)))
    vol_iso = ndi_zoom(vol, zoom=(fz, fy, fx), order=interp, mode='nearest', grid_mode=True)
    vol_iso = vol_iso[:newZ, :newY, :newX]

    # windowing
    if window == "auto" or wl is None or ww is None:
        Z2, Y2, X2 = vol_iso.shape
        samp = vol_iso[::max(Z2 // 16, 1), ::max(Y2 // 16, 1), ::max(X2 // 16, 1)].ravel()
        vmin, vmax = np.percentile(samp, 0.5), np.percentile(samp, 99.5)
    else:
        vmin, vmax = wl - ww / 2.0, wl + ww / 2.0
    vol01 = np.clip((vol_iso - vmin) / max(1e-6, (vmax - vmin)), 0.0, 1.0)

    # save stacks
    ext = ".jpg" if fmt.lower() in ("jpg", "jpeg") else ".png"
    def _save_stack(stack3d: np.ndarray, out_dir: Path, flip_lr=False, flip_ud=False):
        out_dir.mkdir(parents=True, exist_ok=True)
        num_width = max(3, len(str(stack3d.shape[0])))
        for i, slice2d in enumerate(stack3d, start=1):
            img = (slice2d * 255.0 + 0.5).astype(np.uint8)
            if flip_lr:
                img = np.fliplr(img)
            if flip_ud:
                img = np.flipud(img)
            fname = f"{i:0{num_width}d}{ext}"
            _save_image(img, out_dir / fname, fmt=fmt, jpeg_quality=jpeg_quality)

    if save_axial:
        _save_stack(vol01, output_dir / "axial", flip_lr=flip_ax_lr, flip_ud=flip_ax_ud)
    if save_coronal:
        cor = np.transpose(vol01, (1, 0, 2))
        _save_stack(cor, output_dir / "coronal", flip_lr=flip_cor_lr, flip_ud=flip_cor_ud)
    if save_sagittal:
        sag = np.transpose(vol01, (2, 0, 1))
        _save_stack(sag, output_dir / "sagittal", flip_lr=flip_sag_lr, flip_ud=flip_sag_ud)

    if write_metadata:
        import csv
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "metadata.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["SeriesInstanceUID", key[1]])
            w.writerow(["Original_Shape_ZYX", Z, Y, X])
            w.writerow(["Original_Spacing_mm_XYZ", px, py, pz])
            w.writerow(["Isotropic_Spacing_mm", iso])
            Z2, Y2, X2 = vol01.shape
            w.writerow(["Isotropic_Shape_ZYX", Z2, Y2, X2])
            w.writerow(["Axial_pixel_mm_(H,W)_&slice_step_mm", (iso, iso), iso])
            w.writerow(["Coronal_pixel_mm_(H,W)_&slice_step_mm", (iso, iso), iso])
            w.writerow(["Sagittal_pixel_mm_(H,W)_&slice_step_mm", (iso, iso), iso])

    total = 0
    if save_axial:
        total += len(list((output_dir / "axial").glob(f"*{ext}")))
    if save_coronal:
        total += len(list((output_dir / "coronal").glob(f"*{ext}")))
    if save_sagittal:
        total += len(list((output_dir / "sagittal").glob(f"*{ext}")))
    return int(total)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convert DICOM to PNG/JPEG with windowing (Mac/Windows compatible)")
    p.add_argument("-i", "--input", required=True, type=Path, help="DICOM file or directory")
    p.add_argument("-o", "--output", required=True, type=Path, help="Output directory for PNGs")
    p.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    p.add_argument("--window", choices=["auto", "custom", "soft", "abdomen", "bone", "lung", "brain"], default="auto",
                   help="Window preset or auto")
    p.add_argument("--wl", type=float, default=None, help="Window level (used with --window custom)")
    p.add_argument("--ww", type=float, default=None, help="Window width (used with --window custom)")
    p.add_argument("--prefix", type=str, default=None, help="Optional filename prefix, e.g. case ID")
    p.add_argument("--format", choices=["png", "jpeg", "jpg"], default="png", help="Output format")
    p.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality (if --format jpeg)")
    # 3-plane options
    p.add_argument("--multi-views", action="store_true", help="Export axial/coronal/sagittal stacks (isotropic resampling)")
    p.add_argument("--iso-mm", type=float, default=None, help="Target isotropic spacing (mm). Default=min(original spacings)")
    p.add_argument("--interp", type=int, choices=[0, 1, 3], default=1, help="Interpolation order: 0=nearest, 1=linear, 3=cubic")
    p.add_argument("--no-axial", dest="axial", action="store_false", help="Do not save axial view")
    p.add_argument("--no-coronal", dest="coronal", action="store_false", help="Do not save coronal view")
    p.add_argument("--no-sagittal", dest="sagittal", action="store_false", help="Do not save sagittal view")
    p.set_defaults(axial=True, coronal=True, sagittal=True)
    p.add_argument("--flip-axial-lr", action="store_true", help="Flip axial left-right")
    p.add_argument("--flip-axial-ud", action="store_true", help="Flip axial up-down")
    p.add_argument("--flip-coronal-lr", action="store_true", help="Flip coronal left-right")
    p.add_argument("--flip-coronal-ud", action="store_true", help="Flip coronal up-down")
    p.add_argument("--flip-sagittal-lr", action="store_true", help="Flip sagittal left-right")
    p.add_argument("--flip-sagittal-ud", action="store_true", help="Flip sagittal up-down")
    p.add_argument("--metadata", action="store_true", help="Write metadata.csv with shapes and spacings")
    return p


def main():
    args = build_argparser().parse_args()
    try:
        n = convert_dicom_to_png(
            input_path=args.input,
            output_dir=args.output,
            recursive=args.recursive,
            window=args.window,
            wl=args.wl,
            ww=args.ww,
            prefix=args.prefix,
            fmt=args.format,
            jpeg_quality=args.jpeg_quality,
            multi_views=args.multi_views,
            iso_mm=args.iso_mm,
            interp=args.interp,
            save_axial=args.axial,
            save_coronal=args.coronal,
            save_sagittal=args.sagittal,
            flip_ax_lr=args.flip_axial_lr,
            flip_ax_ud=args.flip_axial_ud,
            flip_cor_lr=args.flip_coronal_lr,
            flip_cor_ud=args.flip_coronal_ud,
            flip_sag_lr=args.flip_sagittal_lr,
            flip_sag_ud=args.flip_sagittal_ud,
            write_metadata=args.metadata,
        )
        print(f"Wrote {n} file(s) to {args.output}")
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
