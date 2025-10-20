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
    from pydicom.pixel_data_handlers.util import apply_voi_lut
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
    # Apply Modality LUT (rescale slope/intercept) if present
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
) -> int:
    _check_deps()

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
            vol = ds.pixel_array  # shape (frames, H, W) or (H, W, frames)
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
        )
        print(f"Wrote {n} {args.format.upper()} files to {args.output}")
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
