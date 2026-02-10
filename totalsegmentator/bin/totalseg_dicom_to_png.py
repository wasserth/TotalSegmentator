#!/usr/bin/env python3
"""
DICOM -> PNG/JPG (doctor-provided isotropic export)

- Selects series with most slices
- Sorts by ImagePositionPatient Z (fallback InstanceNumber)
- Applies modality LUT (HU) and optional VOI LUT for auto mode
- Resamples to isotropic spacing (default min spacing)
- Exports axial/coronal/sagittal stacks as 001.png
- Writes metadata.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import csv
import numpy as np

import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
from PIL import Image
from scipy.ndimage import zoom


DEFAULT_PIXEL = 512
DEFAULT_AX_COUNT = 419
DEFAULT_AX_SPACING = 2.0
DEFAULT_PITCH = 0.683
DEFAULT_SIZE = 0.02
DEFAULT_FOLDER = ''


def _is_dicom(p: Path) -> bool:
    try:
        with open(p, "rb") as f:
            pre = f.read(132)
            return pre[128:132] == b"DICM"
    except Exception:
        return False


def _safe_spacing(ds):
    # X=列方向, Y=行方向, Z=スライス方向
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


def convert_dicom_to_png(
    src_dir: Path,
    out_dir: Path,
    fmt: str = "png",
    wl: float | None = 40,
    ww: float | None = 400,
    auto: bool = False,
    iso_spacing_mm: float | None = None,
    interp_order: int = 1,
    save_axial: bool = True,
    save_coronal: bool = True,
    save_sagittal: bool = True,
    flip_axial_lr: bool = False,
    flip_axial_ud: bool = False,
    flip_cor_lr: bool = False,
    flip_cor_ud: bool = False,
    flip_sag_lr: bool = False,
    flip_sag_ud: bool = False,
) -> None:
    src = Path(src_dir)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) DICOM収集 & 最大スライスのシリーズを選択
    series = {}
    for p in src.rglob("*"):
        if p.is_file() and _is_dicom(p):
            try:
                ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
                key = (getattr(ds, "StudyInstanceUID", None), getattr(ds, "SeriesInstanceUID", None))
                if key[1]:
                    series.setdefault(key, []).append(p)
            except Exception:
                pass
    if not series:
        raise RuntimeError(f"No DICOM series found under: {src}")
    key = max(series, key=lambda k: len(series[k]))
    files = series[key]

    # 2) 並び順（ImagePositionPatientのZ優先、なければInstanceNumber）
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

    # 3) 読み込み & HU変換
    ds0 = pydicom.dcmread(str(files[0]), force=True)
    px, py, pz = _safe_spacing(ds0)   # mm
    Y, X = int(ds0.Rows), int(ds0.Columns)
    Z = len(files)

    vol = np.zeros((Z, Y, X), dtype=np.float32)
    for i, p in enumerate(files):
        ds = pydicom.dcmread(str(p), force=True)
        arr = ds.pixel_array
        try:
            arr = apply_modality_lut(arr, ds)  # HU相当に
        except Exception:
            arr = arr.astype(np.float32)
        else:
            arr = arr.astype(np.float32)
        if auto and hasattr(ds, "VOILUTSequence"):
            arr = apply_voi_lut(arr, ds).astype(np.float32)
        vol[i] = arr

    # 4) 等方化
    if iso_spacing_mm is None:
        iso = min(px, py, pz)
    else:
        iso = float(iso_spacing_mm)

    fz = pz / iso
    fy = py / iso
    fx = px / iso
    newZ = max(1, int(round(Z * fz)))
    newY = max(1, int(round(Y * fy)))
    newX = max(1, int(round(X * fx)))

    vol_iso = zoom(vol, zoom=(fz, fy, fx), order=interp_order, mode='nearest', grid_mode=True)
    vol_iso = vol_iso[:newZ, :newY, :newX]

    # 5) ウィンドウ処理
    if auto or wl is None or ww is None:
        Z2, Y2, X2 = vol_iso.shape
        samp = vol_iso[::max(Z2 // 16, 1), ::max(Y2 // 16, 1), ::max(X2 // 16, 1)].ravel()
        vmin, vmax = np.percentile(samp, 0.5), np.percentile(samp, 99.5)
    else:
        vmin, vmax = wl - ww / 2.0, wl + ww / 2.0

    vol01 = np.clip((vol_iso - vmin) / max(1e-6, (vmax - vmin)), 0.0, 1.0)

    def _save_stack(stack3d, out_dir: Path, flip_lr=False, flip_ud=False):
        out_dir.mkdir(parents=True, exist_ok=True)
        num_width = max(3, len(str(stack3d.shape[0])))
        for i, slice2d in enumerate(stack3d, start=1):
            img = (slice2d * 255.0 + 0.5).astype(np.uint8)
            if flip_lr:
                img = np.fliplr(img)
            if flip_ud:
                img = np.flipud(img)
            fname = f"{i:0{num_width}d}"
            if fmt.lower() in ("jpg", "jpeg"):
                Image.fromarray(img, mode="L").save(out_dir / f"{fname}.jpg", quality=95, subsampling=0)
            else:
                Image.fromarray(img, mode="L").save(out_dir / f"{fname}.png")

    # 6) 3面出力
    if save_axial:
        _save_stack(vol01, out_root / "axial", flip_lr=flip_axial_lr, flip_ud=flip_axial_ud)

    if save_coronal:
        cor = np.transpose(vol01, (1, 0, 2))
        _save_stack(cor, out_root / "coronal", flip_lr=flip_cor_lr, flip_ud=flip_cor_ud)

    if save_sagittal:
        sag = np.transpose(vol01, (2, 0, 1))
        _save_stack(sag, out_root / "sagittal", flip_lr=flip_sag_lr, flip_ud=flip_sag_ud)

    # 7) メタデータ
    with open(out_root / "metadata.csv", "w", newline="") as f:
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

    print("✅ Done.")
    print("Series:", key[1])
    print("Original  (Z,Y,X):", (Z, Y, X), "Spacing(mm XYZ):", (px, py, pz))
    print("Isotropic (Z,Y,X):", vol01.shape, "Spacing(mm):", iso)
    print("Saved to:", str(out_root))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DICOM -> PNG/JPG (doctor isotropic export)")
    p.add_argument("-i", "--input", required=True, type=Path, help="DICOM folder")
    p.add_argument("-o", "--output", required=True, type=Path, help="Output folder")
    p.add_argument("--format", choices=["png", "jpg", "jpeg"], default="png")
    p.add_argument("--wl", type=float, default=40)
    p.add_argument("--ww", type=float, default=400)
    p.add_argument("--auto", action="store_true", help="Auto WL/WW (0.5-99.5%%)")
    p.add_argument("--iso-mm", type=float, default=None)
    p.add_argument("--interp", type=int, choices=[0, 1, 3], default=1)
    p.add_argument("--no-axial", dest="axial", action="store_false")
    p.add_argument("--no-coronal", dest="coronal", action="store_false")
    p.add_argument("--no-sagittal", dest="sagittal", action="store_false")
    p.set_defaults(axial=True, coronal=True, sagittal=True)
    p.add_argument("--flip-axial-lr", action="store_true")
    p.add_argument("--flip-axial-ud", action="store_true")
    p.add_argument("--flip-cor-lr", action="store_true")
    p.add_argument("--flip-cor-ud", action="store_true")
    p.add_argument("--flip-sag-lr", action="store_true")
    p.add_argument("--flip-sag-ud", action="store_true")
    return p


def main():
    args = build_argparser().parse_args()
    convert_dicom_to_png(
        src_dir=args.input,
        out_dir=args.output,
        fmt=args.format,
        wl=args.wl,
        ww=args.ww,
        auto=args.auto,
        iso_spacing_mm=args.iso_mm,
        interp_order=args.interp,
        save_axial=args.axial,
        save_coronal=args.coronal,
        save_sagittal=args.sagittal,
        flip_axial_lr=args.flip_axial_lr,
        flip_axial_ud=args.flip_axial_ud,
        flip_cor_lr=args.flip_cor_lr,
        flip_cor_ud=args.flip_cor_ud,
        flip_sag_lr=args.flip_sag_lr,
        flip_sag_ud=args.flip_sag_ud,
    )


if __name__ == "__main__":
    main()
