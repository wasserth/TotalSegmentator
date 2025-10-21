#!/usr/bin/env python3
"""
Interactive MPR (axial/coronal/sagittal) preview + export with sliders.

Two modes:
1) Jupyter (ipywidgets) — rich sliders and Save button.
2) Fallback (matplotlib Sliders) — if not in Jupyter.

Dependencies (install as needed):
  pip install pydicom numpy scipy pillow matplotlib ipywidgets

Usage (Jupyter):
  %run -m totalsegmentator.bin.totalseg_mpr_widget --dicom <dicom_dir>

Usage (script window):
  totalseg_mpr_widget --dicom <dicom_dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
except Exception as e:
    pydicom = None
    _pydicom_err = e

try:
    from scipy.ndimage import zoom as ndi_zoom
except Exception as e:
    ndi_zoom = None
    _scipy_err = e

try:
    from PIL import Image
except Exception as e:
    Image = None
    _pillow_err = e


def _is_dicom_file(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            pre = f.read(132)
            return pre[128:132] == b"DICM"
    except Exception:
        return False


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


def load_dicom_volume(dicom_dir: Path, recursive: bool = True) -> Tuple[np.ndarray, Tuple[float, float, float], Dict[str, str]]:
    if pydicom is None:
        raise RuntimeError(f"pydicom not available: {_pydicom_err}")
    series_map: Dict[Tuple[str, str], List[Path]] = {}
    iterator = dicom_dir.rglob("*") if recursive else dicom_dir.iterdir()
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
        raise RuntimeError(f"No DICOM series found under: {dicom_dir}")
    key = max(series_map, key=lambda k: len(series_map[k]))
    files = series_map[key]

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

    ds0 = pydicom.dcmread(str(files[0]), force=True)
    px, py, pz = _safe_spacing(ds0)
    Y, X = int(ds0.Rows), int(ds0.Columns)
    Z = len(files)
    vol = np.zeros((Z, Y, X), dtype=np.float32)
    for i, p in enumerate(files):
        ds = pydicom.dcmread(str(p), force=True)
        arr = ds.pixel_array
        try:
            arr = apply_modality_lut(arr, ds).astype(np.float32)
        except Exception:
            arr = arr.astype(np.float32)
            slope = getattr(ds, "RescaleSlope", 1.0)
            intercept = getattr(ds, "RescaleIntercept", 0.0)
            arr = arr * float(slope) + float(intercept)
        vol[i] = arr
    meta = {
        "StudyInstanceUID": str(key[0]),
        "SeriesInstanceUID": str(key[1]),
        "Original_Shape_ZYX": str((Z, Y, X)),
        "Original_Spacing_mm_XYZ": str((px, py, pz)),
    }
    return vol, (px, py, pz), meta


def isotropic_resample(vol: np.ndarray, spacing_xyz: Tuple[float, float, float], iso_mm: Optional[float], order: int = 1) -> Tuple[np.ndarray, float]:
    if ndi_zoom is None:
        raise RuntimeError(f"scipy is required for isotropic resampling: {_scipy_err}")
    px, py, pz = spacing_xyz
    iso = min(px, py, pz) if iso_mm is None else float(iso_mm)
    fz, fy, fx = pz / iso, py / iso, px / iso
    Z, Y, X = vol.shape
    newZ = max(1, int(round(Z * fz)))
    newY = max(1, int(round(Y * fy)))
    newX = max(1, int(round(X * fx)))
    vol_iso = ndi_zoom(vol, zoom=(fz, fy, fx), order=order, mode='nearest', grid_mode=True)
    vol_iso = vol_iso[:newZ, :newY, :newX]
    return vol_iso, iso


def window_volume(vol: np.ndarray, wl: float, ww: float) -> np.ndarray:
    vmin, vmax = wl - ww / 2.0, wl + ww / 2.0
    out = np.clip((vol - vmin) / max(1e-6, (vmax - vmin)), 0.0, 1.0)
    return out


def save_stacks(vol01: np.ndarray, out_dir: Path, fmt: str = "png", jpeg_quality: int = 95,
                save_axial=True, save_coronal=True, save_sagittal=True,
                flip_ax_lr=False, flip_ax_ud=False, flip_cor_lr=False, flip_cor_ud=False,
                flip_sag_lr=False, flip_sag_ud=False,
                iso: Optional[float] = None, meta: Optional[Dict[str, str]] = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = ".jpg" if fmt.lower() in ("jpg", "jpeg") else ".png"

    def _save_stack(stack3d: np.ndarray, sub: str, flip_lr=False, flip_ud=False):
        subdir = out_dir / sub
        subdir.mkdir(parents=True, exist_ok=True)
        num_width = max(3, len(str(stack3d.shape[0])))
        for i, sl in enumerate(stack3d, start=1):
            img = (sl * 255.0 + 0.5).astype(np.uint8)
            if flip_lr:
                img = np.fliplr(img)
            if flip_ud:
                img = np.flipud(img)
            fname = f"{i:0{num_width}d}{ext}"
            im = Image.fromarray(img, mode="L")
            if ext == ".jpg":
                im.save(subdir / fname, quality=jpeg_quality)
            else:
                im.save(subdir / fname)

    if save_axial:
        _save_stack(vol01, "axial", flip_lr=flip_ax_lr, flip_ud=flip_ax_ud)
    if save_coronal:
        cor = np.transpose(vol01, (1, 0, 2))
        _save_stack(cor, "coronal", flip_lr=flip_cor_lr, flip_ud=flip_cor_ud)
    if save_sagittal:
        sag = np.transpose(vol01, (2, 0, 1))
        _save_stack(sag, "sagittal", flip_lr=flip_sag_lr, flip_ud=flip_sag_ud)

    if meta is not None:
        import csv
        with open(out_dir / "metadata.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["SeriesInstanceUID", meta.get("SeriesInstanceUID")])
            w.writerow(["Original_Shape_ZYX", meta.get("Original_Shape_ZYX")])
            w.writerow(["Original_Spacing_mm_XYZ", meta.get("Original_Spacing_mm_XYZ")])
            if iso is not None:
                Z2, Y2, X2 = vol01.shape
                w.writerow(["Isotropic_Spacing_mm", iso])
                w.writerow(["Isotropic_Shape_ZYX", (Z2, Y2, X2)])


def jupyter_ui(dicom_dir: Path):
    try:
        import ipywidgets as W
        from IPython.display import display, clear_output
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(f"Jupyter/ipywidgets/matplotlib not available: {e}")

    vol, spacing, meta = load_dicom_volume(dicom_dir)
    vol_iso, iso = isotropic_resample(vol, spacing, iso_mm=None, order=1)

    # Defaults
    plane = W.Dropdown(options=["axial", "coronal", "sagittal"], value="axial", description="Plane")
    k = W.IntSlider(value=400, min=50, max=3000, step=10, description="WW (k)")
    l = W.IntSlider(value=40, min=-500, max=2000, step=10, description="WL (l)")
    iso_box = W.FloatText(value=iso, description="iso mm")
    fmt = W.Dropdown(options=["png", "jpeg"], value="png", description="Format")
    quality = W.IntSlider(value=95, min=60, max=100, step=1, description="JPEG q")
    flip_ax_lr = W.Checkbox(False, description="Flip AX LR")
    flip_ax_ud = W.Checkbox(False, description="Flip AX UD")
    flip_cor_lr = W.Checkbox(False, description="Flip CO LR")
    flip_cor_ud = W.Checkbox(False, description="Flip CO UD")
    flip_sag_lr = W.Checkbox(False, description="Flip SA LR")
    flip_sag_ud = W.Checkbox(False, description="Flip SA UD")
    save_ax = W.Checkbox(True, description="Save AX")
    save_co = W.Checkbox(True, description="Save CO")
    save_sa = W.Checkbox(True, description="Save SA")
    out_dir = W.Text(value=str(dicom_dir.parent / "mpr_views"), description="Out dir")
    save_btn = W.Button(description="Save Images", button_style="success")

    # Slice slider depends on plane
    def _n_slices():
        Z, Y, X = vol_iso.shape
        return {"axial": Z, "coronal": Y, "sagittal": X}[plane.value]

    slice_idx = W.IntSlider(value=1, min=1, max=_n_slices(), step=1, description="Slice")

    fig, ax = plt.subplots(figsize=(6, 6))
    img_artist = ax.imshow(np.zeros((10, 10)), cmap='gray', vmin=0, vmax=255)
    ax.axis('off')
    plt.tight_layout()

    cached = {"vol_iso": vol_iso, "iso": iso}

    def update_slice_range(*_):
        slice_idx.max = _n_slices()
        slice_idx.value = min(slice_idx.value, slice_idx.max)

    plane.observe(update_slice_range, names='value')

    def maybe_resample(*_):
        target_iso = float(iso_box.value)
        if abs(target_iso - cached["iso"]) > 1e-6:
            cached["vol_iso"], cached["iso"] = isotropic_resample(vol, spacing, iso_mm=target_iso, order=1)
            update_slice_range()
            redraw()

    def get_current_slice() -> np.ndarray:
        vol01 = window_volume(cached["vol_iso"], wl=float(l.value), ww=float(k.value))
        idx = int(slice_idx.value) - 1
        if plane.value == "axial":
            sl = vol01[idx]
            if flip_ax_lr.value: sl = np.fliplr(sl)
            if flip_ax_ud.value: sl = np.flipud(sl)
            return sl
        elif plane.value == "coronal":
            sl = np.transpose(vol01, (1, 0, 2))[idx]
            if flip_cor_lr.value: sl = np.fliplr(sl)
            if flip_cor_ud.value: sl = np.flipud(sl)
            return sl
        else:
            sl = np.transpose(vol01, (2, 0, 1))[idx]
            if flip_sag_lr.value: sl = np.fliplr(sl)
            if flip_sag_ud.value: sl = np.flipud(sl)
            return sl

    def redraw(*_):
        sl = get_current_slice()
        img_artist.set_data((sl * 255.0).astype(np.uint8))
        fig.canvas.draw_idle()

    def on_save_clicked(_):
        vol01 = window_volume(cached["vol_iso"], wl=float(l.value), ww=float(k.value))
        save_stacks(
            vol01,
            Path(out_dir.value),
            fmt=fmt.value,
            jpeg_quality=int(quality.value),
            save_axial=bool(save_ax.value), save_coronal=bool(save_co.value), save_sagittal=bool(save_sa.value),
            flip_ax_lr=bool(flip_ax_lr.value), flip_ax_ud=bool(flip_ax_ud.value),
            flip_cor_lr=bool(flip_cor_lr.value), flip_cor_ud=bool(flip_cor_ud.value),
            flip_sag_lr=bool(flip_sag_lr.value), flip_sag_ud=bool(flip_sag_ud.value),
            iso=cached["iso"], meta=meta,
        )
        with out_box:
            clear_output(wait=True)
            print(f"Saved to: {out_dir.value}")

    # Wire up events
    for w in [k, l, slice_idx, plane, flip_ax_lr, flip_ax_ud, flip_cor_lr, flip_cor_ud, flip_sag_lr, flip_sag_ud]:
        w.observe(redraw, names='value')
    iso_box.observe(maybe_resample, names='value')
    save_btn.on_click(on_save_clicked)

    controls_left = W.VBox([plane, slice_idx, k, l, iso_box])
    controls_right = W.VBox([
        W.HBox([fmt, quality]),
        W.HBox([save_ax, save_co, save_sa]),
        W.HBox([flip_ax_lr, flip_ax_ud]),
        W.HBox([flip_cor_lr, flip_cor_ud]),
        W.HBox([flip_sag_lr, flip_sag_ud]),
        out_dir,
        save_btn,
    ])
    ui = W.HBox([controls_left, controls_right])
    out_box = W.Output()
    display(ui)
    display(out_box)

    redraw()


def mpl_ui(dicom_dir: Path, out_dir: Optional[Path] = None, fmt: str = "png", jpeg_quality: int = 95, close_on_save: bool = False, iso_mm: Optional[float] = None):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider, Button
    except Exception as e:
        raise RuntimeError(f"matplotlib not available: {e}")

    vol, spacing, meta = load_dicom_volume(dicom_dir)
    vol_iso, iso = isotropic_resample(vol, spacing, iso_mm=iso_mm, order=1)

    plane_names = ["axial", "coronal", "sagittal"]
    plane_idx = 0

    def get_slices(vol01):
        return [vol01, np.transpose(vol01, (1, 0, 2)), np.transpose(vol01, (2, 0, 1))]

    wl, ww = 40.0, 400.0
    vol01 = window_volume(vol_iso, wl, ww)
    stacks = get_slices(vol01)

    fig, ax = plt.subplots(figsize=(7, 7))
    plt.subplots_adjust(left=0.25, bottom=0.25)
    img = ax.imshow((stacks[plane_idx][0] * 255.0).astype(np.uint8), cmap='gray', vmin=0, vmax=255)
    ax.set_title(plane_names[plane_idx])
    ax.axis('off')

    ax_wl = plt.axes([0.25, 0.17, 0.65, 0.03])
    ax_ww = plt.axes([0.25, 0.12, 0.65, 0.03])
    ax_slice = plt.axes([0.25, 0.07, 0.65, 0.03])
    s_wl = Slider(ax_wl, 'WL (l)', -500, 2000, valinit=wl, valstep=10)
    s_ww = Slider(ax_ww, 'WW (k)', 50, 3000, valinit=ww, valstep=10)
    s_slice = Slider(ax_slice, 'Slice', 1, stacks[plane_idx].shape[0], valinit=1, valstep=1)

    # Plane buttons
    bax_prev = plt.axes([0.025, 0.55, 0.1, 0.04])
    bax_next = plt.axes([0.135, 0.55, 0.1, 0.04])
    bax_save = plt.axes([0.025, 0.48, 0.21, 0.05])
    b_prev = Button(bax_prev, 'Prev Plane')
    b_next = Button(bax_next, 'Next Plane')
    b_save = Button(bax_save, 'Save All Views', color='#6aa84f', hovercolor='#93c47d')

    def update(_=None):
        nonlocal vol01, stacks
        vol01 = window_volume(vol_iso, s_wl.val, s_ww.val)
        stacks = get_slices(vol01)
        si = int(s_slice.val) - 1
        img.set_data((stacks[plane_idx][si] * 255.0).astype(np.uint8))
        ax.set_title(plane_names[plane_idx])
        fig.canvas.draw_idle()

    def change_plane(delta):
        nonlocal plane_idx
        plane_idx = (plane_idx + delta) % 3
        s_slice.valmax = stacks[plane_idx].shape[0]
        s_slice.set_val(min(s_slice.val, s_slice.valmax))
        update()

    def do_save(event=None):
        # Export axial/coronal/sagittal using current WL/WW
        target = out_dir if out_dir is not None else (dicom_dir.parent / "mpr_views")
        v01 = window_volume(vol_iso, s_wl.val, s_ww.val)
        save_stacks(v01, target, fmt=fmt, jpeg_quality=jpeg_quality,
                    save_axial=True, save_coronal=True, save_sagittal=True,
                    iso=iso, meta=meta)
        print(f"Saved MPR stacks to: {target}")
        if close_on_save:
            plt.close(fig)

    s_wl.on_changed(update)
    s_ww.on_changed(update)
    s_slice.on_changed(update)
    b_prev.on_clicked(lambda e: change_plane(-1))
    b_next.on_clicked(lambda e: change_plane(1))
    b_save.on_clicked(do_save)

    plt.show()


def parse_args():
    p = argparse.ArgumentParser(description="Interactive MPR preview + export with sliders")
    p.add_argument("--dicom", required=True, type=Path, help="DICOM directory (study)")
    p.add_argument("--out", type=Path, default=None, help="Output directory for saved MPR images")
    p.add_argument("--format", choices=["png", "jpeg", "jpg"], default="png", help="Output image format")
    p.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality (if format is jpeg/jpg)")
    p.add_argument("--close-on-save", action="store_true", help="Close window after saving")
    p.add_argument("--iso-mm", type=float, default=None, help="Target isotropic spacing in mm (default=min spacings)")
    return p.parse_args()


def main():
    args = parse_args()
    # Try Jupyter widgets first (if running in notebook)
    try:
        from IPython import get_ipython  # type: ignore
        ip = get_ipython()
        if ip and ip.has_trait('kernel'):
            jupyter_ui(args.dicom)
            return
    except Exception:
        pass
    mpl_ui(args.dicom, out_dir=args.out, fmt=args.format, jpeg_quality=args.jpeg_quality, close_on_save=args.close_on_save, iso_mm=args.iso_mm)


if __name__ == "__main__":
    main()
