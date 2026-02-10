#!/usr/bin/env python3
"""
totalseg_blender_slider.py

Configures the CT slider addon in the current Blender scene:
  - ensures addon is enabled
  - applies CT metadata (pixel size / slice count)
  - sets PNG folder path and CT offset

Usage:
    blender scene.blend -P totalseg_blender_slider.py -- \
        --png-dir /path/to/slices --nifti-path /path/to/scan.nii.gz \
        --scale 0.01 --addon-module ct_slicer_doctor --save output.blend
"""

import sys
import csv
from pathlib import Path
import bpy
import addon_utils


def parse_args():
    if "--" not in sys.argv:
        return {}
    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = {}
    i = 0
    while i < len(argv):
        if argv[i].startswith("--"):
            key = argv[i][2:]
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                args[key] = argv[i + 1]
                i += 2
            else:
                args[key] = True
                i += 1
        else:
            i += 1
    return args


def _make_logger(log_file: str | None):
    log_path = Path(log_file).resolve() if log_file else None

    def _log(msg: str):
        print(msg)
        if log_path:
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(msg.rstrip() + "\n")
            except Exception:
                pass

    return _log


def _addons_dir():
    try:
        scripts_dir = Path(bpy.utils.user_resource("SCRIPTS")).resolve()
        return (scripts_dir / "addons").resolve()
    except Exception:
        return None


def _read_metadata(png_dir_val: str):
    meta_path = Path(png_dir_val) / "metadata.csv"
    if not meta_path.exists():
        return {}
    data = {}
    try:
        with open(meta_path, newline="") as f:
            for row in csv.reader(f):
                if not row:
                    continue
                data[row[0]] = row[1:]
    except Exception:
        return {}
    return data


def _apply_ct_metadata(scene, meta, log):
    shape = meta.get("Isotropic_Shape_ZYX", [])
    iso = meta.get("Isotropic_Spacing_mm", [])
    try:
        if len(shape) >= 3 and len(iso) >= 1:
            z = int(float(shape[0]))
            y = int(float(shape[1]))
            x = int(float(shape[2]))
            iso_mm = float(iso[0])
            if hasattr(scene, "ax_slice_count"):
                scene.ax_slice_count = max(1, z)
                log(f"✓ ax_slice_count = {scene.ax_slice_count}")
            if hasattr(scene, "slice_spacing_ax"):
                scene.slice_spacing_ax = iso_mm
                log(f"✓ slice_spacing_ax = {scene.slice_spacing_ax}")
            if hasattr(scene, "pixel_pitch"):
                scene.pixel_pitch = iso_mm
                log(f"✓ pixel_pitch = {scene.pixel_pitch}")
            if hasattr(scene, "pixel"):
                scene.pixel = max(1, int(max(x, y)))
                log(f"✓ pixel = {scene.pixel}")
    except Exception as e:
        log(f"⚠️ Failed to apply CT metadata: {e}")


def _calc_mesh_center(collection_name="Organs"):
    try:
        coll = bpy.data.collections.get(collection_name)
        if not coll:
            return None
        points = []
        for obj in coll.all_objects:
            if obj.type != "MESH":
                continue
            for v in obj.bound_box:
                p = obj.matrix_world @ bpy.mathutils.Vector(v)
                points.append(p)
        if not points:
            return None
        min_x = min(p.x for p in points)
        min_y = min(p.y for p in points)
        min_z = min(p.z for p in points)
        max_x = max(p.x for p in points)
        max_y = max(p.y for p in points)
        max_z = max(p.z for p in points)
        return ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0, (min_z + max_z) / 2.0)
    except Exception:
        return None


def _ensure_addon_enabled(module_name: str, log):
    try:
        loaded, enabled = addon_utils.check(module_name)
        if enabled and loaded:
            log(f"ℹ️ Addon already enabled: {module_name}")
            return True
        addon_utils.enable(module_name, default_set=False)
        log(f"✅ Addon enabled: {module_name}")
        return True
    except Exception as e:
        log(f"❌ Failed to enable addon {module_name}: {e}")
        return False


def _get_addon_module(module_name: str):
    try:
        for mod in addon_utils.modules():
            if getattr(mod, "__name__", "") == module_name:
                return mod
    except Exception:
        pass
    return None


def main():
    args = parse_args()

    png_dir = args.get("png-dir", "")
    nifti_path = args.get("nifti-path", "")
    scale = float(args.get("scale", 0.01))
    save_path = args.get("save", "")
    log_file = args.get("log-file", "")
    addon_module = args.get("addon-module", "ct_slicer_doctor")
    log = _make_logger(log_file if log_file else None)

    log("=" * 70)
    log("🎬 CT Slider Setup")
    log("=" * 70)

    if not png_dir:
        log("❌ ERROR: --png-dir is required")
        sys.exit(1)

    png_dir = str(Path(png_dir).resolve())
    if not Path(png_dir).exists():
        log(f"❌ ERROR: PNG directory not found: {png_dir}")
        sys.exit(1)

    log(f"📁 PNG_DIR: {png_dir}")
    log(f"🧩 Addon module: {addon_module}")
    log(f"📏 Scale: {scale}")

    adir = _addons_dir()
    if adir:
        log(f"📂 User addons dir: {adir}")
        cand = adir / f"{addon_module}.py"
        log(f"📄 Addon file exists: {cand.exists()} ({cand})")
    if not _ensure_addon_enabled(addon_module, log):
        sys.exit(1)

    try:
        addon_keys = list(bpy.context.preferences.addons.keys())
        log(f"ℹ️ Enabled addons: {', '.join(addon_keys) if addon_keys else '(none)'}")
    except Exception:
        pass

    # Align CT units (units per mm) with mesh scale: mm -> m (0.001) * SCALE
    try:
        mod = _get_addon_module(addon_module)
        if mod and hasattr(mod, "DEFAULT_SIZE"):
            mod.DEFAULT_SIZE = float(scale) * 0.001
            log(f"✓ DEFAULT_SIZE set to {mod.DEFAULT_SIZE} (scale-aligned)")
        elif not mod:
            log("⚠️ Addon module not found in addon_utils.modules(); skip DEFAULT_SIZE")
    except Exception as e:
        log(f"⚠️ Failed to set DEFAULT_SIZE: {e}")

    scene = bpy.context.scene
    if hasattr(scene, "folder_path_png"):
        scene.folder_path_png = png_dir
        log(f"✓ folder_path_png = {png_dir}")
    if hasattr(scene, "folder_path_dicom"):
        scene.folder_path_dicom = png_dir
        log(f"✓ folder_path_dicom = {png_dir}")

    if png_dir:
        meta = _read_metadata(png_dir)
        if meta:
            _apply_ct_metadata(scene, meta, log)

    mesh_center = _calc_mesh_center("Organs")
    if mesh_center and hasattr(scene, "ct_offset"):
        scene.ct_offset = mesh_center
        log(f"✓ CT offset set to mesh center: {mesh_center}")

    try:
        mod = _get_addon_module(addon_module)
        if mod and callable(getattr(mod, "update_axis", None)):
            mod.update_axis(None, bpy.context)
            log("✓ update_axis executed")
        elif not mod:
            log("⚠️ Addon module not found for update_axis; skip")
    except Exception as e:
        log(f"⚠️ update_axis failed: {e}")

    if save_path:
        save_path = Path(save_path).resolve()
        bpy.ops.wm.save_as_mainfile(filepath=str(save_path))
        log(f"💾 Saved to: {save_path}")
    else:
        bpy.ops.wm.save_mainfile()
        log("💾 Saved current file")

    log("=" * 70)
    log("✅ CT Slider configured successfully!")
    log("=" * 70)


if __name__ == "__main__":
    main()

