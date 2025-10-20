#!/usr/bin/env python3
"""
totalseg_blender_import

Import organ STL/OBJ/PLY meshes into Blender with sensible defaults:
- Groups meshes into a collection
- Applies units (mm → meters) and optional uniform scale
- Assigns basic materials/colors by organ name
- Optionally saves a .blend file

Usage (inside Blender):
  blender -b -P totalseg_blender_import.py -- \
    --stl-dir path/to/stl_dir --save scene.blend --units m --collection Organs

If run outside Blender, prints the exact command to run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import bpy  # type: ignore
except Exception:
    bpy = None  # type: ignore


DEFAULT_COLLECTION = "Organs"

COLOR_PALETTE = {
    "liver": (0.85, 0.35, 0.35, 1.0),
    "spleen": (0.55, 0.2, 0.65, 1.0),
    "kidney": (0.35, 0.5, 0.85, 1.0),
    "heart": (0.9, 0.2, 0.2, 1.0),
    "aorta": (0.9, 0.4, 0.2, 1.0),
    "portal": (0.2, 0.6, 0.9, 1.0),
    "hepatic": (0.1, 0.7, 0.7, 1.0),
    "ivc": (0.2, 0.8, 0.8, 1.0),
    "lung": (0.8, 0.8, 0.8, 1.0),
    "bone": (0.9, 0.9, 0.8, 1.0),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Import organ meshes into Blender")
    p.add_argument("--stl-dir", required=True, type=Path, help="Directory with meshes (stl/obj/ply)")
    p.add_argument("--collection", default=DEFAULT_COLLECTION, help="Collection to create/use")
    p.add_argument("--units", choices=["m", "mm"], default="m", help="Units used in mesh files")
    p.add_argument("--scale", type=float, default=1.0, help="Extra uniform scale after unit conversion")
    p.add_argument("--recenter", action="store_true", help="Recenter imported meshes to origin")
    p.add_argument("--save", type=Path, default=None, help="Save scene to this .blend file")
    return p.parse_args(argv)


def ensure_collection(name: str):
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(coll)
    return coll


def color_for_name(name: str):
    key = name.lower()
    for k, col in COLOR_PALETTE.items():
        if k in key:
            return col
    return (0.7, 0.7, 0.7, 1.0)


def material_for(name: str):
    mat_name = f"mat_{name}"
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(name=mat_name)
        mat.diffuse_color = color_for_name(name)
    return mat


def import_mesh(filepath: Path, units: str = "m", extra_scale: float = 1.0, recenter: bool = False, collection=None):
    ext = filepath.suffix.lower()
    before = set(bpy.data.objects)
    if ext == ".stl":
        bpy.ops.import_mesh.stl(filepath=str(filepath))
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=str(filepath))
    elif ext == ".ply":
        bpy.ops.import_mesh.ply(filepath=str(filepath))
    else:
        return None
    after = set(bpy.data.objects)
    new_objs = list(after - before)
    for obj in new_objs:
        # Unit conversion (mm -> m)
        if units == "mm":
            obj.scale = (obj.scale[0] * 0.001, obj.scale[1] * 0.001, obj.scale[2] * 0.001)
        # Extra uniform scale
        if abs(extra_scale - 1.0) > 1e-6:
            obj.scale = (obj.scale[0] * extra_scale, obj.scale[1] * extra_scale, obj.scale[2] * extra_scale)
        # Recenter
        if recenter:
            obj.location = (0.0, 0.0, 0.0)
        # Material
        mat = material_for(filepath.stem)
        if obj.data and hasattr(obj.data, "materials"):
            if len(obj.data.materials) == 0:
                obj.data.materials.append(mat)
            else:
                obj.data.materials[0] = mat
        # Move to collection
        if collection is not None:
            for coll in obj.users_collection:
                coll.objects.unlink(obj)
            collection.objects.link(obj)
    return new_objs


def run_inside_blender(args):
    coll = ensure_collection(args.collection)
    imported = []
    for ext in ("*.stl", "*.obj", "*.ply"):
        for p in sorted(args.stl_dir.glob(ext)):
            imported.extend(import_mesh(p, units=args.units, extra_scale=args.scale, recenter=args.recenter, collection=coll))
    print(f"Imported {len(imported)} objects into collection '{args.collection}'")
    if args.save:
        bpy.ops.wm.save_mainfile(filepath=str(args.save))
        print(f"Saved scene to {args.save}")


def run_outside_blender(args):
    script_path = Path(__file__).resolve()
    print("Blender is required to run this command.")
    print("Run the following from your terminal:")
    print(
        "blender -b -P " + str(script_path) + " -- "
        f"--stl-dir {args.stl_dir} --collection {args.collection} --units {args.units} --scale {args.scale}"
        + (" --recenter" if args.recenter else "")
        + (f" --save {args.save}" if args.save else "")
    )


def main():
    # In Blender, arguments begin after '--'
    argv = sys.argv
    if "--" in argv:
        idx = argv.index("--")
        ns = parse_args(argv[idx + 1 :])
    else:
        ns = parse_args()

    if bpy is None:
        run_outside_blender(ns)
    else:
        run_inside_blender(ns)


if __name__ == "__main__":
    main()

