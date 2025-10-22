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
    import mathutils  # type: ignore
except Exception:
    bpy = None  # type: ignore
    mathutils = None  # type: ignore


DEFAULT_COLLECTION = "Organs"

# Deterministic color helpers and semantic palette
import math
import re

# Add the center_collection function HERE - before it's used
def center_collection(collection):
    """Move the entire collection to the world origin (0,0,0)"""
    if not collection.objects and not collection.children:
        print("No objects found to center.")
        return
    
    # Collect all mesh objects from the collection and its child collections
    all_objects = []
    for obj in collection.objects:
        if obj.type == 'MESH':
            all_objects.append(obj)
    
    # Also collect objects from child collections
    for child_coll in collection.children:
        for obj in child_coll.objects:
            if obj.type == 'MESH':
                all_objects.append(obj)
    
    if not all_objects:
        print(f"No mesh objects found in collection '{collection.name}' to center")
        return
    
    print(f"Moving {len(all_objects)} objects to exact origin (0,0,0)...")
    
    # Calculate the current geometrical center
    sum_x, sum_y, sum_z = 0, 0, 0
    for obj in all_objects:
        # Use object world position
        sum_x += obj.location.x
        sum_y += obj.location.y
        sum_z += obj.location.z
    
    # Calculate average center
    center_x = sum_x / len(all_objects)
    center_y = sum_y / len(all_objects)
    center_z = sum_z / len(all_objects)
    
    print(f"Current average center: ({center_x:.4f}, {center_y:.4f}, {center_z:.4f})")
    
    # Move all objects to offset from current position
    for obj in all_objects:
        obj.location.x -= center_x  # Move to X=0
        obj.location.y -= center_y  # Move to Y=0 
        obj.location.z -= center_z  # Move to Z=0
    
    print(f"All objects moved to origin (0,0,0)")
    return True

def _hash_color(name: str) -> tuple[float, float, float, float]:
    # Stable, pleasant color from name using golden ratio
    h = 0
    for ch in name.lower():
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    hue = ((h % 360) / 360.0)
    sat = 0.55
    val = 0.85
    # HSV to RGB
    i = int(hue * 6)
    f = hue * 6 - i
    p = val * (1 - sat)
    q = val * (1 - f * sat)
    t = val * (1 - (1 - f) * sat)
    i = i % 6
    if i == 0:
        r, g, b = val, t, p
    elif i == 1:
        r, g, b = q, val, p
    elif i == 2:
        r, g, b = p, val, t
    elif i == 3:
        r, g, b = p, q, val
    elif i == 4:
        r, g, b = t, p, val
    else:
        r, g, b = val, p, q
    return (r, g, b, 1.0)

SEMANTIC_COLORS = [
    (re.compile(r"\b(liver)\b"), (0.85, 0.35, 0.35, 1.0)),
    (re.compile(r"\b(spleen)\b"), (0.55, 0.2, 0.65, 1.0)),
    (re.compile(r"\b(kidney|renal)\b"), (0.35, 0.5, 0.85, 1.0)),
    (re.compile(r"\b(heart|atrium|ventricle|cardiac)\b"), (0.9, 0.2, 0.2, 1.0)),
    (re.compile(r"\b(aorta|artery|carotid|subclavian|brachiocephalic)\b"), (0.9, 0.4, 0.2, 1.0)),
    (re.compile(r"\b(portal|hepatic|vein|vena|ivc|svc)\b"), (0.2, 0.6, 0.9, 1.0)),
    (re.compile(r"\b(lung|bronch)\b"), (0.8, 0.8, 0.85, 1.0)),
    (re.compile(r"\b(stomach)\b"), (0.95, 0.6, 0.6, 1.0)),
    (re.compile(r"\b(pancreas)\b"), (0.95, 0.7, 0.3, 1.0)),
    (re.compile(r"\b(gallbladder)\b"), (0.2, 0.8, 0.4, 1.0)),
    (re.compile(r"\b(colon|bowel|intestin|duodenum)\b"), (0.95, 0.85, 0.4, 1.0)),
    (re.compile(r"\b(esophag)\b"), (0.95, 0.7, 0.7, 1.0)),
    (re.compile(r"\b(thyroid|trachea)\b"), (0.6, 0.8, 0.9, 1.0)),
    (re.compile(r"\b(supr?arenal|adrenal)\b"), (0.95, 0.5, 0.2, 1.0)),
    (re.compile(r"\b(spleen)\b"), (0.55, 0.2, 0.65, 1.0)),
    (re.compile(r"\b(prostate|bladder)\b"), (0.4, 0.7, 0.9, 1.0)),
    (re.compile(r"\b(bone|skull|vertebra|rib|clavicle|scapula|femur|sacrum|hip|pelvis|humerus)\b"), (0.9, 0.9, 0.83, 1.0)),
    (re.compile(r"\b(muscle|gluteus|iliopsoas|back muscle)\b"), (0.9, 0.6, 0.6, 1.0)),
]

# Exact palette requested by user: material name -> RGBA
EXACT_MATERIALS: dict[str, tuple[float, float, float, float]] = {
    "Bone": (0.509338, 0.448805, 0.390992, 1.0),
    "Muscle": (0.458575, 0.114023, 0.099804, 1.0),
    "Liver": (0.359082, 0.052501, 0.044477, 1.0),
    "Stomach": (0.483567, 0.277414, 0.269021, 1.0),
    "Artery": (0.675526, 0.020398, 0.041993, 1.0),
    "Vein": (0.071473, 0.014120, 0.373470, 1.0),
    "Kidney": (0.359082, 0.084555, 0.060850, 1.0),
    "Adrenal": (0.799999, 0.254006, 0.030540, 1.0),
    "Pancreas": (0.450415, 0.259994, 0.102502, 1.0),
    "GB": (0.127960, 0.162910, 0.069646, 1.0),
    "Heart": (0.675526, 0.020398, 0.041993, 1.0),
    "Portal": (0.047439, 0.046528, 0.434352, 1.0),
    "Lung": (0.475151, 0.316953, 0.299059, 1.0),
    "Thyroid": (0.373470, 0.246540, 0.067987, 1.0),
    "Bladder": (0.591379, 0.383078, 0.371987, 1.0),
    "Spleen": (0.110568, 0.021969, 0.025012, 1.0),
    "Prostate": (0.366235, 0.160072, 0.063098, 1.0),
    "Colon": (0.403241, 0.212665, 0.103747, 1.0),
}

def _principled_set_color(mat, rgba: tuple[float, float, float, float]):
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    principled = None
    # Try to get existing Principled BSDF
    for n in nodes:
        if getattr(n, 'type', '') == 'BSDF_PRINCIPLED' or n.name == 'Principled BSDF':
            principled = n
            break
    if principled is None:
        principled = nodes.get('Principled BSDF')
    if principled is not None:
        principled.inputs['Base Color'].default_value = (rgba[0], rgba[1], rgba[2], 1.0)
        try:
            principled.inputs['Alpha'].default_value = rgba[3]
        except Exception:
            pass
    # Also set diffuse_color for viewport consistency
    try:
        mat.diffuse_color = rgba
    except Exception:
        pass

def ensure_principled_material(name: str, rgba: tuple[float, float, float, float], replace: bool = False):
    mat = bpy.data.materials.get(name)
    if mat is not None and replace:
        bpy.data.materials.remove(mat)
        mat = None
    if mat is None:
        mat = bpy.data.materials.new(name=name)
    _principled_set_color(mat, rgba)
    return mat

def setup_exact_materials(replace: bool = True):
    for mname, col in EXACT_MATERIALS.items():
        ensure_principled_material(mname, col, replace=replace)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Import organ meshes into Blender")
    p.add_argument("--stl-dir", required=True, type=Path, help="Directory with meshes (stl/obj/ply)")
    p.add_argument("--collection", default=DEFAULT_COLLECTION, help="Top-level collection to create/use")
    p.add_argument("--units", choices=["m", "mm"], default="m", help="Units used in mesh files")
    p.add_argument("--scale", type=float, default=1.0, help="Extra uniform scale after unit conversion")
    p.add_argument("--recenter", action="store_true", help="Recenter imported meshes to origin")
    p.add_argument("--group-categories", action="store_true", help="Create category subcollections and place objects accordingly")
    p.add_argument("--mirror-x", type=str, choices=["true", "false"], default="true", help="Mirror imported meshes across global X (fix mirrored exports)")
    p.add_argument("--rotate-x-deg", type=float, default=90.0, help="Rotate imported meshes around X in degrees (e.g., -90)")
    p.add_argument("--remesh", choices=["none", "voxel", "quad", "smooth", "sharp"], default="none", help="Apply a remesh modifier to imported meshes")
    p.add_argument("--voxel-size", type=float, default=0.003, help="Voxel size for voxel remesh (in scene units; 0.003 ≈ 3 mm if scene is meters)")
    p.add_argument("--palette", choices=["exact", "auto"], default="exact", help="Color palette: exact (fixed organ colors) or auto (semantic + distinct)")
    p.add_argument("--save", type=Path, default=None, help="Save scene to this .blend file")
    return p.parse_args(argv)


def ensure_collection(name: str, parent: "bpy.types.Collection" | None = None):
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        if parent is None:
            bpy.context.scene.collection.children.link(coll)
        else:
            parent.children.link(coll)
    return coll


def _normalize_name(s: str) -> str:
    s = s.lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    return s

def _strip_suffix(name: str) -> str:
    # Remove Blender's .001, .002 suffixes
    return re.sub(r"\.[0-9]{3}$", "", name)

def color_for_name(name: str):
    key = _normalize_name(name)
    for pat, col in SEMANTIC_COLORS:
        if pat.search(key):
            return col
    return _hash_color(key)


def material_for(name: str):
    mat_name = f"mat_{name}"
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(name=mat_name)
        _principled_set_color(mat, color_for_name(name))
    return mat


def _apply_remesh(obj, mode: str, voxel_size: float):
    if mode == "none":
        return
    # Ensure active/selected
    view = bpy.context.view_layer
    for o in bpy.context.selected_objects:
        o.select_set(False)
    obj.select_set(True)
    view.objects.active = obj
    bpy.ops.object.modifier_add(type='REMESH')
    mod = obj.modifiers[-1]
    mod.mode = mode.upper()
    if mode.lower() == 'voxel':
        mod.voxel_size = float(voxel_size)
    bpy.ops.object.modifier_apply(modifier=mod.name)


def _apply_rotate_and_mirror(obj, rotate_x_deg: float = 0.0, mirror_x: bool = False):
    # Apply rotation around X (degrees)
    if abs(rotate_x_deg) > 1e-6:
        obj.rotation_euler[0] += math.radians(rotate_x_deg)
    
    # Mirror across global X axis
    if mirror_x:
        # Ensure only this obj is selected and active
        view = bpy.context.view_layer
        for o in bpy.context.selected_objects:
            o.select_set(False)
        obj.select_set(True)
        view.objects.active = obj
        try:
            bpy.ops.transform.mirror(
                orient_type='GLOBAL',
                orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                orient_matrix_type='GLOBAL',
                constraint_axis=(True, False, False)
            )
        except Exception:
            # Fallback: negative X scale
            obj.scale[0] *= -1.0
        # Match user's intent to flip X location sign
        obj.location.x = -obj.location.x
    
    # ADDED: Mirror across global Y axis by default
    # Ensure only this obj is selected and active
    view = bpy.context.view_layer
    for o in bpy.context.selected_objects:
        o.select_set(False)
    obj.select_set(True)
    view.objects.active = obj
    try:
        bpy.ops.transform.mirror(
            orient_type='GLOBAL',
            orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            orient_matrix_type='GLOBAL',
            constraint_axis=(False, True, False)
        )
    except Exception:
        # Fallback: negative Y scale
        obj.scale[1] *= -1.0
    # Flip Y location sign
    obj.location.y = -obj.location.y


def import_mesh(filepath: Path, units: str = "m", extra_scale: float = 1.0, recenter: bool = False, collection=None, remesh_mode: str = "none", voxel_size: float = 0.003, palette: str = "exact", rotate_x_deg: float = 0.0, mirror_x: bool = False):
    ext = filepath.suffix.lower()
    before = set(bpy.data.objects)

    def _call_op(op, **kwargs):
        try:
            op(**kwargs)
            return True
        except Exception:
            return False

    called = False
    if ext == ".stl":
        # Try Blender 4.x style operator first
        op = getattr(getattr(bpy.ops, "wm", None), "stl_import", None)
        if op:
            called = _call_op(op, filepath=str(filepath))
        if not called:
            # Fallback to classic addon operator
            op = getattr(getattr(bpy.ops, "import_mesh", None), "stl", None)
            if op:
                called = _call_op(op, filepath=str(filepath))
        if not called:
            raise RuntimeError(
                "STL import operator not found. Enable the 'Import-Export: STL format' add-on (io_mesh_stl) in Blender Preferences, or install it."
            )
    elif ext == ".obj":
        # Prefer modern operator, fallback to legacy
        op = getattr(getattr(bpy.ops, "wm", None), "obj_import", None)
        if op:
            called = _call_op(op, filepath=str(filepath))
        if not called:
            op = getattr(getattr(bpy.ops, "import_scene", None), "obj", None)
            if op:
                called = _call_op(op, filepath=str(filepath))
    elif ext == ".ply":
        # Try both possible names
        op = getattr(getattr(bpy.ops, "wm", None), "ply_import", None)
        if op:
            called = _call_op(op, filepath=str(filepath))
        if not called:
            op = getattr(getattr(bpy.ops, "import_mesh", None), "ply", None)
            if op:
                called = _call_op(op, filepath=str(filepath))
    else:
        return None

    if not called:
        # Unknown or unsupported format in the current Blender build
        raise RuntimeError(f"No importer found for extension '{ext}' in this Blender build.")
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
        # Optional rotation / mirror (axis fix or mirrored exports)
        _apply_rotate_and_mirror(obj, rotate_x_deg=rotate_x_deg if 'rotate_x_deg' in locals() else 0.0, mirror_x=mirror_x if 'mirror_x' in locals() else False)
        # Optional remesh
        if remesh_mode and remesh_mode != 'none':
            _apply_remesh(obj, remesh_mode, voxel_size)
        # Material
        mat = None
        if palette == 'exact':
            # Try exact mapping first (object name or file stem)
            mname = lookup_exact_material(obj.name) or lookup_exact_material(filepath.stem)
            if mname:
                col = EXACT_MATERIALS.get(mname)
                if col:
                    mat = ensure_principled_material(mname, col, replace=False)
        if mat is None and palette != 'exact':
            mat = material_for(filepath.stem)
        if obj.data and hasattr(obj.data, "materials"):
            if len(obj.data.materials) == 0:
                if mat is not None:
                    obj.data.materials.append(mat)
            else:
                if mat is not None:
                    obj.data.materials[0] = mat
        # Move to collection
        if collection is not None:
            for coll in obj.users_collection:
                coll.objects.unlink(obj)
            collection.objects.link(obj)
    return new_objs


CATEGORY_MAP = {
    "Bone": [
        r"\b(skull|costal cartilage|sternum|clavicle|scapula|humerus|vertebra|vertebrae|vertebral|rib|sacrum|hip|pelvis|femur|spine|spinal cord|spinal)\b",
    ],
    "Muscle": [
        r"\b(iliopsoas|gluteus|deep back muscle|muscle)\b",
    ],
    "Thoracic": [
        r"\b(trachea|heart|atrial appendage|pulmonary venous|thyroid|lung)\b",
    ],
    "Abdominal": [
        r"\b(urinary bladder|prostate|colon|duodenum|esophag|gallbladder|adrenal|kidney|liver|pancreas|small bowel|spleen|stomach)\b",
    ],
    "Vessel": [
        r"\b(aorta|vena|vein|artery|carotid|subclavian|brachiocephalic|iliac)\b",
    ],
}

def guess_category(name: str) -> str | None:
    key = _normalize_name(name)
    for cat, patterns in CATEGORY_MAP.items():
        for pat in patterns:
            if re.search(pat, key):
                return cat
    return None

# Exact object-to-material mapping using normalized names
EXACT_OBJECT_TO_MATERIAL: dict[str, str] = {
    # Bone
    _normalize_name("skull"): "Bone",
    _normalize_name("costal cartilage"): "Bone",
    _normalize_name("sternum"): "Bone",
    _normalize_name("left clavicle"): "Bone",
    _normalize_name("right clavicle"): "Bone",
    _normalize_name("left scapula"): "Bone",
    _normalize_name("right scapula"): "Bone",
    _normalize_name("left humerus"): "Bone",
    _normalize_name("right humerus"): "Bone",
    _normalize_name("C3 vertebra"): "Bone",
    _normalize_name("C4 vertebra"): "Bone",
    _normalize_name("C5 vertebra"): "Bone",
    _normalize_name("C6 vertebra"): "Bone",
    _normalize_name("C7 vertebra"): "Bone",
    _normalize_name("T1 vertebra"): "Bone",
    _normalize_name("T2 vertebra"): "Bone",
    _normalize_name("T3 vertebra"): "Bone",
    _normalize_name("T4 vertebra"): "Bone",
    _normalize_name("T5 vertebra"): "Bone",
    _normalize_name("T6 vertebra"): "Bone",
    _normalize_name("T7 vertebra"): "Bone",
    _normalize_name("T8 vertebra"): "Bone",
    _normalize_name("T9 vertebra"): "Bone",
    _normalize_name("T10 vertebra"): "Bone",
    _normalize_name("T11 vertebra"): "Bone",
    _normalize_name("T12 vertebra"): "Bone",
    _normalize_name("L1 vertebra"): "Bone",
    _normalize_name("L2 vertebra"): "Bone",
    _normalize_name("L3 vertebra"): "Bone",
    _normalize_name("L4 vertebra"): "Bone",
    _normalize_name("L5 vertebra"): "Bone",
    _normalize_name("S1 vertebra"): "Bone",
    _normalize_name("Sacrum"): "Bone",
    _normalize_name("left rib 1"): "Bone",
    _normalize_name("left rib 2"): "Bone",
    _normalize_name("left rib 3"): "Bone",
    _normalize_name("left rib 4"): "Bone",
    _normalize_name("left rib 5"): "Bone",
    _normalize_name("left rib 6"): "Bone",
    _normalize_name("left rib 7"): "Bone",
    _normalize_name("left rib 8"): "Bone",
    _normalize_name("left rib 9"): "Bone",
    _normalize_name("left rib 10"): "Bone",
    _normalize_name("left rib 11"): "Bone",
    _normalize_name("left rib 12"): "Bone",
    _normalize_name("right rib 1"): "Bone",
    _normalize_name("right rib 2"): "Bone",
    _normalize_name("right rib 3"): "Bone",
    _normalize_name("right rib 4"): "Bone",
    _normalize_name("right rib 5"): "Bone",
    _normalize_name("right rib 6"): "Bone",
    _normalize_name("right rib 7"): "Bone",
    _normalize_name("right rib 8"): "Bone",
    _normalize_name("right rib 9"): "Bone",
    _normalize_name("right rib 10"): "Bone",
    _normalize_name("right rib 11"): "Bone",
    _normalize_name("right rib 12"): "Bone",
    _normalize_name("left hip"): "Bone",
    _normalize_name("right hip"): "Bone",
    _normalize_name("left femur"): "Bone",
    _normalize_name("right femur"): "Bone",
    _normalize_name("spinal cord"): "Bone",

    # Muscle
    _normalize_name("left iliopsoas muscle"): "Muscle",
    _normalize_name("right iliopsoas muscle"): "Muscle",
    _normalize_name("left deep back muscle"): "Muscle",
    _normalize_name("right deep back muscle"): "Muscle",
    _normalize_name("left gluteus maximus"): "Muscle",
    _normalize_name("left gluteus medius"): "Muscle",
    _normalize_name("left gluteus minimus"): "Muscle",
    _normalize_name("right gluteus maximus"): "Muscle",
    _normalize_name("right gluteus medius"): "Muscle",
    _normalize_name("right gluteus minimus"): "Muscle",

    # Thoracic
    _normalize_name("trachea"): "",
    _normalize_name("heart"): "Heart",
    _normalize_name("left atrial appendage"): "Heart",
    _normalize_name("pulmonary venous system"): "Vein",
    _normalize_name("thyroid"): "Thyroid",
    _normalize_name("superior lobe of right lung"): "Lung",
    _normalize_name("inferior lobe of left lung"): "Lung",
    _normalize_name("inferior lobe of right lung"): "Lung",
    _normalize_name("middle lobe of right lung"): "Lung",
    _normalize_name("superior lobe of left lung"): "Lung",

    # Abdominal
    _normalize_name("urinary bladder"): "Bladder",
    _normalize_name("prostate"): "Prostate",
    _normalize_name("colon"): "Colon",
    _normalize_name("duodenum"): "Stomach",
    _normalize_name("esophagus"): "Stomach",
    _normalize_name("gallbladder"): "GB",
    _normalize_name("left adrenal gland"): "Adrenal",  # corrected from "Adrenall"
    _normalize_name("right adrenal gland"): "Adrenal",
    _normalize_name("left kidney"): "Kidney",
    _normalize_name("right kidney"): "Kidney",
    _normalize_name("liver"): "Liver",
    _normalize_name("pancreas"): "Pancreas",
    _normalize_name("small bowel"): "Stomach",
    _normalize_name("spleen"): "Spleen",
    _normalize_name("stomach"): "Stomach",

    # Vessel
    _normalize_name("aorta"): "Artery",
    _normalize_name("superior vena cava"): "Vein",
    _normalize_name("inferior vena cava"): "Vein",
    _normalize_name("left brachiocephalic vein"): "Vein",
    _normalize_name("right brachiocephalic vein"): "Vein",
    _normalize_name("left subclavian artery"): "Artery",
    _normalize_name("right subclavian artery"): "Artery",
    _normalize_name("brachiocephalic trunk"): "Vein",
    _normalize_name("left common iliac artery"): "Artery",
    _normalize_name("right common iliac artery"): "Artery",
    _normalize_name("left common carotid artery"): "Artery",
    _normalize_name("right common carotid artery"): "Artery",
    _normalize_name("left common iliac vein"): "Vein",
    _normalize_name("right common iliac vein"): "Vein",
    _normalize_name("portal vein and splenic vein"): "Vein",
}

def lookup_exact_material(name: str) -> str | None:
    key = _normalize_name(_strip_suffix(name))
    mname = EXACT_OBJECT_TO_MATERIAL.get(key)
    if mname is not None and len(mname) == 0:
        return None
    return mname


def run_inside_blender(args):
    # Delete the default cube, light and camera
    for obj_name in ["Cube", "Light", "Camera"]:
        if obj_name in bpy.data.objects:
            obj = bpy.data.objects[obj_name]
            bpy.data.objects.remove(obj)
    
    # Convert mirror-x string to boolean
    mirror_x = args.mirror_x.lower() == "true"
    
    # Prepare exact materials if requested
    if args.palette == 'exact':
        setup_exact_materials(replace=True)
    parent_coll = ensure_collection(args.collection)
    cat_collections = {}
    if args.group_categories:
        for cat in CATEGORY_MAP.keys():
            cat_collections[cat] = ensure_collection(cat, parent=parent_coll)
    imported = []
    for ext in ("*.stl", "*.obj", "*.ply"):
        for p in sorted(args.stl_dir.glob(ext)):
            # Choose collection per object if grouping
            target_coll = parent_coll
            if args.group_categories:
                cat = guess_category(p.stem)
                if cat is None and args.palette == 'exact':
                    mname = lookup_exact_material(p.stem)
                    if mname in {"Bone"}:
                        cat = "Bone"
                    elif mname in {"Muscle"}:
                        cat = "Muscle"
                    elif mname in {"Artery", "Vein", "Portal"}:
                        cat = "Vessel"
                    elif mname in {"Heart", "Lung", "Thyroid"}:  # trachea intentionally has no material
                        cat = "Thoracic"
                    elif mname in {"Liver", "Stomach", "Colon", "Kidney", "Spleen", "Pancreas", "GB", "Bladder", "Prostate", "Adrenal"}:
                        cat = "Abdominal"
                if cat is not None:
                    target_coll = cat_collections.get(cat, parent_coll)
            imported.extend(
                import_mesh(
                    p,
                    units=args.units,
                    extra_scale=args.scale,
                    recenter=args.recenter,
                    collection=target_coll,
                    remesh_mode=args.remesh,
                    voxel_size=args.voxel_size,
                    palette=args.palette,
                    rotate_x_deg=args.rotate_x_deg,
                    mirror_x=mirror_x,
                )
            )
    print(f"Imported {len(imported)} objects into collection '{args.collection}'")
    
    # Center the entire collection after all objects are imported
    center_collection(parent_coll)
    
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
        + (" --group-categories" if getattr(args, 'group_categories', False) else "")
        + (f" --palette {args.palette}" if getattr(args, 'palette', None) else "")
        + (f" --remesh {args.remesh}" if getattr(args, 'remesh', None) else "")
        + (f" --voxel-size {args.voxel_size}" if getattr(args, 'voxel_size', None) else "")
        + (f" --mirror-x {args.mirror_x}" if getattr(args, 'mirror_x', "true") != "true" else "")
        + (f" --rotate-x-deg {args.rotate_x_deg}" if getattr(args, 'rotate_x_deg', 90.0) != 90.0 else "")
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

