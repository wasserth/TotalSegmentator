#!/usr/bin/env python3
"""
totalseg_blender_slider

Create a simple visibility slider or a timeline sequence to cycle through
imported organ meshes in Blender.

Usage (timeline sequence, headless):
  blender -b -P totalseg_blender_slider.py -- 
    --collection Organs --make-timeline --start 1 --step 10 --save scene_slider.blend

Usage (UI panel; run interactively without -b):
  blender -P totalseg_blender_slider.py -- --collection Organs --panel
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import bpy  # type: ignore
except Exception:
    bpy = None  # type: ignore


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Add visibility slider or timeline sequence for a collection")
    p.add_argument("--collection", default="Organs", help="Collection name containing organ meshes")
    p.add_argument("--make-timeline", action="store_true", help="Animate visibility across frames (headless friendly)")
    p.add_argument("--start", type=int, default=1, help="Start frame for timeline sequence")
    p.add_argument("--step", type=int, default=10, help="Frames per object visibility window")
    p.add_argument("--panel", action="store_true", help="Add a simple UI panel with slider")
    p.add_argument("--save", type=Path, default=None, help="Save scene to this .blend file")
    return p.parse_args(argv)


def get_collection(name: str):
    return bpy.data.collections.get(name)


def set_only_visible(obj, visible: bool):
    obj.hide_viewport = not visible
    obj.hide_render = not visible


def make_timeline(collection_name: str, start: int, step: int):
    coll = get_collection(collection_name)
    if coll is None:
        print(f"Collection '{collection_name}' not found")
        return
    objs = [o for o in coll.objects if o.type == 'MESH']
    objs.sort(key=lambda o: o.name)
    scene = bpy.context.scene
    frame = start
    for i, o in enumerate(objs):
        for j, other in enumerate(objs):
            vis = (i == j)
            set_only_visible(other, vis)
            other.keyframe_insert(data_path="hide_viewport", frame=frame)
            other.keyframe_insert(data_path="hide_render", frame=frame)
        frame += step
    scene.frame_start = start
    scene.frame_end = max(start + step * max(0, len(objs) - 1), start)
    print(f"Timeline created for {len(objs)} objects, frames {scene.frame_start}-{scene.frame_end}")


def register_panel(collection_name: str):
    # Define a simple property on the scene to select object index
    class TS_OT_UpdateVisibility(bpy.types.Operator):
        bl_idname = "ts.update_visibility"
        bl_label = "Update Visibility"

        def execute(self, context):
            coll = get_collection(collection_name)
            if coll is None:
                return {'CANCELLED'}
            objs = [o for o in coll.objects if o.type == 'MESH']
            objs.sort(key=lambda o: o.name)
            idx = int(context.scene.ts_vis_index)
            for i, o in enumerate(objs):
                set_only_visible(o, i == idx)
            return {'FINISHED'}

    class TS_PT_VisibilityPanel(bpy.types.Panel):
        bl_label = "TotalSeg Visibility"
        bl_idname = "TS_PT_visibility"
        bl_space_type = 'VIEW_3D'
        bl_region_type = 'UI'
        bl_category = 'Totalseg'

        def draw(self, context):
            layout = self.layout
            layout.prop(context.scene, "ts_vis_index", slider=True)
            layout.operator("ts.update_visibility", text="Apply")

    # Register types
    if not hasattr(bpy.types.Scene, "ts_vis_index"):
        bpy.types.Scene.ts_vis_index = bpy.props.IntProperty(name="Index", default=0, min=0, soft_max=100)

    bpy.utils.register_class(TS_OT_UpdateVisibility)
    bpy.utils.register_class(TS_PT_VisibilityPanel)
    print("Panel registered under View3D > Totalseg")


def run_inside_blender(args):
    if args.make_timeline:
        make_timeline(args.collection, args.start, args.step)
    if args.panel:
        register_panel(args.collection)
    if args.save:
        bpy.ops.wm.save_mainfile(filepath=str(args.save))
        print(f"Saved scene to {args.save}")


def run_outside_blender(args):
    script = Path(__file__).resolve()
    print("Blender is required to run this command.")
    print("Example (timeline):")
    print(f"  blender -b -P {script} -- --collection {args.collection} --make-timeline --start {args.start} --step {args.step} --save scene_slider.blend")
    print("Example (interactive UI panel):")
    print(f"  blender -P {script} -- --collection {args.collection} --panel")


def main():
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

