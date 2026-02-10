bl_info = {
    "name": "CT Slider + Vessel Path Placer (Unified Panel)",
    "author": "Sora",
    "version": (3, 6, 2),  # fix: no drift + no out-of-plane clicks + ignore UI panel clicks
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar(N) > CT Vessel",
    "description": "Unified UI: CT image slider (persistent Image Empty per axis) + click-to-add NURBS Path points. FIX: click-plane drift on fast sliding + restrict clicks to plane bounds + ignore UI panel clicks while placing.",
    "category": "3D View",
}

import bpy
import os
from mathutils import Vector
from bpy_extras import view3d_utils

# =========================================================
# Defaults
# =========================================================
DEFAULT_PIXEL = 512
DEFAULT_AX_COUNT = 419
DEFAULT_AX_SPACING = 2.0
DEFAULT_PITCH = 0.683
DEFAULT_SIZE = 0.02
DEFAULT_FOLDER = ''

RAW_KEY = "vessel_raw_points"


# =========================================================
# Small utilities (added)
# =========================================================
def safe_viewlayer_update():
    """Force depsgraph evaluation to avoid 1-frame lag (critical for fast slider moves)."""
    try:
        bpy.context.view_layer.update()
    except Exception:
        pass


def tag_redraw_3d():
    try:
        for area in bpy.context.window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
    except Exception:
        pass


def mouse_in_view3d_window_region(context, event):
    """
    True only if mouse is inside the VIEW_3D WINDOW region (not on UI panels).
    Prevents placing points when clicking N-panel / UI.
    """
    area = context.area
    if area is None or area.type != 'VIEW_3D':
        return False

    win_region = None
    for r in area.regions:
        if r.type == 'WINDOW':
            win_region = r
            break
    if win_region is None:
        return False

    # event.mouse_x/mouse_y are window coords; region has x,y origin in window coords
    mx = event.mouse_x - win_region.x
    my = event.mouse_y - win_region.y
    return (0 <= mx < win_region.width) and (0 <= my < win_region.height)


# =========================================================
# CT core
# =========================================================
def compute_scales(ctx):
    ax_count = ctx.scene.ax_slice_count
    ax_spacing = ctx.scene.slice_spacing_ax
    pixel_pitch = ctx.scene.pixel_pitch
    pixel = ctx.scene.pixel
    size = DEFAULT_SIZE

    cc_scale = ax_count * ax_spacing * size
    ax_scale = size * pixel_pitch * pixel
    common_scale = cc_scale if (pixel_pitch * pixel) < (ax_count * ax_spacing) else ax_scale
    return ax_scale, common_scale


def load_reference_image(image_path):
    try:
        return bpy.data.images.load(image_path, check_existing=True)
    except Exception as e:
        print(f"[CT] Failed to load image: {image_path}, Error: {e}")
        return None


def get_or_create_ct_empty(context, axis_type: str) -> bpy.types.Object:
    """Persistent Image Empty per axis: CT_AX / CT_COR / CT_SAG"""
    name = f"CT_{axis_type}"
    obj = bpy.data.objects.get(name)

    if obj is None or obj.type != 'EMPTY':
        bpy.ops.object.add(type='EMPTY', location=(0, 0, 0))
        obj = bpy.context.object
        obj.name = name
        obj.empty_display_type = 'IMAGE'
        obj.empty_display_size = 1.0

    obj.empty_display_type = 'IMAGE'
    return obj


def get_or_create_click_plane(context, axis_type: str) -> bpy.types.Object:
    """Persistent click plane per axis: CT_ClickPlane_AX / _COR / _SAG (2x2 base plane)."""
    name = f"CT_ClickPlane_{axis_type}"
    plane = bpy.data.objects.get(name)

    if plane is None or plane.type != 'MESH':
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, 0, 0))
        plane = bpy.context.object
        plane.name = name

    plane.hide_select = True
    plane.hide_render = True
    plane.display_type = 'WIRE'
    plane.show_in_front = True

    plane["ct_click_plane"] = True
    plane["ct_axis"] = axis_type
    return plane


def sync_click_plane_to_ct(context, axis_type: str, ct_obj: bpy.types.Object):
    """No parenting: click plane follows CT by copying matrix_world every update."""
    plane = get_or_create_click_plane(context, axis_type)
    plane.matrix_world = ct_obj.matrix_world.copy()
    return plane


def ensure_all_planes_synced(context):
    for ax in ("AX", "COR", "SAG"):
        ct = bpy.data.objects.get(f"CT_{ax}")
        if ct and ct.type == 'EMPTY':
            try:
                sync_click_plane_to_ct(context, ax, ct)
            except Exception:
                pass


def update_slider(self, context):
    bpy.ops.object.image_slider_operator()


def update_axis(self, context):
    if context.scene.image_slider_axis == 'AX':
        maxv = max(1, context.scene.ax_slice_count)
    else:
        maxv = max(1, context.scene.pixel)

    bpy.types.Scene.image_slider_property = bpy.props.IntProperty(
        name="Slide",
        min=1,
        max=maxv,
        default=min(getattr(context.scene, 'image_slider_property', 1), maxv),
        update=update_slider
    )

    # Instantiate/sync for new axis
    try:
        bpy.ops.object.image_slider_operator()
    except Exception:
        pass

    tag_redraw_3d()


class ImageSliderOperator(bpy.types.Operator):
    bl_idname = "object.image_slider_operator"
    bl_label = "Update CT Slice"

    def execute(self, context):
        sc = context.scene
        axis_type = sc.image_slider_axis
        slider_value = sc.image_slider_property
        folder_path = sc.folder_path_png

        pixel_pitch = sc.pixel_pitch
        pixel = sc.pixel
        ax_slides = sc.ax_slice_count
        ax_distance = sc.slice_spacing_ax

        if not folder_path:
            self.report({'WARNING'}, "Parent Folder is empty.")
            return {'CANCELLED'}

        ax_scale, common_scale = compute_scales(context)
        fname = f"{slider_value:03d}.png"

        if axis_type == 'AX':
            subdir = 'axial'
        elif axis_type == 'COR':
            subdir = 'coronal'
        else:
            subdir = 'sagittal'

        image_path = os.path.join(folder_path, subdir, fname)
        img = load_reference_image(image_path)
        if not img:
            self.report({'WARNING'}, f"Image not found: {image_path}")
            return {'CANCELLED'}

        ct_obj = get_or_create_ct_empty(context, axis_type)
        ct_obj.data = img

        if axis_type == 'AX':
            ct_obj.rotation_euler = (1.5708, 0, 0)
            ct_obj.location = (0, -(ax_distance / 50) * ((ax_slides + 1) / 2 - slider_value), 0)
            ct_obj.scale = (ax_scale, ax_scale, ax_scale)

        elif axis_type == 'COR':
            ct_obj.rotation_euler = (0, 3.14159, 3.14159)
            ct_obj.location = (0, 0, (pixel_pitch / 50) * (pixel / 2 + 0.5 - slider_value))
            ct_obj.scale = (common_scale, common_scale, common_scale)

        elif axis_type == 'SAG':
            ct_obj.rotation_euler = (0, 1.5708, 3.14159)
            ct_obj.location = (-(pixel_pitch / 50) * (pixel / 2 + 0.5 - slider_value), 0, 0)
            ct_obj.scale = (common_scale, common_scale, common_scale)

        # Apply global offset (computed from meshes) if present
        try:
            if hasattr(sc, "ct_offset"):
                ox, oy, oz = sc.ct_offset
                ct_obj.location.x += ox
                ct_obj.location.y += oy
                ct_obj.location.z += oz
        except Exception:
            pass

        sync_click_plane_to_ct(context, axis_type, ct_obj)
        if sc.ct_sync_all_axes:
            ensure_all_planes_synced(context)

        safe_viewlayer_update()
        tag_redraw_3d()

        return {'FINISHED'}


# =========================================================
# Vessel appearance
# =========================================================
def palette_rgba(key: str):
    pal = {
        "CYAN":   (0.2, 0.95, 0.95, 1.0),
        "RED":    (1.0, 0.15, 0.15, 1.0),
        "GREEN":  (0.2, 1.0, 0.2, 1.0),
        "BLUE":   (0.2, 0.5, 1.0, 1.0),
        "YELLOW": (1.0, 0.9, 0.2, 1.0),
        "ORANGE": (1.0, 0.55, 0.1, 1.0),
        "PURPLE": (0.7, 0.3, 1.0, 1.0),
        "WHITE":  (0.95, 0.95, 0.95, 1.0),
    }
    return pal.get(key, (0.95, 0.95, 0.95, 1.0))


def ensure_material(name_prefix: str, color_key: str, rgba):
    mat_name = f"{name_prefix}_{color_key}"
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(mat_name)
        mat.use_nodes = True

    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = rgba
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = 0.35
    return mat


def apply_path_appearance(context, curve_obj: bpy.types.Object):
    if curve_obj is None or curve_obj.type != 'CURVE':
        return
    sc = context.scene
    cd = curve_obj.data

    cd.dimensions = '3D'
    cd.use_path = True

    cd.bevel_depth = float(sc.vessel_path_depth)
    cd.bevel_resolution = int(sc.vessel_path_bevel_resolution)
    cd.use_fill_caps = bool(sc.vessel_path_fill_caps)

    rgba = palette_rgba(sc.vessel_path_color)
    mat = ensure_material("VesselPathMat", sc.vessel_path_color, rgba)
    if cd.materials:
        cd.materials[0] = mat
    else:
        cd.materials.append(mat)

    try:
        curve_obj.color = (rgba[0], rgba[1], rgba[2], 1.0)
    except Exception:
        pass


# =========================================================
# Vessel Path core
# =========================================================
def _object_is_alive(obj) -> bool:
    """PointerProperty may still reference a deleted datablock; this checks reliably."""
    try:
        return (obj is not None) and (obj.name in bpy.data.objects) and (bpy.data.objects.get(obj.name) == obj)
    except Exception:
        return False


def make_unique_curve_name(base: str) -> str:
    base = (base or "VesselPath").strip() or "VesselPath"
    if bpy.data.objects.get(base) is None:
        return base
    i = 1
    while True:
        name = f"{base}_{i:03d}"
        if bpy.data.objects.get(name) is None:
            return name
        i += 1


def create_new_path_object(context, base_name: str) -> bpy.types.Object:
    name = make_unique_curve_name(base_name)
    curve_data = bpy.data.curves.new(name=f"{name}Data", type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.use_path = True

    spline = curve_data.splines.new(type='NURBS')
    spline.order_u = 2  # will be adjusted in rebuild

    obj = bpy.data.objects.new(name, curve_data)
    context.scene.collection.objects.link(obj)

    obj[RAW_KEY] = []
    apply_path_appearance(context, obj)
    return obj


def ensure_current_path(context) -> bpy.types.Object:
    sc = context.scene

    if _object_is_alive(sc.vessel_curve_object) and sc.vessel_curve_object.type == 'CURVE':
        if RAW_KEY not in sc.vessel_curve_object:
            sc.vessel_curve_object[RAW_KEY] = []
        apply_path_appearance(context, sc.vessel_curve_object)
        return sc.vessel_curve_object

    obj = create_new_path_object(context, sc.vessel_new_curve_name)
    sc.vessel_curve_object = obj
    return obj


def get_target_path(context):
    sc = context.scene
    if sc.vessel_curve_mode == "NEW":
        return ensure_current_path(context)

    if _object_is_alive(sc.vessel_target_curve) and sc.vessel_target_curve.type == 'CURVE':
        return sc.vessel_target_curve

    ao = context.active_object
    if ao and ao.type == 'CURVE':
        return ao

    return None


def get_raw_points(curve_obj: bpy.types.Object):
    flat = list(curve_obj.get(RAW_KEY, []))
    pts = []
    for i in range(0, len(flat), 3):
        pts.append(Vector((flat[i], flat[i + 1], flat[i + 2])))
    return pts


def append_raw_point(curve_obj: bpy.types.Object, p: Vector):
    flat = list(curve_obj.get(RAW_KEY, []))
    flat.extend([float(p.x), float(p.y), float(p.z)])
    curve_obj[RAW_KEY] = flat


def rebuild_nurbs_path(context, curve_obj: bpy.types.Object, points, order_u: int, resolution_u: int):
    if curve_obj is None or curve_obj.type != 'CURVE':
        return

    cd = curve_obj.data
    cd.splines.clear()
    cd.resolution_u = max(1, int(resolution_u))

    spline = cd.splines.new(type='NURBS')

    if not points:
        spline.order_u = 2
        spline.use_endpoint_u = True
        if len(spline.points) == 0:
            spline.points.add(count=1)
        spline.points[0].co = (0.0, 0.0, 0.0, 1.0)
        apply_path_appearance(context, curve_obj)
        return

    need = len(points)
    have = len(spline.points)
    if need > have:
        spline.points.add(count=need - have)

    ou = max(2, int(order_u))
    ou = min(ou, len(points))
    spline.order_u = ou
    spline.use_endpoint_u = True

    for i, p in enumerate(points):
        spline.points[i].co = (float(p.x), float(p.y), float(p.z), 1.0)

    apply_path_appearance(context, curve_obj)


# =========================================================
# click location: analytic intersection with CURRENT axis click plane
# =========================================================
def intersect_current_axis_click_plane(context, origin: Vector, direction: Vector):
    """
    Intersect the view ray with CURRENT axis click plane.
    Returns hit point ONLY if inside the plane bounds (local x/y within +/-0.5 for size=1 plane).
    Uses evaluated depsgraph matrix_world to avoid drift during fast slider changes.
    """
    axis = context.scene.image_slider_axis
    plane = bpy.data.objects.get(f"CT_ClickPlane_{axis}")
    if plane is None:
        return None

    safe_viewlayer_update()

    deps = context.evaluated_depsgraph_get()
    pe = plane.evaluated_get(deps)
    mw = pe.matrix_world.copy()
    inv = mw.inverted()

    p0 = mw.translation
    n = (mw.to_3x3() @ Vector((0, 0, 1))).normalized()

    denom = n.dot(direction)
    if abs(denom) < 1e-10:
        return None

    t = n.dot(p0 - origin) / denom
    if t < 0:
        return None

    hit = origin + direction * t

    # Bound check: primitive_plane_add(size=1.0) => local x,y in [-0.5, +0.5]
    hit_local = inv @ hit
    if abs(hit_local.x) > 0.5 or abs(hit_local.y) > 0.5:
        return None

    return hit


# =========================================================
# Operators
# =========================================================
class VIEW3D_OT_new_vessel_path(bpy.types.Operator):
    """Create a new (2nd/3rd/...) path and make it current (Refresh)."""
    bl_idname = "view3d.new_vessel_path"
    bl_label = "New/Refresh Path"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        sc = context.scene
        obj = create_new_path_object(context, sc.vessel_new_curve_name)
        sc.vessel_curve_object = obj
        self.report({'INFO'}, f"New path created: {obj.name}")
        return {'FINISHED'}


class VIEW3D_OT_apply_path_appearance(bpy.types.Operator):
    bl_idname = "view3d.apply_path_appearance"
    bl_label = "Apply Appearance"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        sc = context.scene
        curve_obj = get_target_path(context)
        if curve_obj is None or curve_obj.type != 'CURVE':
            self.report({'WARNING'}, "No target curve found.")
            return {'CANCELLED'}
        apply_path_appearance(context, curve_obj)
        self.report({'INFO'}, "Applied path appearance.")
        return {'FINISHED'}


class VIEW3D_OT_click_add_path_points(bpy.types.Operator):
    """Click to add points to NURBS Path on the CURRENT axis CT click plane"""
    bl_idname = "view3d.click_add_path_points"
    bl_label = "Start Placing Path"
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        if context.area.type != 'VIEW_3D':
            self.report({'WARNING'}, "Run this operator in a 3D View")
            return {'CANCELLED'}

        sc = context.scene
        if sc.vessel_place_active:
            self.report({'INFO'}, "Already active. Use Stop.")
            return {'CANCELLED'}

        if sc.vessel_curve_mode == "NEW":
            ensure_current_path(context)

        sc.vessel_place_active = True
        context.window_manager.modal_handler_add(self)
        self.report({'INFO'}, "Placing ON: Left Click to add point on CURRENT axis CT plane. ESC/Right Click or Stop to finish.")
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        sc = context.scene

        if not sc.vessel_place_active:
            return {'CANCELLED'}

        if event.type in {'RIGHTMOUSE', 'ESC'}:
            sc.vessel_place_active = False
            self.report({'INFO'}, "Placing OFF.")
            return {'CANCELLED'}

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            # Avoid UI panel clicks
            if not mouse_in_view3d_window_region(context, event):
                return {'PASS_THROUGH'}

            region = context.region
            rv3d = context.region_data
            if region is None or rv3d is None or region.type != 'WINDOW':
                return {'PASS_THROUGH'}

            coord = (event.mouse_region_x, event.mouse_region_y)
            origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
            direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)

            location = intersect_current_axis_click_plane(context, origin, direction)
            if location is None:
                return {'PASS_THROUGH'}

            curve_obj = ensure_current_path(context) if sc.vessel_curve_mode == "NEW" else get_target_path(context)
            if curve_obj is None:
                self.report({'WARNING'}, "No target curve found.")
                return {'RUNNING_MODAL'}

            if RAW_KEY not in curve_obj:
                curve_obj[RAW_KEY] = []

            append_raw_point(curve_obj, location)
            raw = get_raw_points(curve_obj)

            rebuild_nurbs_path(
                context, curve_obj, raw,
                order_u=sc.vessel_nurbs_order,
                resolution_u=sc.vessel_curve_resolution
            )

            tag_redraw_3d()
            return {'RUNNING_MODAL'}

        return {'PASS_THROUGH'}


class VIEW3D_OT_stop_place(bpy.types.Operator):
    bl_idname = "view3d.stop_place"
    bl_label = "Stop Placing"
    bl_options = {'REGISTER'}

    def execute(self, context):
        context.scene.vessel_place_active = False
        self.report({'INFO'}, "Placing OFF.")
        return {'FINISHED'}


# =========================================================
# Panel
# =========================================================
class VIEW3D_PT_ct_vessel_unified(bpy.types.Panel):
    bl_label = "CT + Vessel Tools"
    bl_idname = "VIEW3D_PT_ct_vessel_unified"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "CT Vessel"

    def draw(self, context):
        layout = self.layout
        sc = context.scene

        # ---- CT slider ----
        box = layout.box()
        box.label(text="CT Slider")
        box.prop(sc, "folder_path_png")
        box.prop(sc, "image_slider_axis", expand=True)
        box.prop(sc, "image_slider_property", slider=True)

        col = box.column(align=True)
        col.label(text="Parameters")
        col.prop(sc, "ax_slice_count")
        col.prop(sc, "slice_spacing_ax")
        col.prop(sc, "pixel_pitch")
        col.prop(sc, "pixel")

        box.prop(sc, "ct_sync_all_axes")
        box.label(text="Click-plane follows CT by matrix_world (no parenting).")
        box.label(text="Raycast accepts CURRENT axis plane only.")

        # ---- Vessel ----
        box2 = layout.box()
        box2.label(text="Vessel Path")
        box2.prop(sc, "vessel_curve_mode", expand=True)

        if sc.vessel_curve_mode == "NEW":
            row = box2.row(align=True)
            row.prop(sc, "vessel_new_curve_name", text="Base Name")
            row.operator("view3d.new_vessel_path", text="Refresh/New", icon="FILE_REFRESH")

            box2.prop(sc, "vessel_curve_object", text="Current")
            box2.label(text="Tip: If you delete a path, a new one will auto-create on next click.")

        else:
            box2.prop(sc, "vessel_target_curve", text="Target Path")
            box2.label(text="Tip: If empty, active curve used.")

        # Appearance (requested depth 0.04 default)
        box2.separator()
        box2.label(text="Appearance")
        colA = box2.column(align=True)
        colA.prop(sc, "vessel_path_depth")
        colA.prop(sc, "vessel_path_bevel_resolution")
        colA.prop(sc, "vessel_path_fill_caps")
        colA.prop(sc, "vessel_path_color")
        colA.operator("view3d.apply_path_appearance", icon="MATERIAL")

        # NURBS params
        box2.separator()
        box2.label(text="NURBS")
        colN = box2.column(align=True)
        colN.prop(sc, "vessel_nurbs_order")
        colN.prop(sc, "vessel_curve_resolution")

        row = box2.row(align=True)
        if sc.vessel_place_active:
            row.operator("view3d.stop_place", icon="CANCEL")
        else:
            row.operator("view3d.click_add_path_points", icon="CURVE_PATH")


# =========================================================
# Props
# =========================================================
def ensure_props():
    S = bpy.types.Scene

    # CT
    if not hasattr(S, "image_slider_axis"):
        S.image_slider_axis = bpy.props.EnumProperty(
            name="Axis",
            items=[('AX', 'Axial', ''), ('COR', 'Coronal', ''), ('SAG', 'Sagittal', '')],
            default='AX',
            update=update_axis
        )

    if not hasattr(S, "image_slider_property"):
        S.image_slider_property = bpy.props.IntProperty(
            name="Slide",
            min=1, max=DEFAULT_AX_COUNT, default=1,
            update=update_slider
        )

    if not hasattr(S, "ax_slice_count"):
        S.ax_slice_count = bpy.props.IntProperty(name="AX Slices (Z)", min=1, max=10000, default=DEFAULT_AX_COUNT)
    if not hasattr(S, "slice_spacing_ax"):
        S.slice_spacing_ax = bpy.props.FloatProperty(name="AX Slice Spacing", min=0.0, soft_max=100.0, default=DEFAULT_AX_SPACING)
    if not hasattr(S, "pixel_pitch"):
        S.pixel_pitch = bpy.props.FloatProperty(name="Pixel Pitch", min=0.0, soft_max=10.0, default=DEFAULT_PITCH)
    if not hasattr(S, "pixel"):
        S.pixel = bpy.props.IntProperty(name="Image Pixel (XY)", min=1, max=16384, default=DEFAULT_PIXEL)
    if not hasattr(S, "folder_path_png"):
        S.folder_path_png = bpy.props.StringProperty(name="Parent Folder", subtype='DIR_PATH', default=DEFAULT_FOLDER)

    if not hasattr(S, "ct_sync_all_axes"):
        S.ct_sync_all_axes = bpy.props.BoolProperty(
            name="Sync All Axes Planes",
            description="If ON, keep AX/COR/SAG click planes synced whenever you move the slider (if their CT empties exist).",
            default=True
        )
    if not hasattr(S, "ct_offset"):
        S.ct_offset = bpy.props.FloatVectorProperty(
            name="CT Offset",
            size=3,
            default=(0.0, 0.0, 0.0),
            subtype='TRANSLATION',
        )

    # Vessel
    if not hasattr(S, "vessel_place_active"):
        S.vessel_place_active = bpy.props.BoolProperty(default=False, options={'HIDDEN'})

    if not hasattr(S, "vessel_curve_mode"):
        S.vessel_curve_mode = bpy.props.EnumProperty(
            name="Mode",
            items=[("NEW", "NEW PATH", ""), ("APPEND", "APPEND PATH", "")],
            default="NEW",
        )
    if not hasattr(S, "vessel_new_curve_name"):
        S.vessel_new_curve_name = bpy.props.StringProperty(name="New Path Base Name", default="VesselPath")
    if not hasattr(S, "vessel_curve_object"):
        S.vessel_curve_object = bpy.props.PointerProperty(name="Current Path", type=bpy.types.Object)
    if not hasattr(S, "vessel_target_curve"):
        S.vessel_target_curve = bpy.props.PointerProperty(name="Target Path", type=bpy.types.Object)

    if not hasattr(S, "vessel_nurbs_order"):
        S.vessel_nurbs_order = bpy.props.IntProperty(name="NURBS Order", default=4, min=2, max=6)
    if not hasattr(S, "vessel_curve_resolution"):
        S.vessel_curve_resolution = bpy.props.IntProperty(name="Display Resolution", default=24, min=1, max=128)

    # Appearance (Depth default 0.04 requested)
    if not hasattr(S, "vessel_path_depth"):
        S.vessel_path_depth = bpy.props.FloatProperty(
            name="Geometry Depth",
            description="Curve bevel depth (tube thickness).",
            default=0.04, min=0.0, soft_max=1.0, precision=4, subtype='DISTANCE'
        )
    if not hasattr(S, "vessel_path_bevel_resolution"):
        S.vessel_path_bevel_resolution = bpy.props.IntProperty(name="Bevel Resolution", default=3, min=0, max=12)
    if not hasattr(S, "vessel_path_fill_caps"):
        S.vessel_path_fill_caps = bpy.props.BoolProperty(name="Fill Caps", default=True)
    if not hasattr(S, "vessel_path_color"):
        S.vessel_path_color = bpy.props.EnumProperty(
            name="Path Color",
            items=[
                ("CYAN", "Cyan", ""), ("RED", "Red", ""), ("GREEN", "Green", ""),
                ("BLUE", "Blue", ""), ("YELLOW", "Yellow", ""), ("ORANGE", "Orange", ""),
                ("PURPLE", "Purple", ""), ("WHITE", "White", ""),
            ],
            default="CYAN"
        )


def cleanup_props():
    S = bpy.types.Scene
    attrs = (
        "image_slider_axis", "image_slider_property", "ax_slice_count", "slice_spacing_ax",
        "pixel_pitch", "pixel", "folder_path_png", "ct_sync_all_axes",
        "ct_offset",
        "vessel_place_active", "vessel_curve_mode", "vessel_new_curve_name",
        "vessel_curve_object", "vessel_target_curve",
        "vessel_nurbs_order", "vessel_curve_resolution",
        "vessel_path_depth", "vessel_path_bevel_resolution", "vessel_path_fill_caps", "vessel_path_color",
    )
    for a in attrs:
        if hasattr(S, a):
            delattr(S, a)


# =========================================================
# Register
# =========================================================
classes = (
    ImageSliderOperator,
    VIEW3D_OT_new_vessel_path,
    VIEW3D_OT_apply_path_appearance,
    VIEW3D_OT_click_add_path_points,
    VIEW3D_OT_stop_place,
    VIEW3D_PT_ct_vessel_unified,
)


def register():
    ensure_props()
    for c in classes:
        bpy.utils.register_class(c)
    try:
        update_axis(None, bpy.context)
    except Exception:
        pass


def unregister():
    try:
        bpy.context.scene.vessel_place_active = False
    except Exception:
        pass

    for c in reversed(classes):
        bpy.utils.unregister_class(c)
    cleanup_props()


if __name__ == "__main__":
    register()
