import bpy
import os
import sys
from pathlib import Path

# Default parameters
DEFAULT_PIXEL = 512          # Image dimension (used for COR/SAG slider max)
DEFAULT_AX_COUNT = 419       # Number of axial (Z-direction) slices
DEFAULT_AX_SPACING = 2.0     # Spacing between axial slices (relative units)
DEFAULT_PITCH = 0.683        # In-plane pixel pitch (e.g., mm/pixel)
DEFAULT_SIZE = 0.02          # Global scaling factor - MATCHES YOUR SCRIPT NOW
DEFAULT_FOLDER = ''          # Contains axial/coronal/sagittal subfolders

# Center offset - where are your organs actually located?
DEFAULT_CENTER_X = 0.0
DEFAULT_CENTER_Y = 0.0  # Your script uses 7.5 but blender import centers at 0
DEFAULT_CENTER_Z = 0.0

def compute_scales(ctx):
    """Recalculate and return size (scaling factors)"""
    ax_count = ctx.scene.ax_slide_count
    ax_spacing = ctx.scene.slice_spacing_ax
    pixel_pitch = ctx.scene.pixel_pitch
    pixel = ctx.scene.pixel
    size = ctx.scene.base_size

    cc_scale = ax_count * ax_spacing * size
    ax_scale = size * pixel_pitch * pixel
    common_scale = cc_scale if (pixel_pitch * pixel) < (ax_count * ax_spacing) else ax_scale
    return ax_scale, common_scale

def load_reference_image(image_path, object_name):
    """Load an image as an Empty (Image) and assign the specified name"""
    try:
        abs_path = os.path.abspath(image_path)
        
        # Check if already loaded
        img = None
        for existing in bpy.data.images:
            if existing.filepath == abs_path:
                img = existing
                break
                
        if not img:
            img = bpy.data.images.load(abs_path)
            
        # Add an empty object
        bpy.ops.object.empty_add(type='IMAGE', location=(0, 0, 0))
        empty_obj = bpy.context.object
        empty_obj.data = img
        empty_obj.name = object_name
        return empty_obj
        
    except Exception as e:
        print(f"Failed to load image: {image_path}, Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def find_image_files(folder_path, axis_type):
    """Find image files in the specified directory for the given axis"""
    if not folder_path or not os.path.exists(folder_path):
        print(f"Invalid folder path: {folder_path}")
        return []
        
    print(f"Searching for {axis_type} images in {folder_path}")
    
    if axis_type == 'AX':
        subdir = 'axial'
    elif axis_type == 'COR':
        subdir = 'coronal'
    else:  # SAG
        subdir = 'sagittal'
    
    paths_to_check = [
        os.path.join(folder_path, subdir),
        os.path.join(folder_path, subdir.upper()),
        folder_path
    ]
    
    image_dir = None
    for path in paths_to_check:
        if os.path.isdir(path):
            print(f"Found valid directory: {path}")
            image_dir = path
            break
    
    if not image_dir:
        print(f"Could not find image directory for {axis_type} slices")
        return []
    
    extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
    
    # Find numbered files like "001.png"
    for ext in extensions:
        pattern_files = sorted([f for f in os.listdir(image_dir) 
                              if f.lower().endswith(ext) and f[:-len(ext)].isdigit()])
        if pattern_files:
            files = [os.path.join(image_dir, f) for f in pattern_files]
            print(f"Found {len(files)} numbered image files")
            return files
    
    print(f"No image files found in {image_dir}")
    return []

class ImageSliderOperator(bpy.types.Operator):
    bl_idname = "object.image_slider_operator"
    bl_label = "Image Slider Operator"

    def execute(self, context):
        print("\n" + "="*70)
        print("🎬 EXECUTING ImageSliderOperator")
        print("="*70)
        
        axis_type = context.scene.image_slider_axis
        slider_value = context.scene.image_slider_property
        folder_path = context.scene.folder_path_dicom
        
        print(f"Axis: {axis_type}, Slider: {slider_value}, Folder: {folder_path}")
        
        # Convert to absolute path
        if folder_path:
            folder_path = os.path.abspath(bpy.path.abspath(folder_path))
            print(f"🔍 Resolved folder path: {folder_path}")
            
            # CHECK IF PATH EXISTS
            if not os.path.exists(folder_path):
                error_msg = f"Folder does not exist: {folder_path}"
                print(f"❌ {error_msg}")
                self.report({'ERROR'}, error_msg)
                return {'CANCELLED'}
        else:
            error_msg = "No image folder specified"
            print(f"❌ {error_msg}")
            self.report({'ERROR'}, error_msg)
            return {'CANCELLED'}
        
        pixel_pitch = context.scene.pixel_pitch
        pixel = context.scene.pixel_dim
        ax_slides = context.scene.ax_slide_count
        ax_distance = context.scene.ax_slide_spacing

        ax_scale, common_scale = compute_scales(context)
        print(f"Computed scales - ax_scale: {ax_scale:.4f}, common_scale: {common_scale:.4f}")
        
        # Find image files
        images = find_image_files(folder_path, axis_type)
        
        print(f"Found {len(images)} images")
        
        if not images:
            error_msg = f"No images found for {axis_type} axis in {folder_path}"
            print(f"❌ {error_msg}")
            
            # List what's actually in the folder
            try:
                print(f"📁 Contents of {folder_path}:")
                for item in os.listdir(folder_path):
                    item_path = os.path.join(folder_path, item)
                    item_type = "DIR" if os.path.isdir(item_path) else "FILE"
                    print(f"   [{item_type}] {item}")
            except Exception as e:
                print(f"   Could not list directory: {e}")
            
            self.report({'ERROR'}, error_msg)
            return {'CANCELLED'}
            
        if slider_value < 1 or slider_value > len(images):
            slider_value = max(1, min(slider_value, len(images)))
            print(f"⚠️ Adjusted slider value to: {slider_value}")
        
        image_path = images[slider_value - 1]
        
        print(f"📸 Loading image: {image_path}")
        
        # CHECK IF IMAGE FILE EXISTS
        if not os.path.exists(image_path):
            error_msg = f"Image file not found: {image_path}"
            print(f"❌ {error_msg}")
            self.report({'ERROR'}, error_msg)
            return {'CANCELLED'}

        # Get center offset
        center_x = context.scene.center_offset_x
        center_y = context.scene.center_offset_y
        center_z = context.scene.center_offset_z
        
        print(f"Center offset: ({center_x}, {center_y}, {center_z})")

        if axis_type == 'AX':      # Axial (slides along Y-axis)
            prefix = 'iAx'

            # Remove existing axial image objects
            removed_count = 0
            for obj in list(bpy.context.scene.objects):
                if obj.name.startswith(prefix):
                    bpy.data.objects.remove(obj, do_unlink=True)
                    removed_count += 1
            if removed_count > 0:
                print(f"🗑️ Removed {removed_count} existing {prefix} objects")

            image_obj = load_reference_image(image_path, f"{prefix}{slider_value:03d}")
            if not image_obj:
                print("❌ Failed to load reference image")
                return {'CANCELLED'}

            # Rotation (preserved from original)
            image_obj.rotation_euler = (1.5708, 0, 0)

            # Translate along Y-axis (equivalent to medical Z) WITH CENTER OFFSET
            y_pos = center_y - (ax_distance / 50) * ((ax_slides + 1) / 2 - slider_value)
            image_obj.location = (center_x, y_pos, center_z)

            # Scale
            image_obj.scale = (ax_scale, ax_scale, ax_scale)
            print(f"✅ Axial slice created at Y={y_pos:.4f}, scale={ax_scale:.4f}")

        elif axis_type == 'COR':   # Coronal (slides along Z-axis)
            prefix = 'iCor'

            removed_count = 0
            for obj in list(bpy.context.scene.objects):
                if obj.name.startswith(prefix):
                    bpy.data.objects.remove(obj, do_unlink=True)
                    removed_count += 1
            if removed_count > 0:
                print(f"🗑️ Removed {removed_count} existing {prefix} objects")

            image_obj = load_reference_image(image_path, f"{prefix}{slider_value:03d}")
            if not image_obj:
                print("❌ Failed to load reference image")
                return {'CANCELLED'}

            image_obj.rotation_euler = (0, 3.14159, 3.14159)
            z_pos = center_z + (pixel_pitch / 50) * (pixel / 2 + 0.5 - slider_value)
            image_obj.location = (center_x, center_y, z_pos)
            image_obj.scale = (common_scale, common_scale, common_scale)
            print(f"✅ Coronal slice created at Z={z_pos:.4f}, scale={common_scale:.4f}")

        elif axis_type == 'SAG':   # Sagittal (slides along X-axis)
            prefix = 'iSag'

            removed_count = 0
            for obj in list(bpy.context.scene.objects):
                if obj.name.startswith(prefix):
                    bpy.data.objects.remove(obj, do_unlink=True)
                    removed_count += 1
            if removed_count > 0:
                print(f"🗑️ Removed {removed_count} existing {prefix} objects")

            image_obj = load_reference_image(image_path, f"{prefix}{slider_value:03d}")
            if not image_obj:
                print("❌ Failed to load reference image")
                return {'CANCELLED'}

            image_obj.rotation_euler = (0, 1.5708, 3.14159)
            x_pos = center_x - (pixel_pitch / 50) * (pixel / 2 + 0.5 - slider_value)
            image_obj.location = (x_pos, center_y, center_z)
            image_obj.scale = (common_scale, common_scale, common_scale)
            print(f"✅ Sagittal slice created at X={x_pos:.4f}, scale={common_scale:.4f}")
        
        # Update slider max
        max_slices = len(images)
        if context.scene.slider_max != max_slices:
            context.scene.slider_max = max_slices
            print(f"Updated slider max to: {max_slices}")

        print("="*70)
        print("✅ ImageSliderOperator FINISHED")
        print("="*70 + "\n")
        
        return {'FINISHED'}

class ImageSliderPanel(bpy.types.Panel):
    bl_label = "DICOM Slices"
    bl_idname = "OBJECT_PT_dicom_slider"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'DICOM'

    def draw(self, context):
        layout = self.layout
        
        # Folder path
        box = layout.box()
        box.label(text="Image Folder:", icon='FILE_FOLDER')
        box.prop(context.scene, 'folder_path_dicom', text="")
        
        # Show resolved absolute path
        if context.scene.folder_path_dicom:
            abs_path = os.path.abspath(bpy.path.abspath(context.scene.folder_path_dicom))
            box.label(text=f"...{abs_path[-50:]}", icon='INFO')
        
        layout.separator()
        
        # Axis selection
        row = layout.row()
        row.prop(context.scene, 'image_slider_axis', expand=True)
        
        # Slider
        layout.prop(context.scene, 'image_slider_property', slider=True)
        
        layout.separator()
        
        # Parameters
        box = layout.box()
        box.label(text="Parameters", icon='PREFERENCES')
        col = box.column(align=True)
        col.prop(context.scene, 'ax_slide_count')
        col.prop(context.scene, 'ax_slide_spacing')
        col.prop(context.scene, 'pixel_pitch')
        col.prop(context.scene, 'pixel_dim')
        col.prop(context.scene, 'base_size')
        
        # ADD CENTER OFFSET CONTROLS
        box2 = layout.box()
        box2.label(text="Center Offset", icon='OBJECT_ORIGIN')
        col2 = box2.column(align=True)
        col2.prop(context.scene, 'center_offset_x')
        col2.prop(context.scene, 'center_offset_y')
        col2.prop(context.scene, 'center_offset_z')
        
        layout.separator()
        layout.operator("object.image_slider_operator", text="Reload Slice", icon='FILE_REFRESH')

def update_axis(self, context):
    """Update the slider's maximum value when the axis or parameters change"""
    try:
        folder_path = context.scene.folder_path_dicom
        if folder_path:
            folder_path = os.path.abspath(bpy.path.abspath(folder_path))
        
        images = find_image_files(folder_path, context.scene.image_slider_axis)
        
        if context.scene.image_slider_axis == 'AX':
            maxv = len(images) if images else max(1, context.scene.ax_slide_count)
        else:
            maxv = len(images) if images else max(1, context.scene.pixel_dim)

        context.scene.slider_max = maxv
        
        # Re-declare the property to update its maximum value
        bpy.types.Scene.image_slider_property = bpy.props.IntProperty(
            name="Slice",
            min=1,
            max=maxv,
            default=min(getattr(context.scene, 'image_slider_property', 1), maxv),
            update=update_slider
        )
        
        # Load initial slice
        bpy.ops.object.image_slider_operator()
    except Exception as e:
        print(f"Error updating axis: {e}")
    
    return None

def update_slider(self, context):
    """Update image when slider changes"""
    try:
        print(f"🎚️ Slider changed to: {context.scene.image_slider_property}")
        # Make sure we're not in the middle of another operation
        if context.scene.is_property_set("image_slider_property"):
            result = bpy.ops.object.image_slider_operator()
            print(f"   Operator result: {result}")
    except Exception as e:
        print(f"❌ Error in update_slider: {e}")
        import traceback
        traceback.print_exc()
    return None

def register_properties():
    """Register all required properties"""
    try:
        # Find initial maximum values for sliders
        dicom_dir = ""
        if "--" in sys.argv:
            try:
                argv = sys.argv[sys.argv.index("--") + 1:]
                for i, arg in enumerate(argv):
                    if arg == "--image-dir" and i + 1 < len(argv):
                        dicom_dir = os.path.abspath(argv[i + 1])
                        print(f"Using image directory from command line: {dicom_dir}")
            except Exception as e:
                print(f"Error processing command line args: {e}")
        
        # Count files in axial directory if it exists
        max_slices = DEFAULT_AX_COUNT
        if dicom_dir and os.path.exists(dicom_dir):
            axial_images = find_image_files(dicom_dir, 'AX')
            max_slices = len(axial_images) if axial_images else DEFAULT_AX_COUNT
        
        # Register the properties
        bpy.types.Scene.folder_path_dicom = bpy.props.StringProperty(
            name="Parent Folder",
            subtype='DIR_PATH',
            default=dicom_dir,
            description="Folder containing axial/coronal/sagittal subfolders"
        )
        
        bpy.types.Scene.image_slider_axis = bpy.props.EnumProperty(
            name="Axis",
            items=[
                ('AX', 'Axial', 'Axial slices'),
                ('COR', 'Coronal', 'Coronal slices'),
                ('SAG', 'Sagittal', 'Sagittal slices')
            ],
            default='AX',
            update=update_axis
        )
        
        bpy.types.Scene.image_slider_property = bpy.props.IntProperty(
            name="Slice", 
            min=1,
            max=max_slices,
            default=1,
            description="Select slice number",
            update=update_slider
        )
        
        bpy.types.Scene.slider_max = bpy.props.IntProperty(
            name="Max Slices",
            default=max_slices
        )
        
        # Parameters
        bpy.types.Scene.ax_slide_count = bpy.props.IntProperty(
            name="AX Slices (Z)",
            min=1,
            max=10000,
            default=max_slices,
            description="Number of axial slices (Z direction)"
        )
        
        bpy.types.Scene.ax_slide_spacing = bpy.props.FloatProperty(
            name="AX Slice Spacing",
            min=0.0,
            soft_max=100.0,
            default=DEFAULT_AX_SPACING,
            description="Spacing between axial slices (relative units)"
        )
        
        bpy.types.Scene.pixel_pitch = bpy.props.FloatProperty(
            name="Pixel Pitch",
            min=0.0,
            soft_max=10.0,
            default=DEFAULT_PITCH,
            description="In-plane pixel pitch (e.g., mm/pixel)"
        )
        
        bpy.types.Scene.pixel_dim = bpy.props.IntProperty(
            name="Image Pixel (XY)",
            min=1,
            max=16384,
            default=DEFAULT_PIXEL,
            description="Image resolution used for COR/SAG slider max"
        )
        
        bpy.types.Scene.base_size = bpy.props.FloatProperty(
            name="Base Scale",
            min=0.001,
            max=1.0,
            default=DEFAULT_SIZE,
            description="Global scaling factor"
        )
        
        # ADD CENTER OFFSET PROPERTIES
        bpy.types.Scene.center_offset_x = bpy.props.FloatProperty(
            name="Center X",
            default=DEFAULT_CENTER_X,
            description="X offset of organ center"
        )
        
        bpy.types.Scene.center_offset_y = bpy.props.FloatProperty(
            name="Center Y",
            default=DEFAULT_CENTER_Y,
            description="Y offset of organ center"
        )
        
        bpy.types.Scene.center_offset_z = bpy.props.FloatProperty(
            name="Center Z",
            default=DEFAULT_CENTER_Z,
            description="Z offset of organ center"
        )
        
    except Exception as e:
        print(f"Error registering properties: {e}")
        import traceback
        traceback.print_exc()

def register():
    """Register the add-on"""
    try:
        register_properties()
        
        bpy.utils.register_class(ImageSliderOperator)
        bpy.utils.register_class(ImageSliderPanel)
        
        print("✅ DICOM Slider add-on registered successfully")
        print(f"   Operator available: {hasattr(bpy.ops.object, 'image_slider_operator')}")
        
        # DON'T auto-load on startup - let user trigger it
        # This prevents errors if folder path isn't set yet
        print("⚠️ Note: Click 'Reload Slice' button to load first image")
        
    except Exception as e:
        print(f"Registration error: {e}")
        import traceback
        traceback.print_exc()

def unregister():
    """Unregister the add-on"""
    try:
        bpy.utils.unregister_class(ImageSliderOperator)
        bpy.utils.unregister_class(ImageSliderPanel)
        
        S = bpy.types.Scene
        for attr in ("folder_path_dicom", "image_slider_axis", "image_slider_property", 
                     "slider_max", "ax_slide_count", "ax_slide_spacing", 
                     "pixel_pitch", "pixel_dim", "base_size",
                     "center_offset_x", "center_offset_y", "center_offset_z"):
            if hasattr(S, attr):
                delattr(S, attr)
    except Exception as e:
        print(f"Error during unregister: {e}")

@bpy.app.handlers.persistent
def load_handler(dummy):
    try:
        register()
    except Exception as e:
        print(f"Error in load_handler: {e}")
    return None

# Register the handler only if not already registered
if load_handler not in bpy.app.handlers.load_post:
    bpy.app.handlers.load_post.append(load_handler)

if __name__ == "__main__":
    print("Starting DICOM Slider add-on")
    register()