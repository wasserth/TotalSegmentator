import bpy
import os
import sys
from pathlib import Path

# Default parameters (will be auto-calculated when possible)
DEFAULT_PIXEL_PITCH = 0.683  # mm/pixel
DEFAULT_PIXEL = 512          # Image dimension
DEFAULT_SIZE = 0.02          # Base scale factor
DEFAULT_AX_COUNT = 419       # Number of axial slices
DEFAULT_AX_SPACING = 2.0     # Spacing between axial slices

def calculate_organ_dimensions():
    """Calculate bounding box of all mesh objects in scene"""
    try:
        import mathutils
        mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
        
        if not mesh_objects:
            print("No mesh objects found for calibration")
            return (0, 0, 0), 0.1  # Default center and scale
        
        # Get bounds
        min_x = min_y = min_z = float('inf')
        max_x = max_y = max_z = float('-inf')
        
        for obj in mesh_objects:
            # Get world-space coordinates of bounding box
            for v in obj.bound_box:
                # Convert local coordinates to world space
                world_v = obj.matrix_world @ mathutils.Vector((v[0], v[1], v[2]))
                
                min_x = min(min_x, world_v.x)
                min_y = min(min_y, world_v.y)
                min_z = min(min_z, world_v.z)
                
                max_x = max(max_x, world_v.x)
                max_y = max(max_y, world_v.y)
                max_z = max(max_z, world_v.z)
        
        # Calculate center and dimensions
        center = ((min_x + max_x)/2, (min_y + max_y)/2, (min_z + max_z)/2)
        
        # Get the maximum dimension for scale
        x_size = max_x - min_x
        y_size = max_y - min_y
        z_size = max_z - min_z
        max_size = max(x_size, y_size, z_size)
        
        # Scale is half the maximum dimension
        scale = max_size / 2 if max_size > 0 else 0.1
        
        return center, scale
    except Exception as e:
        print(f"Error calculating organ dimensions: {e}")
        import traceback
        traceback.print_exc()
        return (0, 0, 0), 0.1

def load_reference_image(image_path, object_name):
    """Load an image as an empty object and assign the specified name"""
    try:
        # Make sure we have an absolute path
        abs_path = os.path.abspath(image_path)
        print(f"Loading image from: {abs_path}")
        
        # Check if already loaded
        img = None
        for existing in bpy.data.images:
            if existing.filepath == abs_path:
                img = existing
                break
                
        if not img:
            img = bpy.data.images.load(abs_path)
            
        # Add an empty object - FIXED FOR BLENDER 4.5
        # In Blender 4.5, use 'IMAGE' directly as the type
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
    
    # Check both specified subdir and potential alternative paths
    paths_to_check = [
        os.path.join(folder_path, subdir),         # Standard /axial folder
        os.path.join(folder_path, axis_type[0]),   # /A folder (abbreviation)
        os.path.join(folder_path, subdir.upper()), # /AXIAL folder (uppercase)
        os.path.join(folder_path, axis_type[0].upper()),  # /A folder (uppercase)
        folder_path  # Direct folder (no subdirectory)
    ]
    
    # Find the first valid directory
    image_dir = None
    for path in paths_to_check:
        if os.path.isdir(path):
            print(f"Found valid directory: {path}")
            image_dir = path
            break
    
    if not image_dir:
        print(f"Could not find image directory for {axis_type} slices")
        return []
    
    # Find all image files
    extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
    image_files = []
    
    # First check for numbered files like "001.jpg"
    for ext in extensions:
        pattern_files = sorted([f for f in os.listdir(image_dir) 
                              if f.lower().endswith(ext) and f[:-len(ext)].isdigit()])
        if pattern_files:
            files = [os.path.join(image_dir, f) for f in pattern_files]
            print(f"Found {len(files)} numbered image files")
            return files
    
    # Then check for prefixed files like "A1.jpg" 
    for ext in extensions:
        prefix = axis_type[0].upper()
        pattern_files = sorted([f for f in os.listdir(image_dir) 
                              if f.lower().endswith(ext) and f.startswith(prefix) and 
                              f[len(prefix):-len(ext)].isdigit()],
                              key=lambda x: int(x[len(prefix):-len(ext)]))
        if pattern_files:
            files = [os.path.join(image_dir, f) for f in pattern_files]
            print(f"Found {len(files)} prefixed image files")
            return files
    
    # Finally just get all image files
    all_files = []
    for ext in extensions:
        files = [f for f in os.listdir(image_dir) if f.lower().endswith(ext)]
        all_files.extend(files)
        
    if all_files:
        files = [os.path.join(image_dir, f) for f in sorted(all_files)]
        print(f"Found {len(files)} image files")
        return files
    
    print(f"No image files found in {image_dir}")
    return []

class ImageSliderOperator(bpy.types.Operator):
    bl_idname = "object.image_slider_operator"
    bl_label = "Image Slider Operator"

    def execute(self, context):
        # Get settings
        axis_type = context.scene.image_slider_axis
        slider_value = context.scene.image_slider_property
        folder_path = context.scene.folder_path_dicom
        
        # Calculate center position and scale from organ models
        try:
            # Make sure mathutils is available
            import mathutils
        except ImportError:
            self.report({'ERROR'}, "Mathutils module not available")
            return {'CANCELLED'}
            
        center, scale = calculate_organ_dimensions()
        
        # Get pixel dimensions from scene
        pixel_pitch = context.scene.pixel_pitch
        pixel = context.scene.pixel_dim
        ax_slide_number = context.scene.ax_slide_count
        ax_slide_distance = context.scene.ax_slide_spacing
        size = context.scene.base_size
        
        # Calculate scale factors
        cc_scale = ax_slide_number * ax_slide_distance * size
        ax_scale = size * pixel_pitch * pixel
        
        if pixel_pitch * pixel < ax_slide_number * ax_slide_distance:
            common_scale = cc_scale
        else:
            common_scale = ax_scale
            
        # Find image files
        images = find_image_files(folder_path, axis_type)
        
        if not images:
            self.report({'ERROR'}, f"No images found for {axis_type} axis in {folder_path}")
            return {'CANCELLED'}
            
        # Make sure slider_value is in valid range
        if slider_value < 1 or slider_value > len(images):
            slider_value = max(1, min(slider_value, len(images)))
        
        # Get image path
        image_path = images[slider_value - 1]
        
        if axis_type == 'AX':  # Axial
            prefix = "iA"
            
            # Remove existing image objects
            for obj in bpy.context.scene.objects:
                if obj.name.startswith(prefix):
                    bpy.data.objects.remove(obj, do_unlink=True)
                    
            # Load the image
            image_obj = load_reference_image(image_path, f"{prefix}{slider_value:03d}")
            if not image_obj:
                return {'CANCELLED'}
                
            # Set orientation and position
            image_obj.rotation_euler = (1.5708, 0, 0)  # 90 deg on X
            
            # Center on organs with appropriate offset
            fraction = (slider_value - 1) / max(1, len(images) - 1)  # 0 to 1
            slice_range = scale * 2  # Full range of the model
            slice_offset = (fraction - 0.5) * slice_range  # Center offset (-0.5 to 0.5)
            
            # Position image - for axial, move along Y axis
            image_obj.location = (center[0], center[1] + slice_offset, center[2])
            
            # Set scale
            if context.scene.use_auto_scale:
                image_obj.scale = (scale, scale, scale)
            else:
                image_obj.scale = (ax_scale, ax_scale, ax_scale)
                
        elif axis_type == 'COR':  # Coronal
            prefix = "iC"
            
            # Remove existing image objects
            for obj in bpy.context.scene.objects:
                if obj.name.startswith(prefix):
                    bpy.data.objects.remove(obj, do_unlink=True)
                    
            # Load the image
            image_obj = load_reference_image(image_path, f"{prefix}{slider_value:03d}")
            if not image_obj:
                return {'CANCELLED'}
                
            # Set orientation and position
            image_obj.rotation_euler = (0, 3.14159, 3.14159)  # 180 on Y and Z
            
            # Center on organs with appropriate offset
            fraction = (slider_value - 1) / max(1, len(images) - 1)
            slice_range = scale * 2
            slice_offset = (fraction - 0.5) * slice_range
            
            # Position image - for coronal, move along Z axis
            image_obj.location = (center[0], center[1], center[2] + slice_offset)
            
            # Set scale
            if context.scene.use_auto_scale:
                image_obj.scale = (scale, scale, scale)
            else:
                image_obj.scale = (common_scale, common_scale, common_scale)
                
        elif axis_type == 'SAG':  # Sagittal
            prefix = "iS"
            
            # Remove existing image objects
            for obj in bpy.context.scene.objects:
                if obj.name.startswith(prefix):
                    bpy.data.objects.remove(obj, do_unlink=True)
                    
            # Load the image
            image_obj = load_reference_image(image_path, f"{prefix}{slider_value:03d}")
            if not image_obj:
                return {'CANCELLED'}
                
            # Set orientation and position
            image_obj.rotation_euler = (0, 1.5708, 3.14159)  # 90 on Y, 180 on Z
            
            # Center on organs with appropriate offset
            fraction = (slider_value - 1) / max(1, len(images) - 1)
            slice_range = scale * 2
            slice_offset = (fraction - 0.5) * slice_range
            
            # Position image - for sagittal, move along X axis
            image_obj.location = (center[0] + slice_offset, center[1], center[2])
            
            # Set scale
            if context.scene.use_auto_scale:
                image_obj.scale = (scale, scale, scale)
            else:
                image_obj.scale = (common_scale, common_scale, common_scale)
        
        # Update slider max value if needed
        max_slices = len(images)
        if context.scene.slider_max != max_slices:
            context.scene.slider_max = max_slices
            # Update the property definition
            bpy.types.Scene.image_slider_property = bpy.props.IntProperty(
                name="Slice",
                min=1,
                max=max_slices,
                default=min(slider_value, max_slices),
                update=update_slider
            )
        
        return {'FINISHED'}

class ImageSliderPanel(bpy.types.Panel):
    bl_label = "DICOM Slices"
    bl_idname = "OBJECT_PT_dicom_slider"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'DICOM'

    def draw(self, context):
        layout = self.layout
        
        # Folder selection
        layout.prop(context.scene, 'folder_path_dicom')
        
        # Axis selection with radio buttons
        row = layout.row()
        row.prop(context.scene, 'image_slider_axis', expand=True)
        
        # Slice slider
        layout.prop(context.scene, 'image_slider_property', slider=True)
        
        # Auto-scale option
        layout.prop(context.scene, 'use_auto_scale')
        
        # Parameters section
        box = layout.box()
        box.label(text="Advanced Parameters")
        box.prop(context.scene, 'pixel_pitch')
        box.prop(context.scene, 'pixel_dim')
        box.prop(context.scene, 'ax_slide_count')
        box.prop(context.scene, 'ax_slide_spacing')
        box.prop(context.scene, 'base_size')
        
        # Information about detected organs
        center, scale = calculate_organ_dimensions()
        box = layout.box()
        box.label(text="Detected Organ Dimensions")
        box.label(text=f"Center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
        box.label(text=f"Scale: {scale:.3f}")
        
        # Load button
        layout.operator("object.image_slider_operator", text="Reload Slice")
        
def update_axis(self, context):
    """Update slider properties when axis changes"""
    try:
        # Find images for the new axis
        images = find_image_files(context.scene.folder_path_dicom, context.scene.image_slider_axis)
        max_slices = len(images) if images else 1
        
        # Update the max slider value
        context.scene.slider_max = max_slices
        
        # Update the property definition
        bpy.types.Scene.image_slider_property = bpy.props.IntProperty(
            name="Slice",
            min=1,
            max=max_slices,
            default=min(context.scene.image_slider_property, max_slices),
            update=update_slider
        )
        
        # Load the first image for the new axis
        context.scene.image_slider_property = 1
        bpy.ops.object.image_slider_operator()
    except Exception as e:
        print(f"Error updating axis: {e}")
    
    return None

def update_slider(self, context):
    """Update image when slider changes"""
    bpy.ops.object.image_slider_operator()
    return None

def register_properties():
    """Register all required properties"""
    try:
        import mathutils
        
        # Try to detect organ dimensions
        center, scale = calculate_organ_dimensions()
        
        # Find initial maximum values for sliders
        dicom_dir = ""
        # Try to get from command line if available
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
        max_slices = 1
        if dicom_dir and os.path.exists(dicom_dir):
            axial_images = find_image_files(dicom_dir, 'AX')
            max_slices = len(axial_images) if axial_images else DEFAULT_AX_COUNT
        
        # Register the properties
        bpy.types.Scene.folder_path_dicom = bpy.props.StringProperty(
            name="Image Folder",
            subtype='DIR_PATH',
            default=dicom_dir
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
        
        bpy.types.Scene.use_auto_scale = bpy.props.BoolProperty(
            name="Auto-Scale to Organs",
            default=True,
            description="Automatically scale slices to match organ dimensions",
            update=update_slider
        )
        
        # Advanced parameters
        bpy.types.Scene.pixel_pitch = bpy.props.FloatProperty(
            name="Pixel Pitch (mm)",
            min=0.01,
            max=10.0,
            default=DEFAULT_PIXEL_PITCH,
            description="Size of each pixel in mm",
            update=update_slider
        )
        
        bpy.types.Scene.pixel_dim = bpy.props.IntProperty(
            name="Image Width (px)",
            min=1,
            max=4096,
            default=DEFAULT_PIXEL,
            description="Width of the image in pixels",
            update=update_slider
        )
        
        bpy.types.Scene.ax_slide_count = bpy.props.IntProperty(
            name="Axial Slices",
            min=1,
            max=2000,
            default=max_slices,
            description="Number of axial slices",
            update=update_slider
        )
        
        bpy.types.Scene.ax_slide_spacing = bpy.props.FloatProperty(
            name="Slice Spacing",
            min=0.1,
            max=10.0,
            default=DEFAULT_AX_SPACING,
            description="Distance between slices",
            update=update_slider
        )
        
        bpy.types.Scene.base_size = bpy.props.FloatProperty(
            name="Base Scale",
            min=0.001,
            max=1.0,
            default=DEFAULT_SIZE,
            description="Base scaling factor",
            update=update_slider
        )
    except Exception as e:
        print(f"Error registering properties: {e}")
        import traceback
        traceback.print_exc()

def register():
    """Register the add-on"""
    try:
        # Make sure mathutils is imported (needed for calculations)
        import mathutils
        globals()['mathutils'] = mathutils
        
        # Register properties
        register_properties()
        
        # Register classes
        bpy.utils.register_class(ImageSliderOperator)
        bpy.utils.register_class(ImageSliderPanel)
        
        print("DICOM Slider add-on registered successfully")
        
        # Load first slice - delay this to avoid issues during initial registration
        def delayed_load():
            try:
                bpy.ops.object.image_slider_operator()
            except Exception as e:
                print(f"Error loading initial slice: {e}")
                import traceback
                traceback.print_exc()
        bpy.app.timers.register(delayed_load, first_interval=1.0)
        
    except Exception as e:
        print(f"Registration error: {e}")
        import traceback
        traceback.print_exc()

def unregister():
    """Unregister the add-on"""
    try:
        # Unregister classes
        bpy.utils.unregister_class(ImageSliderOperator)
        bpy.utils.unregister_class(ImageSliderPanel)
        
        # Remove properties
        del bpy.types.Scene.folder_path_dicom
        del bpy.types.Scene.image_slider_axis
        del bpy.types.Scene.image_slider_property
        del bpy.types.Scene.slider_max
        del bpy.types.Scene.use_auto_scale
        del bpy.types.Scene.pixel_pitch
        del bpy.types.Scene.pixel_dim
        del bpy.types.Scene.ax_slide_count
        del bpy.types.Scene.ax_slide_spacing
        del bpy.types.Scene.base_size
    except Exception as e:
        print(f"Error during unregister: {e}")

# For automatic registration on file load
@bpy.app.handlers.persistent
def load_handler(dummy):
    register()
    return None

# Add the handler to load_post if not already there
if load_handler not in bpy.app.handlers.load_post:
    bpy.app.handlers.load_post.append(load_handler)

# Run add-on when executed
if __name__ == "__main__":
    print("Starting DICOM Slider add-on")
    
    # Check if script is running standalone or as an add-on
    if __file__ and bpy.context is not None:
        # Get image directory from command line arguments if available
        if "--" in sys.argv:
            try:
                argv = sys.argv[sys.argv.index("--") + 1:]
                for i, arg in enumerate(argv):
                    if arg == "--image-dir" and i + 1 < len(argv):
                        image_dir = os.path.abspath(argv[i + 1])
                        print(f"Command line image directory: {image_dir}")
            except Exception as e:
                print(f"Error processing command line arguments: {e}")
        
        register()