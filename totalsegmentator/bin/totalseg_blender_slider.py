#!/usr/bin/env python3
"""
totalseg_blender_slider.py

Installs the DICOM slider addon into a Blender file and configures it
with the correct image paths and scaling parameters.

Usage:
    blender scene.blend -P totalseg_blender_slider.py -- \
        --png-dir /path/to/slices --nifti-path /path/to/scan.nii.gz \
        --scale 0.01 --save output.blend
"""

import bpy
import sys
import os
from pathlib import Path
import shutil

def parse_args():
    """Parse command line arguments after '--'"""
    if "--" not in sys.argv:
        return {}
    
    argv = sys.argv[sys.argv.index("--") + 1:]
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

def install_addon_to_blend():
    """Copy the addon script into the Blender file's text blocks"""
    addon_source = Path(__file__).parent.parent.parent / "dicom_slider_addon.py"
    
    if not addon_source.exists():
        print(f"❌ ERROR: Addon source not found: {addon_source}")
        return False
    
    print(f"📦 Installing addon from: {addon_source}")
    
    # Remove old version if exists
    for txt in bpy.data.texts:
        if txt.name == "dicom_slider_addon.py":
            bpy.data.texts.remove(txt)
    
    # Load addon into text block
    with open(addon_source, 'r') as f:
        addon_code = f.read()
    
    txt = bpy.data.texts.new("dicom_slider_addon.py")
    txt.from_string(addon_code)
    txt.use_module = True  # Enable as Python module
    
    print("✅ Addon installed into .blend file")
    return True

def setup_startup_script(png_dir, nifti_path, scale):
    """Create a startup script that registers the addon on file open"""
    # Ensure paths are absolute
    png_dir = str(Path(png_dir).resolve())
    nifti_path = str(Path(nifti_path).resolve()) if nifti_path else ""
    
    startup_code = f'''
import bpy
import os

# Configuration from pipeline (ABSOLUTE PATHS)
PNG_DIR = r"{png_dir}"
NIFTI_PATH = r"{nifti_path}"
SCALE = {scale}

print(f"📁 Configured paths:")
print(f"   PNG_DIR: {{PNG_DIR}}")
print(f"   NIFTI_PATH: {{NIFTI_PATH}}")
print(f"   SCALE: {{SCALE}}")

# Auto-register the DICOM slider addon when file is opened
def register_dicom_slider():
    try:
        addon_text = bpy.data.texts.get("dicom_slider_addon.py")
        if addon_text:
            # Execute the addon code
            exec(compile(addon_text.as_string(), "dicom_slider_addon.py", 'exec'))
            
            # Configure it with ABSOLUTE paths
            if hasattr(bpy.context.scene, 'folder_path_dicom'):
                bpy.context.scene.folder_path_dicom = PNG_DIR
                print(f"✓ Set folder_path_dicom to: {{PNG_DIR}}")
            
            if hasattr(bpy.context.scene, 'base_size'):
                bpy.context.scene.base_size = SCALE
                print(f"✓ Set base_size to: {{SCALE}}")
            
            if hasattr(bpy.context.scene, 'use_auto_scale'):
                bpy.context.scene.use_auto_scale = True
                print(f"✓ Enabled auto-scale")
            
            print("✅ DICOM Slider addon loaded and configured")
    except Exception as e:
        print(f"❌ Error loading DICOM Slider addon: {{e}}")
        import traceback
        traceback.print_exc()

# Register handler
@bpy.app.handlers.persistent
def load_post_handler(dummy):
    register_dicom_slider()

if load_post_handler not in bpy.app.handlers.load_post:
    bpy.app.handlers.load_post.append(load_post_handler)

# Also register now
register_dicom_slider()
'''
    
    # Remove old startup script if exists
    for txt in bpy.data.texts:
        if txt.name == "startup_dicom.py":
            bpy.data.texts.remove(txt)
    
    txt = bpy.data.texts.new("startup_dicom.py")
    txt.from_string(startup_code)
    txt.use_module = True
    
    print("✅ Startup script created")
    return True

def setup_addon_parameters(png_dir, nifti_path, scale):
    """Configure the addon with correct paths and scale"""
    try:
        # Execute the startup script which will register and configure the addon
        startup_text = bpy.data.texts.get("startup_dicom.py")
        if startup_text:
            exec(compile(startup_text.as_string(), "startup_dicom.py", 'exec'))
            print("✅ Addon configured successfully")
            return True
        else:
            print("❌ ERROR: Startup script not found")
            return False
        
    except Exception as e:
        print(f"❌ ERROR configuring addon: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    args = parse_args()
    
    print("=" * 70)
    print("🎬 DICOM Slice Viewer Setup")
    print("=" * 70)
    
    png_dir = args.get('png-dir', '')
    nifti_path = args.get('nifti-path', '')
    scale = float(args.get('scale', 0.01))
    save_path = args.get('save', '')
    
    if not png_dir:
        print("❌ ERROR: --png-dir is required")
        sys.exit(1)
    
    png_dir = Path(png_dir).resolve()
    if not png_dir.exists():
        print(f"❌ ERROR: PNG directory not found: {png_dir}")
        sys.exit(1)
    
    # Install addon
    if not install_addon_to_blend():
        sys.exit(1)
    
    # Setup startup script
    if not setup_startup_script(str(png_dir), str(nifti_path), scale):
        sys.exit(1)
    
    # Configure addon (this will also execute it)
    if not setup_addon_parameters(png_dir, nifti_path, scale):
        print("⚠️ Warning: Addon configuration had issues, but continuing...")
    
    # Save file
    if save_path:
        save_path = Path(save_path).resolve()
        bpy.ops.wm.save_as_mainfile(filepath=str(save_path))
        print(f"💾 Saved to: {save_path}")
    else:
        bpy.ops.wm.save_mainfile()
        print(f"💾 Saved current file")
    
    print("=" * 70)
    print("✅ DICOM Slice Viewer installed successfully!")
    print("=" * 70)
    print("\nUsage:")
    print("  1. Open the .blend file in Blender")
    print("  2. Look for 'DICOM' tab in the right sidebar (press N)")
    print("  3. Use the axis buttons and slider to navigate slices")
    print("  4. Slices will auto-scale to match organ dimensions")
    print("=" * 70)

if __name__ == "__main__":
    main()

