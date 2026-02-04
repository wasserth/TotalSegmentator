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

def install_addon_to_blend(addon_source: Path, text_name: str = "dicom_slider_addon.py"):
    """Copy an addon script into Blender text blocks"""
    if not addon_source.exists():
        print(f"❌ ERROR: Addon source not found: {addon_source}")
        return False
    
    print(f"📦 Installing addon from: {addon_source}")
    
    # Remove old version if exists
    for txt in bpy.data.texts:
        if txt.name == text_name:
            bpy.data.texts.remove(txt)
    
    # Load addon into text block
    with open(addon_source, 'r') as f:
        addon_code = f.read()
    
    txt = bpy.data.texts.new(text_name)
    txt.from_string(addon_code)
    txt.use_module = True  # Enable as Python module
    
    print(f"✅ Addon installed into .blend file as: {text_name}")
    return True

def setup_startup_script(png_dir, nifti_path, scale, addon_text_names):
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

# Auto-register addons when file is opened
def register_addons():
    try:
        addon_texts = {repr(addon_text_names)}
        for addon_name in addon_texts:
            addon_text = bpy.data.texts.get(addon_name)
            if addon_text:
                exec(compile(addon_text.as_string(), addon_name, 'exec'))
            
        # Configure DICOM slider fields if this addon exposes them
        if hasattr(bpy.context.scene, 'folder_path_dicom'):
            bpy.context.scene.folder_path_dicom = PNG_DIR
            print(f"✓ Set folder_path_dicom to: {{PNG_DIR}}")
        if hasattr(bpy.context.scene, 'base_size'):
            bpy.context.scene.base_size = SCALE
            print(f"✓ Set base_size to: {{SCALE}}")
        if hasattr(bpy.context.scene, 'use_auto_scale'):
            bpy.context.scene.use_auto_scale = True
            print(f"✓ Enabled auto-scale")
        print("✅ Addons loaded and configured")
    except Exception as e:
        print(f"❌ Error loading addons: {{e}}")
        import traceback
        traceback.print_exc()

# Register handler
@bpy.app.handlers.persistent
def load_post_handler(dummy):
    register_addons()

if load_post_handler not in bpy.app.handlers.load_post:
    bpy.app.handlers.load_post.append(load_post_handler)

# Also register now
register_addons()
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
    addon_path_arg = args.get('addon-path', '')
    extra_addon_path_arg = args.get('extra-addon-path', '')

    default_addon_source = Path(__file__).parent.parent.parent / "dicom_slider_addon.py"
    if addon_path_arg:
        addon_source = Path(addon_path_arg).resolve()
        if not addon_source.exists():
            print(f"⚠️ Provided addon path not found, fallback to default: {addon_source}")
            addon_source = default_addon_source
    else:
        addon_source = default_addon_source
    
    if not png_dir:
        print("❌ ERROR: --png-dir is required")
        sys.exit(1)
    
    png_dir = Path(png_dir).resolve()
    if not png_dir.exists():
        print(f"❌ ERROR: PNG directory not found: {png_dir}")
        sys.exit(1)
    
    addon_text_names = []

    # Install primary addon
    if not install_addon_to_blend(addon_source, "dicom_slider_addon.py"):
        sys.exit(1)
    addon_text_names.append("dicom_slider_addon.py")

    # Install optional extra addon (e.g., vessel tool)
    if extra_addon_path_arg:
        extra_source = Path(extra_addon_path_arg).resolve()
        if not extra_source.exists():
            print(f"⚠️ Extra addon path not found, skipping: {extra_source}")
        else:
            extra_text_name = extra_source.name if extra_source.suffix == ".py" else f"{extra_source.name}.py"
            if install_addon_to_blend(extra_source, extra_text_name):
                addon_text_names.append(extra_text_name)
    
    # Setup startup script
    if not setup_startup_script(str(png_dir), str(nifti_path), scale, addon_text_names):
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

