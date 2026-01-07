#!/usr/bin/env python3
"""
Setup Web Viewer - One-time setup script

This script:
1. Checks if node_modules exists
2. Copies vtk.js UMD build to web_viewer/lib/
3. Verifies all required files are in place

Usage:
    python setup_web_viewer.py
"""

from pathlib import Path
import shutil
import sys


def check_file(path: Path, description: str) -> bool:
    """Check if a file exists and print status"""
    if path.exists():
        size = path.stat().st_size
        size_mb = size / (1024 * 1024)
        print(f"✅ {description}")
        print(f"   {path}")
        print(f"   Size: {size_mb:.2f} MB")
        return True
    else:
        print(f"❌ {description}")
        print(f"   Expected: {path}")
        return False


def main():
    # Determine project root (where node_modules should be)
    script_dir = Path(__file__).parent.resolve()
    
    # Assuming this script is in totalsegmentator/bin/
    if script_dir.name == "bin":
        project_root = script_dir.parent.parent
    else:
        # Fallback: current directory
        project_root = Path.cwd()
    
    print("=" * 60)
    print("🔧 TotalSegmentator Web Viewer Setup")
    print("=" * 60)
    print(f"\n📁 Project root: {project_root}\n")
    
    # Check node_modules
    node_modules = project_root / "node_modules"
    if not node_modules.exists():
        print("❌ node_modules not found!")
        print(f"   Expected: {node_modules}")
        print("\n💡 Please run first:")
        print("   npm install")
        print("   or")
        print("   npm install @kitware/vtk.js")
        return 1
    
    print(f"✅ node_modules found: {node_modules}\n")
    
    # Destination: web_viewer/lib/
    web_viewer_dir = project_root / "totalsegmentator" / "bin" / "web_viewer"
    lib_dir = web_viewer_dir / "lib"
    vtk_dest = lib_dir / "vtk.js"
    
    # Check if vtk.js already exists at destination
    if vtk_dest.exists():
        size = vtk_dest.stat().st_size / (1024 * 1024)
        print(f"✅ vtk.js already exists at destination!")
        print(f"   Location: {vtk_dest}")
        print(f"   Size: {size:.2f} MB")
        print("\n💡 No need to copy - setup already complete!")
        return 0
    
    # Source: vtk.js UMD build
    vtk_source = node_modules / "@kitware" / "vtk.js" / "dist" / "vtk.js"
    
    if not vtk_source.exists():
        print("❌ vtk.js not found in node_modules!")
        print(f"   Expected: {vtk_source}")
        print("\n💡 Please run:")
        print("   npm install @kitware/vtk.js")
        return 1
    
    print(f"✅ Found vtk.js source:")
    print(f"   {vtk_source}")
    size = vtk_source.stat().st_size / (1024 * 1024)
    print(f"   Size: {size:.2f} MB\n")
    
    # Create lib directory if it doesn't exist
    lib_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Created directory: {lib_dir}\n")
    
    # Copy vtk.js
    print(f"📋 Copying vtk.js...")
    print(f"   From: {vtk_source}")
    print(f"   To:   {vtk_dest}")
    
    try:
        shutil.copy2(vtk_source, vtk_dest)
        print(f"✅ Copy successful!\n")
    except Exception as e:
        print(f"❌ Copy failed: {e}")
        return 1
    
    # Verify all required files
    print("=" * 60)
    print("🔍 Verifying Setup")
    print("=" * 60 + "\n")
    
    all_ok = True
    
    # Check vtk.js
    all_ok &= check_file(vtk_dest, "vtk.js (UMD build)")
    print()
    
    # Check viewer_mvp.html
    viewer_html = web_viewer_dir / "viewer_mvp.html"
    all_ok &= check_file(viewer_html, "viewer_mvp.html")
    print()
    
    # Check totalseg_gui.py
    gui_script = project_root / "totalsegmentator" / "bin" / "totalseg_gui.py"
    all_ok &= check_file(gui_script, "totalseg_gui.py")
    print()
    
    # Summary
    print("=" * 60)
    if all_ok:
        print("✅ Setup Complete!")
        print("=" * 60)
        print("\n🚀 You can now run:")
        print("   python -m totalsegmentator.bin.totalseg_gui")
        print("\nThe web viewer will open automatically after segmentation.")
        return 0
    else:
        print("⚠️  Setup Incomplete")
        print("=" * 60)
        print("\n❌ Some files are missing. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())