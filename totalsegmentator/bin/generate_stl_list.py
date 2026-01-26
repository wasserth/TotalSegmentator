#!/usr/bin/env python3
"""
Generate stl_list.json for web viewer

Usage:
    python generate_stl_list.py /path/to/stl/folder
    
Example:
    python generate_stl_list.py data/2/output/out_total_all/total_all
"""

import sys
import json
from pathlib import Path


def generate_stl_list_json(stl_dir: Path) -> Path:
    """Generate stl_list.json for web viewer"""
    if not stl_dir.exists():
        raise FileNotFoundError(f"Directory not found: {stl_dir}")
    
    stl_files = sorted([f.name for f in stl_dir.glob("*.stl")])
    
    if not stl_files:
        raise ValueError(f"No STL files found in: {stl_dir}")
    
    list_json = {
        "files": stl_files,
        "count": len(stl_files)
    }
    
    output_path = stl_dir / "stl_list.json"
    with open(output_path, "w") as f:
        json.dump(list_json, f, indent=2)
    
    print(f"✅ Generated {output_path}")
    print(f"   Found {len(stl_files)} STL files")
    return output_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_stl_list.py <stl_directory>")
        print("\nExample:")
        print("  python generate_stl_list.py data/2/output/out_total_all/total_all")
        sys.exit(1)
    
    stl_dir = Path(sys.argv[1])
    
    try:
        generate_stl_list_json(stl_dir)
        print("\n💡 To view in browser:")
        print(f"   Open: totalsegmentator/bin/web_viewer/viewer_mvp.html?dir=<relative-path-to-stl-dir>")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()