#!/usr/bin/env python3
"""
Example script demonstrating TotalSegmentatorImproved usage.
"""

import subprocess
import sys
from pathlib import Path


def run_example(input_file, output_dir):
    """
    Run an example of TotalSegmentatorImproved with sample parameters.
    """
    print("🔬 Running TotalSegmentatorImproved Example")
    print(f"Input: {input_file}")
    print(f"Output: {output_dir}")
    
    # Example 1: Complete processing with medium smoothing
    print("\n📋 Example 1: Complete Processing")
    cmd1 = [
        "python", "-m", "totalsegmentator.bin.TotalSegmentatorImproved",
        "-i", str(input_file),
        "-o", str(output_dir / "complete"),
        "--smoothing", "medium",
        "--robust-crop"
    ]
    print(f"Command: {' '.join(cmd1)}")
    
    # Example 2: Liver analysis only with STL export  
    print("\n🫀 Example 2: Liver Analysis Only")
    cmd2 = [
        "python", "-m", "totalsegmentator.bin.TotalSegmentatorImproved",
        "-i", str(input_file),
        "-o", str(output_dir / "liver_only"),
        "--tasks", "liver_segments", "liver_vessels",
        "--smoothing", "heavy",
        "--export-stl"
    ]
    print(f"Command: {' '.join(cmd2)}")
    
    # Example 3: Vascular analysis with light smoothing
    print("\n🩸 Example 3: Vascular Analysis")  
    cmd3 = [
        "python", "-m", "totalsegmentator.bin.TotalSegmentatorImproved",
        "-i", str(input_file),
        "-o", str(output_dir / "vascular"),
        "--tasks", "liver_vessels", "total_vessels", 
        "--smoothing", "light"
    ]
    print(f"Command: {' '.join(cmd3)}")
    
    print("\n⚠️  Note: These commands will only work if you have:")
    print("   - A valid CT NIfTI input file")
    print("   - TotalSegmentator model weights downloaded")
    print("   - Required dependencies installed (numpy, nibabel, etc.)")
    
    print(f"\n📁 Expected Output Structure in {output_dir}:")
    print("""
    complete/
    ├── overall_summary.json
    ├── liver_segments/
    │   ├── liver_segment_1.nii.gz
    │   ├── ...
    │   └── task_summary.json
    ├── liver_vessels/
    │   ├── blood_vessel.nii.gz       # (renamed from liver_vessels)
    │   ├── neoplasm.nii.gz           # (renamed from liver_tumor)
    │   └── task_summary.json
    └── total_vessels/
        ├── inferior_vena_cava.nii.gz
        ├── portal_vein_and_splenic_vein.nii.gz
        └── task_summary.json
    
    liver_only/
    ├── overall_summary.json
    ├── liver_segments/
    │   ├── liver_segment_*.nii.gz
    │   └── task_summary.json
    └── liver_vessels/
        ├── blood_vessel.nii.gz
        ├── blood_vessel.stl          # STL for Blender
        ├── neoplasm.nii.gz
        ├── neoplasm.stl              # STL for Blender
        └── task_summary.json
    """)


def main():
    """Main function."""
    if len(sys.argv) < 3:
        print("Usage: python examples.py <input_file.nii.gz> <output_directory>")
        print("\nExample:")
        print("  python examples.py patient_ct.nii.gz ./results")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    
    if not input_file.exists():
        print(f"Error: Input file {input_file} does not exist")
        sys.exit(1)
    
    run_example(input_file, output_dir)
    
    print(f"\n✨ To run these examples, copy and paste the commands above")
    print(f"   (after ensuring you have the required dependencies and model weights)")


if __name__ == "__main__":
    main()