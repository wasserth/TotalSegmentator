#!/usr/bin/env python3
"""
Enhanced TotalSegmentator for Mac
Optimized for macOS with Apple Silicon support
"""

import argparse
import sys
import os
from pathlib import Path
import platform

def main():
    print("🍎 Enhanced TotalSegmentator for Mac")
    print(f"🖥️ Running on: {platform.platform()}")
    
    parser = argparse.ArgumentParser(
        description="Enhanced TotalSegmentator with improved liver vessel segmentation (Mac optimized)",
        epilog="Enhanced by AbeezUrRehman - macOS optimized version"
    )
    
    parser.add_argument("-i", "--input", required=True, type=Path,
                        help="Input CT image (NIfTI format)")
    
    parser.add_argument("-o", "--output", required=True, type=Path,
                        help="Output directory")
    
    parser.add_argument("--mode", choices=["standard", "enhanced_liver", "liver_vessels_only"],
                        default="enhanced_liver",
                        help="Processing mode")
    
    parser.add_argument("--liver_mask", type=Path,
                        help="Pre-computed liver mask (optional)")
    
    parser.add_argument("--contrast_phase", choices=["arterial", "portal", "delayed"],
                        help="Manually specify contrast phase")
    
    parser.add_argument("--device", choices=["auto", "mps", "cpu"],
                        default="auto",
                        help="Device to use (auto=detect best available)")
    
    parser.add_argument("--robust_crop", action="store_true",
                        help="Use robust cropping for better accuracy")
    
    args = parser.parse_args()
    
    # Validate input file
    if not args.input.exists():
        print(f"❌ Error: Input file {args.input} not found")
        sys.exit(1)
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {args.output}")
    
    # Import after argument parsing to speed up help
    from totalsegmentator.enhanced_liver_vessels import process_enhanced_liver_vessels
    from totalsegmentator.python_api import totalsegmentator
    
    # Determine device
    if args.device == "auto":
        try:
            import torch
            if torch.backends.mps.is_available():
                device = "mps"
                print("🚀 Using Apple Silicon GPU (MPS)")
            else:
                device = "cpu"
                print("🖥️ Using CPU")
        except:
            device = "cpu"
            print("🖥️ Using CPU (PyTorch not available)")
    else:
        device = args.device
    
    if args.mode == "enhanced_liver":
        print("🔬 Running enhanced liver vessel segmentation...")
        
        # Create subdirectories
        liver_output = args.output / "liver_standard"
        liver_output.mkdir(exist_ok=True)
        
        # First run standard liver segmentation
        print("1️⃣ Running liver segmentation...")
        totalsegmentator(
            str(args.input), 
            str(liver_output),
            roi_subset=["liver"],
            robust_crop=args.robust_crop,
            ml=True,
            device=device
        )
        
        # Then run enhanced vessel segmentation
        print("2️⃣ Running enhanced vessel processing...")
        vessel_output = args.output / "enhanced_liver_vessels.nii.gz"
        liver_mask_path = liver_output / "liver.nii.gz" if not args.liver_mask else args.liver_mask
        
        enhanced_vessels, metadata = process_enhanced_liver_vessels(
            args.input,
            vessel_output,
            liver_mask_path
        )
        
        print(f"✅ Enhanced liver vessels saved to: {vessel_output}")
        print(f"📊 Vessel volume: {metadata['vessel_volume_mm3']:.2f} mm³")
        print(f"🩸 Contrast phase: {metadata['contrast_phase']}")
        print(f"💻 Device used: {metadata['device_used']}")
        
    elif args.mode == "liver_vessels_only":
        print("🔬 Running liver vessels only...")
        totalsegmentator(
            str(args.input),
            str(args.output),
            task="liver_vessels",
            robust_crop=args.robust_crop,
            device=device
        )
        
    else:  # standard mode
        print("🔬 Running standard TotalSegmentator...")
        totalsegmentator(
            str(args.input),
            str(args.output),
            robust_crop=args.robust_crop,
            device=device
        )
    
    print("🎉 Processing complete!")

if __name__ == "__main__":
    main()