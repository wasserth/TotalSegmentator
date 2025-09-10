#!/usr/bin/env python3
"""
Improved TotalSegmentator CLI with organized outputs and enhanced functionality.

This script provides:
1. Organized outputs with clear task titles and result mappings
2. Smoothing options for better 3D Slicer and Blender compatibility  
3. Automatic renaming of outputs to match user specifications
4. Optional export to Blender-compatible formats
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
import warnings

try:
    import numpy as np
    import nibabel as nib
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    warnings.warn("numpy/nibabel not available - some functionality will be limited")

try:
    from totalsegmentator.python_api import totalsegmentator
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False
    warnings.warn("totalsegmentator API not available - segmentation will not work")


# Task definitions with titles and output mappings
SEGMENTATION_TASKS = {
    "liver_segments": {
        "title": "liver: segments",
        "task_name": "liver_segments", 
        "results": [
            "liver_segment_1",
            "liver_segment_2", 
            "liver_segment_3",
            "liver_segment_4",
            "liver_segment_5",
            "liver_segment_6",
            "liver_segment_7",
            "liver_segment_8"
        ],
        "output_mapping": {}  # No renaming needed
    },
    "liver_vessels": {
        "title": "liver: vessels",
        "task_name": "liver_vessels",
        "results": [
            "blood_vessel",
            "neoplasm"
        ],
        "output_mapping": {
            "liver_vessels.nii.gz": "blood_vessel.nii.gz",
            "liver_tumor.nii.gz": "neoplasm.nii.gz"
        }
    },
    "total_vessels": {
        "title": "total", 
        "task_name": "total",
        "results": [
            "inferior_vena_cava",
            "portal_vein_and_splenic_vein"
        ],
        "roi_subset": ["inferior_vena_cava", "portal_vein_and_splenic_vein"],
        "output_mapping": {}  # No renaming needed
    }
}


def apply_smoothing(image_path, smoothing_level="medium"):
    """
    Apply smoothing to segmentation masks for better 3D visualization.
    
    Parameters:
    -----------
    image_path : Path
        Path to the NIfTI segmentation file
    smoothing_level : str
        Level of smoothing: "none", "light", "medium", "heavy"
    """
    if smoothing_level == "none":
        return
        
    if not DEPENDENCIES_AVAILABLE:
        warnings.warn("numpy/nibabel not available, skipping smoothing")
        return
        
    try:
        from scipy import ndimage
        from scipy.ndimage import gaussian_filter
    except ImportError:
        warnings.warn("scipy not available, skipping smoothing")
        return
    
    # Load image
    img = nib.load(str(image_path))
    data = img.get_fdata()
    
    # Apply smoothing based on level
    smoothing_params = {
        "light": {"sigma": 0.5, "iterations": 1},
        "medium": {"sigma": 1.0, "iterations": 1}, 
        "heavy": {"sigma": 1.5, "iterations": 2}
    }
    
    if smoothing_level not in smoothing_params:
        return
        
    params = smoothing_params[smoothing_level]
    
    # For binary masks, apply gaussian smoothing and threshold
    if len(np.unique(data)) <= 2:  # Binary mask
        smoothed = gaussian_filter(data.astype(float), sigma=params["sigma"])
        smoothed = (smoothed > 0.5).astype(np.uint8)
    else:  # Multi-label mask
        smoothed = np.zeros_like(data)
        unique_labels = np.unique(data)
        for label in unique_labels:
            if label == 0:
                continue
            mask = (data == label).astype(float)
            smoothed_mask = gaussian_filter(mask, sigma=params["sigma"])
            smoothed[smoothed_mask > 0.5] = label
    
    # Save smoothed image
    smoothed_img = nib.Nifti1Image(smoothed.astype(data.dtype), img.affine, img.header)
    nib.save(smoothed_img, str(image_path))


def export_to_blender_format(segmentation_path, output_path=None):
    """
    Export segmentation to Blender-compatible format (STL).
    
    Parameters:
    -----------
    segmentation_path : Path
        Path to the NIfTI segmentation file
    output_path : Path, optional
        Output path for STL file. If None, uses same name with .stl extension
    """
    if not DEPENDENCIES_AVAILABLE:
        warnings.warn("numpy/nibabel not available, skipping STL export")
        return None
        
    try:
        import trimesh
        from skimage import measure
    except ImportError:
        warnings.warn("trimesh and/or scikit-image not available, skipping STL export")
        return None
        
    if output_path is None:
        output_path = segmentation_path.with_suffix('.stl')
    
    # Load segmentation
    img = nib.load(str(segmentation_path))
    data = img.get_fdata()
    
    # Extract mesh using marching cubes
    try:
        verts, faces, _, _ = measure.marching_cubes(data, level=0.5)
        
        # Create mesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        
        # Export to STL
        mesh.export(str(output_path))
        return output_path
    except Exception as e:
        warnings.warn(f"Failed to export to STL: {e}")
        return None


def run_segmentation_task(input_path, output_dir, task_config, smoothing="medium", 
                         export_stl=False, device="auto", robust_crop=False):
    """
    Run a specific segmentation task with the given configuration.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temporary directory for raw outputs
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Run TotalSegmentator
        if not API_AVAILABLE:
            raise RuntimeError("TotalSegmentator API not available")
            
        if "roi_subset" in task_config:
            # For total task with specific ROIs
            totalsegmentator(
                input=input_path,
                output=temp_dir,
                task=task_config["task_name"],
                roi_subset=task_config["roi_subset"],
                device=device,
                robust_crop=robust_crop
            )
        else:
            # For other tasks
            totalsegmentator(
                input=input_path,
                output=temp_dir, 
                task=task_config["task_name"],
                device=device,
                robust_crop=robust_crop
            )
        
        # Process and rename outputs
        processed_files = []
        mapping = task_config.get("output_mapping", {})
        
        for temp_file in temp_dir.glob("*.nii.gz"):
            # Determine output filename
            if temp_file.name in mapping:
                output_name = mapping[temp_file.name]
            else:
                output_name = temp_file.name
                
            output_path = output_dir / output_name
            
            # Copy file
            shutil.copy2(temp_file, output_path)
            
            # Apply smoothing
            if smoothing != "none":
                apply_smoothing(output_path, smoothing)
            
            # Export to STL if requested
            if export_stl:
                stl_path = export_to_blender_format(output_path)
                if stl_path:
                    processed_files.append(str(stl_path))
            
            processed_files.append(str(output_path))
        
        # Create task summary
        summary = {
            "title": task_config["title"],
            "task_name": task_config["task_name"],
            "expected_results": task_config["results"],
            "processed_files": processed_files,
            "smoothing_applied": smoothing,
            "stl_export": export_stl,
            "timestamp": str(Path(input_path).stat().st_mtime)
        }
        
        summary_path = output_dir / "task_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
            
        return summary
        
    finally:
        # Clean up temporary directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Improved TotalSegmentator with organized outputs and enhanced functionality",
        epilog="Provides organized segmentation outputs with task titles, smoothing options, and Blender compatibility"
    )
    
    parser.add_argument("-i", "--input", required=True, type=Path,
                        help="Input CT NIfTI file")
    parser.add_argument("-o", "--output", required=True, type=Path,
                        help="Output directory for organized results")
    
    # Task selection
    parser.add_argument("--tasks", nargs="+", 
                        choices=list(SEGMENTATION_TASKS.keys()) + ["all"],
                        default=["all"],
                        help="Segmentation tasks to run")
    
    # Smoothing options
    parser.add_argument("--smoothing", choices=["none", "light", "medium", "heavy"],
                        default="medium",
                        help="Smoothing level for better 3D Slicer/Blender visualization")
    
    # Export options
    parser.add_argument("--export-stl", action="store_true",
                        help="Export segmentations to STL format for Blender")
    
    # Device and performance options
    parser.add_argument("--device", default="auto",
                        help="Device to use: auto, cpu, cuda, etc.")
    parser.add_argument("--robust-crop", action="store_true",
                        help="Use robust cropping for better accuracy")
    
    args = parser.parse_args()
    
    # Validate input
    if not args.input.exists():
        print(f"Error: Input file {args.input} does not exist")
        sys.exit(1)
    
    # Determine tasks to run
    if "all" in args.tasks:
        tasks_to_run = list(SEGMENTATION_TASKS.keys())
    else:
        tasks_to_run = args.tasks
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Run each task
    results = {}
    for task_name in tasks_to_run:
        print(f"\n🔬 Running task: {SEGMENTATION_TASKS[task_name]['title']}")
        print(f"   Expected results: {', '.join(SEGMENTATION_TASKS[task_name]['results'])}")
        
        task_output_dir = args.output / task_name
        
        try:
            summary = run_segmentation_task(
                input_path=args.input,
                output_dir=task_output_dir,
                task_config=SEGMENTATION_TASKS[task_name],
                smoothing=args.smoothing,
                export_stl=args.export_stl,
                device=args.device,
                robust_crop=args.robust_crop
            )
            results[task_name] = summary
            print(f"✅ Completed: {summary['title']}")
            
        except Exception as e:
            print(f"❌ Failed task {task_name}: {e}")
            results[task_name] = {"error": str(e)}
    
    # Create overall summary
    overall_summary = {
        "input_file": str(args.input),
        "output_directory": str(args.output),
        "tasks_requested": tasks_to_run,
        "smoothing_level": args.smoothing,
        "stl_export_enabled": args.export_stl,
        "device_used": args.device,
        "results": results,
        "task_definitions": SEGMENTATION_TASKS
    }
    
    summary_path = args.output / "overall_summary.json"
    with open(summary_path, "w") as f:
        json.dump(overall_summary, f, indent=2)
    
    print(f"\n📊 Overall summary saved to: {summary_path}")
    print(f"📁 All results saved to: {args.output}")
    
    # Print final summary
    successful_tasks = [name for name, result in results.items() if "error" not in result]
    failed_tasks = [name for name, result in results.items() if "error" in result]
    
    print(f"\n✅ Successful tasks: {len(successful_tasks)}")
    for task in successful_tasks:
        print(f"   - {SEGMENTATION_TASKS[task]['title']}")
    
    if failed_tasks:
        print(f"\n❌ Failed tasks: {len(failed_tasks)}")
        for task in failed_tasks:
            print(f"   - {SEGMENTATION_TASKS[task]['title']}: {results[task]['error']}")


if __name__ == "__main__":
    main()