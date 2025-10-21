#!/usr/bin/env python3
"""
Improved TotalSegmentator CLI with organized outputs and enhanced functionality.

Features:
1) Organized outputs with clear task titles and result mappings
2) Mask smoothing (NIfTI) for cleaner meshes in Slicer/Blender
3) Robust, unit-correct mesh export (STL/OBJ/PLY) using voxel spacing from NIfTI affine
4) Clean filenames (no ".nii.stl"), per-label export for multi-label masks
5) Optional Laplacian mesh smoothing for nicer surfaces in Blender
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
import warnings
from typing import Optional, Iterable
import time
from datetime import datetime

# -----------------------------
# Optional dependencies
# -----------------------------
try:
    import numpy as np
    import nibabel as nib
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    warnings.warn("numpy/nibabel not available - smoothing and mesh export limited")

try:
    from totalsegmentator.python_api import totalsegmentator
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False
    warnings.warn("totalsegmentator API not available - segmentation will not work")


# -----------------------------
# Task definitions
# -----------------------------
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
            "liver_segment_8",
        ],
        "output_mapping": {},  # No renaming needed
    },
    "liver_vessels": {
        "title": "liver: vessels",
        "task_name": "liver_vessels",
        "results": [
            "blood_vessel",
            "neoplasm",
        ],
        # Map raw TS filenames to your preferred ones if needed.
        "output_mapping": {
            "liver_vessels.nii.gz": "blood_bessel.nii.gz".replace("bessel", "vessel"),  # safety typo fix
            "liver_tumor.nii.gz": "neoplasm.nii.gz",
            # If your TS version already outputs blood_vessels.nii.gz, you can remove this mapping.
        },
    },
    "total_vessels": {
        "title": "total",
        "task_name": "total",
        "results": [
            "inferior_vena_cava",
            "portal_vein_and_splenic_vein",
        ],
        "roi_subset": ["inferior_vena_cava", "portal_vein_and_splenic_vein"],
        "output_mapping": {},  # No renaming needed
    },
    "total_all": {
        "title": "total: all classes",
        "task_name": "total",
        # Full CT task produces 100+ classes; keep display concise.
        "results": ["(many classes)"],
        "output_mapping": {},
    },
}


# -----------------------------
# Utilities for mesh export
# -----------------------------
def _voxel_sizes_from_affine(affine):
    """Return (sx, sy, sz) in mm from a NIfTI affine."""
    import numpy as _np
    return tuple(_np.sqrt((affine[:3, :3] ** 2).sum(axis=0)).tolist())


def _mesh_outpath_for(seg_path: Path, ext: str = ".stl") -> Path:
    """Replace .nii or .nii.gz with mesh extension, avoiding '.nii.stl'."""
    name = seg_path.name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    return seg_path.with_name(f"{name}{ext}")


def _laplacian_smooth_trimesh(mesh, iterations: int = 0):
    if iterations <= 0:
        return mesh
    try:
        from trimesh.smoothing import filter_laplacian
        m = mesh.copy()
        filter_laplacian(m, lamb=0.5, iterations=iterations)
        return m
    except Exception:
        return mesh


# -----------------------------
# Mask smoothing (NIfTI domain)
# -----------------------------
def apply_smoothing(image_path, smoothing_level="medium"):
    """
    Apply smoothing to segmentation masks for better 3D visualization (NIfTI).
    For binary masks: gaussian blur + threshold.
    For multi-label: per-label smoothing.
    """
    if smoothing_level == "none":
        return

    if not DEPENDENCIES_AVAILABLE:
        warnings.warn("numpy/nibabel not available, skipping smoothing")
        return

    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        warnings.warn("scipy not available, skipping smoothing")
        return

    img = nib.load(str(image_path))
    data = img.get_fdata()

    smoothing_params = {
        "light": {"sigma": 0.5},
        "medium": {"sigma": 1.0},
        "heavy": {"sigma": 1.5},
    }

    if smoothing_level not in smoothing_params:
        return

    sigma = smoothing_params[smoothing_level]["sigma"]

    unique_vals = np.unique(data)
    if len(unique_vals) <= 2:
        # Binary
        smoothed = gaussian_filter(data.astype(float), sigma=sigma)
        smoothed = (smoothed > 0.5).astype(np.uint8)
    else:
        # Multi-label
        smoothed = np.zeros_like(data)
        for label in unique_vals:
            if label == 0:
                continue
            mask = (data == label).astype(float)
            smoothed_mask = gaussian_filter(mask, sigma=sigma)
            smoothed[smoothed_mask > 0.5] = label

    smoothed_img = nib.Nifti1Image(smoothed.astype(data.dtype), img.affine, img.header)
    nib.save(smoothed_img, str(image_path))


# -----------------------------
# Mesh export (Blender-friendly)
# -----------------------------
def export_to_blender_format(
    segmentation_path: Path,
    output_path: Optional[Path] = None,
    export_format: str = "stl",            # 'stl' | 'obj' | 'ply'
    mm_to_meters: bool = True,             # Blender uses meters
    laplacian_iters: int = 0,              # optional surface smoothing on mesh
    is_binary: Optional[bool] = None,      # auto-detect if None
    level: float = 0.5,                    # iso-level for marching cubes
) -> Optional[Path]:
    """
    Convert a NIfTI segmentation (binary or multi-label) into a mesh in a Blender-friendly format.
    - Uses voxel spacing from NIfTI affine (geometry to real size in mm)
    - Optionally scales mm -> meters for Blender
    - Clean output filenames (no '.nii.stl')
    - Exports one mesh per label for multi-label masks
    """
    if not DEPENDENCIES_AVAILABLE:
        warnings.warn("numpy/nibabel not available, skipping mesh export")
        return None

    try:
        import numpy as np
        import trimesh
        from skimage import measure
    except ImportError as e:
        warnings.warn(f"Missing deps for mesh export: {e}")
        return None

    ext = f".{export_format.lower()}"
    if output_path is None:
        output_path = _mesh_outpath_for(segmentation_path, ext=ext)

    img = nib.load(str(segmentation_path))
    data = img.get_fdata()
    affine = img.affine
    sx, sy, sz = _voxel_sizes_from_affine(affine)  # in mm

    # Decide binary vs multi-label
    if is_binary is None:
        unique = np.unique(data)
        is_binary = (len(unique) <= 2)

    def _march(mask: np.ndarray) -> Optional["trimesh.Trimesh"]:
        # Marching cubes with correct physical spacing
        verts, faces, _, _ = measure.marching_cubes(
            mask.astype(np.float32), level=level, spacing=(sx, sy, sz)
        )
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
        if mm_to_meters:
            mesh.apply_scale(0.001)  # mm -> m
        mesh = _laplacian_smooth_trimesh(mesh, iterations=laplacian_iters)
        return mesh if (mesh.vertices.size and mesh.faces.size) else None

    written_paths: list[Path] = []

    if is_binary:
        mask = (data > 0.5)
        if mask.any():
            mesh = _march(mask)
            if mesh:
                mesh.export(str(output_path))
                written_paths.append(output_path)
    else:
        labels: Iterable[float] = [l for l in np.unique(data) if l != 0]
        for l in labels:
            mask = (data == l)
            if not mask.any():
                continue
            mesh = _march(mask)
            if mesh:
                per_label = output_path.with_name(f"{output_path.stem}_label{int(l)}{ext}")
                mesh.export(str(per_label))
                written_paths.append(per_label)

    if not written_paths:
        warnings.warn(f"No meshes written for {segmentation_path}")
        return None

    return written_paths[0] if len(written_paths) == 1 else output_path.parent


# -----------------------------
# Task runner
# -----------------------------
def run_segmentation_task(
    input_path,
    output_dir,
    task_config,
    smoothing="medium",
    export_mesh=False,
    device="auto",
    robust_crop=False,
    export_format="stl",
    units="m",
    mesh_smooth_iters=0,
):
    """
    Run a specific segmentation task with the given configuration.
    """
    task_t0 = time.time()
    started_at = datetime.utcnow().isoformat() + "Z"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = output_dir / "temp"
    temp_dir.mkdir(exist_ok=True)

    try:
        if not API_AVAILABLE:
            raise RuntimeError("TotalSegmentator API not available")

        # Run TotalSegmentator
        ts_kwargs = dict(
            input=input_path,
            output=temp_dir,
            task=task_config["task_name"],
            device=device,
            robust_crop=robust_crop,
        )
        if "roi_subset" in task_config:
            ts_kwargs["roi_subset"] = task_config["roi_subset"]

        totalsegmentator(**ts_kwargs)

        # Process and rename outputs
        processed_files = []
        mapping = task_config.get("output_mapping", {})

        for temp_file in temp_dir.glob("*.nii.gz"):
            # Determine final NIfTI filename
            output_name = mapping.get(temp_file.name, temp_file.name)
            output_path = output_dir / output_name

            # Copy NIfTI
            shutil.copy2(temp_file, output_path)

            # Smooth NIfTI masks (optional)
            if smoothing != "none":
                apply_smoothing(output_path, smoothing)

            # Export to mesh (optional)
            if export_mesh:
                mesh_path = export_to_blender_format(
                    output_path,
                    output_path=None,
                    export_format=export_format,
                    mm_to_meters=(units == "m"),
                    laplacian_iters=mesh_smooth_iters,
                    is_binary=None,  # auto
                )
                if mesh_path:
                    processed_files.append(str(mesh_path))

            processed_files.append(str(output_path))

        # Create task summary
        duration_s = max(0.0, time.time() - task_t0)
        finished_at = datetime.utcnow().isoformat() + "Z"
        summary = {
            "title": task_config["title"],
            "task_name": task_config["task_name"],
            "expected_results": task_config["results"],
            "processed_files": processed_files,
            "smoothing_applied": smoothing,
            "mesh_export": export_mesh,
            "mesh_format": export_format if export_mesh else None,
            "units": units if export_mesh else None,
            "mesh_smooth_iters": mesh_smooth_iters if export_mesh else None,
            "timestamp": str(Path(input_path).stat().st_mtime),
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_seconds": round(duration_s, 3),
        }

        summary_path = output_dir / "task_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        return summary

    finally:
        # Clean temp
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Improved TotalSegmentator with organized outputs and enhanced functionality",
        epilog="Organized outputs + NIfTI smoothing + Blender-ready mesh export (unit-correct)",
    )

    parser.add_argument("-i", "--input", required=True, type=Path,
                        help="Input CT NIfTI file (e.g., input_ct.nii.gz)")
    parser.add_argument("-o", "--output", required=True, type=Path,
                        help="Output directory for organized results")

    # Task selection
    parser.add_argument("--tasks", nargs="+",
                        choices=list(SEGMENTATION_TASKS.keys()) + ["all"],
                        default=["all"],
                        help="Segmentation tasks to run")

    # NIfTI smoothing (mask domain)
    parser.add_argument("--smoothing", choices=["none", "light", "medium", "heavy"],
                        default="medium",
                        help="Mask smoothing level (pre-mesh)")

    # Mesh export options
    parser.add_argument("--export-mesh", action="store_true",
                        help="Export segmentations to a mesh format (Blender-compatible)")
    parser.add_argument("--export-format", choices=["stl", "obj", "ply"],
                        default="stl", help="Mesh format for Blender")
    parser.add_argument("--units", choices=["mm", "m"], default="m",
                        help="Scale meshes to these units (Blender default is meters)")
    parser.add_argument("--mesh-smooth-iters", type=int, default=0,
                        help="Laplacian smoothing iterations on the surface mesh")

    # Device and performance options
    parser.add_argument("--device", default="gpu",
                        help="Device: 'gpu', 'cpu', 'mps', or 'gpu:X' (e.g., gpu:0)")
    parser.add_argument("--robust-crop", action="store_true",
                        help="Use robust cropping for better accuracy")

    # Add-ons for composed tasks
    parser.add_argument("--with-liver-vessels", action="store_true",
                        help="When using 'total_all', also run the liver_vessels subtask and include outputs in total_all/")

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        print(f"Error: Input file {args.input} does not exist")
        sys.exit(1)

    # Determine tasks
    tasks_to_run = list(SEGMENTATION_TASKS.keys()) if "all" in args.tasks else args.tasks

    # Prepare out dir
    args.output.mkdir(parents=True, exist_ok=True)

    results = {}
    overall_t0 = time.time()
    overall_started_at = datetime.utcnow().isoformat() + "Z"
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
                export_mesh=args.export_mesh,
                device=args.device,
                robust_crop=args.robust_crop,
                export_format=args.export_format,
                units=args.units,
                mesh_smooth_iters=args.mesh_smooth_iters,
            )
            results[task_name] = summary
            print(f"✅ Completed: {summary['title']}")
        except Exception as e:
            print(f"❌ Failed task {task_name}: {e}")
            results[task_name] = {"error": str(e)}

    # If total_all is selected and addon requested, run liver_vessels into the same folder
    if "total_all" in tasks_to_run and args.with_liver_vessels:
        addon_task = "liver_vessels"
        print(f"\n➕ Running addon task into total_all/: {SEGMENTATION_TASKS[addon_task]['title']}")
        try:
            addon_summary = run_segmentation_task(
                input_path=args.input,
                output_dir=args.output / "total_all",
                task_config=SEGMENTATION_TASKS[addon_task],
                smoothing=args.smoothing,
                export_mesh=args.export_mesh,
                device=args.device,
                robust_crop=args.robust_crop,
                export_format=args.export_format,
                units=args.units,
                mesh_smooth_iters=args.mesh_smooth_iters,
            )
            results[addon_task] = addon_summary
            print("✅ Added liver_vessels outputs to total_all/")
        except Exception as e:
            print(f"❌ Addon liver_vessels failed: {e}")
            results[addon_task] = {"error": str(e)}

    overall_duration_s = max(0.0, time.time() - overall_t0)
    overall_finished_at = datetime.utcnow().isoformat() + "Z"
    overall_summary = {
        "input_file": str(args.input),
        "output_directory": str(args.output),
        "tasks_requested": tasks_to_run,
        "addons": {"with_liver_vessels": bool(args.with_liver_vessels)} if "total_all" in tasks_to_run else {},
        "smoothing_level": args.smoothing,
        "mesh_export_enabled": args.export_mesh,
        "mesh_format": args.export_format if args.export_mesh else None,
        "units": args.units if args.export_mesh else None,
        "mesh_smooth_iters": args.mesh_smooth_iters if args.export_mesh else None,
        "device_used": args.device,
        "results": results,
        "task_definitions": SEGMENTATION_TASKS,
        "started_at": overall_started_at,
        "finished_at": overall_finished_at,
        "duration_seconds": round(overall_duration_s, 3),
    }

    summary_path = args.output / "overall_summary.json"
    with open(summary_path, "w") as f:
        json.dump(overall_summary, f, indent=2)

    print(f"\n📊 Overall summary saved to: {summary_path}")
    print(f"📁 All results saved to: {args.output}")

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
