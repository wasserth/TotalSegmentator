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
    pad_edges: bool = True,                # pad volume to avoid open caps at scan bounds
    fill_holes: bool = True,               # try to fill mesh holes (watertight)
    pre_dilate_mm: float = 0.0,            # thicken mask before meshing (mm)
    min_mask_voxels: int = 1,              # minimum voxels to consider non-empty
    force_empty_stl: bool = False,         # write placeholder STL even if empty
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

    def _maybe_dilate(mask: np.ndarray) -> np.ndarray:
        if pre_dilate_mm > 0:
            try:
                from scipy.ndimage import binary_dilation
                # Use voxel units based on smallest spacing to approximate a spherical dilation
                vox = max(1, int(np.ceil(pre_dilate_mm / min(sx, sy, sz))))
                for _ in range(vox):
                    mask = binary_dilation(mask)
            except Exception:
                pass
        return mask

    def _march(mask: np.ndarray) -> Optional["trimesh.Trimesh"]:
        m = mask.astype(np.float32)
        offset = np.array([0.0, 0.0, 0.0], dtype=float)
        if pad_edges:
            # Pad by 1 voxel so surfaces are closed at array borders
            m = np.pad(m, 1, mode='constant', constant_values=0)
            offset = np.array([sx, sy, sz], dtype=float)
        # Marching cubes with correct physical spacing
        verts, faces, _, _ = measure.marching_cubes(m, level=level, spacing=(sx, sy, sz))
        # Shift back if padded
        if pad_edges:
            verts = verts - offset
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
        # Optional repairs to make watertight
        if fill_holes:
            try:
                import trimesh.repair as trepair
                trepair.fill_holes(mesh)
                trepair.fix_normals(mesh)
                mesh.remove_unreferenced_vertices()
                mesh.remove_degenerate_faces()
            except Exception:
                pass
        if mm_to_meters:
            mesh.apply_scale(0.001)  # mm -> m
        mesh = _laplacian_smooth_trimesh(mesh, iterations=laplacian_iters)
        return mesh if (mesh.vertices.size and mesh.faces.size) else None

    written_paths: list[Path] = []

    def _write_empty_placeholder(path: Path):
        try:
            text = f"solid {path.stem}\nendsolid {path.stem}\n"
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            pass

    if is_binary:
        mask = (data > 0.5)
        if mask.sum() < max(1, int(min_mask_voxels)):
            if force_empty_stl:
                _write_empty_placeholder(output_path)
                written_paths.append(output_path)
        else:
            mask = _maybe_dilate(mask)
            mesh = _march(mask)
            if mesh:
                mesh.export(str(output_path))
                written_paths.append(output_path)
            elif force_empty_stl:
                _write_empty_placeholder(output_path)
                written_paths.append(output_path)
    else:
        labels: Iterable[float] = [l for l in np.unique(data) if l != 0]
        for l in labels:
            mask = (data == l)
            per_label = output_path.with_name(f"{output_path.stem}_label{int(l)}{ext}")
            if mask.sum() < max(1, int(min_mask_voxels)):
                if force_empty_stl:
                    _write_empty_placeholder(per_label)
                    written_paths.append(per_label)
                continue
            mask = _maybe_dilate(mask)
            mesh = _march(mask)
            if mesh:
                mesh.export(str(per_label))
                written_paths.append(per_label)
            elif force_empty_stl:
                _write_empty_placeholder(per_label)
                written_paths.append(per_label)

    if not written_paths:
        # Only warn if we truly produced nothing and didn't request placeholders
        if not force_empty_stl:
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
    mesh_pad_edges=True,
    mesh_fill_holes=True,
    dilate_mm=0.0,
    min_mask_voxels=1,
    write_empty_stl=False,
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
                    pad_edges=mesh_pad_edges,
                    fill_holes=mesh_fill_holes,
                    pre_dilate_mm=float(dilate_mm),
                    min_mask_voxels=int(min_mask_voxels),
                    force_empty_stl=bool(write_empty_stl),
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
    parser.add_argument("--no-mesh-pad-edges", action="store_true",
                        help="Disable 1-voxel padding before meshing (may leave open caps at scan bounds)")
    parser.add_argument("--no-mesh-fill-holes", action="store_true",
                        help="Disable mesh hole filling/repairs during export")
    parser.add_argument("--dilate-mm", type=float, default=0.0,
                        help="Pre-dilate masks by this many millimeters before meshing (thickens thin structures)")
    parser.add_argument("--min-mask-voxels", type=int, default=1,
                        help="Minimum number of voxels to treat a mask as non-empty for meshing")
    parser.add_argument("--write-empty-stl", action="store_true",
                        help="Write placeholder empty STL files when a mask is empty or cannot be meshed")

    # Mesh-only export mode (skip inference)
    parser.add_argument("--export-only-dir", type=Path, default=None,
                        help="Export meshes only from NIfTI masks in this directory (skip inference)")
    parser.add_argument("--export-recursive", action="store_true",
                        help="Recurse into subdirectories when using --export-only-dir")
    parser.add_argument("--export-pattern", type=str, default="*.nii.gz",
                        help="Glob pattern for NIfTI files in --export-only-dir (default: *.nii.gz)")

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

    # Mesh-only export mode
    if args.export_only_dir is not None:
        src = args.export_only_dir
        if not src.exists():
            print(f"Error: export-only dir {src} does not exist")
            sys.exit(1)
        args.output.mkdir(parents=True, exist_ok=True)
        files = list(src.rglob(args.export_pattern)) if args.export_recursive else list(src.glob(args.export_pattern))
        exported = []
        for f in sorted(files):
            if f.suffix not in [".gz", ".nii", ".nii.gz"] and not str(f).endswith(".nii.gz"):
                continue
            # Compute output path in target dir
            name = f.name[:-7] if f.name.endswith('.nii.gz') else f.stem
            out_mesh = args.output / f"{name}.{args.export_format}"
            mesh_path = export_to_blender_format(
                f,
                output_path=out_mesh,
                export_format=args.export_format,
                mm_to_meters=(args.units == "m"),
                laplacian_iters=args.mesh_smooth_iters,
                is_binary=None,
                pad_edges=not args.no_mesh_pad_edges,
                fill_holes=not args.no_mesh_fill_holes,
                pre_dilate_mm=float(args.dilate_mm),
                min_mask_voxels=int(args.min_mask_voxels),
                force_empty_stl=bool(args.write_empty_stl),
            )
            if mesh_path:
                exported.append(str(mesh_path))
        summary = {
            "mode": "export_only",
            "source_dir": str(src),
            "pattern": args.export_pattern,
            "recursive": bool(args.export_recursive),
            "export_format": args.export_format,
            "units": args.units,
            "mesh_smooth_iters": args.mesh_smooth_iters,
            "pad_edges": not args.no_mesh_pad_edges,
            "fill_holes": not args.no_mesh_fill_holes,
            "dilate_mm": args.dilate_mm,
            "min_mask_voxels": args.min_mask_voxels,
            "write_empty_stl": bool(args.write_empty_stl),
            "exported_count": len(exported),
        }
        with open(args.output / "export_only_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Exported {len(exported)} mesh(es) to {args.output}")
        return

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
                mesh_pad_edges=not args.no_mesh_pad_edges,
                mesh_fill_holes=not args.no_mesh_fill_holes,
                dilate_mm=args.dilate_mm,
                min_mask_voxels=args.min_mask_voxels,
                write_empty_stl=args.write_empty_stl,
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
                mesh_pad_edges=not args.no_mesh_pad_edges,
                mesh_fill_holes=not args.no_mesh_fill_holes,
                dilate_mm=args.dilate_mm,
                min_mask_voxels=args.min_mask_voxels,
                write_empty_stl=args.write_empty_stl,
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
