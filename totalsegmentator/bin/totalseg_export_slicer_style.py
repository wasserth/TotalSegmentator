#!/usr/bin/env python3
"""
Slicer-like segmentation→model (surface mesh) exporter.

Implements a VTK pipeline similar to 3D Slicer's "Export segmentation to model":
- vtkDiscreteMarchingCubes to extract a closed surface from a binary labelmap
- Optional vtkWindowedSincPolyDataFilter smoothing (Slicer-like smoothing)
- Optional vtkDecimatePro decimation
- vtkPolyDataNormals to generate consistent normals
- Optional LPS coordinate system (Slicer uses LPS)

Usage examples
  # Export all NIfTI masks in a folder to STL, Slicer-like smoothing, LPS coords
  totalseg_export_slicer_style \
    --input-dir out_total_all/total_all \
    --output-dir meshes_slicer \
    --format stl \
    --smoothing-iterations 30 --smoothing-passband 0.1 \
    --decimate 0.0 \
    --lps

  # Export a single file to OBJ with mild decimation
  totalseg_export_slicer_style -i seg.nii.gz -o seg.obj --format obj --decimate 0.3
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import nibabel as nib
except Exception as e:
    raise SystemExit(f"nibabel is required: {e}")

try:
    import vtk
    from vtk.util import numpy_support
except Exception as e:
    raise SystemExit(
        "vtk is required for Slicer-like export. Install with 'pip install vtk'.\n"
        f"Import error: {e}"
    )


def _voxel_sizes_from_affine(affine):
    return tuple(np.sqrt((affine[:3, :3] ** 2).sum(axis=0)).tolist())


def _vtk_matrix_from_numpy(m44: np.ndarray) -> vtk.vtkMatrix4x4:
    M = vtk.vtkMatrix4x4()
    flat = m44.astype(float).ravel(order="C")
    for r in range(4):
        for c in range(4):
            M.SetElement(r, c, float(m44[r, c]))
    return M


def export_single(
    nifti_path: Path,
    out_path: Path,
    label: int = 1,
    smoothing_iterations: int = 30,
    smoothing_passband: float = 0.1,
    decimate: float = 0.0,  # 0..1 target reduction
    lps: bool = False,
    pad_edges: bool = True,
    min_voxels: int = 1,
    write_empty: bool = False,
):
    img = nib.load(str(nifti_path))
    data = img.get_fdata()
    if data.ndim != 3:
        raise ValueError(f"Only 3D volumes supported: {nifti_path}")

    # Binary mask for the label (>0 for general case)
    if min_voxels > 1 and (data > 0).sum() < min_voxels:
        if write_empty:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(f"solid {out_path.stem}\nendsolid {out_path.stem}\n")
        return

    mask = (data == label) if (np.unique(data).size > 2 and label != 1) else (data > 0)
    if mask.sum() == 0:
        if write_empty:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(f"solid {out_path.stem}\nendsolid {out_path.stem}\n")
        return

    # Prepare VTK image (X,Y,Z ordering). We keep unit spacing and apply full affine later.
    vol_xyz = np.transpose(mask.astype(np.uint8), (2, 1, 0))  # (X,Y,Z)
    X, Y, Z = vol_xyz.shape
    vol_flat = np.ascontiguousarray(vol_xyz.ravel(order="C"))

    image = vtk.vtkImageData()
    image.SetDimensions(int(X), int(Y), int(Z))
    image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    vtk_arr = numpy_support.numpy_to_vtk(vol_flat, deep=1, array_type=vtk.VTK_UNSIGNED_CHAR)
    image.GetPointData().SetScalars(vtk_arr)

    # Optionally pad by 1 voxel to avoid open caps
    if pad_edges:
        pad = vtk.vtkImageConstantPad()
        pad.SetInputData(image)
        pad.SetOutputWholeExtent(0, X, 0, Y, 0, Z)
        pad.SetConstant(0)
        pad.Update()
        image = pad.GetOutput()

    # Discrete marching cubes
    dmc = vtk.vtkDiscreteMarchingCubes()
    dmc.SetInputData(image)
    dmc.SetValue(0, 1)  # foreground=1
    dmc.Update()
    poly = dmc.GetOutput()

    # Transform by full affine (voxel index -> RAS mm)
    aff = img.affine.copy()
    # Convert to LPS if requested: LPS = diag(-1,-1,1) * RAS
    if lps:
        R2L = np.diag([-1.0, -1.0, 1.0, 1.0])
        aff = R2L @ aff
    tfm = vtk.vtkTransform()
    tfm.SetMatrix(_vtk_matrix_from_numpy(aff))
    tfm_f = vtk.vtkTransformPolyDataFilter()
    tfm_f.SetInputData(poly)
    tfm_f.SetTransform(tfm)
    tfm_f.Update()
    poly = tfm_f.GetOutput()

    # Smoothing (Slicer-like: windowed sinc)
    if smoothing_iterations > 0 and smoothing_passband > 0:
        smooth = vtk.vtkWindowedSincPolyDataFilter()
        smooth.SetInputData(poly)
        smooth.SetNumberOfIterations(int(smoothing_iterations))
        smooth.SetPassBand(float(smoothing_passband))
        smooth.FeatureEdgeSmoothingOff()
        smooth.BoundarySmoothingOff()
        smooth.NonManifoldSmoothingOn()
        smooth.NormalizeCoordinatesOn()
        smooth.Update()
        poly = smooth.GetOutput()

    # Decimation (optional)
    if decimate > 0.0:
        dec = vtk.vtkDecimatePro()
        dec.SetInputData(poly)
        dec.SetTargetReduction(float(decimate))  # fraction to remove
        dec.SetPreserveTopology(1)
        dec.BoundaryVertexDeletionOff()
        dec.Update()
        poly = dec.GetOutput()

    # Normals
    norms = vtk.vtkPolyDataNormals()
    norms.SetInputData(poly)
    norms.SplittingOff()
    norms.ConsistencyOn()
    norms.AutoOrientNormalsOn()
    norms.Update()
    poly = norms.GetOutput()

    # Write
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower()
    if ext == ".stl":
        w = vtk.vtkSTLWriter()
        w.SetInputData(poly)
        w.SetFileName(str(out_path))
        w.SetFileTypeToBinary()
        w.Write()
    elif ext == ".obj":
        w = vtk.vtkOBJWriter()
        w.SetInputData(poly)
        w.SetFileName(str(out_path))
        w.Write()
    else:
        raise ValueError(f"Unsupported output format: {ext}")


def parse_args():
    p = argparse.ArgumentParser(description="Slicer-like export of segmentation masks to mesh (STL/OBJ)")
    g_in = p.add_argument_group("Input")
    g_in.add_argument("-i", "--input", type=Path, help="Single NIfTI mask file (.nii.gz)")
    g_in.add_argument("--input-dir", type=Path, help="Directory of NIfTI masks to export")
    g_in.add_argument("--pattern", default="*.nii.gz", help="Glob pattern for input-dir (default: *.nii.gz)")

    g_out = p.add_argument_group("Output")
    g_out.add_argument("-o", "--output", type=Path, required=True, help="Output file (if -i) or output directory (if --input-dir)")
    g_out.add_argument("--format", choices=["stl", "obj"], default="stl", help="Output mesh format")

    g_opt = p.add_argument_group("Options")
    g_opt.add_argument("--smoothing-iterations", type=int, default=30, help="Windowed-sinc smoothing iterations")
    g_opt.add_argument("--smoothing-passband", type=float, default=0.1, help="Windowed-sinc passband (lower=more smoothing)")
    g_opt.add_argument("--decimate", type=float, default=0.0, help="Target reduction (0..1) for vtkDecimatePro")
    g_opt.add_argument("--lps", action="store_true", help="Export coordinates in LPS system (Slicer default)")
    g_opt.add_argument("--no-pad", action="store_true", help="Do not pad 1 voxel around image before marching cubes")
    g_opt.add_argument("--min-voxels", type=int, default=1, help="Minimum voxels to export a mesh")
    g_opt.add_argument("--write-empty", action="store_true", help="Write empty placeholder STL when below threshold")

    return p.parse_args()


def main():
    args = parse_args()
    if (args.input is None) == (args.input_dir is None):
        raise SystemExit("Specify exactly one of --input or --input-dir")

    if args.input:
        # Single file mode
        ext = "." + args.format.lower()
        out_path = args.output
        if out_path.is_dir():
            name = args.input.name[:-7] if args.input.name.endswith('.nii.gz') else args.input.stem
            out_path = out_path / f"{name}{ext}"
        export_single(
            args.input,
            out_path,
            smoothing_iterations=args.smoothing_iterations,
            smoothing_passband=args.smoothing_passband,
            decimate=args.decimate,
            lps=bool(args.lps),
            pad_edges=not args.no_pad,
            min_voxels=args.min_voxels,
            write_empty=args.write_empty,
        )
        print(f"Exported {out_path}")
        return

    # Directory mode
    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = "." + args.format.lower()
    files = sorted(args.input_dir.glob(args.pattern))
    count = 0
    for f in files:
        if not (f.name.endswith('.nii') or f.name.endswith('.nii.gz')):
            continue
        name = f.name[:-7] if f.name.endswith('.nii.gz') else f.stem
        out_path = out_dir / f"{name}{ext}"
        export_single(
            f,
            out_path,
            smoothing_iterations=args.smoothing_iterations,
            smoothing_passband=args.smoothing_passband,
            decimate=args.decimate,
            lps=bool(args.lps),
            pad_edges=not args.no_pad,
            min_voxels=args.min_voxels,
            write_empty=args.write_empty,
        )
        count += 1
    print(f"Exported {count} mesh(es) to {out_dir}")


if __name__ == "__main__":
    main()

