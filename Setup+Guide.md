
---

## End-to-End: DICOM ➜ NIfTI ➜ PNG ➜ Segmentation ➜ STL ➜ Blender (NEW)

This section documents a complete, reproducible pipeline from a folder of DICOM slices to a colored, labeled 3D model in Blender with optional slice overlays and interactive sliders.

### Requirements

- dcm2niix (or dcm2nii) for DICOM ➜ NIfTI conversion
- Python 3.9+ with TotalSegmentator installed (`pip install TotalSegmentator`)
- Blender 4.5+ (with the STL add-on enabled or installed)
- Optional: `nibabel` and `imageio` for PNG slice export (`pip install nibabel imageio`)

> On Windows, run commands in PowerShell inside your virtual environment. On Linux/Mac, adapt paths accordingly.

### 1) Convert DICOM ➜ NIfTI

Windows (PowerShell):

```powershell
# Create output folder for NIfTI
New-Item -ItemType Directory -Force -Path .\out_nii | Out-Null

# Convert one DICOM series to NIfTI (gzipped)
dcm2niix -z y -o .\out_nii -f case01 .\path\to\dicom_series

# Result: .\out_nii\case01.nii.gz
```

Notes:
- `-z y` writes compressed NIfTI (`.nii.gz`).
- `-f case01` controls the output filename base.

### 2) Export PNG Slices (axial / coronal / sagittal)

This creates folders `axial/`, `coronal/`, `sagittal/` under a chosen output directory for use with the Blender DICOM slider panel.

Windows (PowerShell):

```powershell
pip install nibabel imageio

$nii = ".\out_nii\case01.nii.gz"
$out = ".\dicom_slices"
New-Item -ItemType Directory -Force -Path $out | Out-Null

python - << 'PY'
import os, sys, numpy as np
import nibabel as nib
import imageio.v2 as iio

nii_path, out_dir = sys.argv[1], sys.argv[2]
img = nib.load(nii_path)
data = img.get_fdata()

def write_axis(arr, subfolder):
    d = os.path.join(out_dir, subfolder)
    os.makedirs(d, exist_ok=True)
    vmin, vmax = np.percentile(arr, (1, 99))
    for i, sl in enumerate(arr, start=1):
        s = np.clip((sl - vmin) / (vmax - vmin + 1e-9), 0, 1)
        iio.imwrite(os.path.join(d, f"{i:03d}.png"), (s*255).astype(np.uint8))

# Axial (Z axis first)
write_axis(np.moveaxis(data, 2, 0), 'axial')
# Coronal (Y axis first)
write_axis(np.moveaxis(data, 1, 0), 'coronal')
# Sagittal (X axis first)
write_axis(np.moveaxis(data, 0, 0), 'sagittal')
PY
"$nii" "$out"
```

Notes:
- The script performs simple windowing using the 1st–99th percentiles. Adjust as needed.
- The Blender panel also works if you place all PNGs in a single folder (no subfolders).

### 3) Segmentation and Mesh (STL) Export

Make use of a improved CLI to generate segmentation, organize outputs, and export STL directly for Blender.

```powershell
# All classes; export STL meshes with smoothing for Blender
TotalSegmentatorImproved -i .\out_nii\case01.nii.gz -o .\out_total_all --export-mesh --export-format stl --units m `
--mesh-smooth-iters 10

# Meshed organ STL files will be under something like:
# .\out_total_all\total_all\
```

Key parameters (export):
- `--export-format stl|obj|ply`: mesh format for Blender; STL is simplest.
- `--units m|mm`: sets geometry units; prefer `m` then scale if needed in Blender.
- `--mesh-smooth-iters`: Laplacian smoothing iterations for cleaner meshes.
- `--write-empty`: emit placeholder STL when a mask is empty (helps batch import).

### 4) Import Meshes in Blender (collections, transforms)

Use the provided importer to load all meshes, organize collections, and color with the exact palette. Always separate Blender args from script args with `--`.

```powershell
# Create scene and save it
blender -b -P totalsegmentator\bin\totalseg_blender_import.py -- `
  --stl-dir .\out_total_all\total_all `
  --units m `
  --collection Organs `
  --group-categories `
  --palette exact `
  --scale 0.01 `
  --save .\out\scene-setup.blend
```

Key parameters (import):
- `--stl-dir`: folder containing your `.stl`/`.obj`/`.ply` meshes.
- `--units m|mm`: unit used in your meshes; if `mm`, the script converts to meters.
- `--scale`: uniform scale applied after units; `0.01` produces smaller scenes.
- `--group-categories`: creates sub-collections (Bone, Muscle, Thoracic, Abdominal, Vessel) and places objects.
- `--palette exact|auto`: use the fixed color palette (exact) you provided, or semantic + distinct colors (auto). Default is `exact`.
- `--rotate-x-deg`: correct orientation (e.g., `-90` from Slicer to Blender coordinates).
- `--mirror-x`: mirror across global X and flip X-location to correct left/right.
- `--remesh voxel|quad|smooth|sharp|none`: optional cleanup of mesh topology; use `--voxel-size 0.003` for ~3 mm at meter units.

Troubleshooting (Blender):
- If STL import fails, enable the add-on “Import-Export: STL format” in Blender Preferences or install the `io_mesh_stl` add-on for your version.
- Always include `--` before script flags so Blender doesn’t treat them as file paths.

### 5) Apply Materials with Semantic Coloring in Blender

Once the scene is set up with proper geometry, add consistent materials with anatomical coloring using the included material script:

```powershell
# Apply anatomically-consistent materials to the scene
blender -b ".\out\scene-setup.blend" -P totalsegmentator\bin\totalseg_material.py -- .\out\colored-organs.blend
```

The material script applies semantic coloring to each segmentation class:
- Bones (white/cream)
- Muscles (dark red)
- Vessels (arteries: bright red, veins: blue)
- Organs (liver: reddish-brown, kidneys: maroon, etc.)

If you need to customize the materials or add your own color schemes:

```python
# Example custom material mapping in your script
custom_mapping = {
  "liver": (0.35, 0.05, 0.04, 1.0),  # Dark red-brown
  "portal_vein_branches": (0.04, 0.04, 0.43, 1.0),  # Deep blue
  "hepatic_veins": (0.07, 0.01, 0.37, 1.0)  # Purple-blue
}
```

For custom materials or if the automatic mapping fails, provide an explicit mapping file with:
```powershell
blender -b ".\out\scene-setup.blend" -P totalsegmentator\bin\totalseg_material.py -- \
  --mapping-json .\custom_materials.json .\out\custom-colored.blend
```

### 6) Overlay DICOM/PNG Slices in Blender (interactive slider)

Open your saved scene and run the DICOM slider panel script with your image folder. It auto-centers to organ bounds and adds an interactive UI to scroll slices.

```powershell
blender .\out\scene-setup.blend -P .\dicom_slider_addon.py -- --image-dir "C:\Users\<you>\Documents\GitHub\TotalSegmentator\dicom_slices"
```

Usage inside Blender:
- Panel: Viewport → Sidebar (N) → DICOM → “DICOM Slices”.
- Fields: select folder, axis (AX/COR/SAG), slice number, and whether to auto-scale to the model.
- Advanced: pixel pitch (mm), image width, axial slice count and spacing, and base scale.

What the slider script does:
- Detects the overall organ bounding box to center and scale images automatically.
- Loads one image as an image-empty at a time and positions it along the chosen axis.
- Supports folders containing subfolders `axial/`, `coronal/`, `sagittal/`, or a flat folder of images.

### Putting It All Together (Minimal)

```powershell
# 1) DICOM ➜ NIfTI
dcm2niix -z y -o .\out_nii -f case01 .\path\to\dicom_series

# 2) NIfTI ➜ PNG slices
pip install nibabel imageio
python - << 'PY'
import os, sys, numpy as np
import nibabel as nib
import imageio.v2 as iio
nii_path, out_dir = sys.argv[1], sys.argv[2]
img = nib.load(nii_path); data = img.get_fdata()
def W(a,sf):
  import numpy as np, os, imageio.v2 as iio
  os.makedirs(sf, exist_ok=True)
  vmin, vmax = np.percentile(a,(1,99))
  for i,sl in enumerate(a,1):
    s=np.clip((sl-vmin)/(vmax-vmin+1e-9),0,1)
    iio.imwrite(os.path.join(sf,f"{i:03d}.png"),(s*255).astype(np.uint8))
W(np.moveaxis(data,2,0), os.path.join(out_dir,'axial'))
W(np.moveaxis(data,1,0), os.path.join(out_dir,'coronal'))
W(np.moveaxis(data,0,0), os.path.join(out_dir,'sagittal'))
PY
".\out_nii\case01.nii.gz" ".\dicom_slices"

# 3) Segmentation + STL export
TotalSegmentatorImproved -i .\out_nii\case01.nii.gz -o .\out_total_all `
  --export-mesh --export-format stl --units m --mesh-smooth-iters 10

# 4) Blender import (exact colors, collections)
blender -b -P totalsegmentator\bin\totalseg_blender_import.py -- `
  --stl-dir .\out_total_all\total_all `
  --units m --collection Organs --group-categories --palette exact `
  --scale 0.01 --save .\out\scene-setup.blend

# 5) Slice overlay / slider
blender .\out\scene-setup.blend -P .\dicom_slider_addon.py -- --image-dir ".\dicom_slices"
```

### Next Steps

- A simple desktop GUI is planned to wrap these steps for non-technical users.
- If your dataset has different naming for STL files, extend the exact mapping list so materials and collections match perfectly (see `totalsegmentator/bin/totalseg_blender_import.py`).


---