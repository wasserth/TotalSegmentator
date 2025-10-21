# TotalSegmentator Unified Toolkit

End-to-end quickstart for the new tools added alongside TotalSegmentatorImproved: DICOM → 2D images, dataset building, mesh export, and Blender import/slider.

This guide is cross-platform (macOS + Windows). Commands use forward slashes; on Windows, PowerShell/CMD work the same for these examples.

---

## Install / Update

Editable install (recommended for trying the new tools):

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .\.venv\Scripts\activate      # Windows

pip install --upgrade pip setuptools wheel
pip install -e .
```

Optional extras:
- Blender (app) must be installed separately from blender.org
- For mesh smoothing/processing extras: `pip install scikit-image trimesh scipy`

---

## What’s Included

- `TotalSegmentatorImproved` — Organized outputs, optional NIfTI smoothing, STL/OBJ/PLY export (Blender-compatible). See docs/TotalSegmentatorImproved.md.
- `totalseg_dicom_to_png` — DICOM → PNG/JPEG with window presets and custom WL/WW.
- `totalseg_mpr_widget` — Interactive sliders (Jupyter or window) to preview WL/WW, slice and export 3-plane MPR.
- `totalseg_dataset_build` — Build a 2D PNG dataset from a volume + segmentation (per-class or multi-label PNGs, manifest).
- `totalseg_blender_import` — Import meshes into Blender with units, materials, and collection grouping.
- `totalseg_blender_slider` — Create a visibility slider or timeline sequence to cycle through organs.

Script locations for reference:
- `totalsegmentator/bin/TotalSegmentatorImproved.py`
- `totalsegmentator/bin/totalseg_dicom_to_png.py`
- `totalsegmentator/bin/totalseg_dataset_build.py`
- `totalsegmentator/bin/totalseg_blender_import.py`
- `totalsegmentator/bin/totalseg_blender_slider.py`

---

## Quick Command Reference

1) DICOM → PNG/JPEG
- Auto windowing: `totalseg_dicom_to_png -i <dicom_dir> -o <out_dir>`
- Abdomen preset: `totalseg_dicom_to_png -i <dicom_dir> -o <out_dir> --window abdomen`
- Custom W/L: `totalseg_dicom_to_png -i <dicom_dir> -o <out_dir> --window custom --wl 40 --ww 400`
- JPEG output: `--format jpeg --jpeg-quality 95`
- 3-plane MPR folders: `totalseg_dicom_to_png -i <dicom_dir> -o out_iso_views --multi-views --metadata`

2) Segment + export meshes (improved)
- `TotalSegmentatorImproved -i <ct.nii.gz> -o <results_dir> --smoothing medium --export-mesh`
  - Per-task and overall runtimes are saved into `task_summary.json` and `overall_summary.json` (`duration_seconds`, `started_at`, `finished_at`).

3) Build 2D PNG dataset (images + masks)
- From multi-label NIfTI: `totalseg_dataset_build -i <ct.nii.gz> -s <seg_multilabel.nii.gz> -o <dataset_dir> --mode both`
- From per-class NIfTI masks: `totalseg_dataset_build -i <ct.nii.gz> -s <seg_dir> -o <dataset_dir> --mode both`

4) Blender import
- `blender -b -P totalsegmentator/bin/totalseg_blender_import.py -- --stl-dir <results_dir> --units m --collection Organs --save scene.blend`

5) Blender slider/timeline
- Timeline (one organ visible per step):
  `blender -b -P totalsegmentator/bin/totalseg_blender_slider.py -- --collection Organs --make-timeline --start 1 --step 10 --save scene_slider.blend`
- Interactive UI panel (run without `-b`):
  `blender -P totalsegmentator/bin/totalseg_blender_slider.py -- --collection Organs --panel`

---

## End-to-End Test (Using Repo Samples)

The repository includes small test datasets to validate the pipeline.

1) Convert DICOM to PNG (QC)

```bash
totalseg_dicom_to_png \
  -i tests/reference_files/example_ct_dicom \
  -o out/pngs --window abdomen
```

2) Segment with organized outputs and export meshes

```bash
TotalSegmentatorImproved \
  -i tests/reference_files/example_ct.nii.gz \
  -o out/seg_improved \
  --smoothing medium --export-mesh
```

3) Build a 2D PNG dataset

Option A: If you have a multi-label NIfTI (e.g., `example_seg.nii.gz`):
```bash
totalseg_dataset_build \
  -i tests/reference_files/example_ct.nii.gz \
  -s tests/reference_files/example_seg.nii.gz \
  -o out/dataset --mode both --skip-empty
```

Option B: From per-class masks produced by TotalSegmentatorImproved/TotalSegmentator:
```bash
totalseg_dataset_build \
  -i tests/reference_files/example_ct.nii.gz \
  -s out/seg_improved \
  -o out/dataset --mode both --skip-empty
```

4) Import meshes into Blender (headless) and save a scene

```bash
blender -b -P totalsegmentator/bin/totalseg_blender_import.py -- \
  --stl-dir out/seg_improved \
  --units m \
  --collection Organs \
  --save out/scene.blend
```

5) Create a visibility timeline

```bash
blender -b -P totalsegmentator/bin/totalseg_blender_slider.py -- \
  --collection Organs --make-timeline --start 1 --step 10 \
  --save out/scene_slider.blend
```

You can open `out/scene.blend` or `out/scene_slider.blend` in Blender to inspect the results.

---

## Notes and Tips

- Windows paths: If Blender isn’t in PATH, use the full path to `blender.exe`.
- Units and scale: `TotalSegmentatorImproved` exports meshes with physical spacing; Blender import scales mm → meters by default.
- PNG vs JPEG: Use PNG for masks; JPEG is available for image previews if needed.
- Class subsets: To reduce runtime and dataset size, use `--class-filter` in `totalseg_dataset_build`.
- Overlay QC: If you want PNG overlays (image + colored mask), open an issue; a helper can be added quickly.

---

## Troubleshooting

- `ModuleNotFoundError` for Pillow/pydicom: Run `pip install -e .` again to ensure dependencies are installed.
- Blender import says “Blender is required”: Run the printed command that starts with `blender -b -P ... --`.
- Dataset shape mismatch: Ensure the segmentation aligns with the volume used; both should be on the same grid.

---

## License and Attribution

These tools integrate DICOM conversion, dataset building, and Blender viewing concepts inspired by the referenced repositories, adapted to work natively within TotalSegmentator’s CLI and file conventions.
6) Interactive MPR (sliders)
- In Jupyter: `%run -m totalsegmentator.bin.totalseg_mpr_widget --dicom <dicom_dir>`
- As a window: `totalseg_mpr_widget --dicom <dicom_dir>`
  - Adjust WL (l), WW (k), slice index, plane and export.
  - Install extras: `pip install ipywidgets matplotlib scipy`
