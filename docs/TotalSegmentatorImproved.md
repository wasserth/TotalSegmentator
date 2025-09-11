# TotalSegmentatorImproved

An enhanced version of TotalSegmentator that provides organized outputs with task titles, smoothing options, and Blender compatibility.

## Features

- **Organized Output Structure**: Clear task titles and result mappings
- **Automatic Output Renaming**: Maps internal names to user-friendly names
- **Smoothing Options**: Improves 3D visualization in Slicer and Blender
- **STL Export**: Direct export to Blender-compatible format
- **Comprehensive Reporting**: JSON summaries of all processing

## Supported Tasks

### 1. Liver Segments
- **Title**: "liver: segments"
- **Results**: liver_segment_1 through liver_segment_8
- **Task**: Uses the `liver_segments` model

### 2. Liver Vessels
- **Title**: "liver: vessels"  
- **Results**: 
  - `blood_vessel` (mapped from `liver_vessels`)
  - `neoplasm` (mapped from `liver_tumor`)
- **Task**: Uses the `liver_vessels` model

### 3. Total Vessels
- **Title**: "total"
- **Results**:
  - `inferior_vena_cava`
  - `portal_vein_and_splenic_vein`
- **Task**: Uses the `total` model with ROI subset

## Installation

The improved CLI is included when you install TotalSegmentator:

```bash
pip install TotalSegmentator
```

For enhanced features (smoothing and STL export):

```bash
pip install TotalSegmentator[enhanced]
```

## Usage

### Basic Usage

Run all segmentation tasks:

```bash
TotalSegmentatorImproved -i input.nii.gz -o output_directory
```

### Task-Specific Usage

Run specific tasks:

```bash
# Liver segments only
TotalSegmentatorImproved -i input.nii.gz -o output_directory --tasks liver_segments

# Liver vessels only
TotalSegmentatorImproved -i input.nii.gz -o output_directory --tasks liver_vessels

# Multiple specific tasks
TotalSegmentatorImproved -i input.nii.gz -o output_directory --tasks liver_segments liver_vessels
```

### Smoothing Options

Apply different smoothing levels for better 3D visualization:

```bash
# Light smoothing (good for detailed analysis)
TotalSegmentatorImproved -i input.nii.gz -o output_directory --smoothing light

# Medium smoothing (balanced, default)
TotalSegmentatorImproved -i input.nii.gz -o output_directory --smoothing medium

# Heavy smoothing (best for visualization)
TotalSegmentatorImproved -i input.nii.gz -o output_directory --smoothing heavy

# No smoothing
TotalSegmentatorImproved -i input.nii.gz -o output_directory --smoothing none
```

### STL Export for Blender

Export segmentations directly to STL format:

```bash
TotalSegmentatorImproved -i input.nii.gz -o output_directory --export-stl
```

### Performance Options

```bash
# Use GPU (if available)
TotalSegmentatorImproved -i input.nii.gz -o output_directory --device cuda

# Use robust cropping for better accuracy
TotalSegmentatorImproved -i input.nii.gz -o output_directory --robust-crop
```

## Output Structure

The improved CLI creates an organized output structure:

```
output_directory/
├── overall_summary.json          # Complete processing summary
├── liver_segments/               # Liver segments task results
│   ├── liver_segment_1.nii.gz
│   ├── liver_segment_2.nii.gz
│   ├── ...
│   ├── liver_segment_8.nii.gz
│   └── task_summary.json
├── liver_vessels/                # Liver vessels task results  
│   ├── blood_vessel.nii.gz       # (renamed from liver_vessels.nii.gz)
│   ├── neoplasm.nii.gz           # (renamed from liver_tumor.nii.gz)
│   ├── blood_vessel.stl          # (if --export-stl used)
│   ├── neoplasm.stl
│   └── task_summary.json
└── total_vessels/                # Total vessels task results
    ├── inferior_vena_cava.nii.gz
    ├── portal_vein_and_splenic_vein.nii.gz
    └── task_summary.json
```

## Using Results in 3D Slicer

1. Load your original CT image in 3D Slicer
2. Load the smoothed segmentation files as Label Maps
3. Import into Segmentations module
4. Enable 3D display for better visualization
5. Use the Export function to create STL files for Blender (or use the --export-stl option)

## Using Results in Blender

1. Use the `--export-stl` option to generate STL files directly
2. In Blender: File → Import → STL
3. The smoothed segmentations will provide better mesh quality

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-i, --input` | Input CT NIfTI file | Required |
| `-o, --output` | Output directory | Required |
| `--tasks` | Tasks to run: liver_segments, liver_vessels, total_vessels, all | all |
| `--smoothing` | Smoothing level: none, light, medium, heavy | medium |
| `--export-stl` | Export to STL format for Blender | False |
| `--device` | Device: auto, cpu, cuda, etc. | auto |
| `--robust-crop` | Use robust cropping | False |

## Summary Files

### Task Summary (task_summary.json)
Contains information about each task:
- Task title and name
- Expected results
- Processed files
- Smoothing settings
- Processing metadata

### Overall Summary (overall_summary.json)
Contains complete processing information:
- Input file details
- All task results
- Configuration used
- Task definitions reference

## Troubleshooting

### Dependencies Not Available
If you see warnings about missing dependencies:
- For basic functionality: `pip install numpy nibabel`
- For smoothing: `pip install scipy scikit-image`  
- For STL export: `pip install trimesh`

### Performance Issues
- Use `--device cuda` for GPU acceleration
- Use `--robust-crop` for better accuracy (slower)
- Use `--smoothing none` to skip smoothing processing

### Memory Issues
- Process tasks individually instead of all at once
- Use lower resolution models with the original TotalSegmentator CLI first

## Examples

### Example 1: Complete Processing for 3D Slicer

```bash
TotalSegmentatorImproved \
  -i patient_ct.nii.gz \
  -o results \
  --smoothing medium \
  --robust-crop \
  --device cuda
```

### Example 2: Liver Analysis Only

```bash
TotalSegmentatorImproved \
  -i liver_ct.nii.gz \
  -o liver_results \
  --tasks liver_segments liver_vessels \
  --smoothing heavy \
  --export-stl
```

### Example 3: Vascular Analysis

```bash
TotalSegmentatorImproved \
  -i vascular_ct.nii.gz \
  -o vascular_results \
  --tasks liver_vessels total_vessels \
  --smoothing light
```

This improved CLI provides a streamlined workflow for getting organized, smoothed segmentations ready for 3D visualization and analysis.