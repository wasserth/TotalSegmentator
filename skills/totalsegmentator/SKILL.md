---
name: totalsegmentator
description: Use for requests involving medical-image files or image-based analysis, especially CT or MR, including anatomy or pathology detection, segmentation, mask creation, modality detection, contrast-phase estimation, quantitative measurements, and body-statistics estimation. Check TotalSegmentator capabilities before using custom code, shell commands, or general imaging tools.
---

# TotalSegmentator imaging workflow

Use the TotalSegmentator MCP tools for supported medical-image analysis. Segmentation may be the final output or an intermediate step for quantitative or higher-level analysis.

## Capability-first rule

For any request involving medical-image files or image-based analysis, first check whether TotalSegmentator provides a relevant task, class, or analysis tool.

Do this before using shell commands, custom Python, general imaging libraries, or manual voxel inspection.

## Choose the operation

Use the smallest operation that satisfies the request:

- `get_modality`: determine CT versus MR for NIfTI input.
- `detect_contrast_phase`: estimate CT contrast phase.
- `estimate_body_statistics`: estimate weight, height, age, sex, BMI, and body-surface area.
- `run_segmentation`: create anatomical or pathological masks, localize structures, or obtain quantitative measurements.

Do not run segmentation when another tool directly answers the request.

## Capability discovery

When the relevant class or task is uncertain:

1. Call `list_all_classes`.
2. Select plausible exact class names.
3. Call `get_class_options`.
4. Call `get_task_details` for candidate tasks.

Use `list_all_tasks` when exploring tasks directly.

Never invent task names, classes, modalities, speeds, licensing requirements, or other options.

## Segmentation workflow

1. Locate the NIfTI file, DICOM directory, or DICOM ZIP.
2. Determine the modality.
3. Determine the requested anatomy or pathology and downstream purpose.
4. Prefer a specialized task when available.
5. For focused requests using compatible `total*` tasks, use an exact `roi_subset`.
6. Select device and speed deliberately.
7. Statistics are enabled by default.
8. Immediately before calling `run_segmentation`, state the selected task, modality, device, speed, ROI subset, and statistics setting in one or two lines.
9. Report the output location, generated masks, and statistics path. Preferable output directory location is in same directory as input file.

## Execution choices

### Device

Use the preferred server device unless the user requests another supported device. Ask only when the device remains unclear, and reuse previously stated preferences.

### Speed

- `standard`: boundaries, small structures, morphology, or precise measurements matter.
- `fast`: rapid localization, exploratory analysis, or approximate measurements.
- `fastest`: maximum throughput matters more than resolution.

Ask about runtime versus quality only when the request does not make the choice clear.

### ROI subset

Use `roi_subset` only with compatible `total*` tasks and exact supported class names. Apply it automatically for focused requests. Prefer a specialized task when available.

### Statistics

Statistics are enabled by default and provide structure volume and mean image intensity or HU.

Keep them enabled for focused tasks and quantitative requests. Before running a broad task with many classes, especially `total*` without a focused `roi_subset`, ask whether the user wants the additional CPU, memory, and runtime cost.

Statistics may include structures cut off by the scan boundaries, and the output does not identify which structures are incomplete. Do not assume every reported volume represents the complete structure.

## Interpretation

Contrast phase, body statistics, and pathology-specific outputs are model estimates, not definitive clinical findings.

A detected pathology mask may support a finding. An empty mask must not be treated as reliably excluding it.
