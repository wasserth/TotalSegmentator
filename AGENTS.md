# Using TotalSegmentator from automation and AI coding agents

This file is for scripts, pipelines and AI coding agents (e.g. Codex, Claude Code) that drive
TotalSegmentator non-interactively. It documents how to discover the tool's capabilities,
invoke it reproducibly and parse its results without scraping human-readable stdout.

For the full feature documentation see [README.md](README.md).

## TL;DR

```bash
# 1. Discover what the tool can do (instant, no GPU, no model download):
totalseg_info --json                       # full capability registry as JSON

# 2. Run a segmentation and capture a machine-readable run report:
TotalSegmentator -i ct.nii.gz -o seg/ --report seg/run_report.json

# 3. Read seg/run_report.json to learn versions, device, classes and output files.
```

## Discovering capabilities

Do not hard-code task names or class names from the source. Query them instead — the data is
generated from the same registry the CLI validates against, so it never drifts.

- `totalseg_info --list-tasks` — table of every task (modality CT/MR, whether a license is
  required, number of classes).
- `totalseg_info --classes -ta <task>` — the `index -> class_name` map a task outputs. The
  class names are exactly what `--roi_subset` expects.
- `totalseg_info --json` — full registry as JSON: `{ "totalsegmentator_version", "tasks": {
  <task>: { "modality", "license_required", "classes": { "<index>": "<name>" } } } }`.
- Add `--json` to `--list-tasks` or `--classes` to scope the JSON to just that view.

`totalseg_info` imports no heavy dependencies, needs no GPU and downloads no weights, so it is
safe to call at planning time to decide which task/classes to request.

The same lists are available on the main command (`TotalSegmentator --list-tasks`,
`TotalSegmentator --list-classes [task]`), but those load the full runtime; prefer
`totalseg_info` for quick discovery.

## Running a segmentation

```bash
TotalSegmentator -i <input> -o <output> -ta <task> [options]
```

- `-i` accepts a NIfTI file (`.nii` / `.nii.gz`), a folder of DICOM slices, or a zip of DICOM
  slices. `-o` is a directory of per-class masks, or a single file when `--ml` (multilabel),
  `dicom_seg` or `dicom_rtstruct` output is used.
- `-d cpu|gpu|gpu:N|mps` selects the device. Without a GPU the run falls back to CPU (slow);
  `--fast` (3mm) or `--roi_subset <classes>` reduce runtime and memory.
- Some tasks require a license (see `license_required` in the registry). Set it once with
  `totalseg_set_license -l <key>`; free non-commercial licenses:
  https://backend.totalsegmentator.com/license-academic/

## Reading the run report

Pass `--report <path.json>` to write a manifest after the run completes. It contains:

| field | meaning |
|-------|---------|
| `totalsegmentator_version`, `nnunetv2_version`, `torch_version` | software versions used |
| `task`, `modality`, `license_required` | what was run |
| `device` | resolved device (`gpu` / `cpu` / `mps`) |
| `fast`, `fastest`, `save_lowres`, `multilabel`, `output_type`, `roi_subset` | run options |
| `input`, `output` | resolved paths (or `"Nifti1Image"` for an in-memory input) |
| `num_classes`, `classes` | classes produced (`index -> name`), filtered by `roi_subset` |
| `runtime_seconds` | wall-clock segmentation time |
| `output_files` | `*.nii.gz` files written to the output directory |

This lets a pipeline verify a run and chain the next step (e.g. feed `output_files` into
`totalseg_combine_masks` or statistics) without parsing stdout.

For per-class volume and intensity, add `--statistics` (writes `statistics.json`).
Add `--statistics_extra` for `n_voxels`, intensity std/min/max and the morphometric
`centroid_vox` / `bbox_vox` (voxel coordinates).

## Exit codes

- `0` — success.
- `1` — runtime error (e.g. a licensed task without a valid license; any unhandled exception).
- `2` — argument error (unknown task, missing `-i`/`-o` for a segmentation run, etc.).

Add `--quiet` to suppress progress text and `--debug` to print the input path and task on
error. Stdout is intended for humans; rely on `--report` and `--statistics` JSON for parsing.

## Python API

```python
from totalsegmentator.python_api import totalsegmentator
seg_img = totalsegmentator("ct.nii.gz", "seg/", task="total")
```

The task registry is importable directly (no torch required):

```python
from totalsegmentator.registry import list_tasks, get_task_classes, task_registry
```
