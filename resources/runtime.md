# Runtime and memory measurements (v3.0.0)

Measured on one CT image with shape **512 x 512 x 807** and spacing **0.82 x 0.82 x 1.00 mm**.

Hardware:

- GPU: NVIDIA GeForce RTX 3090 (24 GB)
- CPU: Intel Core i9-9900X @ 3.50 GHz (10 cores / 20 threads)
- RAM: 125 GB


## GPU


| Command                                                                    | Runtime     | RAM     | GPU memory |
| -------------------------------------------------------------------------- | ----------- | ------- | ---------- |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -d gpu`                   | 2 min 35 s  | 12.0 GB | 7.6 GB     |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -ms small -d gpu`         | 2 min 37 s  | 12.0 GB | 6.8 GB     |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -f -d gpu`                | 54 s        | 8.1 GB  | 3.4 GB     |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -f -ms small -d gpu`      | 38 s        | 8.2 GB  | 2.8 GB     |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -f -ms small -sl -d gpu`  | 26 s        | 8.2 GB  | 2.9 GB     |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -ta total_highres -d gpu` | 16 min 44 s | 37.4 GB | 23.0 GB    |


## CPU


| Command                                                                   | Runtime     | RAM     |
| ------------------------------------------------------------------------- | ----------- | ------- |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -d cpu`                  | 14 min 45 s | 11.3 GB |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -ms small -d cpu`        | 5 min 55 s  | 11.4 GB |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -f -d cpu`               | 1 min 3 s   | 7.9 GB  |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -f -ms small -d cpu`     | 50 s        | 8.1 GB  |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -f -ms small -sl -d cpu` | 40 s        | 7.5 GB  |


## Notes

- `--model_size small` (`-ms small`) barely changes GPU runtime (pre/postprocessing dominates). On CPU it is about **2.5x faster** than the default `total` model (14 min 45 s → 5 min 55 s).
- `--fast` (`-f`) uses the 3 mm model and cuts runtime a lot on both devices.
- `--save_lowres` (`-sl`) skips resampling the segmentation back to the input resolution.
- `total_highres` uses much more RAM, GPU memory and time. On this image it nearly filled the 24 GB GPU. Do not run it on large CTs, and do not run it on CPU.



## Runtime impact of in-memory inference

The optimized inference path:

- passes resampled inputs directly to nnU-Net in memory instead of writing and
  rereading temporary NIfTI files;
- returns predictions in memory, avoiding background export processes and
  intermediate prediction files;
- loads the first model checkpoint in parallel with input resampling;
- retains at most one preprocessed volume, reusing it only when consecutive
  models have identical preprocessing and normalization settings;
- maps and combines multi-model predictions in memory;
- splits and rejoins large images in memory;
- avoids an eager full-volume input copy and keeps nearest-neighbor label maps,
  final output, and statistics inputs in compact native dtypes where safe;
- keeps the file-based path for probability export and test/reference modes.

Before and after measurements use the same input, hardware, command, and
resource monitor.

### Runtime reduction

| Command                                                                   | Before | After | Runtime reduction |
| ------------------------------------------------------------------------- | ------ | ----- | ----------------- |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -d gpu`                  | 155 s  | 75 s  | 52%               |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -f -d gpu`               | 54 s   | 24 s  | 56%               |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -d cpu`                  | 885 s  | 799 s | 10%               |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -f -d cpu`               | 63 s   | 52 s  | 18%               |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -f -ms small -sl -d cpu` | 40 s   | 32 s  | 21%               |

### RAM reduction

| Command                                                                   | Before | After | RAM reduction |
| ------------------------------------------------------------------------- | ------ | ----- | ------------- |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -d gpu`                  | 12.0 GB | 9.0 GB | 25%          |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -f -d gpu`               | 8.1 GB | 5.6 GB | 31%           |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -d cpu`                  | 11.3 GB | 8.9 GB | 21%          |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -f -d cpu`               | 7.9 GB | 6.2 GB | 22%           |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -f -ms small -sl -d cpu` | 7.5 GB | 6.6 GB | 12%           |

CPU outputs were voxel-identical before and after. GPU inference is
non-deterministic at a small number of boundary voxels; mean per-class Dice
against the before outputs was 0.99989 (default) and 0.99982 (`--fast`).




