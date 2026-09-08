# Runtime and memory measurements (v3.0.0)

Measured on one CT image with shape **512 x 512 x 807** and spacing **0.82 x 0.82 x 1.00 mm**.

Hardware:

- GPU: NVIDIA GeForce RTX 3090 (24 GB)
- CPU: Intel Core i9-9900X @ 3.50 GHz (10 cores / 20 threads)
- RAM: 125 GB

## GPU


| Command                                                                    | Runtime     | RAM     | GPU memory |
| -------------------------------------------------------------------------- | ----------- | ------- | ---------- |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -d gpu`                   | 1 min 45 s  | 8.6 GB  | 7.1 GB     |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -ms small -d gpu`         | 1 min 35 s  | 8.5 GB  | 8.1 GB     |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -f -d gpu`                | 25 s        | 5.6 GB  | 3.4 GB     |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -f -ms small -d gpu`      | 25 s        | 5.6 GB  | 2.9 GB     |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -f -ms small -sl -d gpu`  | 17 s        | 5.6 GB  | 2.9 GB     |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -ta total_highres -d gpu` | 11 min 11 s | 31.5 GB | 23.4 GB    |


## CPU


| Command                                                                   | Runtime     | RAM     |
| ------------------------------------------------------------------------- | ----------- | ------- |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -d cpu`                  | 14 min 32 s | 8.8 GB  |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -ms small -d cpu`        | 5 min 13 s  | 9.1 GB  |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -f -d cpu`               | 57 s        | 6.5 GB  |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -f -ms small -d cpu`     | 43 s        | 6.6 GB  |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -f -ms small -sl -d cpu` | 34 s        | 6.7 GB  |


## Notes

- `--model_size small` (`-ms small`) barely changes GPU runtime (pre/postprocessing dominates). On CPU it is about **2.8x faster** than the default `total` model (14 min 32 s → 5 min 13 s).
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
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -d gpu`                  | 155 s  | 105 s | 32%               |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -f -d gpu`               | 54 s   | 25 s  | 54%               |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -d cpu`                  | 885 s  | 872 s | 1%                |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -f -d cpu`               | 63 s   | 57 s  | 10%               |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -f -ms small -sl -d cpu` | 40 s   | 34 s  | 15%               |


### RAM reduction


| Command                                                                   | Before  | After  | RAM reduction |
| ------------------------------------------------------------------------- | ------- | ------ | ------------- |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -d gpu`                  | 12.0 GB | 8.6 GB | 28%           |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -f -d gpu`               | 8.1 GB  | 5.6 GB | 31%           |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -d cpu`                  | 11.3 GB | 8.8 GB | 22%           |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -f -d cpu`               | 7.9 GB  | 6.5 GB | 18%           |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -f -ms small -sl -d cpu` | 7.5 GB  | 6.7 GB | 11%           |


CPU outputs were voxel-identical before and after. GPU inference is
non-deterministic at a small number of boundary voxels; mean per-class Dice
against the before outputs was 0.99989 (default) and 0.99982 (`--fast`).

## Apple Silicon (MPS)

The same CT was also measured on:

- Apple M2 Pro with 32 GB unified memory;
- macOS 15.7.4;
- PyTorch 2.12.0 using the `mps` device.

The before and after runs used the parent and final runtime-improvement commits,
respectively, with the same environment and model weights. Runtime was measured
without a memory sampler because macOS process-footprint inspection noticeably
affects execution time.

### Runtime reduction


| Command                                                     | Before | After | Runtime reduction |
| ----------------------------------------------------------- | ------ | ----- | ----------------- |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -d mps`    | 490 s  | 443 s | 10%               |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -f -d mps` | 34 s   | 26 s  | 25%               |


### Peak physical memory

Peak physical memory was sampled every second for the complete process tree
using macOS `footprint`. Since MPS uses unified memory, this includes memory
used for both CPU and GPU processing and is not directly comparable with the
Linux PSS measurements above.


| Command                                                     | Before  | After   | Change       |
| ----------------------------------------------------------- | ------- | ------- | ------------ |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -d mps`    | 11.8 GB | 12.2 GB | 3% increase  |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -f -d mps` | 10.1 GB | 10.0 GB | 1% reduction |


Both MPS outputs were voxel-identical before and after the optimization.

## Optimized runtime by device

These are the runtimes after all improvements. GPU and CPU were measured on
the Linux workstation described above; MPS was measured on the Apple M2 Pro.


| Command                                                            | GPU  | CPU   | MPS   |
| ------------------------------------------------------------------ | ---- | ----- | ----- |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml`                  | 105 s | 872 s | 443 s |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -f`               | 25 s  | 57 s  | 26 s  |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -f -ms small -sl` | 17 s  | 34 s  | 15 s  |


## Optimized memory by device

Linux RAM is the peak process-tree PSS. MPS is the peak process-tree physical
footprint and includes both CPU and GPU allocations in unified memory, so it is
not directly comparable with the Linux RAM columns.


| Command                                                            | GPU RAM | GPU memory | CPU RAM | MPS unified memory |
| ------------------------------------------------------------------ | ------- | ---------- | ------- | ------------------ |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml`                  | 8.6 GB  | 7.1 GB     | 8.8 GB  | 12.2 GB            |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -f`               | 5.6 GB  | 3.4 GB     | 6.5 GB  | 10.0 GB            |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -f -ms small -sl` | 5.6 GB  | 2.9 GB     | 6.7 GB  | 9.9 GB             |


