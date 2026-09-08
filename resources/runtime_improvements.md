Measured on one CT image with shape **512 x 512 x 807** and spacing **0.82 x 0.82 x 1.00 mm**.

Hardware:

- GPU: NVIDIA GeForce RTX 3090 (24 GB)
- CPU: Intel Core i9-9900X @ 3.50 GHz (10 cores / 20 threads)
- RAM: 125 GB

## Runtime impact of in-memory inference

Changes: https://github.com/wasserth/TotalSegmentator/commit/25d0dbc31aa062be5850898b4560bf57f40eb93f

Optimizations:

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
