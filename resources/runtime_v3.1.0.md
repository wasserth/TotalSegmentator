# Runtime and memory measurements (v3.1.0)

Measured with nnU-Net's streaming inference (TotalSegmentator commit `2640c03`, nnU-Net master `13128943`).
Default mode uses `-ta total_single` (the 117 classes of `total` in one model at 1.5 mm); `--fast` and `--fastest`
use the 3 mm and 6 mm `total` models. All runs predict all 117 classes (no `--roi_subset`), except the lung example.
For the measurements with the old inference pipeline (incl. `-ms small`, `-ho`, `-sl` and `total_highres`) see
[runtime_v3.0.1.md](runtime_v3.0.1.md).

Images:

- **Big CT**: shape **512 x 512 x 807**, spacing **0.82 x 0.82 x 1.00 mm**.
- **Chest CT**: shape **512 x 512 x 139**, spacing **0.76 x 0.76 x 2.50 mm**.

Hardware setup:

GPU and CPU:

- GPU: NVIDIA GeForce RTX 3090 (24 GB)
- CPU: Intel Core i9-9900X @ 3.50 GHz (10 cores / 20 threads)
- RAM: 125 GB
- PyTorch 2.4.1 (CUDA 12.4)

MPS:

- Apple M2 Pro with 32 GB unified memory;
- macOS 15.7.4;
- PyTorch 2.12.0 using the `mps` device.

Runtime is the wall-clock time of the whole command (incl. loading the image and the model and saving the output).

Linux RAM is the peak process-tree PSS. MPS is the peak process-tree physical
footprint and includes both CPU and GPU allocations in unified memory, so it is
not directly comparable with the Linux RAM columns.

## Big CT

### Runtime

| Command                                                         | GPU    | CPU    | MPS    |
| --------------------------------------------------------------- | ------ | ------ | ------ |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ta total_single`  | 14 s   | 280 s  | 143 s  |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz --fast`            | 12 s   | 108 s  | 36 s   |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz --fastest`         | 12 s   | 85 s   | 24 s   |

### Memory

| Command                                                         | System RAM (GPU run) | GPU VRAM | System RAM (CPU run) | MPS unified memory |
| --------------------------------------------------------------- | -------------------- | -------- | -------------------- | ------------------ |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ta total_single`  | 2.8 GB               | 6.2 GB   | 8.4 GB               | 25.2 GB            |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz --fast`            | 2.4 GB               | 1.7 GB   | 3.0 GB               | 3.7 GB             |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz --fastest`         | 2.4 GB               | 1.3 GB   | 2.6 GB               | 2.3 GB             |

## Chest CT

### Runtime

| Command                                                         | GPU    | CPU    | MPS    |
| --------------------------------------------------------------- | ------ | ------ | ------ |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ta total_single`  | 10 s   | 109 s  | 70 s   |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz --fast`            | 7 s    | 30 s   | 16 s   |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz --fastest`         | 7 s    | 17 s   | 7 s    |

### Memory

| Command                                                         | System RAM (GPU run) | GPU VRAM | System RAM (CPU run) | MPS unified memory |
| --------------------------------------------------------------- | -------------------- | -------- | -------------------- | ------------------ |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ta total_single`  | 1.7 GB               | 5.6 GB   | 7.9 GB               | 18.0 GB            |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz --fast`            | 1.3 GB               | 1.5 GB   | 2.3 GB               | 2.9 GB             |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz --fastest`         | 1.3 GB               | 0.5 GB   | 1.1 GB               | 1.6 GB             |

## Real life example: Get lungs in Chest CT

The same chest CT, but only the five lung-lobe classes. All commands also used

`--roi_subset lung_upper_lobe_left lung_lower_lobe_left lung_upper_lobe_right lung_middle_lobe_right lung_lower_lobe_right`.

### Runtime

| Command                                                         | GPU    | CPU    | MPS    |
| --------------------------------------------------------------- | ------ | ------ | ------ |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ta total_single`  | 11 s   | 68 s   | 42 s   |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz --fast`            | 8 s    | 26 s   | 10 s   |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz --fastest`         | 7 s    | 27 s   | 8 s    |

### Memory

| Command                                                         | System RAM (GPU run) | GPU VRAM | System RAM (CPU run) | MPS unified memory |
| --------------------------------------------------------------- | -------------------- | -------- | -------------------- | ------------------ |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ta total_single`  | 2.0 GB               | 5.4 GB   | 7.8 GB               | 13.0 GB            |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz --fast`            | 1.6 GB               | 1.4 GB   | 2.4 GB               | 3.2 GB             |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz --fastest`         | 1.4 GB               | 0.6 GB   | 1.4 GB               | 1.7 GB             |

## Summary

- If you have a GPU: use default mode (`total_single`). On a big CT it takes 14 s, barely more than `--fast`.
- If you have only CPU or MPS (Mac): default mode takes 2-5 min on a big CT; use `--fast` or `--fastest` if that is too slow.
- Default mode on MPS needs a lot of unified memory (18-25 GB); on Macs with 16 GB use `--fast`.
- If you do not need all classes, use `--roi_subset`. In default mode on CPU/MPS this cuts runtime by about 40% (chest CT); with `--fast`/`--fastest` and on GPU it makes little difference.

## Running the MPS measurements remotely

To access the Mac open a reverse tunnel to the Linux machine.

On the Mac:

```bash
ssh -N -o ExitOnForwardFailure=yes -R 2222:localhost:22 <user>@<linux-host-ip>
```

On the Linux machine:

```bash
ssh -p 2222 <user>@localhost
```
