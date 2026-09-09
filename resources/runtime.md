# Runtime and memory measurements (v3.0.0)

Measured on one CT image with shape **512 x 512 x 807** and spacing **0.82 x 0.82 x 1.00 mm**.

Hardware setup:

GPU and CPU:

- GPU: NVIDIA GeForce RTX 3090 (24 GB)
- CPU: Intel Core i9-9900X @ 3.50 GHz (10 cores / 20 threads)
- RAM: 125 GB

MPS:

- Apple M2 Pro with 32 GB unified memory;
- macOS 15.7.4;
- PyTorch 2.12.0 using the `mps` device.

## Runtime by device


| Command                                                             | GPU         | CPU    | MPS   |
| ------------------------------------------------------------------- | ----------- | ------ | ----- |
| **Default**                                                         |             |        |       |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml`                   | 105 s       | 872 s  | 443 s |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -ms small`         | 95 s        | 313 s  | 133 s |
| **Fast mode (`-f`)**                                                |             |        |       |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -f`                | 25 s        | 57 s   | 26 s  |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -f -ms small`      | 25 s        | 43 s   | 16 s  |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -f -ms small -sl`  | 17 s        | 34 s   | 15 s  |
| **High-resolution (`-ta total_highres`)**                           |             |        |       |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -ta total_highres` | 11 min 11 s |        |       |
| **Higher-order resampling (`-ho`)**                                 |             |        |       |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -ho`               | 309 s       | 1039 s | 607 s |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -ms small -ho`     | 312 s       | 679 s  | 303 s |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -f -ho`            | 68 s        | 95 s   | 59 s  |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -f -ms small -ho`  | 70 s        | 84 s   | 50 s  |



## Memory by device

Linux RAM is the peak process-tree PSS. MPS is the peak process-tree physical
footprint and includes both CPU and GPU allocations in unified memory, so it is
not directly comparable with the Linux RAM columns.


| Command                                                             | System RAM (GPU run) | GPU VRAM | System RAM (CPU run) | MPS unified memory |
| ------------------------------------------------------------------- | -------------------- | -------- | -------------------- | ------------------ |
| **Default**                                                         |                      |          |                      |                    |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml`                   | 8.6 GB               | 7.1 GB   | 8.8 GB               | 12.2 GB            |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -ms small`         | 8.5 GB               | 8.1 GB   | 9.1 GB               | 11.9 GB            |
| **Fast mode (`-f`)**                                                |                      |          |                      |                    |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -f`                | 5.6 GB               | 3.4 GB   | 6.5 GB               | 10.0 GB            |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -f -ms small`      | 5.6 GB               | 2.9 GB   | 6.6 GB               | 9.1 GB             |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -f -ms small -sl`  | 5.6 GB               | 2.9 GB   | 6.7 GB               | 9.9 GB             |
| **High-resolution (`-ta total_highres`)**                           |                      |          |                      |                    |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -ta total_highres` | 31.5 GB              | 23.4 GB  |                      |                    |
| **Higher-order resampling (`-ho`)**                                 |                      |          |                      |                    |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -ho`               | 10.3 GB              | 7.2 GB   | 10.0 GB              | 11.6 GB            |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -ms small -ho`     | 10.1 GB              | 8.1 GB   | 9.6 GB               | 12.9 GB            |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -f -ho`            | 7.9 GB               | 5.0 GB   | 7.9 GB               | 8.7 GB             |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -f -ms small -ho`  | 7.9 GB               | 2.9 GB   | 7.8 GB               | 9.1 GB             |



## Real life example: Get lungs in Chest CT

A more typical case: a chest CT (**512 x 512 x 139**, spacing **0.76 x 0.76 x 2.50 mm**)
and only the five lung-lobe classes. The first three commands also used

`--roi_subset lung_upper_lobe_left lung_lower_lobe_left lung_upper_lobe_right lung_middle_lobe_right lung_lower_lobe_right`.

`total_highres` was run without `--roi_subset` (it always body-crops, then predicts all classes).


| Command                                                             | GPU   | CPU    | MPS   |
| ------------------------------------------------------------------- | ----- | ------ | ----- |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml`                   | 23 s  | 37 s   | 21 s  |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -ms small`         | 25 s  | 29 s   | 14 s  |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -f -ho`            | 31 s  | 27 s   | 16 s  |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -ta total_highres` | 191 s | 1662 s | 847 s |


| Command                                                             | System RAM (GPU run) | GPU VRAM | System RAM (CPU run) | MPS unified memory |
| ------------------------------------------------------------------- | -------------------- | -------- | -------------------- | ------------------ |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml`                   | 3.8 GB               | 2.4 GB   | 4.7 GB               | 6.5 GB             |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -ms small`         | 3.6 GB               | 4.7 GB   | 5.1 GB               | 7.3 GB             |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -f -ho`            | 4.0 GB               | 1.7 GB   | 4.7 GB               | 6.2 GB             |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -ta total_highres` | 12.2 GB              | 9.3 GB   | 14.5 GB              | 18.5 GB            |



## Summary

- If you have a GPU: use default mode
- If you have only CPU or MPS (Mac): use `--model_size small` or if you need even faster use `--fast`
- If you do not need all classes, use `--roi_subset`. This makes a huge difference in runtime.
- For smoother outlies higher-order resampling (`-ho`) makes a big difference. But it comes with a high runtime. (if you use `--roi_subset` it becomes a lot faster)
- `total_highres` uses much more RAM, GPU memory and time. Do not run it on large CTs, and do not run it on CPU/MPS. In 99% of the cases default mode + `-ho`is sufficient for your needs.
- `--save_lowres` (`-sl`) skips resampling the segmentation back to the input resolution. Use this if you do not need the segmentation in the same resolution as the input image.

