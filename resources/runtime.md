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
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -ho`               | 309 s       | 1039 s |       |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -ms small -ho`     | 312 s       | 679 s  |       |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -f -ho`            | 68 s        | 95 s   |       |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -f -ms small -ho`  | 70 s        | 84 s   |       |




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
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -ho`               | 10.3 GB              | 7.2 GB   | 10.0 GB              |                    |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -ms small -ho`     | 10.1 GB              | 8.1 GB   | 9.6 GB               |                    |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -f -ho`            | 7.9 GB               | 5.0 GB   | 7.9 GB               |                    |
| `TotalSegmentator -i ct.nii.gz -o seg.nii.gz -ml -f -ms small -ho`  | 7.9 GB               | 2.9 GB   | 7.8 GB               |                    |



## Summary
- 


## Notes

- `--model_size small` (`-ms small`) barely changes GPU runtime (pre/postprocessing dominates). On CPU it is about **2.8x faster** than the default `total` model.
- `--fast` (`-f`) uses the 3 mm model and cuts runtime a lot on all devices.
- `--save_lowres` (`-sl`) skips resampling the segmentation back to the input resolution. Use this if you do not need the segmentation in the same resolution as the input image.
- `total_highres` uses much more RAM, GPU memory and time. On this image it nearly filled the 24 GB GPU. Do not run it on large CTs, and do not run it on CPU.

