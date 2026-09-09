# Torch resampling ([PR #606](https://github.com/wasserth/TotalSegmentator/pull/606)) on eiger

Measured on `/home/jakob/Downloads/nnunet_test/ct.nii.gz`: shape **512 x 512 x 807**, spacing **0.818 x 0.818 x 1.00 mm**.

This is the forward **image** resample only (`change_spacing` / `resample_img` vs `resample_img_torch`). Label resampling is unchanged by the PR.

## Hardware (eiger)

- GPU: NVIDIA GeForce RTX 3090 (24 GB), CUDA device `cuda:0`
- CPU: Intel Core i9-9900X @ 3.50 GHz (10 cores / 20 threads)
- RAM: 125 GB
- PyTorch 2.4.1+cu124, scipy 1.14.1, numpy 2.0.1

The GTX 1050 Ti in this machine was not used.

## Method

- **Default** target spacing: 1.5 mm (output **279 x 279 x 538**)
- **`--fast`** target spacing: 3.0 mm (output **140 x 140 x 269**)
- scipy path: `resample_img(..., nr_cpus=1)` (TotalSegmentator default)
- torch path: `resample_img_torch` on `cuda:0` or `cpu`
- 1 warmup + 3 timed runs; table values are the **median**
- Input already in RAM as C-contiguous float64 (same isolation as the PR). NIfTI decode is excluded.

Default was timed on GPU only. `--fast` was timed on GPU and CPU. Both interpolation orders used by TotalSegmentator were timed (`-ro 1` default, `-ro 3` / `-ho`).

PR tests: `tests/test_resampling.py` → **15 passed**, 2 skipped (MPS not present).

## Isolated resample (kernel)

| Mode | Order | scipy (1 CPU) | torch CUDA | CUDA speedup | torch CPU | CPU speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Default (1.5 mm) | 1 | 1.773 s | 0.373 s | **4.8x** | — | — |
| Default (1.5 mm) | 3 | 20.000 s | 0.385 s | **52.0x** | — | — |
| `--fast` (3.0 mm) | 1 | 0.238 s | 0.263 s | 0.91x (slower) | 0.402 s | 0.59x (slower) |
| `--fast` (3.0 mm) | 3 | 8.796 s | 0.261 s | **33.7x** | 0.501 s | **17.6x** |

Torch runtime is almost the same for order 1 and order 3. Scipy order 3 is about **11x** slower than order 1 on this CT (default: 20.0 s vs 1.77 s). That is the case where `--torch-resample` pays off.

`--fast` + order 1 is already cheap on scipy (~0.24 s). Host-to-device copy then dominates, so torch is slightly slower.

## Agreement vs `ndimage.zoom`

On this CT (HU intensities), torch vs scipy:

| Device | Order | max abs diff | mean abs diff | max relative |
| --- | ---: | ---: | ---: | ---: |
| CUDA | 1 | 5.7e-4 HU | 2.7e-5 HU | 3.5e-5 |
| CUDA | 3 | 1.3e-3 HU | 7.9e-5 HU | 7.8e-5 |
| CPU | 1 | 9.1e-13 | 5.3e-14 | — |
| CPU | 3 | 8.2e-12 | 3.0e-13 | — |

CPU matches scipy to float64. GPU residual is float32 arithmetic, far below CT quantization.

## `change_spacing` wrapper (as TotalSegmentator calls it)

`change_spacing` still does `img.get_fdata()` (float64, ~1.69 GB here). The torch backend then copies that to a C-contiguous array (`np.ascontiguousarray` ≈ **2.4 s** on eiger) before the GPU/CPU matmuls. That constant is paid even when the resample itself is 0.3 s.

| Mode | Order | scipy `change_spacing` | torch `change_spacing` | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Default GPU | 1 | 7.38 s | 2.86 s | 2.6x |
| Default GPU | 3 | 18.35 s | 2.86 s | 6.4x |
| `--fast` GPU | 1 | 1.12 s | 2.72 s | 0.41x (slower) |
| `--fast` GPU | 3 | 8.62 s | 2.72 s | 3.2x |
| `--fast` CPU | 1 | 1.12 s | 2.90 s | 0.39x (slower) |
| `--fast` CPU | 3 | 8.62 s | 2.96 s | 2.9x |

For a full TotalSegmentator run the rest of the pipeline (inference, inverse label resample) is unchanged. Default GPU wall time is ~105 s, so shaving ~5–17 s off the forward image resample is noticeable but not a 50x end-to-end gain. Order 3 / `-ho` is where the fraction of runtime spent in this step is large.

## PR notes

- Approach is sound: it applies scipy’s own 1-D zoom operators as matmuls, rather than reimplementing `ndimage.zoom`.
- Scope is correct: intensity data at `order > 0` only. Cascade labels and inverse label resampling keep the existing path.
- Device index is preserved (`cuda:N`), so a multi-GPU box resamples on the same device as inference.
- Opt-in flag (`--torch-resample`) is the right default until the host-copy overhead is reduced.
- On this machine the flag is a clear win for **default + order 3** and **`--fast` + order 3**. It is a small loss for **`--fast` + order 1**.
