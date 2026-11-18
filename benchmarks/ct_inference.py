"""Compare total CT inference on a pre-downloaded model with the same in-memory NIfTI."""

import argparse
import gc
import json
from statistics import median
from time import perf_counter

import nibabel as nib
import numpy as np
import torch

from totalsegmentator.optimized_ct import CTSession
from totalsegmentator.python_api import totalsegmentator


def measure(image, mode, args):
    device = args.device
    session = False if mode == "stock" else CTSession()
    elapsed, peaks = [], []
    for _ in range(args.repeats + 1):
        if device == "gpu":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        elif device == "mps":
            torch.mps.synchronize()
        start = perf_counter()
        result = totalsegmentator(
            image,
            None,
            ml=True,
            nr_thr_saving=1,
            nr_thr_resamp=1,
            device=device,
            fast=args.fast,
            fastest=args.fastest,
            task=args.task,
            model_size=args.model_size,
            optimized_ct=session,
            quiet=True,
        )
        labels = np.asarray(result.dataobj)
        if device == "gpu":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        elapsed.append(perf_counter() - start)
        if device == "gpu":
            peaks.append(torch.cuda.max_memory_allocated() / 2**20)
    return (
        labels,
        result.affine,
        {
            "first_s": elapsed[0],
            "warm_median_s": median(elapsed[1:]),
            "peak_torch_mib": max(peaks, default=None),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ct")
    parser.add_argument("--task", choices=["total", "total_v3"], default="total")
    parser.add_argument("--model-size", choices=["big", "small"], default="big")
    resolution = parser.add_mutually_exclusive_group()
    resolution.add_argument("--fast", action="store_true")
    resolution.add_argument("--fastest", action="store_true")
    parser.add_argument("--device", choices=["gpu", "mps", "cpu"], default="gpu")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if (
        args.repeats < 1
        or (args.device == "gpu" and not torch.cuda.is_available())
        or (args.device == "mps" and not torch.backends.mps.is_available())
    ):
        parser.error("The selected device must be available and repeats must be positive.")
    image = nib.Nifti1Image.load(args.ct)
    image = nib.Nifti1Image(image.get_fdata(), image.affine)
    config = {
        "device": args.device,
        "task": args.task,
        "model_size": args.model_size,
        "spacing_mm": 6 if args.fastest else 3 if args.fast else 1.5,
        "warm_calls": args.repeats,
    }
    reference, affine, baseline = measure(image, "stock", args)
    print(json.dumps({**config, "mode": "stock", **baseline}), flush=True)
    mode = "torch"
    gc.collect()
    if args.device == "gpu":
        torch.cuda.empty_cache()
    elif args.device == "mps":
        torch.mps.empty_cache()
    labels, grid, row = measure(image, mode, args)
    if reference.shape != labels.shape or not np.allclose(affine, grid):
        raise RuntimeError("The two segmentations use different output grids.")
    counts = np.bincount(reference.ravel(), minlength=256) + np.bincount(labels.ravel(), minlength=256)
    hits = np.bincount(reference[reference == labels], minlength=256)
    row["dice_by_label"] = {int(i): float(2 * hits[i] / counts[i]) for i in np.flatnonzero(counts) if i}
    row["voxel_agreement"] = float(np.mean(labels == reference))
    row["warm_speedup"] = baseline["warm_median_s"] / row["warm_median_s"]
    print(json.dumps({**config, "mode": mode, **row}), flush=True)


if __name__ == "__main__":
    main()
