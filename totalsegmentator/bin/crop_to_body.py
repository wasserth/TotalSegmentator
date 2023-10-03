#!/usr/bin/env python
import sys
import os
import argparse
from pkg_resources import require
from pathlib import Path
import time

import numpy as np
import nibabel as nib
import torch

from totalsegmentator.libs import download_pretrained_weights, combine_masks
from totalsegmentator.config import setup_nnunet
from totalsegmentator.cropping import crop_to_mask, undo_crop


def main():
    parser = argparse.ArgumentParser(description="Crop input image to body.",
                                     epilog="Written by Jakob Wasserthal. If you use this tool please cite https://pubs.rsna.org/doi/10.1148/ryai.230024")

    parser.add_argument("-i", metavar="filepath", dest="input",
                        help="CT nifti image", 
                        type=lambda p: Path(p).absolute(), required=True)

    parser.add_argument("-o", metavar="filepath", dest="output",
                        help="Cropped nifti image", 
                        type=lambda p: Path(p).absolute(), required=True)

    parser.add_argument("-t", "--only_trunc", action="store_true", help="Crop to trunc instead of entire body.",
                        default=False)

    parser.add_argument("-nr", "--nr_thr_resamp", type=int, help="Nr of threads for resampling", default=1)

    parser.add_argument("-ns", "--nr_thr_saving", type=int, help="Nr of threads for saving segmentations", 
                        default=6)
    
    parser.add_argument("-d", "--device", choices=["gpu", "cpu"], help="Device to run on (default: gpu).",
                        default="gpu")

    parser.add_argument("-q", "--quiet", action="store_true", help="Print no intermediate outputs",
                        default=False)

    parser.add_argument("-v", "--verbose", action="store_true", help="Show more intermediate output",
                        default=False)

    args = parser.parse_args()

    quiet, verbose = args.quiet, args.verbose

    device = "cuda" if args.device == "gpu" else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("No GPU detected. Running on CPU.")
        device = "cpu"

    setup_nnunet()

    from totalsegmentator.nnunet import nnUNet_predict_image  # this has to be after setting new env vars

    crop_addon = [3, 3, 3]  # default value
    download_pretrained_weights(300)

    st = time.time()
    if not quiet: print("Generating rough body segmentation...")
    body_seg, _ = nnUNet_predict_image(args.input, None, 300, model="3d_fullres", folds=[0],
                        trainer="nnUNetTrainer", tta=False, multilabel_image=True, resample=6.0,
                        crop=None, crop_path=None, task_name="body", nora_tag="None", preview=False, 
                        save_binary=False, nr_threads_resampling=args.nr_thr_resamp, nr_threads_saving=1, 
                        crop_addon=crop_addon, quiet=quiet, verbose=verbose, test=0, device=device)
    if verbose: print(f"Rough body segmentation generated in {time.time()-st:.2f}s")

    body_seg_data = body_seg.get_fdata()
    if args.only_trunc:
        body_seg_data = body_seg_data == 1
    else:
        body_seg_data = body_seg_data > 0.5
    body_seg = nib.Nifti1Image(body_seg_data.astype(np.uint8), body_seg.affine)

    img_in = nib.load(args.input)
    img_out, bbox = crop_to_mask(img_in, body_seg, addon=crop_addon, dtype=np.int32,
                                verbose=verbose)
    if not quiet:
        print(f"  cropping from {img_in.shape} to {img_out.shape}")

    nib.save(img_out, args.output)


if __name__ == "__main__":
    main()
