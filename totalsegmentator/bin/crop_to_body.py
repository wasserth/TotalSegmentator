#!/usr/bin/env python
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

from totalsegmentator.bin.TotalSegmentator import RemovedOption
from totalsegmentator.predictor import TotalSegmentatorPredictor, bounding_box, class_ids, label_isin


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

    parser.add_argument("-nr", "--nr_thr_resamp", action=RemovedOption, nargs=1)

    parser.add_argument("-ns", "--nr_thr_saving", type=int, help="Nr of threads for saving segmentations",
                        default=6)

    parser.add_argument("-d", "--device", choices=["gpu", "cpu"], help="Device to run on (default: gpu).",
                        default="gpu")

    parser.add_argument("-q", "--quiet", action="store_true", help="Print no intermediate outputs",
                        default=False)

    parser.add_argument("-v", "--verbose", action="store_true", help="Show more intermediate output",
                        default=False)

    args = parser.parse_args()

    if not args.quiet: print("Generating rough body segmentation...")
    with TotalSegmentatorPredictor("body", fast=True, device=args.device, quiet=args.quiet,
                                   verbose=args.verbose) as predictor:
        body = np.asanyarray(predictor.predict(args.input).dataobj)
    img = nib.load(args.input)
    classes = ["body_trunc"] if args.only_trunc else ["body_trunc", "body_extremities"]
    box = bounding_box(label_isin(body, class_ids("body", classes)), np.array([3, 3, 3]) / img.header.get_zooms()[:3])
    if box is None:
        print("WARNING: Could not crop because no foreground detected")
    img_out = img if box is None else img.slicer[box]
    if not args.quiet:
        print(f"  cropping from {img.shape} to {img_out.shape}")
    nib.save(img_out, args.output)


if __name__ == "__main__":
    main()
