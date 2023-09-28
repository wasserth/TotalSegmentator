#!/usr/bin/env python
import sys
import os
import argparse
from pkg_resources import require
from pathlib import Path

from totalsegmentator.python_api import totalsegmentator


def main():
    parser = argparse.ArgumentParser(description="Segment 104 anatomical structures in CT images.",
                                     epilog="Written by Jakob Wasserthal. If you use this tool please cite https://pubs.rsna.org/doi/10.1148/ryai.230024")

    parser.add_argument("-i", metavar="filepath", dest="input",
                        help="CT nifti image or folder of dicom slices", 
                        type=lambda p: Path(p).absolute(), required=True)

    parser.add_argument("-o", metavar="directory", dest="output",
                        help="Output directory for segmentation masks", 
                        type=lambda p: Path(p).absolute(), required=True)

    parser.add_argument("-ot", "--output_type", choices=["nifti", "dicom"],
                    help="Select if segmentations shall be saved as Nifti or as Dicom RT Struct image.",
                    default="nifti")
                    
    parser.add_argument("-ml", "--ml", action="store_true", help="Save one multilabel image for all classes",
                        default=False)

    parser.add_argument("-nr", "--nr_thr_resamp", type=int, help="Nr of threads for resampling", default=1)

    parser.add_argument("-ns", "--nr_thr_saving", type=int, help="Nr of threads for saving segmentations", 
                        default=6)

    parser.add_argument("-f", "--fast", action="store_true", help="Run faster lower resolution model",
                        default=False)

    parser.add_argument("-t", "--nora_tag", type=str, 
                        help="tag in nora as mask. Pass nora project id as argument.",
                        default="None")

    parser.add_argument("-p", "--preview", action="store_true", 
                        help="Generate a png preview of segmentation",
                        default=False)

    # cerebral_bleed: Intracerebral hemorrhage 
    # liver_vessels: hepatic vessels
    parser.add_argument("-ta", "--task", choices=["total", "body", "vertebrae_body",
                        "lung_vessels", "cerebral_bleed", "hip_implant", "coronary_arteries", 
                        "pleural_pericard_effusion", "test",
                        "appendicular_bones", "tissue_types", "heartchambers_highres", 
                        "face", 
                        ],
                        # future: liver_vessels, head, 
                        help="Select which model to use. This determines what is predicted.",
                        default="total")

    parser.add_argument("-rs", "--roi_subset", type=str, nargs="+",
                        help="Define a subset of classes to save (space separated list of class names). If running 1.5mm model, will only run the appropriate models for these rois.")

    # When this is used together with --roi_subset and --ml, then statistics will be calculated for all classes
    # in the class_map_part_X with is calculated. Some of the rois will be cropped and therefore the volume will
    # only be of the cropped region, making it an incorrect volume.
    parser.add_argument("-s", "--statistics", action="store_true", 
                        help="Calc volume (in mm3) and mean intensity. Results will be in statistics.json",
                        default=False)

    parser.add_argument("-r", "--radiomics", action="store_true", 
                        help="Calc radiomics features. Requires pyradiomics. Results will be in statistics_radiomics.json",
                        default=False)

    parser.add_argument("-sii", "--stats_include_incomplete", action="store_true", 
                        help="Normally statistics are only calculated for ROIs which are not cut off by the beginning or end of image. Use this option to calc anyways.",
                        default=False)

    parser.add_argument("-cp", "--crop_path", help="Custom path to masks used for cropping. If not set will use output directory.", 
                        type=lambda p: Path(p).absolute(), default=None)

    parser.add_argument("-bs", "--body_seg", action="store_true", 
                        help="Do initial rough body segmentation and crop image to body region",
                        default=False)
    
    parser.add_argument("-fs", "--force_split", action="store_true", help="Process image in 3 chunks for less memory consumption",
                        default=False)

    parser.add_argument("-ss", "--skip_saving", action="store_true", 
                        help="Skip saving of segmentations for faster runtime if you are only interested in statistics.",
                        default=False)

    # Used for server to make statistics file have the same classes as images are created
    parser.add_argument("-ndm", "--no_derived_masks", action="store_true", 
                        help="Do not create derived masks (e.g. skin from body mask).",
                        default=False)

    parser.add_argument("-v1o", "--v1_order", action="store_true", 
                        help="In multilabel file order classes as in v1. New v2 classes will be removed.",
                        default=False)

    # "mps" is for apple silicon; but does not support 3D Conv at the moment. Therefore do not allow here.
    # https://github.com/pytorch/pytorch/issues/77818
    parser.add_argument("-d", "--device", choices=["gpu", "cpu"],
                        help="Device to run on (default: gpu).",
                        default="gpu")

    parser.add_argument("-q", "--quiet", action="store_true", help="Print no intermediate outputs",
                        default=False)

    parser.add_argument("-v", "--verbose", action="store_true", help="Show more intermediate output",
                        default=False)

    parser.add_argument("-l", "--license_number", help="Set license number. Needed for some tasks. Only needed once, then stored in config file.", 
                        type=str, default=None)

    # Tests:
    # 0: no testing behaviour activated
    # 1: total normal
    # 2: total fast -> removed because can run normally with cpu
    # 3: lung_vessels
    parser.add_argument("--test", metavar="0|1|3", choices=[0, 1, 3], type=int,
                        help="Only needed for unittesting.",
                        default=0)

    parser.add_argument('--version', action='version', version=require("TotalSegmentator")[0].version)

    args = parser.parse_args()

    totalsegmentator(args.input, args.output, args.ml, args.nr_thr_resamp, args.nr_thr_saving,
                     args.fast, args.nora_tag, args.preview, args.task, args.roi_subset,
                     args.statistics, args.radiomics, args.crop_path, args.body_seg,
                     args.force_split, args.output_type, args.quiet, args.verbose, args.test, args.skip_saving,
                     args.device, args.license_number, not args.stats_include_incomplete,
                     args.no_derived_masks, args.v1_order)


if __name__ == "__main__":
    main()
