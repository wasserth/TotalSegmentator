#!/usr/bin/env python
import argparse
import importlib.metadata
from pathlib import Path
import re
from totalsegmentator.python_api import totalsegmentator, validate_device_type_api
from totalsegmentator.registry import TASKS, format_tasks_table, format_classes_table


def validate_device_type(value):
    try:
        return validate_device_type_api(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid device type: '{value}'. Must be 'gpu', 'cpu', 'mps', or 'gpu:X' where X is an integer representing the GPU device ID.")


def positive_float(value):
    try:
        value = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid number: '{value}'.")
    if value <= 0:
        raise argparse.ArgumentTypeError("Value must be greater than 0.")
    return value


def resampling_order(value):
    try:
        value = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid resampling order: '{value}'.")
    if value < 0 or value > 5:
        raise argparse.ArgumentTypeError("Resampling order must be between 0 and 5.")
    return value


def normalize_output_types(values):

    VALID_OUTPUT_TYPES = {"nifti", "dicom_rtstruct", "dicom_seg"}

    # Split on commas and flatten
    result = []
    for v in values:
        result.extend(v.split(","))

    # Make "dicom" the same as "dicom_rtstruct" for backward compatibility
    result = ["dicom_rtstruct" if r == "dicom" else r for r in result]
    
    # Validate
    invalid = [x for x in result if x not in VALID_OUTPUT_TYPES]
    if invalid:
        raise ValueError(f"Invalid output types: {invalid}. Allowed are: {sorted(VALID_OUTPUT_TYPES)}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Segment 104 anatomical structures in CT images.",
                                     epilog="Written by Jakob Wasserthal. If you use this tool please cite https://pubs.rsna.org/doi/10.1148/ryai.230024")

    parser.add_argument("-i", metavar="filepath", dest="input",
                        help="CT nifti image or folder of dicom slices or zip file of dicom slices. "
                             "(Required for segmentation; not needed for --list-tasks / --list-classes.)",
                        type=lambda p: Path(p).absolute(), required=False)

    parser.add_argument("-o", metavar="directory", dest="output",
                        help="Output directory for segmentation masks. " +
                             "Or path of multilabel output nifti file if --ml option is used." +
                             "Or path of output dicom seg file if --output_type is set to 'dicom_seg' or 'dicom_rtstruct'. " +
                             "(Required for segmentation; not needed for --list-tasks / --list-classes.)",
                        type=lambda p: Path(p).absolute(), required=False)

    parser.add_argument("-ot", "--output_type", type=str, nargs="+",
                    help="Select output type(s). Choices: nifti, dicom_rtstruct, dicom_seg. Multiple are allowed e.g. -ot nifti dicom_seg OR -ot nifti,dicom_seg).",
                    default=None)

    parser.add_argument("-ml", "--ml", action="store_true", help="Save one multilabel image for all classes",
                        default=False)

    parser.add_argument("-nr", "--nr_thr_resamp", type=int, help="Nr of threads for resampling", default=1)

    parser.add_argument("-ro", "--resampling_order", type=resampling_order, default=3,
                        help="Spline interpolation order for input image resampling (0-5). "
                             "Default: 3. Setting this to 1 can speed up resampling with very similar segmentation accuracy.")

    parser.add_argument("-ns", "--nr_thr_saving", type=int, help="Nr of threads for saving segmentations",
                        default=6)

    parser.add_argument("-f", "--fast", action="store_true", help="Run faster lower resolution model (3mm)",
                        default=False)

    parser.add_argument("-ff", "--fastest", action="store_true", help="Run even faster lower resolution model (6mm)",
                        default=False)

    parser.add_argument("-sl", "--save_lowres", action="store_true",
                        help="When using --fast or --fastest, save the segmentation in the model resolution "
                             "(3mm or 6mm) instead of upsampling to the input resolution.",
                        default=False)

    parser.add_argument("-t", "--nora_tag", type=str,
                        help="tag in nora as mask. Pass nora project id as argument.",
                        default="None")

    parser.add_argument("-p", "--preview", action="store_true",
                        help="Generate a png preview of segmentation",
                        default=False)

    # cerebral_bleed: Intracerebral hemorrhage
    # liver_vessels: hepatic vessels
    # The list of selectable tasks lives in totalsegmentator/registry.py (TASKS) so that the
    # CLI choices and the totalseg_info introspection command can never drift apart.
    parser.add_argument("-ta", "--task", choices=TASKS, metavar="task",
                        help="Select which model to use. This determines what is predicted. "
                             "Run 'totalseg_info --list-tasks' to see all options.",
                        default="total")

    parser.add_argument("-ms", "--model_size", choices=["big", "small"], default="big",
                        help="Select model size. Currently only affects task 'total_v3': "
                             "'small' uses nnUNetResEncUNetLPlans_8. Default: big.")

    parser.add_argument("-rs", "--roi_subset", type=str, nargs="+",
                        help="Define a subset of classes to save (space separated list of class names). If running 1.5mm model, will only run the appropriate models for these rois.")

    # Will use 3mm model instead of 6mm model to crop to the rois specified in this argument.
    # 3mm is slower but more accurate.
    # LEGACY: use --robust_crop now
    parser.add_argument("-rsr", "--roi_subset_robust", type=str, nargs="+",
                        help="Like roi_subset but uses a slower but more robust model to find the rois.")

    parser.add_argument("-rc", "--robust_crop", action="store_true", help="For cropping (which is required for several task) or roi_subset, use the more robust 3mm model instead of the default and faster 6mm model.",
                        default=False)

    parser.add_argument("-ho", "--higher_order_resampling", action="store_true", 
                        help="Use higher order resampling for segmentations. Results in smoother segmentations. Use with e.g. -nr 4 for faster runtime.",
                        default=False)

    parser.add_argument("-s", "--statistics", nargs='?', const=True, default=False,
                        metavar="filepath",
                        help="Calc volume (in mm3) and mean intensity. Results will be in statistics.json in the output directory. Optionally specify a custom output path for statistics.json.")

    parser.add_argument("-r", "--radiomics", action="store_true",
                        help="Calc radiomics features. Requires pyradiomics. Results will be in statistics_radiomics.json",
                        default=False)

    parser.add_argument("-rp", "--report", type=lambda p: Path(p).absolute(), default=None,
                        metavar="filepath",
                        help="Write a machine-readable JSON run report (software/model versions, device, task, "
                             "classes, runtime, output files) to this path. Useful for reproducible pipelines and automation.")

    parser.add_argument("-sii", "--stats_include_incomplete", action="store_true",
                        help="Normally statistics are only calculated for ROIs which are not cut off by the beginning or end of image. Use this option to calc anyways.",
                        default=False)

    parser.add_argument("-sa", "--stats_aggregation", type=str, choices=["mean", "median"],
                        help="Aggregation method for intensity statistics (default: mean).",
                        default="mean")

    parser.add_argument("-sx", "--statistics_extra", action="store_true",
                        help="Add extra per-structure metrics to the statistics: n_voxels, intensity std/min/max, "
                             "and the morphometric centroid_vox and bbox_vox (voxel coordinates). Slightly increases statistics runtime.",
                        default=False)

    parser.add_argument("-cp", "--crop_path", help="Custom path to masks used for cropping. If not set will use output directory.",
                        type=lambda p: Path(p).absolute(), default=None)

    parser.add_argument("-bs", "--body_seg", action="store_true",
                        help="Do initial rough body segmentation and crop image to body region",
                        default=False)

    parser.add_argument("-fs", "--force_split", action="store_true", help="Process image in 3 chunks for less memory consumption. (do not use on small images)",
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

    parser.add_argument("-rmb", "--remove_small_blobs", nargs="?", const=200.0, default=False,
                        type=positive_float, metavar="mm3",
                        help="Remove small connected components from the final segmentations. "
                             "Optionally pass the minimum component size in mm3 (default: 200).")  # ~30s runtime because of the large number of classes
        
    # "mps" is for apple silicon; the latest pytorch nightly version supports 3D Conv but not ConvTranspose3D which is
    # also needed by nnU-Net. So "mps" not working for now.
    # https://github.com/pytorch/pytorch/issues/77818
    parser.add_argument("-d",'--device', type=validate_device_type, default="gpu",
                        help="Device type: 'gpu', 'cpu', 'mps', or 'gpu:X' where X is an integer representing the GPU device ID.")

    parser.add_argument("-q", "--quiet", action="store_true", help="Print no intermediate outputs",
                        default=False)

    parser.add_argument("-sp", "--save_probabilities", help="Save probabilities to this path. Only for experienced users. Python skills required.",
                        type=lambda p: Path(p).absolute())

    parser.add_argument("-v", "--verbose", action="store_true", help="Show more intermediate output",
                        default=False)

    parser.add_argument("--debug", action="store_true", help="Show additional debug information on errors (e.g. input path and task)",
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

    parser.add_argument("-lt", "--list-tasks", action="store_true", dest="list_tasks",
                        help="List all available tasks (modality, license, number of classes) and exit. "
                             "For machine-readable output use 'totalseg_info --json'.")

    parser.add_argument("-lc", "--list-classes", dest="list_classes", nargs="?", const="total",
                        metavar="task",
                        help="List the classes of a task (index -> name) and exit. Defaults to 'total'.")

    parser.add_argument('--version', action='version', version=importlib.metadata.version("TotalSegmentator"))

    args = parser.parse_args()

    # Capability discovery: these short-circuit before any input/output is required and
    # before models are loaded, so scripts/agents can introspect the tool quickly.
    if args.list_tasks:
        print(format_tasks_table())
        return
    if args.list_classes is not None:
        if args.list_classes not in TASKS:
            parser.error(f"unknown task '{args.list_classes}' for --list-classes. "
                         "Run --list-tasks to see all options.")
        print(format_classes_table(args.list_classes))
        return

    # For an actual segmentation run, input and output are required.
    missing = [flag for flag, val in (("-i", args.input), ("-o", args.output)) if val is None]
    if missing:
        parser.error(f"the following arguments are required for segmentation: {', '.join(missing)}")

    normalized_output_type = ["nifti"] if args.output_type is None else normalize_output_types(args.output_type)
    # Backward compatibility: single element stays a string
    args.output_type = normalized_output_type[0] if len(normalized_output_type) == 1 else normalized_output_type

    # Auto-select task from DICOM Modality when input is DICOM
    if args.input.exists() and not str(args.input).endswith((".nii", ".nii.gz")):
        try:
            from totalsegmentator.dicom_io import detect_dicom_modality
            modality = detect_dicom_modality(args.input)
            if modality is not None:
                if modality.upper() == "CT" and args.task == "total_mr":
                    print("WARNING: you tried to run MR model, but input modality is CT. Will run CT model instead.")
                    args.task = "total"
                elif modality.upper() == "MR" and args.task == "total":
                    print("WARNING: you tried to run CT model, but input modality is MR. Will run MR model instead.")
                    args.task = "total_mr"
        except Exception:
            pass

    totalsegmentator(args.input, args.output, args.ml, args.nr_thr_resamp, args.nr_thr_saving,
                     args.fast, args.nora_tag, args.preview, args.task, args.roi_subset,
                     args.statistics, args.radiomics, args.crop_path, args.body_seg,
                     args.force_split, args.output_type, args.quiet, args.verbose, args.test, args.skip_saving,
                     device=args.device, license_number=args.license_number,
                     statistics_exclude_masks_at_border=not args.stats_include_incomplete,
                     no_derived_masks=args.no_derived_masks, v1_order=args.v1_order, fastest=args.fastest,
                     roi_subset_robust=args.roi_subset_robust, stats_aggregation=args.stats_aggregation, 
                     remove_small_blobs=args.remove_small_blobs, statistics_normalized_intensities=False,
                     robust_crop=args.robust_crop, higher_order_resampling=args.higher_order_resampling,
                     save_probabilities=args.save_probabilities, debug=args.debug, report=args.report,
                     statistics_extra=args.statistics_extra, save_lowres=args.save_lowres,
                     resampling_order=args.resampling_order, model_size=args.model_size)


if __name__ == '__main__':
    main()
