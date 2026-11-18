#!/usr/bin/env python
import argparse
import importlib.metadata
from pathlib import Path
import re
from totalsegmentator.predictor import nifti_cases
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


class RemovedOption(argparse.Action):
    """An option of the old inference pipeline. The options that only tuned it are ignored with a note; the ones that
    asked for an output the pipeline no longer has are an error."""

    def __init__(self, option_strings, dest, error=False, nargs=0, **kwargs):
        super().__init__(option_strings, dest, nargs=nargs, help=argparse.SUPPRESS, **kwargs)
        self.error = error

    def __call__(self, parser, namespace, values, option_string=None):
        if self.error:
            parser.error(f"{option_string} was removed (see CHANGELOG.md).")
        print(f"NOTE: {option_string} was removed and is ignored (see CHANGELOG.md).")


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


def resolve_multilabel(mode, output, folder_of_images=False):
    """'auto': multilabel if the output is a file path or the input a folder of images, binary (one file per class)
    if the output is a directory."""
    if mode != "auto":
        return mode == "multilabel"
    if folder_of_images:
        return True
    if output.is_dir():
        return False
    return output.name.endswith((".nii", ".nii.gz", ".dcm"))


def main():
    parser = argparse.ArgumentParser(description="Segment 104 anatomical structures in CT images.",
                                     epilog="Written by Jakob Wasserthal. If you use this tool please cite https://pubs.rsna.org/doi/10.1148/ryai.230024")

    parser.add_argument("-i", metavar="filepath", dest="input",
                        help="CT nifti image or folder of dicom slices or zip file of dicom slices. "
                             "Or a folder of nifti images: all of them are segmented in one run (models loaded once). "
                             "(Required for segmentation; not needed for --list-tasks / --list-classes.)",
                        type=lambda p: Path(p).absolute(), required=False)

    parser.add_argument("-o", metavar="directory", dest="output",
                        help="Output directory for segmentation masks (one file per class). " +
                             "Or path of multilabel output nifti file (ending with .nii / .nii.gz; see --multilabel). " +
                             "For a folder of nifti images: output directory with <case>.nii.gz per image (or <case>/ with '-ml binary'). " +
                             "Or path of output dicom seg file if --output_type is set to 'dicom_seg' or 'dicom_rtstruct'. " +
                             "(Required for segmentation; not needed for --list-tasks / --list-classes.)",
                        type=lambda p: Path(p).absolute(), required=False)

    parser.add_argument("-ot", "--output_type", type=str, nargs="+",
                    help="Select output type(s). Choices: nifti, dicom_rtstruct, dicom_seg. Multiple are allowed e.g. -ot nifti dicom_seg OR -ot nifti,dicom_seg).",
                    default=None)

    parser.add_argument("-ml", "--multilabel", "--ml", dest="ml", nargs="?", const="auto", default="auto",
                        choices=["auto", "multilabel", "binary"],
                        help="'multilabel': save one multilabel image for all classes. 'binary': save one binary "
                             "mask per class into the output directory. 'auto' (default): multilabel if -o is a file "
                             "path (ending with .nii / .nii.gz / .dcm) or -i is a folder of nifti images, binary if "
                             "-o is a directory. "
                             "-ml without a value is the same as 'auto'.")

    parser.add_argument("-ns", "--nr_thr_saving", type=int, help="Nr of threads for saving segmentations",
                        default=6)

    parser.add_argument("-f", "--fast", action="store_true", help="Run faster lower resolution model (3mm)",
                        default=False)

    parser.add_argument("-ff", "--fastest", action="store_true", help="Run even faster lower resolution model (6mm)",
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
                        help="Select model size. Currently only affects task 'total': "
                             "'small' uses model with less parameters. Default: big.")

    parser.add_argument("-rs", "--roi_subset", type=str, nargs="+",
                        help="Define a subset of classes to save (space separated list of class names). The models only run on the region of these classes.")

    # Will use 3mm model instead of 6mm model to crop to the rois specified in this argument.
    # 3mm is slower but more accurate.
    # LEGACY: use --robust_crop now
    parser.add_argument("-rsr", "--roi_subset_robust", type=str, nargs="+",
                        help="Like roi_subset but uses a slower but more robust model to find the rois.")

    parser.add_argument("-rc", "--robust_crop", action="store_true", help="For cropping (which is required for several task) or roi_subset, use the more robust 3mm model instead of the default and faster 6mm model.",
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

    parser.add_argument("-v", "--verbose", action="store_true", help="Show more intermediate output",
                        default=False)

    parser.add_argument("--debug", action="store_true", help="Show additional debug information on errors (e.g. input path and task)",
                        default=False)

    parser.add_argument("-l", "--license_number", help="Set license number. Needed for some tasks. Only needed once, then stored in config file.",
                        type=str, default=None)

    # Still parsed: argparse would read their short forms as other options otherwise (-fs as --fast --statistics,
    # -sl as --statistics l, -ro 1 as --radiomics -o 1).
    parser.add_argument("-nr", "--nr_thr_resamp", action=RemovedOption, nargs=1)
    parser.add_argument("-ro", "--resampling_order", action=RemovedOption, nargs=1)
    parser.add_argument("-fs", "--force_split", action=RemovedOption)
    parser.add_argument("-bs", "--body_seg", action=RemovedOption)
    parser.add_argument("-ho", "--higher_order_resampling", "--high_order_resampling", action=RemovedOption)
    parser.add_argument("--higher_order_resampling_LEGACY", "--high_order_resampling_LEGACY", action=RemovedOption)
    parser.add_argument("-sl", "--save_lowres", action=RemovedOption, error=True)
    parser.add_argument("-sp", "--save_probabilities", action=RemovedOption, nargs=1, error=True)
    parser.add_argument("-cp", "--crop_path", action=RemovedOption, nargs=1, error=True)

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

    folder_of_images = args.input.is_dir() and bool(nifti_cases(args.input))
    if folder_of_images and args.output.name.endswith((".nii", ".nii.gz", ".dcm")):
        parser.error(f"-i is a folder of images, so -o has to be a folder, not a file: {args.output}")
    args.ml = resolve_multilabel(args.ml, args.output, folder_of_images)

    normalized_output_type = ["nifti"] if args.output_type is None else normalize_output_types(args.output_type)
    # Backward compatibility: single element stays a string
    args.output_type = normalized_output_type[0] if len(normalized_output_type) == 1 else normalized_output_type

    # Auto-select task from DICOM Modality when input is DICOM
    if args.input.exists() and not str(args.input).endswith((".nii", ".nii.gz")) and \
            not (args.input.is_dir() and nifti_cases(args.input)):
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

    totalsegmentator(args.input, args.output, ml=args.ml, nr_thr_saving=args.nr_thr_saving, fast=args.fast,
                     nora_tag=args.nora_tag, preview=args.preview, task=args.task, roi_subset=args.roi_subset,
                     statistics=args.statistics, radiomics=args.radiomics, output_type=args.output_type,
                     quiet=args.quiet, verbose=args.verbose, skip_saving=args.skip_saving,
                     device=args.device, license_number=args.license_number,
                     statistics_exclude_masks_at_border=not args.stats_include_incomplete,
                     no_derived_masks=args.no_derived_masks, v1_order=args.v1_order, fastest=args.fastest,
                     roi_subset_robust=args.roi_subset_robust, stats_aggregation=args.stats_aggregation,
                     remove_small_blobs=args.remove_small_blobs, robust_crop=args.robust_crop,
                     debug=args.debug, report=args.report,
                     statistics_extra=args.statistics_extra, model_size=args.model_size)


if __name__ == '__main__':
    main()

