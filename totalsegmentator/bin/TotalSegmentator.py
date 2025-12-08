#!/usr/bin/env python
import argparse
import importlib.metadata
from pathlib import Path
import json
from totalsegmentator.python_api import totalsegmentator, validate_device_type_api
from totalsegmentator.mixed_precision import set_mixed_precision


def validate_device_type(value):
    try:
        return validate_device_type_api(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid device type: '{value}'. Must be 'gpu', 'cpu', 'mps', or 'gpu:X' where X is an integer representing the GPU device ID.")


def _load_project_config(explicit_path: Path | None = None) -> dict:
    """Load JSON config from explicit path, /config, or project root.
    Priority: explicit --config > /config > project root.
    Prints chosen path and the JSON content (sanitized to a single line if large).
    """
    def _read(p: Path):
        with p.open('r', encoding='utf-8') as f:
            return json.load(f)

    cfg = {}
    used_path = None
    if explicit_path is not None:
        p = Path(explicit_path)
        if p.exists():
            try:
                cfg = _read(p)
                used_path = p
            except Exception:
                print(f"TotalSegmentator: failed to load configuration from {p}", flush=True)
                return {}
        else:
            print(f"TotalSegmentator: config path not found: {p}", flush=True)
            return {}
    else:
        docker_cfg = Path('/config') / 'TotalSegmentator_config.json'
        if docker_cfg.exists():
            try:
                cfg = _read(docker_cfg)
                used_path = docker_cfg
            except Exception:
                print(f"TotalSegmentator: failed to load configuration from {docker_cfg}", flush=True)
                return {}
        else:
            root = Path(__file__).resolve().parents[2]
            proj_cfg = root / 'TotalSegmentator_config.json'
            if proj_cfg.exists():
                try:
                    cfg = _read(proj_cfg)
                    used_path = proj_cfg
                except Exception:
                    print(f"TotalSegmentator: failed to load configuration from {proj_cfg}", flush=True)
                    return {}
    if used_path is not None:
        print(f"TotalSegmentator: loaded configuration from {used_path}", flush=True)
        try:
            print("TotalSegmentator: configuration content:\n" + json.dumps(cfg, indent=2), flush=True)
        except Exception:
            print("TotalSegmentator: (unable to pretty-print config)", flush=True)
    return cfg

# Mapping new flat keys to legacy descriptive keys
_CFG_KEY_MAP = {
    "Generate DICOM SEG results": ["generate_dicom_seg"],
    "Generate DICOM RT-STRUCT results": ["generate_dicom_rtstruct"],
    "Split volume to 3 chunks to save GPU memory ": ["split_to_chunks"],
    "Use Cuda Mixed Precision": ["use_mixed_precision"],
}

def _get_cfg_bool(cfg: dict, key: str, default: bool) -> bool:
    """Return boolean for either legacy schema (properties->key->default) or new flat JSON keys.
    Supports mapped alternative flat keys via _CFG_KEY_MAP.
    """
    # Legacy schema style
    try:
        props = cfg.get("properties", {})
        if key in props:
            entry = props[key]
            if isinstance(entry, dict):
                if "default" in entry:
                    return bool(entry.get("default", default))
    except Exception:
        pass
    # New flat style via direct key (exact match)
    if key in cfg and isinstance(cfg[key], bool):
        return bool(cfg[key])
    # Mapped alternative keys
    for alt in _CFG_KEY_MAP.get(key, []):
        if alt in cfg and isinstance(cfg[alt], bool):
            return bool(cfg[alt])
    return default


def main():
    parser = argparse.ArgumentParser(description="Segment 104 anatomical structures in CT images.",
                                     epilog="Written by Jakob Wasserthal. If you use this tool please cite https://pubs.rsna.org/doi/10.1148/ryai.230024")

    parser.add_argument("-i", metavar="filepath", dest="input",
                        help="CT nifti image or folder of dicom slices or zip file of dicom slices.",
                        type=lambda p: Path(p).absolute(), required=True)

    parser.add_argument("-o", metavar="directory", dest="output",
                        help="Output directory for segmentation masks. " + 
                             "Or path of multilabel output nifti file if --ml option is used." + 
                             "Or path of output dicom seg file if --output_type is set to 'dicom_seg' or 'dicom_rtstruct'",
                        type=lambda p: Path(p).absolute(), required=True)

    # Changed: allow multiple output types either via repeated -ot or comma separated list in a single -ot
    parser.add_argument("-ot", "--output_type", type=str, nargs="+",
                    help="Select output type(s). Choices: nifti, dicom_rtstruct, dicom_seg. Provide multiple types by repeating -ot or comma separating (e.g. -ot nifti dicom_seg OR -ot nifti,dicom_seg).",
                    default=None)

    parser.add_argument("-ml", "--ml", action="store_true", help="Save one multilabel image for all classes",
                        default=False)

    parser.add_argument("-nr", "--nr_thr_resamp", type=int, help="Nr of threads for resampling", default=1)

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
    parser.add_argument("-ta", "--task", choices=["total", "body", "body_mr", "vertebrae_mr",
                        "lung_vessels", "cerebral_bleed", "hip_implant", "coronary_arteries",
                        "pleural_pericard_effusion", "test",
                        "appendicular_bones", "appendicular_bones_mr", "tissue_types", "heartchambers_highres",
                        "face", "vertebrae_body", "total_mr", "tissue_types_mr", "tissue_4_types", "face_mr",
                        "head_glands_cavities", "head_muscles", "headneck_bones_vessels", "headneck_muscles",
                        "brain_structures", "liver_vessels", "oculomotor_muscles",
                        "thigh_shoulder_muscles", "thigh_shoulder_muscles_mr", "lung_nodules", "kidney_cysts", 
                        "breasts", "ventricle_parts", "aortic_sinuses", "liver_segments", "liver_segments_mr",
                        "total_highres_test", "craniofacial_structures", "abdominal_muscles", "teeth",
                        "trunk_cavities"],
                        help="Select which model to use. This determines what is predicted.",
                        default="total")

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
                        help="Use higher order resampling for segmentations. Results in smoother segmentations on high resolution images but uses more runtime + memory.",
                        default=False)

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

    parser.add_argument("-rmb", "--remove_small_blobs", action="store_true", help="Remove small connected components (<0.2ml) from the final segmentations.",
                        default=False)  # ~30s runtime because of the large number of classes
        
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

    parser.add_argument("-l", "--license_number", help="Set license number. Needed for some tasks. Only needed once, then stored in config file.",
                        type=str, default=None)

    # Mixed precision control: if set, overrides JSON; if not set, JSON may enable it.
    parser.add_argument("-mp", "--mixed_precision", action="store_true",
                        help="Enable CUDA mixed precision (autocast). If omitted, can be enabled via TotalSegmentator_config.json.")
    # New: explicit configuration path
    parser.add_argument("-c", "--config", type=lambda p: Path(p).absolute(), default=None,
                        help="Path to configuration JSON. Overrides default search locations (/config and project root).")

    # Tests:
    # 0: no testing behaviour activated
    # 1: total normal
    # 2: total fast -> removed because can run normally with cpu
    # 3: lung_vessels
    parser.add_argument("--test", metavar="0|1|3", choices=[0, 1, 3], type=int,
                        help="Only needed for unittesting.",
                        default=0)

    parser.add_argument('--version', action='version', version=importlib.metadata.version("TotalSegmentator"))

    args = parser.parse_args()

    # Load project config (explicit path takes precedence)
    cfg = _load_project_config(args.config)
    # Merge output types
    # Normalize output_type argument (allow repeated -ot or comma separated values)
    if args.output_type is None:
        # Start empty: will only add nifti if nothing else specified via config
        normalized = []
    else:
        raw_types = args.output_type if isinstance(args.output_type, list) else [args.output_type]
        normalized = []
        for item in raw_types:
            normalized.extend([t.strip() for t in item.split(',') if t.strip()])
    valid = {"nifti", "dicom_rtstruct", "dicom_seg"}
    for t in normalized:
        if t not in valid:
            parser.error(f"Invalid output type '{t}'. Valid choices: {', '.join(sorted(valid))}")
    # Augment from JSON config (does not add nifti implicitly)
    added_from_cfg = []
    if _get_cfg_bool(cfg, "Generate DICOM SEG results", False) and "dicom_seg" not in normalized:
        normalized.append("dicom_seg")
        added_from_cfg.append("dicom_seg")
        print("TotalSegmentator: enabling output_type 'dicom_seg' from configuration JSON", flush=True)
    if _get_cfg_bool(cfg, "Generate DICOM RT-STRUCT results", False) and "dicom_rtstruct" not in normalized:
        normalized.append("dicom_rtstruct")
        added_from_cfg.append("dicom_rtstruct")
        print("TotalSegmentator: enabling output_type 'dicom_rtstruct' from configuration JSON", flush=True)
    # If still empty produce default nifti
    if not normalized:
        normalized = ["nifti"]
    if added_from_cfg and args.output_type is None and "nifti" not in added_from_cfg and normalized == added_from_cfg:
        # User relied solely on config, and did not request nifti; ensure nifti not implicitly added
        pass  # already no nifti present
    # Backward compatibility: single element stays a string
    args.output_type = normalized[0] if len(normalized) == 1 else normalized
    print(f"TotalSegmentator: resolved output types -> {args.output_type}", flush=True)

    # Merge force_split from JSON (only set True from JSON if user didn't pass flag)
    if not args.force_split and _get_cfg_bool(cfg, "Split volume to 3 chunks to save GPU memory ", False):
        args.force_split = True
        print("TotalSegmentator: enabling 'force_split' from configuration JSON", flush=True)

    # Mixed precision: CLI overrides JSON. Set runtime flag used by implementation.
    mp_enabled_json = _get_cfg_bool(cfg, "Use Cuda Mixed Precision", False)
    mp_enabled = args.mixed_precision or mp_enabled_json
    set_mixed_precision(mp_enabled)
    if mp_enabled_json and not args.mixed_precision:
        print("TotalSegmentator: enabling mixed precision from configuration JSON", flush=True)

    # New: model resolution selection from JSON (CLI flags take precedence)
    cfg_3mm = _get_cfg_bool(cfg, "3mm_model", False)
    cfg_6mm = _get_cfg_bool(cfg, "6mm_model", False)
    if not args.fast and not args.fastest:
        if cfg_3mm and cfg_6mm:
            # Prefer 3mm over 6mm if both are accidentally enabled
            args.fast = True
            print("TotalSegmentator: both '3mm_model' and '6mm_model' set in config; preferring 3mm (fast).", flush=True)
        elif cfg_3mm:
            args.fast = True
            print("TotalSegmentator: enabling 'fast' (3mm model) from configuration JSON", flush=True)
        elif cfg_6mm:
            args.fastest = True
            print("TotalSegmentator: enabling 'fastest' (6mm model) from configuration JSON", flush=True)
    else:
        # If user passed either flag, ignore config
        pass

    # New config integrations:
    # Device selection via 'use_gpu' (only switch to cpu if config explicitly disables GPU and user did not override)
    if _get_cfg_bool(cfg, "use_gpu", True) is False:
        # Attempt to detect if user specified device manually: if args.device is default 'gpu' treat as not overridden
        if args.device == 'gpu':
            args.device = 'cpu'
            print("TotalSegmentator: setting device to 'cpu' due to configuration JSON (use_gpu=false)", flush=True)

    # Auto-select task from DICOM Modality when input is DICOM
    if args.input.exists() and not str(args.input).endswith((".nii", ".nii.gz")):
        try:
            from totalsegmentator.dicom_io import _detect_dicom_modality
            modality = _detect_dicom_modality(args.input)
            if modality is not None:
                if modality.upper() == 'CT' and (args.task == '' or 'total' in args.task):
                    print("TotalSegmentator: input Modality=CT -> using task 'total'", flush=True)
                    args.task = 'total'
                elif modality.upper() == 'MR'and (args.task == '' or 'total' in args.task):
                    print("TotalSegmentator: input Modality=MR -> using task 'total_mr'", flush=True)
                    args.task = 'total_mr'
        except Exception:
            pass

    # remove_small_blobs
    if not args.remove_small_blobs and _get_cfg_bool(cfg, "remove_small_blobs", False):
        args.remove_small_blobs = True
        print("TotalSegmentator: enabling 'remove_small_blobs' from configuration JSON", flush=True)

    # higher_order_resampling
    if not args.higher_order_resampling and _get_cfg_bool(cfg, "high_order_resampling", False):
        args.higher_order_resampling = True
        print("TotalSegmentator: enabling 'higher_order_resampling' from configuration JSON", flush=True)

    # verbose output (do not enable if user requested quiet)
    if not args.verbose and not args.quiet and _get_cfg_bool(cfg, "verbose", False):
        args.verbose = True
        print("TotalSegmentator: enabling 'verbose' from configuration JSON", flush=True)

    totalsegmentator(args.input, args.output, args.ml, args.nr_thr_resamp, args.nr_thr_saving,
                     args.fast, args.nora_tag, args.preview, args.task, args.roi_subset,
                     args.statistics, args.radiomics, args.crop_path, args.body_seg,
                     args.force_split, args.output_type, args.quiet, args.verbose, args.test, args.skip_saving,
                     device=args.device, license_number=args.license_number,
                     statistics_exclude_masks_at_border=not args.stats_include_incomplete,
                     no_derived_masks=args.no_derived_masks, v1_order=args.v1_order, fastest=args.fastest,
                     roi_subset_robust=args.roi_subset_robust, remove_small_blobs=args.remove_small_blobs,
                     robust_crop=args.robust_crop, higher_order_resampling=args.higher_order_resampling,
                     save_probabilities=args.save_probabilities)


if __name__ == '__main__':
    main()
