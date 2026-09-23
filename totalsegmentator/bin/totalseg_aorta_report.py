import argparse
import json
import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from totalsegmentator.aorta_report import VERSION
from totalsegmentator.aorta_report.report import create_aorta_report
from totalsegmentator.reporting import save_runtime, setup_logger
from totalsegmentator.serialization_utils import (
    convert_to_serializable,
    decompress_and_deserialize,
)


def _load_metadata(path, logger):
    if path is not None and path.exists():
        with path.open() as file:
            metadata = json.load(file, strict=False)
        for source, target in (
            ("PatientsAge", "PatientAge"),
            ("PatientsSex", "PatientSex"),
            ("PatientsBirthDate", "PatientBirthDate"),
        ):
            if source in metadata:
                metadata[target] = metadata.pop(source)
    else:
        logger.info(
            "WARNING: Could not find nodeinfo.json. Using default metadata."
        )
        metadata = {
            "PatientName": "Unknown",
            "PatientSex": "M",
            "PatientAge": "083Y",
            "PatientBirthDate": "19000101",
            "PatientID": "Unknown",
            "StudyID": "Unknown",
            "StudyDescription": "Unknown",
            "StudyDate": "19000101",
        }
    metadata.setdefault("PatientAge", "099Y")
    if "W" in str(metadata["PatientAge"]):
        logger.info(
            "WARNING: Patient age is in weeks. Patient too young. Stopping."
        )
        return None
    metadata["PatientAge"] = int(str(metadata["PatientAge"]).replace("Y", ""))
    metadata["version"] = VERSION
    return metadata


def _build_parser():
    parser = argparse.ArgumentParser(description="Generate aorta report.")

    def resolved_path(value):
        return Path(value).resolve()

    parser.add_argument("-i", "--ct_path", type=resolved_path, required=True, help="Path to CT file.")
    parser.add_argument("-rt", "--rois_totalseg", type=resolved_path, help="Directory containing the TotalSegmentator ROIs.")
    parser.add_argument("-rd", "--rois_details", type=resolved_path, help="Directory containing the detailed aorta ROIs.")
    parser.add_argument("-n", "--nodeinfo", type=resolved_path, help="Optional path to nodeinfo file.")
    parser.add_argument("-o", "--output_nifti", type=resolved_path, required=True, help="Path to output NIfTI file.")
    parser.add_argument("-j", "--output_json", type=resolved_path, required=True, help="Path to output JSON file.")
    parser.add_argument("-l", "--output_log", type=resolved_path, required=True, help="Path to output log file.")
    parser.add_argument("-tmp", "--tmp_dir", type=resolved_path, help="Temporary directory. Uses a system temporary directory if omitted.")
    parser.add_argument("-c", "--cpr", type=resolved_path, help="Path to CPR output PNG file.")
    parser.add_argument("-ca", "--cpr_animated", type=resolved_path, help="Path to animated CPR output NIfTI file (.nii.gz).")
    parser.add_argument("-a", "--aorta_fn", default="aorta.nii.gz", help="Filename of the aorta mask.")
    parser.add_argument("-an", "--annulus_fn", default="annulus_proper.nii.gz", help="Filename of the annulus mask.")
    parser.add_argument("-rm", "--run_models", action="store_true", help="Run all models from within this command.")
    parser.add_argument("-mp", "--models_parallel", action="store_true", help="Run all models in parallel. Faster, but needs more RAM and GPU memory.")
    parser.add_argument("-t", "--test", default="None", help="Define which test to run.")
    parser.add_argument("-e", "--erosion", action="store_true", help="Erode aorta by 0.8 mm to make the aorta fit better.")
    parser.add_argument("-d", "--debug", action="store_true", help="Keep temporary and intermediate files for debugging.")
    parser.add_argument("-sd", "--skip_dissection", action="store_true", help="Skip true/false lumen segmentation even if available.")
    parser.add_argument("-r", "--save_runtime", action="store_true", help="Save runtime to META/runtime.json.")
    parser.add_argument("--version", action="version", version=VERSION)
    return parser


def _prepare_tmp_dir(args):
    if args.debug:
        print("Running in DEBUG mode.")

    if args.tmp_dir is not None:
        tmp_dir = args.tmp_dir
    else:
        tmp_dir = Path(tempfile.mkdtemp())

    tmp_dir.mkdir(parents=True, exist_ok=True)
    delete_tmp = args.tmp_dir is None and not args.debug
    delete_aux_files = not args.debug
    return tmp_dir, delete_tmp, delete_aux_files


def _validate_ct(ct_path, logger):
    ct_img = nib.load(ct_path)
    shape = ct_img.shape
    if len(shape) != 3:
        logger.info("WARNING: Image shape is not 3D. Stopping.")
        return False
    if max(shape) > 1500:
        logger.info("WARNING: Image shape is very large (>1500). Stopping.")
        return False
    if min(shape) < 20:
        logger.info("WARNING: Image shape is very small (<20). Stopping.")
        return False

    linear_affine = ct_img.affine[:3, :3].copy()
    if linear_affine.min() < 0:
        logger.info("WARNING: Affine contains negative values!")
    np.fill_diagonal(linear_affine, 0)
    if np.abs(linear_affine).max() > 0.01:
        logger.info("WARNING: Affine contains significant rotation!")
    return True


def _save_report(final_result, output_nifti, output_json):
    report_image = decompress_and_deserialize(final_result["report_img"])
    nib.save(report_image, output_nifti)
    with output_json.open("w") as file:
        json.dump(
            convert_to_serializable(final_result["report_json"]),
            file,
            indent=4,
        )

"""
Run for one test case:

# Reuse existing masks:

cd /mnt/nvme/data/test_data/aorta_report/28500925
totalseg_aorta_report \
  -i ct_15mm.nii.gz \
  -rt roi \
  -rd roi_cropped_aorta \
  -o aorta_report/aorta_report.nii.gz \
  -j aorta_report/aorta_report.json \
  -l aorta_report/aorta_report.txt \
  -tmp tmp_aorta_report \
  -c aorta_report/cpr.png \
  -a organ_aorta_T264.nii.gz \
  -an annulus.nii.gz \
  -d

# Create new masks:
cd /mnt/nvme/data/test_data/aorta_report/23950963
totalseg_aorta_report \
  -i ct.nii.gz \
  -o aorta_report/aorta_report.nii.gz \
  -j aorta_report/aorta_report.json \
  -l aorta_report/aorta_report.txt \
  -tmp tmp_aorta_report \
  --run_models \
  --erosion \
  -d

"""
def main():
    parser = _build_parser()
    args = parser.parse_args()
    if not args.run_models and (
        args.rois_totalseg is None or args.rois_details is None
    ):
        parser.error("--rois_totalseg and --rois_details are required without --run_models")

    for output_path in (
        args.output_nifti,
        args.output_json,
        args.output_log,
        args.cpr,
        args.cpr_animated,
    ):
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        args.output_log, name=f"totalseg_aorta_report.{args.output_log}"
    )
    started = time.time()
    logger.info("Getting metadata...")
    metadata = _load_metadata(args.nodeinfo, logger)
    if metadata is None:
        return

    if metadata["PatientAge"] < 5 or metadata["PatientAge"] > 150:
        logger.info("WARNING: Patient age is not in range [5, 150]. Stopping.")
        return
    if not _validate_ct(args.ct_path, logger):
        return

    tmp_dir, delete_tmp, delete_aux_files = _prepare_tmp_dir(args)
    logger.info("Creating report...")
    generator = create_aorta_report(
        args.ct_path,
        args.rois_totalseg,
        args.rois_details,
        metadata,
        tmp_dir,
        logger,
        delete_tmp=delete_tmp,
        delete_aux_files=delete_aux_files,
        cpr_path=args.cpr,
        cpr_animated_path=args.cpr_animated,
        aorta_fn=args.aorta_fn,
        annulus_fn=args.annulus_fn,
        test=args.test,
        debug=args.debug,
        run_models=args.run_models,
        models_parallel=args.models_parallel,
        host="local",
        erosion=args.erosion,
        version=VERSION,
        skip_dissection=args.skip_dissection,
    )
    final_result = None
    for result in generator:
        print(f"progress: {result['progress']}, status: {result['status']}")
        if result["progress"] == 100:
            final_result = result
    if final_result is None:
        raise RuntimeError("Aorta report generator did not return a final result.")
    _save_report(final_result, args.output_nifti, args.output_json)
    if args.save_runtime and args.nodeinfo is not None and args.nodeinfo.exists():
        save_runtime(started, "aorta_report", args.nodeinfo)


if __name__ == "__main__":
    main()
