from pathlib import Path
import argparse
import json
import shutil
import tempfile
import time
from datetime import datetime

import nibabel as nib
import numpy as np

from importlib import resources

from totalsegmentator.config import send_usage_stats_application
from totalsegmentator.dicom_io import dcm_to_nifti
from totalsegmentator.nifti_ext_header import load_multilabel_nifti
from totalsegmentator.serialization_utils import (decompress_and_deserialize, filestream_to_nifti,
                                                 serialize_and_compress)
from totalsegmentator.spine_report.html import generate_html
from totalsegmentator.spine_report.logger import setup_logger
from totalsegmentator.spine_report.measure_verte_height import get_verte_height
from totalsegmentator.spine_report.nifti import combine_as_nifti
from totalsegmentator.spine_report.utils import save_runtime
from totalsegmentator.spine_report.utils_report import run_models_abdomen, get_contrast_phase


VERSION = "0.1.0"

HU_THRESHOLD = 90.25  # below this it is osteoporosis
ART_HU_ADDON = 20  # correction factor for contrast
PV_HU_ADDON = 14.75


def create_spine_report(
    ct_bytes,
    metadata,
    tmp_dir,
    logger,
    delete_tmp=True,
    delete_aux_files=True,
    debug=False,
    test="None",
    f_type="nii",
    host="local",
    totalseg_parallel=False,
    spine_range="all",
    original_nifti_path=None
):
    """
    ct_bytes: path to nifti file or 
              bytes object of zip file or
              bytes object of nifti file
    f_type: "niigz" or "nii" or "dicom" 
    """
    yield {"id": 1, "progress": 5, "status": "Loading files"}

    if tmp_dir is None:
        tmp_dir = Path(tempfile.mkdtemp())
        delete_tmp = True
        delete_aux_files = True
    
    if logger is None:
        logger = setup_logger(tmp_dir / "log.txt")
        
    ct_path = tmp_dir / "ct.nii.gz"
    if isinstance(ct_bytes, Path):  # for local usage
        ct_path = ct_bytes
    elif isinstance(ct_bytes, nib.Nifti1Image):  # for local usage
        nib.save(ct_bytes, ct_path)
    elif f_type == "dicom":  # for online zip file bytes
        if ct_path.exists():
            logger.info("Found existing nifti files. Skipping conversion...")
        else:
            logger.info("Converting dicom to nifti...")
            dcm_to_nifti(ct_bytes, ct_path, tmp_dir=tmp_dir, verbose=True)
    elif f_type == "niigz":  # for online nifti bytes
        nib.save(filestream_to_nifti(ct_bytes, gzipped=True), ct_path)
    else:  # for online nifti bytes
        nib.save(filestream_to_nifti(ct_bytes, gzipped=False), ct_path)
        # ct_path = ct_bytes

    yield {"id": 2, "progress": 20, "status": "Running TotalSegmentator"}
    # logger.info("Running TotalSegmentator...")

    # consecutive execution 
    if not totalseg_parallel:
        res = run_models_abdomen(ct_path, tmp_dir, logger)
        for idx, r in enumerate(res):
            yield {"id": 2, "progress": 20+idx*2, "status": r}
    else:
        # parallel execution asyncio
        import asyncio
        from totalsegmentator.spine_report.utils_report import run_models_abdomen_parallel
        _ = asyncio.run(run_models_abdomen_parallel(ct_path, tmp_dir, logger, host))

        # parallel execution threads
        # from totalsegmentator.spine_report.utils_report import run_models_parallel_threads
        # run_models_parallel_threads(ct_path, tmp_dir, logger)

    yield {"id": 3, "progress": 40, "status": "Detecting contrast phase"}
    contrast_phase = get_contrast_phase(ct_path, tmp_dir, logger, host, original_nifti_path=original_nifti_path)

    yield {"id": 4, "progress": 50, "status": "Loading additional data"}

    subject_age = metadata["PatientAge"]
    subject_sex = metadata["PatientSex"]

    # Load CT
    st = time.time()
    ct_img = nib.load(ct_path)
    ct_img = nib.as_closest_canonical(ct_img)
    # label_map: a dictionary {label_id : label_name}
    # vertebrae_pp_refined contains per-vertebra body segmentations (C1-L5).
    verte_img, verte_label_map = load_multilabel_nifti(tmp_dir / "totalseg_vertebrae_pp_refined.nii.gz")
    verte_img = nib.as_closest_canonical(verte_img)
    verte_body_img = verte_img
    verte_body_label_map = {}
    
    yield {"id": 5, "progress": 60, "status": "Measuring vertebral heights"}

    get_verte_height(ct_img, 
                     verte_img, verte_label_map, 
                     verte_body_img, verte_body_label_map, 
                     tmp_dir / "spine_report_preview_DEBUG.png", 
                     tmp_dir / "spine_report_combined_preview.png", 
                     tmp_dir / "spine_report_heights.json", 
                     debug, smoothed_body_out=None, horizontal_mask_type="raw",
                     spine_range=spine_range)

    yield {"id": 6, "progress": 90, "status": "Generating report image"}

    report_json = json.load(open(tmp_dir / "spine_report_heights.json", "r"))

    # Check if lumbar spine contains a fracture
    vertebrae_high_height_diff = []
    vertebrae_low_intensity = []
    # for vertebrae in ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5"]:
    for vertebrae in ["L1", "L2", "L3", "L4", "L5"]:
        vertebrae_name = f"vertebrae_{vertebrae}"
        if f"{vertebrae_name}_intensity" in report_json:
            height_diff = report_json[f"{vertebrae_name}_height_difference"]
            intensity = report_json[f"{vertebrae_name}_intensity_median"]

            if contrast_phase == "arterial":
                hu_thr = HU_THRESHOLD + ART_HU_ADDON
            elif contrast_phase == "portal_venous":
                hu_thr = HU_THRESHOLD + PV_HU_ADDON
            else:
                hu_thr = HU_THRESHOLD

            if height_diff > 0.2:
                vertebrae_high_height_diff.append(vertebrae)
            if intensity < hu_thr:
                vertebrae_low_intensity.append(vertebrae)
    
    found_fracture = len(vertebrae_high_height_diff) > 0 or len(vertebrae_low_intensity) > 0

    tp = str(resources.files("totalsegmentator").joinpath("resources/spine_report"))  # templates_path
    template_vars = {"report_image": str(tmp_dir / "spine_report_combined_preview.png"),
                     "found_fracture": found_fracture,
                     "vertebrae_high_height_diff": vertebrae_high_height_diff,
                     "vertebrae_low_intensity": vertebrae_low_intensity,
                     "metadata": metadata,
                     "contrast_phase": contrast_phase.replace("_", " ")}
    report_img = generate_html(tp, "report_template_frontpage.html",
                               template_vars, tmp_dir / "spine_report_frontpage.png", width=1200)

    nifti_report = combine_as_nifti(tmp_dir, logger, nib.load(ct_path))

    yield {"id": 7, "progress": 95, "status": "Combine masks for output"}
    # Add totalsegmentator masks to output
    masks_output = {}
    download_masks = ["totalseg_vertebrae_pp_refined"]
    for roi_name in download_masks:
        mask = nib.load(tmp_dir / f"{roi_name}.nii.gz")
        # _ = mask.get_fdata()  # load data into cache -> still refers to file -> have to recreate image
        masks_output[roi_name] = nib.Nifti1Image(mask.get_fdata().astype(np.uint8), mask.affine, mask.header)

    if tmp_dir is not None and delete_tmp:
        shutil.rmtree(tmp_dir)

    if delete_aux_files:
        for f in tmp_dir.glob("*.png"):
            f.unlink()

    # Nifti to Dicom(SR) options:
    # - high dicom: https://highdicom.readthedocs.io/en/latest/quickstart.html#creating-structured-report-sr-documents
    # - https://github.com/tomaroberts/nii2dcm
    # - do myself with pydicom + ChatGPT
    yield {"id": 8, "progress": 96, "status": "Returning data"}

    yield {"id": 8, "progress": 100, "status": "Done",
           "report_img": serialize_and_compress(nifti_report), 
           "report_json": report_json, 
           "masks": serialize_and_compress(masks_output)}


def main():
    """
    Command to test locally 

    cd /mnt/nvme/data/test_data/spine_report/29268064
    OR
    cd /mnt/nvme/data/test_data/spine_report/29276218_2

    totalseg_spine_report -i ct.nii.gz -o spine_report.nii.gz -j spine_report.json -l spine_report.log --debug
    """
    parser = argparse.ArgumentParser(description="Generate spine report.")
    parser.add_argument("-i", "--ct_path", type=Path, required=True,
                        help="Path to ct file.")
    parser.add_argument("-n", "--nodeinfo", type=Path, required=False,
                        help="Path to nodeinfo file.")
    parser.add_argument("-o", "--output_nifti", type=Path, required=True,
                        help="Path to output nifti file")
    parser.add_argument("-j", "--output_json", type=Path, required=True,
                        help="Path to output json file")
    parser.add_argument("-l", "--output_log", type=Path, required=True,
                        help="Path to output log file")
    parser.add_argument("-tmp", "--tmp_dir", type=Path, 
                        help="Path to tmp dir. If not set, then use system tmp dir.")
    parser.add_argument("-t", "--test", type=str, default="None",
                        help="Define which test to run.")  # currently not used; has no effect
    parser.add_argument("-on", "--original_nifti_path", type=Path, default=None,
                        help="Path to original nifti file (for contrast phase detection).")
    parser.add_argument("-sr", "--spine_range", choices=['all', 'thoracic_lumbar', 'lumbar', 'l1_l4'], default='all',
                        help="Limit analysis to a specific vertebral range")
    parser.add_argument("-tp", "--totalseg_parallel", action="store_true",
                        help="Run all totalsegmentator runs in parallel. Faster, but needs more RAM + GPU memory.")
    parser.add_argument("-r", "--save_runtime", action="store_true",
                        help="Save runtime to META/runtime.json file.")
    parser.add_argument("--debug", action="store_true",
                        help="If debug use other tmp dir.")
    parser.add_argument('--version', action='version', version=VERSION)
    args = parser.parse_args()

    if args.debug:
        print("Running in DEBUG mode.")
        tmp_dir = args.output_nifti.absolute().parent
        print(f"tmpdir: {tmp_dir}")
        delete_tmp = False
        delete_aux_files = False
    elif args.tmp_dir:
        tmp_dir = args.tmp_dir.absolute()
        delete_tmp = False
        delete_aux_files = True
    else:
        tmp_dir = Path(tempfile.mkdtemp())
        delete_tmp = True
        delete_aux_files = True

    tmp_dir.mkdir(exist_ok=True)

    logger = setup_logger(args.output_log)

    logger.info("Getting metadata...")
    if args.nodeinfo is not None and args.nodeinfo.exists():
        with open(args.nodeinfo, "r") as f:
            metadata = json.load(f)
    else:
        logger.info("WARNING: Could not find nodeinfo.json. Using default metadata.")
        metadata = {
            "PatientName": "Unknown",
            "PatientSex": "M",
            "PatientAge": "099Y",
            "PatientBirthDate": "19010101",
            "PatientID": "Unknown",
            "StudyID": "Unknown",
            "StudyDescription": "Unknown",
            "StudyDate": "20250101",
        } 
    if 'PatientAge' not in metadata or metadata['PatientAge'] in ('', 'Unknown', None):
        patient_birth_date = metadata.get('PatientBirthDate')
        study_date = metadata.get('StudyDate')
        try:
            birth_date = datetime.strptime(patient_birth_date, '%Y%m%d')
            study_date = datetime.strptime(study_date, '%Y%m%d')
        except (TypeError, ValueError):
            metadata['PatientAge'] = 120
            logger.info("Patient age or birth date missing. Using default age: 120 years")
        else:
            metadata['PatientAge'] = int((study_date - birth_date).days / 365.25)
            logger.info(f"Calculated age from dates: {metadata['PatientAge']} years")
    else:
        metadata["PatientAge"] = int(str(metadata["PatientAge"]).replace("Y", ""))
    logger.info(f"age: {metadata['PatientAge']}")
    metadata["version"] = VERSION

    img_shape = nib.load(args.ct_path).shape

    if max(img_shape) > 1500:
        logger.info("WARNING: Image shape is very large. Stopping.")
    elif min(img_shape) < 20:
        logger.info("WARNING: Image shape is very small. Stopping.")
    elif len(img_shape) != 3:
        logger.info("WARNING: Image shape is not 3D. Stopping.")
    else:
        logger.info("Creating report...")
        
        start_time = time.time()

        res = create_spine_report(
            args.ct_path,
            metadata,
            tmp_dir,
            logger,
            delete_tmp,
            delete_aux_files,
            args.debug,
            args.test,
            host="local",
            totalseg_parallel=args.totalseg_parallel,
            spine_range=args.spine_range,
            original_nifti_path=args.original_nifti_path
        )

        for r in res:
            print(f"progress: {r['progress']}, status: {r['status']}")
            if r["progress"] == 100:
                final_result = r

        final_result["report_img"] = decompress_and_deserialize(final_result["report_img"])

        nib.save(final_result["report_img"], args.output_nifti)

        json.dump({"results": final_result["report_json"],
                   "metadata": metadata},
                  open(args.output_json, "w"), indent=4)

        if args.save_runtime and args.nodeinfo is not None and args.nodeinfo.exists():
            save_runtime(start_time, "spine_report", args.nodeinfo)

    send_usage_stats_application("spine_report")

"""
Runtimes:
(total runtime)
(for ct_15mm.nii.gz image)

locally:
- consecutive: 3:08
- asyncio: 1:59
- threads: 1:57

locally, but totalseg on modal:
- consecutive: ?
- asyncio: 2:55
- threads: 3:02

"""


if __name__ == "__main__":
    main()
