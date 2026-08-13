import asyncio
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes

from totalsegmentator.aorta_report.utils import crop_to_masks, keep_largest_blob, lazy_load
from totalsegmentator.dicom_io import dcm_to_nifti
from totalsegmentator.pulmonary_report import VERSION
from totalsegmentator.pulmonary_report.constants import (
    REPORT_FRAMES,
    REPORT_SMOOTHING,
    create_landmarks,
)
from totalsegmentator.pulmonary_report.measurements import (
    create_landmark_planes,
    measure_landmarks,
)
from totalsegmentator.pulmonary_report.model_runner import (
    crop_to_pulmonary,
    get_pulmonary_fast,
    run_models_consecutive,
    run_models_parallel,
)
from totalsegmentator.pulmonary_report.plotting import plot_pulmonary_seg
from totalsegmentator.pulmonary_report.rendering import (
    cleanup_workdir,
    prepare_results,
    render_report_image,
    serialize_outputs,
)
from totalsegmentator.rendering import fury_display_context
from totalsegmentator.reporting import setup_logger
from totalsegmentator.resampling import change_spacing
from totalsegmentator.serialization_utils import filestream_to_nifti


def _prepare_context(tmp_dir, logger, delete_tmp, delete_aux_files):
    if tmp_dir is None:
        tmp_dir = Path(tempfile.mkdtemp())
        delete_tmp = delete_aux_files = True
    tmp_dir = Path(tmp_dir)
    model_dir = tmp_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    if logger is None:
        logger = setup_logger(
            tmp_dir / "log.txt", name=f"totalseg_pulmonary_report.{tmp_dir}"
        )
    return tmp_dir, model_dir, logger, delete_tmp, delete_aux_files


def _resolve_ct_path(ct_input, model_dir, tmp_dir, f_type, logger):
    ct_path = model_dir / "ct.nii.gz"
    if isinstance(ct_input, Path):
        return ct_input
    if isinstance(ct_input, nib.Nifti1Image):
        nib.save(ct_input, ct_path)
    elif f_type == "dicom":
        if ct_path.exists():
            logger.info("Found existing nifti files. Skipping conversion...")
        else:
            logger.info("Converting dicom to nifti...")
            dcm_to_nifti(ct_input, ct_path, tmp_dir=tmp_dir, verbose=True)
    else:
        nib.save(filestream_to_nifti(ct_input, gzipped=f_type == "niigz"), ct_path)
    return ct_path


def _run_segmentation_models(
    ct_path, model_dir, logger, test, models_parallel, host
):
    get_pulmonary_fast(ct_path, model_dir, logger, host)
    ct_path = crop_to_pulmonary(ct_path, model_dir, logger)
    new_spacing = 1.5 if test == "basic" else 0.8
    ct_img = change_spacing(nib.load(ct_path), new_spacing, order=3, dtype=np.int16)
    nib.save(ct_img, ct_path)
    print(f"Resampled CT to {new_spacing}mm: {ct_img.shape}")
    if models_parallel:
        asyncio.run(run_models_parallel(ct_path, model_dir, logger, host))
        statuses = ()
    else:
        statuses = run_models_consecutive(ct_path, model_dir, logger, host)
    return ct_path, statuses


def _validate_inputs(rois_totalseg_dir, rois_details_dir, pulmonary_fn, landmarks):
    if rois_totalseg_dir is None:
        raise FileNotFoundError(
            "--rois_totalseg is required when --run_models is not used."
        )
    if rois_details_dir is None:
        raise FileNotFoundError(
            "--rois_details is required when --run_models is not used."
        )
    required = [rois_totalseg_dir / pulmonary_fn]
    required.extend(
        rois_details_dir / f"{landmark['name']}.nii.gz"
        for landmark in landmarks.values()
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Pulmonary report requires precomputed pulmonary ROI masks when "
            "--run_models is not used.\nMissing files:\n  - "
            + "\n  - ".join(str(path) for path in missing)
        )


def _load_report_data(
    ct_path,
    rois_totalseg_dir,
    rois_details_dir,
    pulmonary_fn,
    tmp_dir,
    logger,
    debug,
):
    landmarks = create_landmarks()
    _validate_inputs(
        rois_totalseg_dir, rois_details_dir, pulmonary_fn, landmarks
    )
    annulus_img = nib.as_closest_canonical(
        nib.load(rois_details_dir / "pul_annulus.nii.gz")
    )
    if annulus_img.affine[:3, :3].min() < 0:
        logger.info("WARNING: Affine contains negative values!")

    pulmonary_crop = nib.as_closest_canonical(
        nib.load(rois_totalseg_dir / pulmonary_fn)
    )
    pulmonary_crop = nib.Nifti1Image(
        keep_largest_blob(pulmonary_crop.get_fdata(dtype=np.float32)),
        pulmonary_crop.affine,
    )
    annulus_img = crop_to_masks(
        annulus_img, [pulmonary_crop], [20, 20, 20], dtype=np.uint8
    )
    annulus_img = change_spacing(annulus_img, 0.8, order=0)
    annulus = annulus_img.get_fdata(dtype=np.float32)
    annulus_volume = annulus.sum() * np.prod(annulus_img.header.get_zooms())
    if annulus_volume < 200:
        logger.info("WARNING: Annulus is empty!")
    if debug:
        nib.save(annulus_img, tmp_dir / "pul_annulus.nii.gz")

    ct_img, _ = lazy_load(
        ct_path, annulus_img, tmp_dir, order=3, is_mask=False
    )
    _, pulmonary_artery = lazy_load(
        rois_totalseg_dir / pulmonary_fn,
        annulus_img,
        tmp_dir,
        order=0,
        is_mask=True,
    )
    pulmonary_artery = binary_fill_holes(pulmonary_artery).astype(np.uint8)
    annulus = binary_closing(annulus, iterations=1)

    for landmark in landmarks.values():
        _, data = lazy_load(
            rois_details_dir / f"{landmark['name']}.nii.gz",
            annulus_img,
            tmp_dir,
            order=3,
            is_mask=True,
        )
        if data.sum() == 0:
            logger.info(f"WARNING: {landmark['name']} is empty!")
            landmark["empty"] = True
        landmark["data"] = data

    return {
        "landmarks": landmarks,
        "ct_img": ct_img,
        "pulmonary_artery": pulmonary_artery,
        "affine": annulus_img.affine,
        "spacing": annulus_img.header.get_zooms(),
    }


def create_pulmonary_report(
    ct_bytes,
    rois_totalseg_dir,
    rois_details_dir,
    metadata,
    tmp_dir,
    logger,
    delete_tmp=True,
    delete_aux_files=True,
    pulmonary_fn="pulmonary_artery.nii.gz",
    test="None",
    debug=False,
    run_models=False,
    models_parallel=False,
    f_type="nii",
    host="local",
    version=VERSION,
):
    """Generate a pulmonary artery report and yield progress dictionaries."""
    yield {"id": 1, "progress": 2, "status": "Loading code"}
    tmp_dir, model_dir, logger, delete_tmp, delete_aux_files = _prepare_context(
        tmp_dir, logger, delete_tmp, delete_aux_files
    )
    completed = False
    try:
        ct_path = _resolve_ct_path(ct_bytes, model_dir, tmp_dir, f_type, logger)
        if run_models:
            yield {"id": 2, "progress": 5, "status": "Running models"}
            ct_path, statuses = _run_segmentation_models(
                ct_path, model_dir, logger, test, models_parallel, host
            )
            for index, status in enumerate(statuses):
                yield {
                    "id": 3,
                    "progress": 10 + index * 2,
                    "status": status,
                }
            rois_totalseg_dir = rois_details_dir = model_dir

        rois_totalseg_dir = (
            Path(rois_totalseg_dir) if rois_totalseg_dir is not None else None
        )
        rois_details_dir = (
            Path(rois_details_dir) if rois_details_dir is not None else None
        )
        yield {"id": 4, "progress": 20, "status": "Loading data"}
        data = _load_report_data(
            ct_path,
            rois_totalseg_dir,
            rois_details_dir,
            pulmonary_fn,
            tmp_dir,
            logger,
            debug,
        )

        yield {"id": 6, "progress": 35, "status": "Getting max diameter"}
        max_diameter = create_landmark_planes(
            data["landmarks"],
            data["pulmonary_artery"],
            data["spacing"],
            logger,
        )
        measure_landmarks(
            data["landmarks"],
            data["ct_img"],
            data["affine"],
            max_diameter,
            tmp_dir,
            logger,
            debug,
        )

        yield {"id": 7, "progress": 85, "status": "Generating report images"}
        with fury_display_context():
            plot_pulmonary_seg(
                data["pulmonary_artery"],
                data["landmarks"],
                tmp_dir,
                REPORT_SMOOTHING,
                REPORT_FRAMES,
                debug,
            )

        yield {"id": 8, "progress": 90, "status": "Generating report html"}
        report_image = render_report_image(data["landmarks"], metadata, tmp_dir)
        results = prepare_results(data["landmarks"], metadata)
        yield {"id": 9, "progress": 95, "status": "Combine masks for output"}
        serialized_report, serialized_masks = serialize_outputs(
            report_image, results, tmp_dir, logger
        )
        cleanup_workdir(tmp_dir, delete_tmp, delete_aux_files)
        completed = True
        yield {"id": 10, "progress": 96, "status": "Returning data"}
        yield {
            "id": 11,
            "progress": 100,
            "status": "Done",
            "report_img": serialized_report,
            "report_json": results,
            "masks": serialized_masks,
        }
    finally:
        if not completed and delete_tmp and tmp_dir.exists():
            cleanup_workdir(tmp_dir, delete_tmp=True, delete_aux_files=False)


__all__ = ["create_pulmonary_report", "VERSION"]
