import asyncio
import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes

from totalsegmentator.aorta_report import VERSION
from totalsegmentator.aorta_report.constants import VESSELS_TO_PLOT
from totalsegmentator.aorta_report.cpr import cpr
from totalsegmentator.aorta_report.landmarks import (
    attach_structure_anchors,
    build_centerline,
    create_landmarks,
    create_structures,
)
from totalsegmentator.aorta_report.measurements import (
    create_landmark_planes,
    diameter_profiles,
    measure_landmarks,
    measure_sections,
)
from totalsegmentator.aorta_report.model_runner import (
    get_aorta_fast,
    get_contrast_phase,
    run_models_consecutive,
    run_models_parallel,
)
from totalsegmentator.aorta_report.plotting import plot_cpr_overview
from totalsegmentator.aorta_report.rendering import (
    cleanup_workdir,
    prepare_results,
    render_previews,
    render_report_image,
    serialize_outputs,
)
from totalsegmentator.aorta_report.utils import (
    crop_to_aorta,
    crop_to_masks,
    keep_largest_blob,
    lazy_load,
    remove_small_blobs,
)
from totalsegmentator.dicom_io import dcm_to_nifti
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
        logger = setup_logger(tmp_dir / "log.txt", name=f"totalseg_aorta_report.{tmp_dir}")
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


def _run_segmentation_models(ct_path, model_dir, logger, test, models_parallel, host):
    get_aorta_fast(ct_path, model_dir, logger, "v2", host)
    ct_path = crop_to_aorta(ct_path, model_dir, logger, "v2")
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


def _load_report_data(
    ct_path,
    rois_totalseg_dir,
    rois_details_dir,
    aorta_fn,
    annulus_fn,
    tmp_dir,
    contrast_phase,
    skip_dissection,
    logger,
    debug,
):
    structures = create_structures()
    annulus_img = nib.as_closest_canonical(nib.load(rois_details_dir / annulus_fn))
    if annulus_img.affine[:3, :3].min() < 0:
        logger.info("WARNING: Affine contains negative values!")
    aorta_crop = nib.as_closest_canonical(nib.load(rois_totalseg_dir / aorta_fn))
    aorta_crop = nib.Nifti1Image(
        keep_largest_blob(aorta_crop.get_fdata(dtype=np.float32)),
        aorta_crop.affine,
    )
    annulus_img = crop_to_masks(annulus_img, [aorta_crop], [20, 20, 20], dtype=np.uint8)
    annulus_img = change_spacing(annulus_img, 0.8, order=0)
    annulus = annulus_img.get_fdata(dtype=np.float32)
    annulus_volume = annulus.sum() * np.prod(annulus_img.header.get_zooms())
    if annulus_volume < 200:
        logger.info("WARNING: Annulus is empty!")
    if debug:
        nib.save(annulus_img, tmp_dir / "annulus.nii.gz")

    affine, spacing = annulus_img.affine, annulus_img.header.get_zooms()
    ct_img, _ = lazy_load(
        ct_path, annulus_img, tmp_dir, order=3, is_mask=False
    )
    aorta_img, aorta = lazy_load(
        rois_totalseg_dir / aorta_fn,
        annulus_img,
        tmp_dir,
        order=0,
        is_mask=True,
    )
    aorta_totalseg = binary_fill_holes(aorta).astype(np.uint8)

    true_path = rois_details_dir / "aorta_true_lumen.nii.gz"
    calculate_lumen = true_path.exists() and not skip_dissection and contrast_phase != "native"
    if not true_path.exists():
        logger.info("WARNING: No true/false lumen found! Using aorta + empty array instead.")
    elif skip_dissection:
        logger.info("WARNING: skip_dissection=True! Using aorta + empty array instead.")
    elif contrast_phase == "native":
        logger.info("WARNING: contrast phase is native. Not using true/false lumen segmentation!")
    if calculate_lumen:
        _, true_lumen = lazy_load(
            true_path, annulus_img, tmp_dir, order=3, is_mask=True
        )
        _, false_lumen = lazy_load(
            rois_details_dir / "aorta_false_lumen.nii.gz",
            annulus_img,
            tmp_dir,
            order=3,
            is_mask=True,
        )
    else:
        true_lumen, false_lumen = aorta.copy(), np.zeros_like(aorta)
    false_lumen = remove_small_blobs(false_lumen, [20, 1e9])
    aorta[true_lumen > 0.5] = 1
    aorta[false_lumen > 0.5] = 1
    aorta = binary_fill_holes(aorta).astype(np.uint8)
    true_lumen = aorta.copy()
    true_lumen[false_lumen > 0.5] = 0
    annulus = binary_closing(annulus, iterations=1)

    for name, structure in structures.items():
        directory = rois_details_dir if structure["roi_dir"] == "details" else rois_totalseg_dir
        _, data = lazy_load(
            directory / f"{structure['name']}.nii.gz",
            annulus_img,
            tmp_dir,
            order=0,
            is_mask=True,
        )
        if data.sum() == 0:
            logger.info(f"WARNING: {name} is empty!")
        structure["data"] = data
    all_vessels = np.zeros_like(aorta)
    for vessel in VESSELS_TO_PLOT:
        _, data = lazy_load(
            rois_totalseg_dir / f"{vessel}.nii.gz",
            annulus_img,
            tmp_dir,
            order=0,
            is_mask=True,
        )
        all_vessels[data > 0.5] = 1
    return {
        "structures": structures,
        "annulus_img": annulus_img,
        "annulus": annulus,
        "annulus_volume": annulus_volume,
        "affine": affine,
        "spacing": spacing,
        "ct_img": ct_img,
        "aorta_img": aorta_img,
        "aorta": aorta,
        "aorta_totalseg": aorta_totalseg,
        "true_lumen": true_lumen,
        "false_lumen": false_lumen,
        "all_vessels": all_vessels,
    }


def create_aorta_report(
    ct_bytes,
    rois_totalseg_dir,
    rois_details_dir,
    metadata,
    tmp_dir,
    logger,
    delete_tmp=True,
    delete_aux_files=True,
    cpr_path=None,
    cpr_animated_path=None,
    aorta_fn="aorta.nii.gz",
    annulus_fn="annulus.nii.gz",
    test="None",
    debug=False,
    run_models=False,
    models_parallel=False,
    f_type="nii",
    host="local",
    erosion=False,
    version=VERSION,
    skip_dissection=False,
):
    """Generate an aorta report and yield source-compatible progress dictionaries."""
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
            for idx, status in enumerate(statuses):
                yield {"id": 3, "progress": 10 + idx * 2, "status": status}
            rois_totalseg_dir = rois_details_dir = model_dir

        rois_totalseg_dir, rois_details_dir = Path(rois_totalseg_dir), Path(rois_details_dir)
        yield {"id": 3, "progress": 19, "status": "Running contrast phase detection"}
        contrast_phase = get_contrast_phase(ct_path, model_dir, logger, host)
        logger.info(f"Contrast phase: {contrast_phase}")

        yield {"id": 4, "progress": 20, "status": "Loading data"}
        data = _load_report_data(
            ct_path,
            rois_totalseg_dir,
            rois_details_dir,
            aorta_fn,
            annulus_fn,
            tmp_dir,
            contrast_phase,
            skip_dissection,
            logger,
            debug,
        )

        yield {"id": 5, "progress": 25, "status": "Creating centerline"}
        started = time.time()
        centerline_image, centerline, centerline_resampled = build_centerline(
            data["aorta"],
            data["annulus"],
            data["affine"],
            data["spacing"],
            logger,
            debug,
        )
        logger.info(f"  get_centerline took {time.time() - started:.2f}s")
        structures = attach_structure_anchors(
            data["structures"], data["aorta"], centerline, data["spacing"], logger
        )
        landmarks = create_landmarks(
            structures,
            data["annulus_volume"],
            centerline,
            data["aorta_img"],
            data["spacing"],
            logger,
        )

        yield {"id": 6, "progress": 35, "status": "Getting max diameter"}
        max_diameter = create_landmark_planes(
            landmarks,
            centerline,
            data["aorta"],
            data["annulus"],
            data["spacing"],
            logger,
            erosion,
        )
        measure_landmarks(
            landmarks,
            data["ct_img"],
            data["true_lumen"],
            data["false_lumen"],
            data["affine"],
            max_diameter,
            tmp_dir,
            logger,
        )
        section_stats, _ = measure_sections(
            landmarks,
            data["aorta_img"],
            data["aorta_totalseg"],
            data["true_lumen"],
            data["false_lumen"],
            centerline,
            data["spacing"],
            max_diameter,
            logger,
        )

        animated_cpr = None
        if (cpr_path is not None or cpr_animated_path is not None) and not centerline:
            logger.info("WARNING: Skipping CPR because the centerline is empty.")
        elif cpr_path is not None or cpr_animated_path is not None:
            logger.info("Calculating CPR...")
            cpr_started = time.time()
            res_ct, res_seg, lumen_images, cpr_info = cpr(
                data["ct_img"],
                data["aorta_img"],
                centerline,
                max_diameter,
                fast=debug,
                debug=debug,
                extra_mask_imgs=[
                    nib.Nifti1Image(data["true_lumen"].astype(np.uint8), data["affine"]),
                    nib.Nifti1Image(data["false_lumen"].astype(np.uint8), data["affine"]),
                ],
                return_info=True,
            )
            true_cpr, false_cpr = lumen_images
            logger.info(f"CPR took {time.time() - cpr_started:.2f}s")
            logger.info("Calculating diameter profile on original masks...")
            profile_started = time.time()
            curve_positions, profiles = diameter_profiles(
                [data["aorta_totalseg"], data["true_lumen"], data["false_lumen"]],
                centerline_resampled,
                data["spacing"],
                max(max_diameter * 1.6, 60),
            )
            aorta_curve, true_curve, false_curve = profiles
            logger.info(f"Diameter profile took {time.time() - profile_started:.2f}s")
            if cpr_path is not None:
                plot_cpr_overview(
                    res_ct,
                    res_seg,
                    cpr_path,
                    -700,
                    1000,
                    true_cpr,
                    false_cpr,
                    landmarks,
                    np.asarray([vertex.point for vertex in centerline]),
                    data["affine"],
                    cpr_info["sample_positions_mm"],
                    curve_positions,
                    aorta_curve,
                    true_curve,
                    false_curve,
                    metadata,
                    tmp_dir,
                )
            if cpr_animated_path is not None:
                animated_cpr = {
                    "aorta": data["aorta"],
                    "cl": centerline,
                    "affine": data["affine"],
                    "res_ct_img": res_ct,
                    "res_seg_img": res_seg,
                    "true_lumen_cpr_img": true_cpr,
                    "false_lumen_cpr_img": false_cpr,
                    "cpr_info": cpr_info,
                    "curve_positions_mm": curve_positions,
                    "aorta_curve_mm": aorta_curve,
                    "true_curve_mm": true_curve,
                    "false_curve_mm": false_curve,
                    "landmarks": landmarks,
                    "max_dia_all": max_diameter,
                    "output_path": cpr_animated_path,
                    "tmp_dir": tmp_dir,
                    "smoothing": 100,
                    "logger": logger,
                    "metadata": metadata,
                }

        yield {"id": 7, "progress": 85, "status": "Generating report images"}
        render_previews(
            data["aorta"],
            data["true_lumen"],
            data["false_lumen"],
            data["all_vessels"],
            centerline_image,
            landmarks,
            tmp_dir,
            debug,
            animated_cpr,
        )
        landmarks = dict(sorted(landmarks.items()))
        yield {"id": 8, "progress": 90, "status": "Generating report html"}
        report_image = render_report_image(landmarks, section_stats, metadata, tmp_dir)
        results = prepare_results(landmarks, section_stats, metadata)

        yield {"id": 9, "progress": 95, "status": "Combine masks for output"}
        serialized_report, serialized_masks = serialize_outputs(
            report_image, results, tmp_dir, aorta_fn.split(".")[0], logger
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


__all__ = ["create_aorta_report", "diameter_profiles", "VERSION"]
