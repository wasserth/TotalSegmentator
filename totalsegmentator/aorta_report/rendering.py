import gc
import shutil
from importlib import resources

import nibabel as nib
import numpy as np
from PIL import Image

from totalsegmentator.aorta_report.constants import REPORT_FRAMES, REPORT_PLOT_TYPES, REPORT_SMOOTHING
from totalsegmentator.aorta_report.cpr_animated import generate_animated_cpr_nifti
from totalsegmentator.aorta_report.nifti import combine_as_nifti
from totalsegmentator.aorta_report.plotting import plot_aorta_3d, plot_masks_3d
from totalsegmentator.rendering import fury_display_context
from totalsegmentator.reporting import generate_html
from totalsegmentator.serialization_utils import serialize_and_compress


def render_previews(
    aorta,
    true_lumen,
    false_lumen,
    all_vessels,
    centerline_image,
    landmarks,
    tmp_dir,
    debug=False,
    animated_cpr=None,
):
    with fury_display_context():
        plot_aorta_3d(
            aorta,
            true_lumen,
            false_lumen,
            all_vessels,
            centerline_image,
            landmarks,
            tmp_dir,
            REPORT_SMOOTHING,
            REPORT_FRAMES,
            debug,
        )
        plot_masks_3d(
            [true_lumen, false_lumen],
            tmp_dir,
            "preview_tf_lumen_",
            REPORT_SMOOTHING,
            REPORT_FRAMES,
            debug,
            ["green", "red"],
        )
        if animated_cpr is not None:
            generate_animated_cpr_nifti(**animated_cpr)
        gc.collect()


def render_report_image(landmarks, section_stats, metadata, tmp_dir):
    assets_dir = resources.files("totalsegmentator").joinpath("resources/aorta_report")
    first_preview = tmp_dir / f"{REPORT_PLOT_TYPES[0]}{0:06d}.png"
    with Image.open(first_preview) as preview:
        placeholder = Image.new("RGB", preview.size, (255, 0, 255))
    placeholder_path = tmp_dir / "aorta_report_preview_placeholder.png"
    placeholder.save(placeholder_path)
    base_path = tmp_dir / "aorta_report_frontpage_base.png"
    base_image = generate_html(
        str(assets_dir),
        "report_template_frontpage.html",
        {
            "landmarks": landmarks,
            "section_stats": section_stats,
            "preview_3d": str(placeholder_path),
            "metadata": metadata,
        },
        base_path,
        width=900,
        png_logo_hosts=("rigi", "rndapollolp01.uhbs.ch"),
    ).convert("RGB")
    base_data = np.asarray(base_image)
    placeholder_mask = np.all(
        np.abs(base_data.astype(np.int16) - np.array([255, 0, 255], dtype=np.int16))
        <= 5,
        axis=2,
    )
    if not placeholder_mask.any():
        raise RuntimeError("Unable to find preview placeholder in rendered aorta report.")
    ys, xs = np.where(placeholder_mask)
    preview_bbox = (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )
    target_size = (
        preview_bbox[2] - preview_bbox[0],
        preview_bbox[3] - preview_bbox[1],
    )

    for plot_idx, plot_type in enumerate(REPORT_PLOT_TYPES):
        for frame_idx in range(REPORT_FRAMES):
            with Image.open(tmp_dir / f"{plot_type}{frame_idx:06d}.png") as preview:
                preview = preview.convert("RGB")
                if preview.size != target_size:
                    preview = preview.resize(target_size, Image.Resampling.LANCZOS)
                frame = base_image.copy()
                frame.paste(preview, preview_bbox[:2])
                frame.save(
                    tmp_dir
                    / f"aorta_report_frontpage_{plot_idx * REPORT_FRAMES + frame_idx:02d}.png"
                )
    slices = [
        tmp_dir / f"aorta_report_frontpage_{idx:02d}.png"
        for idx in range(len(REPORT_PLOT_TYPES) * REPORT_FRAMES)
    ]
    return combine_as_nifti(slices)


def prepare_results(landmarks, section_stats, metadata):
    landmarks = dict(sorted(landmarks.items()))
    for landmark in landmarks.values():
        if not landmark["empty"]:
            for key in ("roi", "plane_img", "cl_idx", "cl_point"):
                landmark.pop(key, None)
        for key in ("diameter_tmp", "color", "txt_offset", "empty", "depends_on"):
            landmark.pop(key, None)
    return {
        "landmarks": landmarks,
        "section_stats": section_stats,
        "metadata": metadata,
    }


def serialize_outputs(report_image, results, tmp_dir, aorta_name, logger):
    masks = {}
    for name in (aorta_name, "aorta_false_lumen", "aorta_true_lumen"):
        path = tmp_dir / f"{name}.nii.gz"
        if path.exists():
            image = nib.load(path)
            masks[name] = nib.Nifti1Image(
                image.get_fdata(dtype=np.float32).astype(np.uint8),
                image.affine,
                image.header,
            )
        else:
            logger.info(f"WARNING: Mask {name} not found for final output. Skipping.")
    return serialize_and_compress(report_image), serialize_and_compress(masks)


def cleanup_workdir(tmp_dir, delete_tmp, delete_aux_files):
    if delete_aux_files:
        for path in tmp_dir.glob("*.png"):
            if path.name != "cpr.png":
                path.unlink()
        for path in tmp_dir.glob("*.html"):
            path.unlink()
    if delete_tmp:
        shutil.rmtree(tmp_dir)
