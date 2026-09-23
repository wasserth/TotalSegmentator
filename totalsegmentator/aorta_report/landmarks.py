from collections import defaultdict

import numpy as np
from scipy.ndimage import binary_dilation

from totalsegmentator.aorta_report.centerline import (
    add_point_to_centerline,
    extend_centerline,
    get_centerline,
    get_closest_centerline_section,
    get_closest_point_of_centerline,
    get_index_for_point,
    get_len_path_mm,
    get_mid_of_points,
    get_next_point_by_distance,
    get_normal_for_cl_point,
    reorder_centerline,
    resample_cl_to_same_distance,
)
from totalsegmentator.aorta_report.constants import (
    LANDMARK_DEPENDENCIES,
    LANDMARK_NAMES,
    STRUCTURE_DEFINITIONS,
)
from totalsegmentator.aorta_report.geometry import find_center, find_most_distant_points, smooth_mask
from totalsegmentator.aorta_report.geometry_report import get_max_diameter_of_centerline_section


def create_structures():
    return {
        key: {
            "name": mask_name,
            "roi_dir": roi_dir,
            "data": None,
            "endpoint": None,
            "cl_point": None,
            "cl_idx": None,
            "cl_normal": None,
            "empty": False,
        }
        for key, (mask_name, roi_dir) in STRUCTURE_DEFINITIONS.items()
    }


def build_centerline(aorta, annulus, affine, spacing, logger, debug=False):
    annulus_center = find_center(annulus)
    aorta_smooth = smooth_mask(binary_dilation(aorta, iterations=4), sigma=6)
    centerline_image, centerline = get_centerline(aorta_smooth, debug=debug)
    centerline = reorder_centerline(centerline)
    if annulus_center is None:
        logger.info(
            "WARNING: Annulus mask is empty. Building the centerline without an annulus anchor."
        )
    else:
        centerline_image, centerline = add_point_to_centerline(
            centerline_image, centerline, annulus_center
        )
        extended_image, extended = extend_centerline(centerline_image, centerline, 1)
        if annulus[tuple(centerline[-1].point.astype(int))]:
            logger.info("  New annulus point is in annulus mask!")
            centerline_image, centerline = extended_image, extended
        else:
            logger.info(
                "WARNING: New annulus point is not in annulus mask! Skipping the extension."
            )

    if get_len_path_mm(centerline, spacing) < 2:
        logger.info("WARNING: Centerline is very short (<2mm). This may indicate issues with the input data.")
    resampled = resample_cl_to_same_distance(centerline, affine, 1)
    return centerline_image, centerline, resampled


def attach_structure_anchors(structures, aorta, centerline, spacing, logger):
    for name, structure in structures.items():
        if name in ("T12", "sinotub_junc", "iliac"):
            intersection = structure["data"] if structure["data"].sum() * np.prod(spacing) >= 200 else np.zeros_like(structure["data"])
        else:
            intersection = (binary_dilation(structure["data"], iterations=1) + aorta) > 1
        structure["intersect_aorta"] = intersection
        if not intersection.any():
            logger.info(f"WARNING: {name} is (almost) empty or has no intersection with aorta! Setting to empty.")
            structure["empty"] = True

    if structures["brachio"]["empty"] or structures["subclavian"]["empty"]:
        structures["brachio"]["empty"] = structures["subclavian"]["empty"] = True
        logger.info("WARNING: Setting subclavian+brachio to empty because one or both are empty.")
        branch_points = {"brachio": None, "subclavian": None}
    else:
        branch_points = dict(
            zip(
                ("brachio", "subclavian"),
                find_most_distant_points(
                    structures["brachio"]["intersect_aorta"],
                    structures["subclavian"]["intersect_aorta"],
                ),
            )
        )

    for name, structure in structures.items():
        if structure["empty"]:
            continue
        intersection = structure["intersect_aorta"]
        if name in ("celiac", "iliac"):
            indexes = np.where(intersection > 0)
            top = np.argmax(indexes[2])
            endpoint = tuple(indexes[axis][top] for axis in range(3))
        elif name in branch_points:
            endpoint = tuple(branch_points[name])
        else:
            endpoint = tuple(find_center(intersection, True))
        structure["endpoint"] = endpoint

        if name == "sinotub_junc" and not structures["brachio"]["empty"]:
            segment = centerline[structures["brachio"]["cl_idx"] :]
        elif name == "T12":
            segment = get_closest_centerline_section(endpoint, centerline, 40, spacing[2])
        else:
            segment = centerline
        point = get_closest_point_of_centerline(endpoint, segment)
        structure["cl_point"] = point
        structure["cl_idx"] = get_index_for_point(centerline, point)
        if structure["cl_idx"] is None:
            structure["empty"] = True
            logger.info(f"WARNING: Could not map {name} to the centerline. Setting to empty.")
            continue
        structure["cl_normal"] = get_normal_for_cl_point(centerline, structure["cl_idx"])
    return structures


def create_landmarks(structures, annulus_volume, centerline, aorta_img, spacing, logger):
    structures["annulus"] = {"empty": annulus_volume < 200}
    landmarks = defaultdict(dict)
    for number, required in LANDMARK_DEPENDENCIES.items():
        landmarks[number]["depends_on"] = list(required)
        landmarks[number]["empty"] = any(structures[name]["empty"] for name in required)
        landmarks[number]["name"] = LANDMARK_NAMES[number]
        for name in required:
            if structures[name]["empty"]:
                logger.info(f"WARNING: landmark {number} is empty because {name} is empty!")

    def set_index(number, index):
        if landmarks[number]["empty"]:
            return
        if index is None:
            landmarks[number]["empty"] = True
            logger.info(f"WARNING: Could not place landmark {number} on the centerline!")
        else:
            landmarks[number]["cl_idx"] = int(index)

    set_index(11, get_next_point_by_distance(centerline, 0, 20, spacing))
    set_index(10, structures["celiac"]["cl_idx"])

    if not landmarks[9]["empty"]:
        if not structures["celiac"]["empty"] and structures["T12"]["cl_idx"] < structures["celiac"]["cl_idx"]:
            logger.info("WARNING: T12 is lower than celiac! Setting to celiac + 10mm.")
            index = get_next_point_by_distance(centerline, structures["celiac"]["cl_idx"], 10, spacing)
        else:
            index = structures["T12"]["cl_idx"]
        set_index(9, index)

    set_index(7, get_next_point_by_distance(centerline, structures["subclavian"]["cl_idx"], -5, spacing))
    if not landmarks[8]["empty"] and not landmarks[7]["empty"] and not landmarks[9]["empty"]:
        set_index(8, get_mid_of_points(centerline, landmarks[7]["cl_idx"], landmarks[9]["cl_idx"]))
    set_index(5, get_next_point_by_distance(centerline, structures["brachio"]["cl_idx"], 5, spacing))
    if not landmarks[6]["empty"] and not landmarks[5]["empty"] and not landmarks[7]["empty"]:
        set_index(6, get_mid_of_points(centerline, landmarks[5]["cl_idx"], landmarks[7]["cl_idx"]))
    set_index(3, structures["sinotub_junc"]["cl_idx"])
    if not landmarks[4]["empty"] and not landmarks[3]["empty"] and not landmarks[5]["empty"]:
        set_index(4, get_mid_of_points(centerline, landmarks[3]["cl_idx"], landmarks[5]["cl_idx"]))
    set_index(1, len(centerline) - 1)

    if not landmarks[2]["empty"] and not landmarks[1]["empty"] and not landmarks[3]["empty"]:
        start = get_next_point_by_distance(centerline, landmarks[3]["cl_idx"], 10, spacing)
        end = get_next_point_by_distance(centerline, landmarks[1]["cl_idx"], -10, spacing)
        if start is not None and end is not None and start < end:
            point = get_max_diameter_of_centerline_section(aorta_img, centerline[start:end], spacing, subsample=1, crop_size=80)
            index = get_index_for_point(centerline, point)
            if index is None:
                logger.info(
                    "WARNING: Sinus centerline section is too short; using its midpoint."
                )
                index = get_mid_of_points(
                    centerline, landmarks[1]["cl_idx"], landmarks[3]["cl_idx"]
                )
            set_index(2, index)
        else:
            logger.info("WARNING: Distance between annulus sinutub. jun. is too small! As robust backup take middle as sinuses of valsalva.")
            set_index(2, get_mid_of_points(centerline, landmarks[1]["cl_idx"], landmarks[3]["cl_idx"]))
    return landmarks
