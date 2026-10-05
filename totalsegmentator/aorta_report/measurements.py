import nibabel as nib
import numpy as np
from scipy.ndimage import binary_closing, binary_erosion, binary_fill_holes
from tqdm import tqdm

from totalsegmentator.aorta_report.centerline import get_distance, get_len_path_mm, get_normal_for_cl_point
from totalsegmentator.aorta_report.constants import SECTION_LANDMARKS
from totalsegmentator.aorta_report.geometry import find_center, find_mask_normal, get_region_diameters, get_region_diameters_pd
from totalsegmentator.aorta_report.geometry_report import (
    get_2d_plane,
    get_aorta_section,
    get_max_diameter_of_centerline_section,
    get_plane_aorta_intersection,
    process_plane,
)
from totalsegmentator.aorta_report.utils import crop_to_bbox, get_bbox_around_center


def diameter_profiles(mask_data, centerline, spacing, crop_size_mm, subsample=5, target_points=70):
    if not centerline:
        return np.empty(0), [np.empty(0) for _ in mask_data]
    radius = np.maximum(1, (np.asarray([crop_size_mm] * 3) / 2 / np.asarray(spacing)).astype(int))
    stride = max(subsample, int(np.ceil(len(centerline) / max(target_points, 1))))
    indexes = list(range(0, len(centerline), stride))
    if indexes[-1] != len(centerline) - 1:
        indexes.append(len(centerline) - 1)

    points = np.asarray([vertex.point for vertex in centerline])
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0) * np.asarray(spacing), axis=1))))
    result = [np.full(len(indexes), np.nan) for _ in mask_data]
    has_data = [np.count_nonzero(mask) >= 5 for mask in mask_data]
    for output_idx, centerline_idx in enumerate(tqdm(indexes, desc="Diameter profiles")):
        point = points[centerline_idx]
        bbox = get_bbox_around_center(point, radius, mask_data[0])
        cropped_point = point - np.asarray([axis[0] for axis in bbox])
        normal = get_normal_for_cl_point(centerline, centerline_idx)
        for mask_idx, mask in enumerate(mask_data):
            if not has_data[mask_idx]:
                continue
            intersection = get_plane_aorta_intersection(crop_to_bbox(mask, bbox).astype(np.uint8), cropped_point, normal)
            try:
                _, start, end = next(get_region_diameters(intersection))
                result[mask_idx][output_idx] = get_distance(start, end, spacing)
            except StopIteration:
                pass
    return cumulative[-1] - cumulative[indexes], result


def create_landmark_planes(landmarks, centerline, aorta, annulus, spacing, logger, erosion=False):
    for number, landmark in landmarks.items():
        if landmark.get("empty") or landmark.get("cl_idx") is None:
            landmark["empty"] = True
            landmark["diameter_tmp"] = 0
            continue
        point = centerline[landmark["cl_idx"]].point
        if landmark["name"] == "annulus":
            normal = find_mask_normal(annulus.astype(np.uint8))
            plane = get_plane_aorta_intersection(annulus.astype(np.uint8), point, normal).astype(np.uint8)
            if erosion:
                plane = binary_erosion(plane, iterations=1).astype(np.uint8)
        else:
            normal = get_normal_for_cl_point(centerline, landmark["cl_idx"])
            source = binary_erosion(aorta, iterations=1) if erosion else aorta
            plane = get_plane_aorta_intersection(source, point, normal).astype(np.uint8)
        landmark["cl_point"], landmark["roi"] = point, plane
        try:
            _, start, end = next(get_region_diameters(plane))
            landmark["diameter_tmp"] = get_distance(start, end, spacing)
        except StopIteration:
            logger.info(f"WARNING: for landmark {number} no diameter was found!")
            landmark["diameter_tmp"] = 0
    return max(item["diameter_tmp"] for item in landmarks.values())


def measure_landmarks(landmarks, ct_img, true_lumen, false_lumen, affine, max_diameter, tmp_dir, logger):
    empty_metrics = (
        "area",
        "area_fl",
        "area_tl",
        "diameter",
        "diameter_fl",
        "diameter_tl",
        "diameter_pd",
        "diameter_fl_pd",
        "diameter_tl_pd",
        "start",
        "end",
        "start_pd",
        "end_pd",
    )
    for number, landmark in landmarks.items():
        logger.info(f"  Processing landmark {number}")
        if landmark["empty"]:
            landmark.update(dict.fromkeys(empty_metrics))
            continue

        plane_img = nib.Nifti1Image(landmark["roi"], affine)
        false_plane = nib.Nifti1Image(np.logical_and(landmark["roi"], false_lumen).astype(np.uint8), affine)
        true_plane = nib.Nifti1Image(np.logical_and(landmark["roi"], true_lumen).astype(np.uint8), affine)
        plane_2d, ct_2d, false_2d, true_2d, inverse_affine = get_2d_plane(
            plane_img,
            ct_img,
            false_plane,
            true_plane,
            order=3,
            dtype=np.int32,
            addon=[max_diameter * 1.4] * 3,
        )
        image_path = tmp_dir / f"plane_{number}.png"
        process_plane(ct_2d, plane_2d, false_2d, image_path)
        landmark["plane_img"] = str(image_path)

        plane_volume = plane_2d.get_fdata().astype(np.uint8)
        false_volume = false_2d.get_fdata().astype(np.uint8)
        true_volume = true_2d.get_fdata().astype(np.uint8)
        center = find_center(plane_volume, True)
        z = plane_volume.shape[2] // 2 if center is None else center[2]
        plane_data, false_data, true_data = (
            binary_fill_holes(binary_closing(volume[:, :, z])).astype(np.uint8)
            for volume in (plane_volume, false_volume, true_volume)
        )
        slice_spacing = np.asarray(plane_2d.header.get_zooms())[:2]
        for key, data in (("area", plane_data), ("area_fl", false_data), ("area_tl", true_data)):
            landmark[key] = (data.sum() * np.prod(slice_spacing) / 100).round(1)

        diameter, perpendicular, endpoints, endpoints_mm = get_region_diameters_pd(plane_data, slice_spacing, inverse_affine, z=z)
        landmark["diameter"] = (diameter / 10).round(1)
        landmark["diameter_pd"] = (perpendicular / 10).round(1)
        landmark["start"], landmark["end"], landmark["start_pd"], landmark["end_pd"] = endpoints_mm or [None] * 4
        (
            landmark["start_voxel"],
            landmark["end_voxel"],
            landmark["start_pd_voxel"],
            landmark["end_pd_voxel"],
        ) = endpoints or [None] * 4
        false_diameter, false_pd, _, _ = get_region_diameters_pd(false_data, slice_spacing, inverse_affine)
        true_diameter, true_pd, _, _ = get_region_diameters_pd(true_data, slice_spacing, inverse_affine)
        landmark["diameter_fl"] = (false_diameter / 10).round(1)
        landmark["diameter_fl_pd"] = (false_pd / 10).round(1)
        landmark["diameter_tl"] = (true_diameter / 10).round(1)
        landmark["diameter_tl_pd"] = (true_pd / 10).round(1)
    return landmarks


def measure_sections(landmarks, aorta_img, aorta_totalseg, true_lumen, false_lumen, centerline, spacing, max_diameter, logger):
    section_stats, section_masks = {}, []
    for name, (start_number, end_number) in SECTION_LANDMARKS.items():
        logger.info(f"  Processing section: {name}...")
        start_landmark, end_landmark = landmarks[start_number], landmarks[end_number]
        if (
            start_landmark.get("empty")
            or end_landmark.get("empty")
            or start_landmark.get("cl_idx") is None
            or end_landmark.get("cl_idx") is None
        ):
            logger.info(f"WARNING: {name} is empty!")
            section_stats[name] = dict.fromkeys(
                ("length", "max_diameter", "max_diameter_perpendicular", "volume", "volume_true_lumen", "volume_false_lumen")
            )
            continue
        section_centerline = centerline[end_landmark["cl_idx"] : start_landmark["cl_idx"]]
        _, section_diameter = get_max_diameter_of_centerline_section(
            aorta_img,
            section_centerline,
            spacing,
            True,
            subsample=1,
            crop_size=max_diameter * 1.4,
        )
        if section_diameter > max_diameter:
            logger.info(f"WARNING: max_dia of aorta section {name} is larger than max_dia from landmarks ({section_diameter:.2f} > {max_diameter:.2f})")
        if name == "aorta_total":
            section = aorta_totalseg
        else:
            section_img = get_aorta_section(
                aorta_img,
                start_landmark["cl_idx"],
                max(0, end_landmark["cl_idx"] - 2),
                centerline,
                crop_size=max_diameter * 1.4,
            )
            section = section_img.get_fdata()
            section_masks.append(section)
        true_section, false_section = section * true_lumen, section * false_lumen
        section_stats[name] = {
            "length": (get_len_path_mm(section_centerline, spacing) / 10).round(1),
            "max_diameter": round(float(section_diameter) / 10, 1),
            "volume": (section.sum() * np.prod(spacing) / 1000).round(1),
            "volume_true_lumen": (true_section.sum() * np.prod(spacing) / 1000).round(1),
            "volume_false_lumen": (false_section.sum() * np.prod(spacing) / 1000).round(1),
        }
    return section_stats, section_masks
