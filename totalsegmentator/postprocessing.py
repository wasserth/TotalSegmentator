import time
from pathlib import Path
import numpy as np
import nibabel as nib
from tqdm import tqdm

from scipy.ndimage import binary_dilation, binary_erosion, binary_closing
from scipy import ndimage

from totalsegmentator.map_to_binary import class_map


def keep_largest_blob(data, debug=False):
    blob_map, nr_of_blobs = ndimage.label(data)
    # Get number of pixels in each blob
    # counts = list(np.bincount(blob_map.flatten()))  # this will also count background -> bug
    counts = [np.sum(blob_map == i) for i in range(1, nr_of_blobs + 1)]  # this will not count background
    if len(counts) == 0: return data  # no foreground
    largest_blob_label = np.argmax(counts) + 1  # +1 because labels start from 1
    if debug: print(f"size of largest blob: {np.max(counts)}")
    return (blob_map == largest_blob_label).astype(np.uint8)


def keep_largest_blob_multilabel(data, class_map, rois, debug=False, quiet=False):
    """
    Keep the largest blob for the classes defined in rois.

    data: multilabel image (np.array)
    class_map: class map {label_idx: label_name}
    rois: list of labels where to filter for the largest blob

    return multilabel image (np.array)
    """
    st = time.time()
    class_map_inv = {v: k for k, v in class_map.items()}
    for roi in tqdm(rois, disable=quiet):
        idx = class_map_inv[roi]
        data_roi = data == idx
        cleaned_roi = keep_largest_blob(data_roi, debug) > 0.5
        data[data_roi] = 0   # Clear the original ROI in data
        data[cleaned_roi] = idx   # Write back the cleaned ROI into data
    # print(f"  keep_largest_blob_multilabel took {time.time() - st:.2f}s")
    return data


def remove_small_blobs(img: np.ndarray, interval=[10, 30], debug=False) -> np.ndarray:
    """
    Find blobs/clusters of same label. Remove all blobs which have a size which is outside of the interval.

    Args:
        img: Binary image.
        interval: Boundaries of the sizes to remove.
        debug: Show debug information.
    Returns:
        Detected blobs.
    """
    mask, number_of_blobs = ndimage.label(img)
    if debug: print('Number of blobs before: ' + str(number_of_blobs))
    counts = np.bincount(mask.flatten())  # number of pixels in each blob

    # If only one blob (only background) abort because nothing to remove
    if len(counts) <= 1: return img

    remove = np.where((counts <= interval[0]) | (counts > interval[1]), True, False)
    remove_idx = np.nonzero(remove)[0]
    mask[np.isin(mask, remove_idx)] = 0
    mask[mask > 0] = 1  # set everything else to 1

    if debug:
        print(f"counts: {sorted(counts)[::-1]}")
        _, number_of_blobs_after = ndimage.label(mask)
        print('Number of blobs after: ' + str(number_of_blobs_after))

    return mask


def remove_small_blobs_multilabel(data, class_map, rois, interval=[10, 30], debug=False, quiet=False):
    """
    Remove small blobs for the classes defined in rois.

    data: multilabel image (np.array)
    class_map: class map {label_idx: label_name}
    rois: list of labels where to filter for the largest blob

    return multilabel image (np.array)
    """
    st = time.time()
    class_map_inv = {v: k for k, v in class_map.items()}

    for roi in tqdm(rois, disable=quiet):
        idx = class_map_inv[roi]
        data_roi = (data == idx)
        cleaned_roi = remove_small_blobs(data_roi, interval, debug) > 0.5  # Remove small blobs from this ROI
        data[data_roi] = 0  # Clear the original ROI in data
        data[cleaned_roi] = idx  # Write back the cleaned ROI into data

    # print(f"  remove_small_blobs_multilabel took {time.time() - st:.2f}s")
    return data


def remove_outside_of_mask(seg_path, mask_path, addon=1):
    """
    Remove all segmentations outside of mask.

    seg_path: path to nifti file or numpy array
    mask_path: path to nifti file or numpy array

    return: numpy array
    """

    # Read segmentation
    if isinstance(seg_path, (str, Path)):
        seg_img = nib.load(seg_path)
        seg = seg_img.get_fdata()
    else:
        seg = seg_path

    # Read mask
    if isinstance(mask_path, (str, Path)):
        mask = nib.load(mask_path).get_fdata()
    else:
        mask = mask_path

    mask = binary_dilation(mask, iterations=addon)
    seg[mask == 0] = 0

    # Save
    if isinstance(seg_path, (str, Path)):
        nib.save(nib.Nifti1Image(seg.astype(np.uint8), seg_img.affine), seg_path)
    else:
        return seg


def extract_skin(ct_img, body_img):
    """
    Extract the skin from a segmentation of the body.

    ct_img: nifti image
    body_img: nifti image

    returns: nifti image
    """
    ct = ct_img.get_fdata()
    body = body_img.get_fdata()

    # Select skin region
    body = binary_dilation(body, iterations=1).astype(np.uint8)  # add 1 voxel margin at the outside
    body_inner = binary_erosion(body, iterations=3).astype(np.uint8)
    skin = body - body_inner

    # Segment by density
    # Roughly the skin density range. Made large to make segmentation not have holes
    # (0 to 250 would have many small holes in skin)
    density_mask = (ct > -200) & (ct < 250)
    skin[~density_mask] = 0

    # Fill holes
    # skin = binary_closing(skin, iterations=1)  # no real difference
    # skin = binary_dilation(skin, iterations=1)  # not good

    # Removing blobs
    skin = remove_small_blobs(skin>0.5, interval=[5,1e10])

    return nib.Nifti1Image(skin.astype(np.uint8), ct_img.affine)


def remove_auxiliary_labels(img, task_name):
    task_name_aux = task_name + "_auxiliary"
    if task_name_aux in class_map:
        class_map_aux = class_map[task_name_aux]
        data = img.get_fdata()
        # remove auxiliary labels
        for idx in class_map_aux.keys():
            data[data == idx] = 0
        return nib.Nifti1Image(data.astype(np.uint8), img.affine)
    else:
        return img


def _multilabel_labels_touch(data):
    for axis in range(data.ndim):
        slicer_a = [slice(None)] * data.ndim
        slicer_b = [slice(None)] * data.ndim
        slicer_a[axis] = slice(1, None)
        slicer_b[axis] = slice(None, -1)
        a = data[tuple(slicer_a)]
        b = data[tuple(slicer_b)]
        if np.any((a > 0) & (b > 0) & (a != b)):
            return True
    return False


def get_ellipsoid_structuring_element(voxel_spacing, radius_mm):
    """
    Create an anisotropic 3D ellipsoid kernel for morphology in physical space.
    """
    if radius_mm <= 0:
        return np.ones((1, 1, 1), dtype=bool)

    radii = [radius_mm / spacing for spacing in voxel_spacing]
    grids = np.ogrid[
        -radii[0]:radii[0]+1,
        -radii[1]:radii[1]+1,
        -radii[2]:radii[2]+1
    ]
    struct_elem = sum((grid * grid) / (radius * radius) for grid, radius in zip(grids, radii)) <= 1
    return struct_elem


def dilate_vertebrae_labels(data, label_map, voxel_spacing=(1.0, 1.0, 1.0), dilation_mm=3, verbose=False):
    """
    Undo the 3 mm training-label erosion after vertebrae labels are separated.

    Each vertebra is dilated independently, but only into background voxels so
    already assigned vertebra labels are not overwritten.
    """
    if dilation_mm <= 0:
        return data

    st = time.time()
    struct_elem = get_ellipsoid_structuring_element(voxel_spacing, dilation_mm)
    radius_vox = np.ceil([dilation_mm / spacing for spacing in voxel_spacing]).astype(int)
    out = data.copy()
    for label in sorted(label_map):
        label_coords = np.where(data == label)
        if len(label_coords[0]) == 0:
            continue

        bbox_min = [max(int(coords.min()) - radius, 0)
                    for coords, radius in zip(label_coords, radius_vox)]
        bbox_max = [min(int(coords.max()) + radius + 1, data.shape[axis])
                    for axis, (coords, radius) in enumerate(zip(label_coords, radius_vox))]
        bbox = tuple(slice(start, stop) for start, stop in zip(bbox_min, bbox_max))

        label_mask = data[bbox] == label
        dilated_mask = binary_dilation(label_mask, structure=struct_elem)
        out_bbox = out[bbox]
        out_bbox[(out_bbox == 0) & dilated_mask] = label

    if verbose:
        print(f"  dilated vertebrae in: {time.time() - st:.2f}s")
    return out


def refine_vertebrae_pp_with_body_mask(data, body_data, label_map, voxel_spacing=(1.0, 1.0, 1.0),
                                       dilation_mm=2, verbose=False):
    """
    Sharpen vertebrae_pp instance borders with the vertebrae_body model.

    The instance labels are dilated slightly further and then clipped to the
    vertebral-body foreground, preserving the per-vertebra labels.
    """
    st = time.time()
    if data.shape != body_data.shape:
        raise ValueError(f"vertebrae_pp and vertebrae_body shapes do not match: {data.shape} vs {body_data.shape}")

    refined = dilate_vertebrae_labels(data, label_map, voxel_spacing, dilation_mm, verbose=False)
    refined[body_data != 1] = 0

    if verbose:
        print("Refining vertebrae_pp with vertebrae_body:")
        print(f"  dilated before intersection: {dilation_mm} mm")
        print(f"  vertebrae labeled in the end: {len(np.unique(refined[refined > 0]))}")
        print(f"  refinement took: {time.time() - st:.2f}s")

    return refined.astype(np.uint8, copy=False)


def postprocess_vertebrae_pp(data, label_map, min_size_mm3=100, voxel_volume=1.0,
                             voxel_spacing=(1.0, 1.0, 1.0), dilation_mm=3, verbose=False):
    """
    Fix neighboring vertebrae labels that leaked into the same vertebral body.

    The vertebrae_pp model only segments vertebral bodies. Different vertebrae
    labels should therefore never touch. If they do, the combined binary mask is
    split into connected bodies and relabeled anatomically from inferior to
    superior, except head-only scans that contain C1 but not L5 are relabeled
    from superior to inferior.
    """
    st = time.time()
    data = data.astype(np.uint8, copy=False)
    mixed_up = _multilabel_labels_touch(data)

    if verbose:
        print("Postprocessing vertebrae_pp:")
        print(f"  mixed up vertebrae found: {mixed_up}")

    if not mixed_up:
        return dilate_vertebrae_labels(data, label_map, voxel_spacing, dilation_mm, verbose)

    component_map, _ = ndimage.label(data > 0)
    component_sizes = np.bincount(component_map.ravel())
    keep_components = np.flatnonzero(component_sizes * voxel_volume >= min_size_mm3)
    keep_components = keep_components[keep_components != 0]

    if len(keep_components) == 0:
        if verbose:
            print(f"  connected components found after removing small ones (<{min_size_mm3} mm3): 0")
            print("  counting starts: none")
            print("  L5 in image: False")
            print("  vertebrae labeled in the end: 0")
            print(f"  postprocessing took: {time.time() - st:.2f}s")
        return np.zeros_like(data, dtype=np.uint8)

    keep_lookup = np.zeros(component_sizes.shape, dtype=bool)
    keep_lookup[keep_components] = True
    keep_mask = keep_lookup[component_map]

    cleaned_data = data.copy()
    cleaned_data[~keep_mask] = 0
    present_labels = sorted(int(label) for label in np.unique(cleaned_data)
                            if int(label) in label_map)
    if len(present_labels) == 0:
        if verbose:
            print(f"  connected components found after removing small ones (<{min_size_mm3} mm3): {len(keep_components)}")
            print("  counting starts: none")
            print("  L5 in image: False")
            print("  vertebrae labeled in the end: 0")
            print(f"  postprocessing took: {time.time() - st:.2f}s")
        return np.zeros_like(data, dtype=np.uint8)

    label_map_inv = {v: k for k, v in label_map.items()}
    c1_label = label_map_inv["vertebrae_C1"]
    l5_label = label_map_inv["vertebrae_L5"]
    l5_in_image = l5_label in present_labels
    count_from_top = not l5_in_image and c1_label in present_labels

    centers = ndimage.center_of_mass(keep_mask, component_map, keep_components)
    if len(keep_components) == 1:
        centers = [centers]
    component_centers = [(component, center[2]) for component, center in zip(keep_components, centers)]

    if count_from_top:
        component_centers = sorted(component_centers, key=lambda item: item[1], reverse=True)
        labels_to_assign = range(c1_label, max(label_map) + 1)
        counting_starts = "top"
    else:
        component_centers = sorted(component_centers, key=lambda item: item[1])
        labels_to_assign = range(max(present_labels), min(label_map) - 1, -1)
        counting_starts = "bottom"

    out = np.zeros_like(data, dtype=np.uint8)
    vertebrae_labeled = 0
    for (component, _), label in zip(component_centers, labels_to_assign):
        out[component_map == component] = label
        vertebrae_labeled += 1

    if verbose:
        print(f"  connected components found after removing small ones (<{min_size_mm3} mm3): {len(keep_components)}")
        print(f"  counting starts: {counting_starts}")
        print(f"  L5 in image: {l5_in_image}")
        print(f"  vertebrae labeled in the end: {vertebrae_labeled}")
        print(f"  postprocessing took: {time.time() - st:.2f}s")

    return dilate_vertebrae_labels(out, label_map, voxel_spacing, dilation_mm, verbose)

