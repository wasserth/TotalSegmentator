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


def keep_largest_blob_multilabel(data, class_map, rois, debug=False):
    """
    Keep the largest blob for the classes defined in rois.

    data: multilabel image (np.array)
    class_map: class map {label_idx: label_name}
    rois: list of labels where to filter for the largest blob

    return multilabel image (np.array)
    """
    st = time.time()
    class_map_inv = {v: k for k, v in class_map.items()}
    for roi in tqdm(rois):
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


def remove_small_blobs_multilabel(data, class_map, rois, interval=[10, 30], debug=False):
    """
    Remove small blobs for the classes defined in rois.

    data: multilabel image (np.array)
    class_map: class map {label_idx: label_name}
    rois: list of labels where to filter for the largest blob

    return multilabel image (np.array)
    """
    st = time.time()
    class_map_inv = {v: k for k, v in class_map.items()}

    for roi in tqdm(rois):
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

    seg_path: path to nifti file
    mask_path: path to nifti file
    """
    seg_img = nib.load(seg_path)
    seg = seg_img.get_fdata()
    mask = nib.load(mask_path).get_fdata()
    mask = binary_dilation(mask, iterations=addon)
    seg[mask == 0] = 0
    nib.save(nib.Nifti1Image(seg.astype(np.uint8), seg_img.affine), seg_path)


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

