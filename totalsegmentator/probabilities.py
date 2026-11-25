"""
Class probabilities written with --save_probabilities (nnU-Net's streaming probability export).

<file>.b2nd: a blosc2 array (num_channels, z, y, x), by default uint16 = round(p * 65535). Channel c is label id c of
the task (0: background), as in class_map[task]. The axes are those of nnU-Net's reader: the image in canonical (RAS)
orientation, transposed to (z, y, x). For a task that runs on a crop (e.g. --roi_subset, or a task with its own crop)
the array covers only the crop. vlmeta['nnunet'] holds 'scale' (probability = stored value / scale), vlmeta
['totalsegmentator'] the task and its classes.
<file>.pkl: nnU-Net's properties of the image; properties['nibabel_stuff'] the geometry of the array.

See resources/probabilities.md.
"""
from pathlib import Path
from typing import Optional, Sequence, Union

import nibabel as nib
import numpy as np
from nibabel.orientations import axcodes2ornt, inv_ornt_aff, io_orientation, ornt_transform

PROBABILITIES_FILE_ENDING = ".b2nd"


def probabilities_base(path: Union[str, Path]) -> str:
    """The path nnU-Net appends .b2nd / .pkl to: path without a .b2nd ending."""
    path = str(Path(path).absolute())
    return path[:-len(PROBABILITIES_FILE_ENDING)] if path.endswith(PROBABILITIES_FILE_ENDING) else path


def return_segmentation(segmentation, properties, output):
    """export_fn for nnU-Net's streaming predictor: the probabilities are written during the GPU pass, the
    segmentation goes back to the caller."""
    return segmentation


def crop_geometry(nibabel_stuff: dict, box: Sequence[slice], shape_zyx: Sequence[int]) -> dict:
    """nibabel_stuff of nnU-Net's NibabelIOWithReorient for a crop (box, in the reader's (z, y, x) axes) of the image."""
    reoriented = np.array(nibabel_stuff["reoriented_affine"], dtype=float)
    start_xyz = np.array([box[2].start, box[1].start, box[0].start], dtype=float)
    cropped = reoriented.copy()
    cropped[:3, 3] = reoriented[:3, :3] @ start_xyz + reoriented[:3, 3]
    from_canonical = ornt_transform(axcodes2ornt("RAS"), io_orientation(nibabel_stuff["original_affine"]))
    return {"original_affine": cropped @ inv_ornt_aff(from_canonical, tuple(shape_zyx)[::-1]),
            "reoriented_affine": cropped}


def add_class_names(file: Union[str, Path], task: str, classes: dict):
    import blosc2
    arr = blosc2.open(str(file), mode="a")
    arr.schunk.vlmeta["totalsegmentator"] = {"task": task, "classes": {str(k): v for k, v in classes.items()}}


# ------------------------------------------------------------------ reading ----------------------------------------
def open_probabilities(file: Union[str, Path]):
    """The blosc2 array, values as stored (divide by arr.schunk.vlmeta['nnunet']['scale']). Read lazily: slicing it
    decompresses only the selection."""
    import blosc2
    return blosc2.open(str(file), mode="r")


def class_names(file: Union[str, Path]) -> dict:
    """{channel: class name} of a probability file."""
    arr = open_probabilities(file)      # blosc2 frees the schunk with the array: no chained call
    meta = arr.schunk.vlmeta["totalsegmentator"]
    return {int(k): v for k, v in meta["classes"].items()}


def _channel(file, cls: Union[int, str]) -> int:
    if isinstance(cls, (int, np.integer)):
        return int(cls)
    ids = {name: c for c, name in class_names(file).items()}
    if cls not in ids:
        raise ValueError(f"Class '{cls}' is not in {file}. Classes: {sorted(ids)}")
    return ids[cls]


def load_probabilities(file: Union[str, Path], classes: Optional[Sequence[Union[int, str]]] = None) -> np.ndarray:
    """Probabilities as float32 (C, z, y, x) in the reader's axes, of all channels or of the given classes (names or
    channels). Everything selected is held in memory: all 118 channels of a whole-body CT can need >10 GB."""
    arr = open_probabilities(file)
    scale = np.float32(arr.schunk.vlmeta["nnunet"]["scale"])
    channels = range(arr.shape[0]) if classes is None else [_channel(file, c) for c in classes]
    out = np.empty((len(channels), *arr.shape[1:]), dtype=np.float32)
    for i, c in enumerate(channels):
        out[i] = arr[c]
        out[i] /= scale
    return out


def probability_to_nifti(file: Union[str, Path], cls: Union[int, str]) -> nib.Nifti1Image:
    """The probability map of one class (name or channel) as a float32 NIfTI image in the orientation of the input
    image (for a cropped task: of the crop), using the geometry in the .pkl next to the file."""
    from batchgenerators.utilities.file_and_folder_operations import load_pickle
    geometry = load_pickle(probabilities_base(file) + ".pkl")["nibabel_stuff"]
    data = load_probabilities(file, [cls])[0].transpose(2, 1, 0)
    img = nib.Nifti1Image(data, geometry["reoriented_affine"])
    from_canonical = ornt_transform(axcodes2ornt("RAS"), io_orientation(geometry["original_affine"]))
    return img.as_reoriented(from_canonical)
