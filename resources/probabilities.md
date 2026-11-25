# Class probabilities (`--save_probabilities`)

```bash
TotalSegmentator -i ct.nii.gz -o seg.nii.gz --save_probabilities probs.b2nd
```

```python
from totalsegmentator.python_api import totalsegmentator
totalsegmentator("ct.nii.gz", "seg.nii.gz", save_probabilities="probs.b2nd")
```

This writes, next to each other:

- `probs.b2nd`: the probabilities, a [blosc2](https://www.blosc.org/python-blosc2/) array of shape
  `(num_channels, z, y, x)`. The `.b2nd` ending is added if you leave it out.
- `probs.pkl`: nnU-Net's properties of the image; `["nibabel_stuff"]` holds the geometry of the array.

The probabilities are written by nnU-Net's streaming inference while the model runs, so they barely add to the
runtime or the memory use.

Limitations:

- one image per call (not a folder of images)
- tasks with one model only. Tasks that combine several models (`total_5parts`, `total_v2`, `total_mr`, `total_v2_mr`,
  `total_highres`, `headneck_muscles`) raise an error; their `--fast` / `--fastest` models are single models and work.
- the probabilities are those of the model, before TotalSegmentator's postprocessing (e.g. removal of auxiliary
  classes, `--remove_small_blobs`, `--roi_subset` filtering, the `vertebrae_pp` relabelling). The `--roi_subset` filtering
  does not reduce the channels either: the file holds all classes of the model.

## Reading with TotalSegmentator

```python
from totalsegmentator.probabilities import (class_names, load_probabilities, open_probabilities,
                                            probability_to_nifti)

class_names("probs.b2nd")                                  # {0: 'background', 1: 'spleen', ...}
liver_img = probability_to_nifti("probs.b2nd", "liver")    # float32 Nifti1Image, geometry and orientation of the input
probs = load_probabilities("probs.b2nd", ["liver", 5])     # float32 (2, z, y, x): class names or channel numbers
arr = open_probabilities("probs.b2nd")                     # the blosc2 array, values as stored (uint16)
```

`probability_to_nifti` is the easiest way to look at a probability map: save it with `nib.save(liver_img,
"liver_prob.nii.gz")` and it overlays the input image in any viewer.

## Reading with blosc2 only

```python
import blosc2
import numpy as np

arr = blosc2.open("probs.b2nd", mode="r", mmap_mode="r")    # leave out mmap_mode on Windows
scale = arr.schunk.vlmeta["nnunet"]["scale"]                # 65535
classes = arr.schunk.vlmeta["totalsegmentator"]["classes"]  # {"0": "background", "1": "spleen", ...}
liver = arr[5].astype(np.float32) / scale                    # one class: only that class is decompressed
band = arr[:, 100:140].astype(np.float32) / scale            # all classes, a band of slices
```

Keep a reference to `arr`: `blosc2.open(...).schunk.vlmeta[...]` in one expression fails, because the array is freed
before the lookup.

## What the values mean

- **Stored values**: `uint16` = `round(p * 65535)`; the error is at most 7.6e-6.
- **Channels**: channel `c` is label id `c` of the task (0 = background), i.e. the ids of `totalseg_info --classes -ta
  <task>`. A few tasks have auxiliary classes, which are in the file but removed from the segmentation. The channels
  are softmax probabilities and sum to 1 (up to rounding). The argmax over the channels is the segmentation the model
  predicted.

## Axis order and geometry

The array is in the axis order of nnU-Net's reader: the input image reoriented to the canonical RAS orientation, with
the axes transposed to `(z, y, x)`. To get an image on the grid of the input, use `probability_to_nifti`, or do what it
does:

```python
import nibabel as nib
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform
from batchgenerators.utilities.file_and_folder_operations import load_pickle

geometry = load_pickle("probs.pkl")["nibabel_stuff"]
img = nib.Nifti1Image(liver.transpose(2, 1, 0), geometry["reoriented_affine"])
img = img.as_reoriented(ornt_transform(axcodes2ornt("RAS"), io_orientation(geometry["original_affine"])))
```

**Cropped tasks**: if the model runs on a region of the image (`--roi_subset`, or a task that crops itself, e.g.
`lung_vessels`, `teeth`, `heartchambers_highres`), the file covers only that region. The geometry in the `.pkl` is that
of the region, so `probability_to_nifti` returns an image of the region that is placed correctly in world space (it
overlays the input image in a viewer). Outside the region the segmentation is background. If none of the classes that
define the region are found, the model does not run and no probabilities are written.

## Memory

The whole file as float32 needs `num_channels × voxels × 4` bytes, e.g. ~18 GB for the 118 channels of `total` on a CT
of 38 million voxels. Load everything only when that fits; otherwise go class by class (`arr[c]`,
`load_probabilities(file, [name])`) or slice by slice.
