from pathlib import Path

import numpy as np
import nibabel as nib


base = Path("/home/jakob/dev/TotalSegmentator/tests/reference_files")

img_1 = nib.load(base / "example_seg" / "lung_vessels.nii.gz")
img_2 = nib.load(base / "example_seg" / "lung_trachea_bronchia.nii.gz")

img_out = np.zeros(img_1.shape, dtype=np.uint8)
img_out[img_1.get_fdata() > 0.5] = 1
img_out[img_2.get_fdata() > 0.5] = 2
nib.save(nib.Nifti1Image(img_out, img_1.affine), base / "lung_vessels.nii.gz")
