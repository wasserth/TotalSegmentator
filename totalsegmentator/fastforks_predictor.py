"""FastForks inference on TotalSegmentator's canonical, resampled CT parts."""

import nibabel as nib
import numpy as np


def predict_parts(model_folder, input_dir, output_dir, device, folds, step_size, tta):
    """Load one trained nnU-Net folder and predict every staged image part."""
    from fastforks import volume

    predictor = volume.load(model_folder, device=device, mode="portable", dtype="float16", folds=folds, step=step_size, mirror=tta)
    for path in sorted(input_dir.glob("*_0000.nii.gz")):
        image = nib.load(path)
        # FastForks uses (z, y, x) for both the array and its voxel spacing.
        ct = np.ascontiguousarray(np.asarray(image.dataobj, dtype=np.float32).T)
        labels = predictor(ct, spacing=image.header.get_zooms()[::-1]).labels.cpu().numpy()
        output = output_dir / path.name.replace("_0000.nii.gz", ".nii.gz")
        nib.save(nib.Nifti1Image(labels.T.astype(np.uint8, copy=False), image.affine), output)
