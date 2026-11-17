import sys
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import torch

from totalsegmentator.fastforks_predictor import predict_parts


def test_split_parts_keep_axes_spacing_and_affine(tmp_path, monkeypatch):
    data = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    affine = np.diag([1.0, 2.0, 3.0, 1.0])
    for i in (1, 2):
        nib.save(nib.Nifti1Image(data + i, affine), tmp_path / f"s0{i}_0000.nii.gz")

    loads = []

    def predict(ct, spacing):
        assert tuple(spacing) == (3.0, 2.0, 1.0)
        assert ct.shape == (4, 3, 2)
        return SimpleNamespace(labels=torch.from_numpy(ct))

    def load(path, **kwargs):
        loads.append((path, kwargs))
        return predict

    monkeypatch.setitem(sys.modules, "fastforks", SimpleNamespace(volume=SimpleNamespace(load=load)))
    predict_parts("model", tmp_path, tmp_path, "cuda:1", [0], 0.8, False)
    assert loads == [("model", dict(device="cuda:1", mode="portable", dtype="float16", folds=[0], step=0.8, mirror=False))]
    for i in (1, 2):
        image = nib.load(tmp_path / f"s0{i}.nii.gz")
        np.testing.assert_array_equal(image.affine, affine)
        np.testing.assert_array_equal(np.asarray(image.dataobj), data + i)
        assert image.get_data_dtype() == np.uint8
