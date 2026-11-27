import importlib.util
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

pytest.importorskip("surface_distance")
pytest.importorskip("p_tqdm")


def _load_evaluate():
    path = Path(__file__).resolve().parents[1] / "resources" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("evaluate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cube(shift):
    data = np.zeros((40, 40, 40), dtype=np.uint8)
    data[10 + shift : 30 + shift, 10:30, 10:30] = 1
    return data


def test_surface_dice_uses_label_spacing(tmp_path):
    evaluate = _load_evaluate()
    gt_dir, pred_dir = tmp_path / "gt", tmp_path / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()
    affine = np.diag([3.0, 3.0, 3.0, 1.0])
    nib.save(nib.Nifti1Image(_cube(0), affine), gt_dir / "s01.nii.gz")
    # 2 voxels at 3 mm = 6 mm shift, outside the 3 mm tolerance on the shifted faces
    nib.save(nib.Nifti1Image(_cube(2), affine), pred_dir / "s01.nii.gz")

    r = evaluate.calc_metrics("s01", gt_dir=gt_dir, pred_dir=pred_dir, class_map={1: "spleen"})

    assert r["surface_dice_3-spleen"] == pytest.approx(0.697, abs=1e-3)
