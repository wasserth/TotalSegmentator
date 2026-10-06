import json
import unittest
from pathlib import Path

import pytest


class TestExtraMetrics(unittest.TestCase):
    """Needs numpy + nibabel (present in CI)."""

    def test_extra_metrics(self):
        np = pytest.importorskip("numpy")
        nib = pytest.importorskip("nibabel")
        from totalsegmentator.statistics import get_basic_statistics

        seg = np.zeros((10, 10, 10), dtype=np.uint8)
        seg[4:7, 4:7, 4:7] = 1  # spleen, interior (not touching border)
        ct = np.full((10, 10, 10), -50.0)
        ct[seg == 1] = 25.0
        img = nib.Nifti1Image(ct, np.eye(4))

        stats = get_basic_statistics(seg, img, None, quiet=True, task="total",
                                     roi_subset=["spleen"], extra_metrics=True)
        sp = stats["spleen"]
        self.assertEqual(sp["n_voxels"], 27)
        self.assertEqual(sp["bbox_vox"], [[4, 6], [4, 6], [4, 6]])
        self.assertEqual(sp["centroid_vox"], [5.0, 5.0, 5.0])
        self.assertEqual(sp["intensity_min"], 25.0)
        self.assertEqual(sp["intensity_max"], 25.0)

    def test_default_has_no_extra_metrics(self):
        np = pytest.importorskip("numpy")
        nib = pytest.importorskip("nibabel")
        from totalsegmentator.statistics import get_basic_statistics

        seg = np.zeros((10, 10, 10), dtype=np.uint8)
        seg[4:7, 4:7, 4:7] = 1
        ct = np.zeros((10, 10, 10))
        img = nib.Nifti1Image(ct, np.eye(4))
        stats = get_basic_statistics(seg, img, None, quiet=True, task="total", roi_subset=["spleen"])
        self.assertEqual(set(stats["spleen"]), {"volume", "intensity"})


@pytest.mark.parametrize(("fast", "save_lowres"), [
    pytest.param(False, False, id="normal"),
    pytest.param(True, False, id="fast"),
    pytest.param(False, True, id="save_lowres"),
])
@pytest.mark.parametrize("statistics_extra", [False, True])
@pytest.mark.parametrize("write_statistics", [False, True], ids=["in_memory", "custom_path"])
def test_statistics_extra_prediction_paths(monkeypatch, tmp_path, fast, save_lowres,
                                          statistics_extra, write_statistics):
    pytest.importorskip("torch")
    pytest.importorskip("nnunetv2")
    np = pytest.importorskip("numpy")
    nib = pytest.importorskip("nibabel")
    for name in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        monkeypatch.setenv(name, str(tmp_path))

    from totalsegmentator import nnunet, python_api

    # Keep configuration, downloads and usage tracking out of this test.
    for name in ("setup_nnunet", "setup_totalseg", "download_pretrained_weights",
                 "increase_prediction_counter", "send_usage_stats"):
        monkeypatch.setattr(python_api, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(python_api, "get_config_key", lambda key: True)

    model_prediction = {}

    def predict(input_dir, output_dir, task_id, *args, **kwargs):
        # Default mode combines several models, each with its own label map.
        labels = nnunet.class_map["total"]
        if task_id in nnunet.map_taskid_to_partname_ct:
            part = nnunet.map_taskid_to_partname_ct[task_id]
            labels = nnunet.class_map_5_parts[part]
        spleen_label = next((idx for idx, name in labels.items() if name == "spleen"), None)
        for path in Path(input_dir).glob("*_0000.nii.gz"):
            img = nib.load(path)
            seg = np.zeros(img.shape, dtype=np.uint8)
            if spleen_label is not None:
                slices = tuple(slice(size // 2 - 1, size // 2 + 2) for size in img.shape)
                seg[slices] = spleen_label
                model_prediction["mask"] = seg == spleen_label
                model_prediction["voxel_volume"] = float(np.prod(img.header.get_zooms()))
            output = Path(output_dir) / path.name.replace("_0000.nii.gz", ".nii.gz")
            nib.save(nib.Nifti1Image(seg, img.affine), output)

    monkeypatch.setattr(nnunet, "nnUNetv2_predict", predict)
    img = nib.Nifti1Image(np.full((10, 10, 10), 25, dtype=np.int16),
                         np.diag([3.0, 3.0, 3.0, 1.0]))
    stats_file = tmp_path / "custom_statistics.json"
    seg_img, stats = python_api.totalsegmentator(
        img, device="cpu", quiet=True, fast=fast, save_lowres=save_lowres,
        statistics=stats_file if write_statistics else True,
        statistics_extra=statistics_extra, resampling_order=0,
    )

    if write_statistics:
        assert json.loads(stats_file.read_text()) == stats

    spleen_label = next(idx for idx, name in nnunet.class_map["total"].items() if name == "spleen")
    if fast or save_lowres:
        mask = model_prediction["mask"]
        voxel_volume = model_prediction["voxel_volume"]
    else:
        mask = seg_img.get_fdata() == spleen_label
        voxel_volume = float(np.prod(img.header.get_zooms()))
    sp = stats["spleen"]
    assert sp["volume"] == float(mask.sum()) * voxel_volume
    assert sp["intensity"] == 25.0

    expected_fields = {"volume", "intensity"}
    if statistics_extra:
        expected_fields.update({"n_voxels", "intensity_std", "intensity_min", "intensity_max",
                                "centroid_vox", "bbox_vox"})
    assert set(sp) == expected_fields
    if statistics_extra:
        idx = np.argwhere(mask)
        assert sp["n_voxels"] == int(mask.sum())
        assert sp["centroid_vox"] == idx.mean(axis=0).round(2).tolist()
        assert sp["bbox_vox"] == [[int(idx[:, axis].min()), int(idx[:, axis].max())] for axis in range(3)]
        assert sp["intensity_std"] == 0.0
        assert sp["intensity_min"] == 25.0
        assert sp["intensity_max"] == 25.0


if __name__ == "__main__":
    unittest.main()
