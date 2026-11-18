"""The pieces of the inference pipeline (totalsegmentator/predictor.py), on the CPU and without model weights."""
import threading
from pathlib import Path

import numpy as np
import nibabel as nib
import pytest
from scipy import ndimage

from totalsegmentator import predictor as P
from totalsegmentator.map_tasks_config import TASK_ID_WEIGHTS_CONFIGS
from totalsegmentator.map_to_binary import class_map


def test_bounding_box_crop_and_paste_back():
    mask = np.zeros((20, 30, 40), dtype=bool)
    mask[5:8, 10:20, 38:40] = True
    box = P.bounding_box(mask, [2.9, 0, 5])                    # margins in voxels, rounded down, clipped to the array
    assert box == (slice(3, 10), slice(10, 20), slice(33, 40))
    full = np.zeros(mask.shape, dtype=np.uint8)
    full[box] = 1
    assert full.sum() == 7 * 10 * 7 and full[mask].all()
    assert P.bounding_box(np.zeros((4, 4, 4), dtype=bool), [1, 1, 1]) is None


def test_merge_parts_later_part_wins():
    first = np.array([[1, 1, 0], [2, 0, 0]], dtype=np.uint8)
    second = np.array([[0, 1, 1], [0, 0, 2]], dtype=np.uint8)
    lut_first, lut_second = np.zeros(256, np.uint8), np.zeros(256, np.uint8)
    lut_first[[1, 2]] = [10, 20]
    lut_second[[1, 2]] = [30, 40]
    merged = P.merge_parts([(first, lut_first), (second, lut_second)])
    assert merged.tolist() == [[10, 30, 30], [20, 0, 40]]
    assert P.merge_parts([(first, None)]) is first              # one model: its label map as it is


def registry_entries():
    """Every (task, mode) entry of the registry."""
    for task, raw in P.TASK_CONFIGS.items():
        for mode in raw.get("sub_modes", {"default": None}):
            yield task, mode


def check_registry_entry(task, mode):
    """Resolve one registry entry through the pipeline's own code, as far as that goes without weights. Returns the
    list of problems (empty: the pipeline can express the entry)."""
    problems = []
    config = P.get_task_config(task, mode, quiet=True)
    labels = set(class_map[task].values())
    if config["input_resampling"] not in ("cubic", "plans"):
        problems.append(f"input_resampling {config['input_resampling']!r}")
    # models: weights are registered; a multi-model task has a part map whose classes are classes of the task
    task_ids = config["task_id"] if isinstance(config["task_id"], list) else [config["task_id"]]
    for task_id in task_ids:
        if "foldername" not in TASK_ID_WEIGHTS_CONFIGS.get(task_id, {}):
            problems.append(f"model {task_id} has no weights entry")
    if isinstance(config["task_id"], list):
        if task not in P.PART_MAPS:
            problems.append("multi-model task without part maps")
        else:
            parts, part_of_model = P.PART_MAPS[task]
            for task_id in task_ids:
                part = parts.get(part_of_model.get(task_id), {})
                if not part or not set(part.values()) <= labels:
                    problems.append(f"part map of model {task_id}: {sorted(set(part.values()) - labels) or 'missing'}")
    # roi: the providing task and mode exist, the classes are its classes, the chain of rois ends
    roi, roi_task, chain = P.get_roi(task, config), task, [task]
    first_roi = roi
    while roi is not None:
        raw = P.TASK_CONFIGS.get(roi["task"])
        if raw is None or (roi["mode"] != "default" and roi["mode"] not in raw.get("sub_modes", {})):
            problems.append(f"roi task {roi['task']} ({roi['mode']}) does not exist")
            break
        if not set(roi["classes"]) <= set(class_map[roi["task"]].values()) or len(roi["margin"]) != 3:
            problems.append(f"roi of {roi_task}: {roi}")
        chain.append(roi["task"])
        if len(chain) > 4:
            problems.append(f"roi chain does not end: {chain}")
            break
        roi_task = roi["task"]
        roi = P.get_roi(roi_task, P.get_task_config(roi_task, roi["mode"], quiet=True))
    if config["remove_outside"] is not None:
        if first_roi is None or not set(config["remove_outside"]) <= set(class_map[first_roi["task"]].values()) \
                or config["remove_outside_dilation"] is None:
            problems.append("remove_outside without a roi that has its classes")
    # a class subset of a total task is located with a model that knows the class
    if task.startswith("total"):
        subset_roi = P.get_roi(task, config, roi_subset=[next(iter(labels))])
        if subset_roi is None or subset_roi["task"] not in P.TASK_CONFIGS or \
                not set(subset_roi["classes"]) <= set(class_map[subset_roi["task"]].values()):
            problems.append(f"roi_subset: {subset_roi}")
    # label rules run on an empty label map
    empty = P.apply_label_rules(np.zeros((4, 5, 6), dtype=np.uint8), task, (1.5, 1.5, 1.5))
    if empty.shape != (4, 5, 6) or empty.any():
        problems.append("label rules")
    return problems


def test_every_registry_entry_resolves(monkeypatch):
    """Every task and mode of map_tasks_config.py can be expressed by the pipeline: models, part maps, roi (task,
    classes, margin), remove_outside, label rules, input resampling."""
    monkeypatch.setattr(P, "show_license_info", lambda: None)      # the walk needs no license
    entries = list(registry_entries())
    failed = {entry: problems for entry in entries if (problems := check_registry_entry(*entry))}
    assert len(entries) >= 60
    assert not failed, failed


def test_roi_examples():
    config = P.get_task_config("teeth", quiet=True)
    assert P.get_roi("teeth", config)["task"] == "craniofacial_structures"         # which itself crops to the skull
    assert P.get_roi("teeth", config)["margin"] == [10, 10, 10]
    config = P.get_task_config("total_v2", quiet=True)
    assert P.get_roi("total_v2", config) is None
    assert P.get_roi("total_v2", config, roi_subset=["liver"]) == \
        {"task": "total_v2", "mode": "fastest", "classes": ["liver"], "margin": [20, 20, 20]}
    assert P.get_roi("total_v2", config, roi_subset=["liver"], robust=True)["mode"] == "fast"
    # roi_subset only crops in the default mode
    assert P.get_roi("total", P.get_task_config("total", "fastest", quiet=True), ["liver"], mode="fastest") is None
    assert P.get_roi("total", P.get_task_config("total", "fast", quiet=True), ["liver"], mode="fast") is None
    assert P.get_roi("total_mr", P.get_task_config("total_mr", "fast", quiet=True), ["liver"], mode="fast") is None
    assert P.get_roi("total_mr", P.get_task_config("total_mr", quiet=True), ["liver"])["mode"] == "fast"
    assert P.get_roi("abdominal_muscles", P.get_task_config("abdominal_muscles", quiet=True))["task"] == "body"
    # a task's crop_addon is the margin of its crop; without one the crop gets 20 mm
    assert P.get_roi("abdominal_muscles", P.get_task_config("abdominal_muscles", quiet=True))["margin"] == [5, 5, 5]
    assert P.get_roi("liver_segments_mr", P.get_task_config("liver_segments_mr", quiet=True))["margin"] == [10, 10, 10]
    assert P.get_roi("ventricle_parts", P.get_task_config("ventricle_parts", quiet=True))["margin"] == [5, 5, 5]
    assert P.get_roi("lung_vessels", P.get_task_config("lung_vessels", quiet=True))["margin"] == [20, 20, 20]


def test_total_highres_crops_to_the_body_with_3mm(monkeypatch):
    """TotalSegmentator v3 cropped total_highres to the body with the task's 3 mm (--body_seg), not with 20 mm."""
    monkeypatch.setattr(P, "show_license_info", lambda: None)      # a commercial task
    roi = P.get_roi("total_highres", P.get_task_config("total_highres", quiet=True))
    assert roi == {"task": "body", "mode": "fast", "classes": ["body_trunc", "body_extremities"], "margin": [3, 3, 3]}


def test_label_isin_equals_np_isin():
    rng = np.random.default_rng(0)
    for dtype in (np.uint8, np.uint16, np.int32):
        for order in ("C", "F"):
            seg = np.asarray(rng.integers(0, 120, size=(7, 9, 11)).astype(dtype), order=order)
            ids = [0, 3, 117, 300]
            mask = P.label_isin(seg, ids)
            assert np.array_equal(mask, np.isin(seg, ids))
            if dtype != np.int32:                   # the lookup table keeps the layout (np.isin does not)
                assert mask.flags.f_contiguous == seg.flags.f_contiguous


def test_output_file():
    assert P.output_file(Path("seg"), True, ["nifti"]) == Path("seg.nii.gz")
    assert P.output_file(Path("seg.nii"), True, ["nifti"]) == Path("seg.nii")
    assert P.output_file(Path("seg"), False, ["nifti"]) == Path("seg")
    assert P.output_file(Path("seg"), True, ["dicom_seg"]) == Path("seg")
    assert P.output_file(None, True, ["nifti"]) is None


def test_roi_subset_names_are_checked_before_anything_runs():
    with pytest.raises(ValueError, match="did you mean 'liver'"):
        P.validate_roi_subset(["livr"], "total")
    P.validate_roi_subset(["liver", "spleen"], "total")


def test_label_rules_remove_auxiliary_labels():
    auxiliary = list(class_map["kidney_cysts_auxiliary"])
    seg = np.array([[[1, auxiliary[0], 2]]], dtype=np.uint8)
    seg = P.apply_label_rules(seg, "kidney_cysts", (1.0, 1.0, 1.0))
    assert seg.tolist() == [[[0 if auxiliary[0] == 1 else 1, 0, 0 if auxiliary[0] == 2 or 2 in auxiliary else 2]]]


@pytest.mark.parametrize("new_shape", [(30, 33, 27), (10, 22, 9), (20, 22, 18)])
def test_resample_cubic_is_center_aligned_scipy_zoom(new_shape):
    """The input resampling of the 'cubic' models is scipy's cubic zoom on the pixel grid (grid_mode=True): the image
    keeps its extent, which is the rule nnU-Net maps the logits back with."""
    rng = np.random.default_rng(0)
    data = rng.normal(size=(2, 20, 22, 18)).astype(np.float32)
    got = P.resample_cubic(data, new_shape, device="cpu")
    zoom = [n / o for n, o in zip(new_shape, data.shape[1:])]
    expected = np.stack([ndimage.zoom(c.astype(np.float64), zoom, order=3, mode="nearest", grid_mode=True)
                         for c in data])
    assert got.shape == (2, *new_shape) and got.dtype == np.float32
    np.testing.assert_allclose(got, expected, atol=1e-4)


class FakeModel:
    """Stands in for predictor.Model: no network, the label map is a function of the image it is given."""

    def __init__(self, label_fn):
        self.label_fn, self.shapes = label_fn, []

    def preprocess(self, data, properties):
        return data[0], properties

    def predict(self, data, properties, num_cases=1):
        self.shapes.append(data.shape)
        return self.label_fn(data).astype(np.uint8)


@pytest.fixture
def liver_vessels(monkeypatch, tmp_path):
    """A liver_vessels predictor whose models are fakes: the rough (total) model labels the voxels of value 100 as
    liver, the task's own model labels its whole input, which shows where it ran. Returns (predictor, rough, fine)."""
    pytest.importorskip("nnunetv2")
    monkeypatch.setenv("TOTALSEG_HOME_DIR", str(tmp_path))
    liver = next(i for i, name in class_map["total"].items() if name == "liver")
    rough = FakeModel(lambda data: (data == 100) * liver)
    fine = FakeModel(lambda data: np.ones(data.shape))
    ts = P.TotalSegmentatorPredictor("liver_vessels", device="cpu", quiet=True)
    monkeypatch.setattr(ts, "models", lambda task, config, roi_subset=None:
                        [(rough if task == "total" else fine, None)])
    return ts, rough, fine


def test_roi_recursion_crop_and_paste_back(liver_vessels):
    """liver_vessels runs on the bounding box of the liver in a rough total segmentation, plus 20 mm; the image is
    stored left-right and up-down flipped (LAI), so reorientation and its inverse are part of the round trip."""
    ts, rough, fine = liver_vessels
    affine = np.diag([-2.0, 2.0, -5.0, 1.0])
    image = np.zeros((40, 50, 30), dtype=np.int16)
    image[10:20, 20:30, 12:16] = 100                            # "the liver", in the stored grid
    seg = ts.predict(nib.Nifti1Image(image, affine))

    expected = np.zeros(image.shape, dtype=np.uint8)
    expected[0:30, 10:40, 8:20] = 1                             # the liver's box + 20 mm = 10 / 10 / 4 voxels
    assert np.array_equal(np.asanyarray(seg.dataobj), expected) and np.array_equal(seg.affine, affine)
    assert seg.dataobj.flags.f_contiguous                       # NIfTI's memory layout
    assert rough.shapes == [(30, 50, 40)] and fine.shapes == [(12, 30, 30)]     # (z, y, x) of whole image and crop

    # nothing found by the rough segmentation: empty result, the task's model does not run
    empty = ts.predict(nib.Nifti1Image(np.zeros_like(image), affine))
    assert not np.asanyarray(empty.dataobj).any() and len(fine.shapes) == 1


def test_load_image_reads_every_kind_of_input_once(tmp_path):
    """A NIfTI file, the same image in memory and a 4D file give the reader's array of the same 3D image. Only an
    input that is not a 3D NIfTI file is kept for statistics / preview, and only if asked for."""
    from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
    reader = NibabelIOWithReorient()
    values = np.random.default_rng(0).integers(-1024, 3000, (20, 18, 16)).astype(np.int16)
    img = nib.Nifti1Image(values, np.diag([-1.5, 1.5, 2.0, 1.0]))       # stored LAS: the reader flips it
    nib.save(img, tmp_path / "ct.nii.gz")
    nib.save(nib.Nifti1Image(np.stack([values, values + 1], -1), img.affine), tmp_path / "ct_4d.nii.gz")

    expected, expected_properties = reader.read_images([str(tmp_path / "ct.nii.gz")])
    for input, is_file in ((tmp_path / "ct.nii.gz", True), (nib.load(tmp_path / "ct.nii.gz"), False),
                           (tmp_path / "ct_4d.nii.gz", False)):
        for keep_image in (False, True):
            data, properties, source = P.load_image(input, reader, keep_image)
            np.testing.assert_array_equal(data, expected)
            assert properties["spacing"] == expected_properties["spacing"]
            assert (source["nifti_path"] is not None) == is_file
            assert (source["image"] is not None) == (keep_image and not is_file)
            if source["image"] is not None:
                assert source["image"].shape == values.shape and P._input_image(source) is source["image"]


def test_statistics_of_a_4d_file(liver_vessels, tmp_path):
    """Statistics need the image again after the prediction: for a 4D file that is its first volume, not the file."""
    ts, _, _ = liver_vessels
    image = np.zeros((40, 50, 30, 2), dtype=np.int16)
    image[10:20, 20:30, 12:16] = 100
    nib.save(nib.Nifti1Image(image, np.diag([2.0, 2.0, 5.0, 1.0])), tmp_path / "ct_4d.nii.gz")
    seg, stats = ts.predict(tmp_path / "ct_4d.nii.gz", statistics=True, statistics_exclude_masks_at_border=False)
    assert seg.shape == (40, 50, 30) and set(stats) == set(class_map["liver_vessels"].values())


def test_v1_order_with_roi_subset_keeps_the_classes_asked_for():
    """The class subset is applied to the ids that are saved: with v1_order those of the v1 label map."""
    task, v1 = class_map["total"], class_map["total_v1"]
    ids = {name: i for i, name in task.items()}
    ids_v1 = {name: i for i, name in v1.items()}
    assert ids["aorta"] != ids_v1["aorta"]
    data = np.array([[[ids["aorta"], ids["liver"], ids_v1["aorta"], 0]]], dtype=np.uint8, order="F")
    source = {"affine": np.eye(4), "header": nib.Nifti1Header(), "nifti_path": None, "dicom_dir": None, "image": None}
    options = {"quiet": True, "ml": True, "roi_subset": ["aorta"], "output_types": ["nifti"], "dicom_output": False,
               "statistics": False, "preview": False, "remove_small_blobs": False, "v1_order": True,
               "skip_saving": True, "return_images": True, "batch": False}
    label_map = {i: name for i, name in v1.items() if name == "aorta"}
    img, _ = P.export_case(data, source, None, "total", label_map, options)
    assert np.asanyarray(img.dataobj).ravel().tolist() == [ids_v1["aorta"], 0, 0, 0]


def test_files_of_one_class_have_no_label_table(tmp_path):
    data = np.zeros((4, 5, 6), dtype=np.uint8, order="F")
    data[1:3, 1:3, 1:3] = 1
    source = {"affine": np.eye(4), "header": nib.Nifti1Header(), "nifti_path": None, "dicom_dir": None, "image": None}
    options = {"quiet": True, "ml": False, "roi_subset": None, "output_types": ["nifti"], "dicom_output": False,
               "statistics": False, "preview": False, "remove_small_blobs": False, "v1_order": False,
               "skip_saving": False, "return_images": True, "nora_tag": "None", "nr_thr_saving": 2,
               "no_derived_masks": False, "batch": False}
    label_map = class_map["liver_vessels"]
    img, _ = P.export_case(data, source, tmp_path / "masks", "liver_vessels", label_map, options)
    assert len(img.header.extensions) == 1                                      # the multilabel image names its labels
    files = sorted(f.name for f in (tmp_path / "masks").iterdir())
    assert files == sorted(f"{name}.nii.gz" for name in label_map.values())
    first = nib.load(tmp_path / "masks" / f"{label_map[1]}.nii.gz")
    assert len(first.header.extensions) == 0 and np.asanyarray(first.dataobj).sum() == 8


def test_a_folder_of_nifti_images_is_a_list_of_cases(tmp_path):
    for name in ("b.nii", "a.nii.gz", ".a_copy.nii.gz", "notes.txt"):
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.nii.gz").write_bytes(b"")
    assert P.nifti_cases(tmp_path) == [("a", tmp_path / "a.nii.gz"), ("b", tmp_path / "b.nii")]
    dicom = tmp_path / "dicom"
    dicom.mkdir()
    (dicom / "IM0001.dcm").write_bytes(b"")
    assert P.nifti_cases(dicom) is None                         # a folder without NIfTI images is one DICOM series
    (tmp_path / "a.nii").write_bytes(b"")
    with pytest.raises(ValueError, match="same name"):
        P.nifti_cases(tmp_path)


def test_a_folder_runs_as_one_batch_with_one_result_per_image(monkeypatch, tmp_path):
    """totalsegmentator() (and the command line) segments a folder of NIfTI images with one predictor call."""
    from totalsegmentator import python_api
    monkeypatch.setenv("TOTALSEG_HOME_DIR", str(tmp_path / "home"))
    calls = []

    class FakePredictor:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *exc): return False
        def predict(self, images, outputs, **kwargs):
            calls.append((images, outputs, kwargs["ml"]))
            return [None] * len(images) if isinstance(images, list) else None

    monkeypatch.setattr(python_api, "TotalSegmentatorPredictor", FakePredictor)
    cts = tmp_path / "cts"
    cts.mkdir()
    for name in ("s2.nii.gz", "s1.nii"):
        (cts / name).write_bytes(b"")
    out = tmp_path / "out"
    assert python_api.totalsegmentator(cts, out, ml=True, device="cpu", quiet=True) == [None, None]
    python_api.totalsegmentator(cts, out, device="cpu", quiet=True)
    assert calls == [([cts / "s1.nii", cts / "s2.nii.gz"], [out / "s1.nii.gz", out / "s2.nii.gz"], True),
                     ([cts / "s1.nii", cts / "s2.nii.gz"], [out / "s1", out / "s2"], False)]
    with pytest.raises(ValueError, match="output is a folder"):
        python_api.totalsegmentator(cts, tmp_path / "seg.nii.gz", ml=True, device="cpu", quiet=True)
    calls.clear()
    python_api.totalsegmentator(cts / "s1.nii", tmp_path / "seg", ml=True, device="cpu", quiet=True)
    assert calls == [(cts / "s1.nii", tmp_path / "seg.nii.gz", True)]       # a multilabel file gets .nii.gz


def test_statistics_of_several_images_in_one_folder_are_named_after_each_image(liver_vessels, tmp_path):
    ts, _, _ = liver_vessels
    image = np.zeros((20, 22, 12), dtype=np.int16)
    image[5:10, 5:10, 3:6] = 100                                # "the liver": the task's model runs
    for name in ("a", "b"):
        nib.save(nib.Nifti1Image(image, np.diag([1.5, 1.5, 3.0, 1.0])), tmp_path / f"{name}.nii.gz")
    out = tmp_path / "out"
    with ts:
        ts.predict([tmp_path / "a.nii.gz", tmp_path / "b.nii.gz"], [out / "a.nii.gz", out / "b.nii.gz"], ml=True,
                   statistics=True)
        assert sorted(f.name for f in out.iterdir()) == ["a.nii.gz", "a_statistics.json", "b.nii.gz", "b_statistics.json"]
        ts.predict(tmp_path / "a.nii.gz", tmp_path / "single" / "a.nii.gz", ml=True, statistics=True)
        assert (tmp_path / "single" / "statistics.json").exists()         # one image: as before
        with pytest.raises(ValueError, match="not a path"):
            ts.predict([tmp_path / "a.nii.gz"], [out / "a.nii.gz"], ml=True, statistics=tmp_path / "s.json")


def test_a_multilabel_output_without_ending_is_compressed(liver_vessels, tmp_path):
    ts, _, _ = liver_vessels
    image = nib.Nifti1Image(np.zeros((8, 9, 10), dtype=np.int16), np.eye(4))
    out = tmp_path / "out"
    ts.predict(image, out / "seg", ml=True)
    ts.predict(image, out / "plain.nii", ml=True)              # an explicit .nii stays uncompressed
    assert sorted(f.name for f in out.iterdir()) == ["plain.nii", "seg.nii.gz"]
    assert (out / "seg.nii.gz").read_bytes()[:2] == b"\x1f\x8b"


def test_wrong_arguments_fail_before_anything_runs(liver_vessels):
    ts, rough, _ = liver_vessels
    image = nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.int16), np.eye(4))
    with pytest.raises(ValueError, match="one output per input"):
        ts.predict([image, image, image], "out")                # not ['o', 'u', 't']
    with pytest.raises(ValueError, match="model_size"):
        P.get_task_config("total", "fast", quiet=True, model_size="Small")
    assert rough.shapes == []


def test_model_inputs_are_dropped_once_their_model_has_run(liver_vessels):
    ts, rough, _ = liver_vessels
    data = np.zeros((1, 6, 7, 8), dtype=np.float32)
    properties = {"spacing": [1.0, 1.0, 1.0]}
    prepared = [rough.preprocess(data, properties)]
    ts.segment(data, properties, "total", prepared=prepared)
    assert prepared == [] and rough.shapes == [(6, 7, 8)]


def test_close_restores_torch_settings_and_a_broken_export_pool_is_replaced(liver_vessels):
    import torch
    from concurrent.futures.process import BrokenProcessPool
    torch.set_num_threads(2)
    before = torch.get_num_threads(), torch.backends.cudnn.benchmark
    ts = P.TotalSegmentatorPredictor("liver_vessels", device="cpu", quiet=True)     # every core on the CPU
    torch.backends.cudnn.benchmark = not before[1]              # as nnU-Net's inference does
    shut_down = []
    ts._export_pool = type("Pool", (), {"shutdown": lambda self, **kwargs: shut_down.append(kwargs)})()
    with pytest.raises(BrokenProcessPool):
        with ts._dropped_if_broken():
            raise BrokenProcessPool("a worker died")
    assert ts._export_pool is None and len(shut_down) == 1      # the next batch starts a new pool
    ts.close()
    assert (torch.get_num_threads(), torch.backends.cudnn.benchmark) == before


def test_dicom_output_needs_dicom_input_before_anything_runs(liver_vessels):
    ts, rough, _ = liver_vessels
    with pytest.raises(ValueError, match="Dicom image as input"):
        ts.predict(nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.int16), np.eye(4)), "out.dcm", output_type="dicom_seg")
    assert rough.shapes == []


def test_weights_that_cannot_be_downloaded(monkeypatch, tmp_path):
    """A model without a released version asks for its folder. A licensed model goes to the license download instead,
    whether or not its weights entry says that it is commercial."""
    from totalsegmentator import libs
    from totalsegmentator.map_to_binary import commercial_models
    monkeypatch.setenv("TOTALSEG_HOME_DIR", str(tmp_path))
    unreleased = next(i for i, info in TASK_ID_WEIGHTS_CONFIGS.items() if info.get("version") is None
                      and not info.get("commercial") and i not in commercial_models.values())
    with pytest.raises(FileNotFoundError, match="not downloaded automatically"):
        libs.download_pretrained_weights(unreleased)
    asked = []
    monkeypatch.setattr(libs, "download_model_with_license_and_unpack", lambda name, path: asked.append(name))
    licensed = next(i for i in commercial_models.values() if not TASK_ID_WEIGHTS_CONFIGS[i].get("commercial"))
    with pytest.raises(RuntimeError, match="weights folder was not created"):   # the fake download creates nothing
        libs.download_pretrained_weights(licensed)
    assert len(asked) == 1


def test_removed_cli_options_are_not_read_as_other_options(monkeypatch, capsys):
    """-fs, -sl and -ro used to be options; without them argparse reads the letters as other options."""
    from totalsegmentator.bin import TotalSegmentator as cli
    calls = []
    monkeypatch.setattr(cli, "totalsegmentator", lambda *args, **kwargs: calls.append(kwargs))
    monkeypatch.setattr("sys.argv", ["TotalSegmentator", "-i", "ct.nii.gz", "-o", "seg", "-fs", "-ro", "1", "-nr", "4",
                                     "-ho", "-bs"])
    cli.main()
    assert calls[0]["fast"] is False and calls[0]["statistics"] is False and calls[0]["radiomics"] is False
    assert "-fs was removed and is ignored" in capsys.readouterr().out
    for removed in (["-sl"], ["-sp", "probabilities"], ["-cp", "crops"]):
        monkeypatch.setattr("sys.argv", ["TotalSegmentator", "-i", "ct.nii.gz", "-o", "seg", *removed])
        with pytest.raises(SystemExit) as exit_info:
            cli.main()
        assert exit_info.value.code == 2
    assert len(calls) == 1


def test_statistics_in_one_pass_equal_the_per_class_definition():
    """Volume, mean, median and the extra metrics of every class, for both memory layouts, against the definition:
    one mask per class."""
    from totalsegmentator.statistics import get_basic_statistics
    rng = np.random.default_rng(0)
    names = class_map["total"]
    seg = np.zeros((24, 20, 16), dtype=np.uint8)
    seg[5:12, 4:15, 4:9], seg[14:20, 6:12, 5:12], seg[0:4, 0:4, 0:4] = 1, 5, 7      # the third touches the border
    ct = rng.integers(-1000, 1000, seg.shape).astype(np.int16)
    img = nib.Nifti1Image(ct, np.diag([1.5, 2.0, 3.0, 1.0]))
    for layout in (np.ascontiguousarray, np.asfortranarray):
        for metric in ("mean", "median"):
            stats = get_basic_statistics(layout(seg), img, None, quiet=True, task="total", metric=metric,
                                         extra_metrics=True)
            for label in (1, 5):
                mask, s = seg == label, stats[names[label]]
                assert s["volume"] == pytest.approx(mask.sum() * 9.0)
                assert s["intensity"] == pytest.approx(getattr(np, metric)(ct[mask]), abs=1e-5)
                index = np.argwhere(mask)
                assert s["n_voxels"] == mask.sum() and s["intensity_min"] == ct[mask].min()
                assert s["bbox_vox"] == [[int(index[:, a].min()), int(index[:, a].max())] for a in range(3)]
                assert s["centroid_vox"] == [round(float(c), 2) for c in index.mean(axis=0)]
            assert stats[names[7]]["volume"] == 0 and stats[names[7]]["centroid_vox"] is None      # at the border
            assert stats[names[2]]["volume"] == 0 and stats[names[2]]["n_voxels"] == 0             # absent


def test_keep_largest_blob_and_v1_order_helpers():
    from scipy import ndimage
    from totalsegmentator.libs import reorder_multilabel_like_v1
    from totalsegmentator.postprocessing import keep_largest_blob
    mask = np.random.default_rng(0).random((30, 30, 30)) > 0.72
    blobs, n = ndimage.label(mask)                      # face connectivity
    sizes = np.bincount(blobs.ravel())[1:]
    assert n > 3 and np.array_equal(keep_largest_blob(mask) > 0, blobs == sizes.argmax() + 1)
    assert not keep_largest_blob(np.zeros((4, 4, 4), dtype=bool)).any()

    task, v1 = class_map["total"], class_map["total_v1"]
    data = np.asfortranarray(np.random.default_rng(0).integers(0, max(task) + 1, (6, 7, 8)).astype(np.uint8))
    expected = np.zeros(data.shape, dtype=np.uint8)
    for label_id, name in v1.items():
        for task_id, task_name in task.items():
            if task_name == name:
                expected[data == task_id] = label_id
    reordered = reorder_multilabel_like_v1(data, task, v1)
    assert np.array_equal(reordered, expected) and reordered.flags.f_contiguous


def test_vertebrae_refinement_runs_on_the_grid_of_the_input(monkeypatch, tmp_path):
    """The dilation kernel of the refinement is not symmetric, so its result depends on the direction of the axes:
    it runs on the input's own grid (here stored left-right flipped, with thick slices), not in the reader's."""
    pytest.importorskip("nnunetv2")
    from totalsegmentator.postprocessing import refine_vertebrae_pp_with_body_mask
    monkeypatch.setenv("TOTALSEG_HOME_DIR", str(tmp_path))
    labels = class_map["vertebrae_pp_refined"]
    image = np.zeros((40, 44, 24), dtype=np.int16)
    image[14:26, 16:28, 4:9], image[14:26, 16:28, 14:19] = 100, 200          # two "vertebral bodies"
    img = nib.Nifti1Image(image, np.diag([-0.85, 0.85, 2.5, 1.0]))
    vertebrae = FakeModel(lambda data: (data == 100) * min(labels) + (data == 200) * (min(labels) + 1))
    body = FakeModel(lambda data: np.ones(data.shape))
    results = {}
    for task in ("vertebrae_pp", "vertebrae_pp_refined"):
        ts = P.TotalSegmentatorPredictor(task, device="cpu", quiet=True)
        monkeypatch.setattr(ts, "models", lambda task, config, roi_subset=None:
                            [(body if task == "vertebrae_body" else vertebrae, None)])
        results[task] = np.asanyarray(ts.predict(img).dataobj)
    zooms = img.header.get_zooms()[:3]
    expected = refine_vertebrae_pp_with_body_mask(results["vertebrae_pp"].copy(), np.ones(image.shape, np.uint8), labels,
                                                  voxel_spacing=zooms, dilation_mm=2)
    assert np.array_equal(results["vertebrae_pp_refined"], expected)
    flipped = refine_vertebrae_pp_with_body_mask(results["vertebrae_pp"][::-1].copy(), np.ones(image.shape, np.uint8),
                                                 labels, voxel_spacing=zooms, dilation_mm=2)[::-1]
    assert not np.array_equal(flipped, expected)                # the same refinement along the flipped axis differs


def test_the_loader_reads_two_images_ahead_unless_the_image_is_kept(liver_vessels, monkeypatch):
    """While an image is predicted, the loader thread reads the next READ_AHEAD images of a task that needs only the
    preprocessed image (a large file takes longer to read than a small image's GPU pass), and one image of a task
    that keeps the image itself (here for its roi), so that no more than that waits in memory."""
    images = [nib.Nifti1Image(np.zeros((20, 22, 12), dtype=np.int16), np.diag([1.5, 1.5, 3.0, 1.0])) for _ in range(5)]

    def loads_while_predicting(ts, ahead):
        """Per image: (the loader got as far ahead as expected, it went further)."""
        started, seen, progress = [], [], threading.Condition()
        load, segment = ts._load, ts.segment

        def counting_load(*args):
            with progress:
                started.append(args[0])
                progress.notify_all()
            return load(*args)

        def watching_segment(data, properties, task, *args, **kwargs):
            if task == ts.task:
                expected = min(len(images), len(seen) + 1 + ahead)
                with progress:
                    seen.append((progress.wait_for(lambda: len(started) >= expected, timeout=60),
                                 progress.wait_for(lambda: len(started) > expected, timeout=0.3)))
            return segment(data, properties, task, *args, **kwargs)

        monkeypatch.setattr(ts, "_load", counting_load)
        monkeypatch.setattr(ts, "segment", watching_segment)
        with ts:
            assert len(ts.predict(images)) == len(images)
        return seen

    ts, _, _ = liver_vessels
    assert loads_while_predicting(ts, ahead=1) == [(True, False)] * len(images)
    total = P.TotalSegmentatorPredictor("total", device="cpu", quiet=True)
    monkeypatch.setattr(total, "models", lambda task, config, roi_subset=None: [(FakeModel(np.zeros_like), None)])
    assert P.READ_AHEAD == 2
    assert loads_while_predicting(total, ahead=P.READ_AHEAD) == [(True, False)] * len(images)

