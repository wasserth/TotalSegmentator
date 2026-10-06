import nibabel as nib
import numpy as np
from PIL import Image

from totalsegmentator.aorta_report.centerline import Vertex
from totalsegmentator.aorta_report import landmarks as landmarks_module
from totalsegmentator.aorta_report.landmarks import (
    attach_structure_anchors,
    build_centerline,
    create_landmarks,
    create_structures,
)
from totalsegmentator.aorta_report.measurements import create_landmark_planes
from totalsegmentator.aorta_report import plotting
from totalsegmentator.aorta_report import rendering


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)


def test_sinotubular_junction_does_not_require_brachio_anchor():
    shape = (10, 10, 12)
    aorta = np.zeros(shape, dtype=np.uint8)
    aorta[4:7, 4:7, 1:11] = 1
    structures = create_structures()
    for structure in structures.values():
        structure["data"] = np.zeros(shape, dtype=np.uint8)
    structures["sinotub_junc"]["data"][2:8, 2:8, 2:10] = 1
    centerline = [Vertex((5, 5, z)) for z in range(1, 11)]
    logger = _Logger()

    result = attach_structure_anchors(
        structures, aorta, centerline, (1, 1, 1), logger
    )

    assert result["brachio"]["empty"]
    assert not result["sinotub_junc"]["empty"]
    assert result["sinotub_junc"]["cl_idx"] is not None


def test_empty_annulus_still_builds_aorta_centerline(monkeypatch):
    path = [Vertex((5, 5, z)) for z in range(4, 20)]
    image = np.zeros((12, 12, 24), dtype=np.uint8)

    def fake_get_centerline(data, debug=False):
        return image.copy(), [Vertex(vertex.point.copy()) for vertex in path]

    monkeypatch.setattr(landmarks_module, "get_centerline", fake_get_centerline)
    aorta = np.zeros_like(image)
    aorta[4:8, 4:8, 4:20] = 1
    logger = _Logger()

    _, centerline, resampled = build_centerline(
        aorta, np.zeros_like(aorta), np.eye(4), (1, 1, 1), logger
    )

    assert len(centerline) == len(path)
    assert np.array_equal(centerline[-1].point, path[-1].point)
    assert len(resampled) > 10
    assert any("Annulus mask is empty" in message for message in logger.messages)


def test_empty_aorta_skips_centerline_without_raising():
    aorta = np.zeros((8, 8, 8), dtype=np.uint8)
    logger = _Logger()

    image, centerline, resampled = build_centerline(
        aorta, aorta.copy(), np.eye(4), (1, 1, 1), logger
    )

    assert image.shape == aorta.shape
    assert centerline == []
    assert resampled == []
    assert any("Aorta mask is empty" in message for message in logger.messages)


def test_small_aorta_keeps_centerline_when_smoothing_erases_it():
    aorta = np.zeros((48, 48, 48), dtype=np.uint8)
    aorta[24, 24, 12:36] = 1
    logger = _Logger()

    _, centerline, resampled = build_centerline(
        aorta, np.zeros_like(aorta), np.eye(4), (1, 1, 1), logger
    )

    assert len(centerline) >= 2
    assert len(resampled) >= 2
    assert any("Smoothed aorta mask is empty" in message for message in logger.messages)


def test_unplaced_landmark_marks_dependents_empty_before_plane_measurement():
    structures = create_structures()
    for structure in structures.values():
        structure["empty"] = True
        structure["cl_idx"] = None
    structures["brachio"]["empty"] = False
    structures["brachio"]["cl_idx"] = 20
    structures["subclavian"]["empty"] = False
    structures["subclavian"]["cl_idx"] = 8
    centerline = [Vertex((6, 6, z)) for z in range(21)]
    aorta = np.zeros((12, 12, 24), dtype=np.uint8)
    aorta[4:9, 4:9, 2:22] = 1
    aorta_img = nib.Nifti1Image(aorta, np.eye(4))
    logger = _Logger()

    landmarks = create_landmarks(
        structures,
        annulus_volume=0,
        centerline=centerline,
        aorta_img=aorta_img,
        spacing=(1, 1, 1),
        logger=logger,
    )

    assert landmarks[5]["empty"]
    assert landmarks[6]["empty"]
    assert landmarks[6].get("cl_idx") is None
    assert any("landmark 6 has no centerline index" in message for message in logger.messages)

    max_diameter = create_landmark_planes(
        landmarks, centerline, aorta, np.zeros_like(aorta), (1, 1, 1), logger
    )

    assert max_diameter >= 0
    assert all("diameter_tmp" in landmark for landmark in landmarks.values())


def test_preview_skips_empty_masks(monkeypatch, tmp_path):
    plotted = []

    def fake_plot_mask(_scene, mask, **_kwargs):
        plotted.append(int(np.count_nonzero(mask)))
        return "actor"

    class _Scene:
        def __init__(self):
            self.actors = []

        def add(self, actor):
            self.actors.append(actor)

        def clear(self):
            self.actors.clear()

    monkeypatch.setattr("totalsegmentator.rendering.plot_mask", fake_plot_mask)
    monkeypatch.setattr("totalsegmentator.rendering.text", lambda *args, **kwargs: "text")
    monkeypatch.setattr("fury.window.Scene", _Scene)
    monkeypatch.setattr(plotting, "_record_rotating_scene", lambda *args, **kwargs: None)

    aorta = np.zeros((8, 8, 8), dtype=np.uint8)
    aorta[2:6, 2:6, 1:7] = 1
    landmarks = {
        number: {"empty": True, "roi": np.zeros_like(aorta), "cl_point": None}
        for number in range(1, 12)
    }

    plotting.plot_aorta_3d(
        aorta,
        aorta,
        np.zeros_like(aorta),
        np.zeros_like(aorta),
        np.zeros_like(aorta),
        landmarks,
        tmp_path,
        smoothing=0,
        nr_frames=1,
    )

    assert plotted == [int(np.count_nonzero(aorta))]


def test_empty_structure_dependencies_produce_empty_landmarks_without_index_errors():
    structures = create_structures()
    for structure in structures.values():
        structure["empty"] = True
    centerline = [Vertex((0, 0, z)) for z in range(5)]
    aorta_img = nib.Nifti1Image(np.ones((3, 3, 5), dtype=np.uint8), np.eye(4))
    logger = _Logger()

    landmarks = create_landmarks(
        structures,
        annulus_volume=0,
        centerline=centerline,
        aorta_img=aorta_img,
        spacing=(1, 1, 1),
        logger=logger,
    )

    assert all(landmark["empty"] for landmark in landmarks.values())
    assert set(landmarks) == set(range(1, 12))


def test_report_frontpage_renders_layout_once(monkeypatch, tmp_path):
    monkeypatch.setattr(rendering, "REPORT_FRAMES", 2)
    monkeypatch.setattr(rendering, "REPORT_PLOT_TYPES", ("first_", "second_"))
    colors = ((10, 20, 30), (40, 50, 60), (70, 80, 90), (100, 110, 120))
    for index, (prefix, frame) in enumerate(
        (("first_", 0), ("first_", 1), ("second_", 0), ("second_", 1))
    ):
        Image.new("RGB", (4, 3), colors[index]).save(
            tmp_path / f"{prefix}{frame:06d}.png"
        )

    render_calls = []

    def fake_generate_html(_assets, _template, values, output_path, **_kwargs):
        render_calls.append(values["preview_3d"])
        image = Image.new("RGB", (10, 8), "black")
        image.paste(Image.open(values["preview_3d"]), (2, 2))
        image.save(output_path)
        return image

    captured = []

    def fake_combine(paths):
        captured.extend(np.asarray(Image.open(path))[3, 3].tolist() for path in paths)
        return "combined"

    monkeypatch.setattr(rendering, "generate_html", fake_generate_html)
    monkeypatch.setattr(rendering, "combine_as_nifti", fake_combine)

    result = rendering.render_report_image({}, {}, {}, tmp_path)

    assert result == "combined"
    assert len(render_calls) == 1
    assert captured == [list(color) for color in colors]
