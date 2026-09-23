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
