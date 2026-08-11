import numpy as np
from fury import window

from totalsegmentator.rendering import (
    FURY_GE_2,
    SceneRenderer,
    plane_actors,
    plot_mask,
    record_rotating_scene,
    remove_actors,
    text,
    volume_slice,
)


def _cube_mask():
    mask = np.zeros((16, 16, 16), dtype=np.uint8)
    mask[3:13, 3:13, 3:13] = 1
    return mask


def test_rendering_helpers_create_backend_compatible_actors():
    scene = window.Scene()
    mask_actor = plot_mask(
        scene,
        _cube_mask(),
        np.eye(4),
        0,
        0,
        smoothing=2,
        opacity=0.5,
    )
    scene.add(mask_actor)
    scene.add(text("1", (0, 0, 0), (2, 2, 2), (1, 1, 1)))

    planes = plane_actors((8, 8, 8), (0, 0, 1), 3)
    scene.add(*planes)
    remove_actors(scene, planes)

    assert mask_actor is not None


def test_offscreen_renderer_returns_requested_image_size():
    scene = window.Scene()
    scene.add(plot_mask(scene, _cube_mask(), np.eye(4), 0, 0, smoothing=2))
    scene.add(text("1", (3, 3, 3), (2, 2, 2), (1, 1, 1)))
    scene.add(*plane_actors((8, 8, 8), (0, 0, 1), 3))
    renderer = SceneRenderer(scene, (128, 96), parallel=True)
    try:
        image = renderer.snapshot()
    finally:
        renderer.close()

    assert image.shape == (96, 128, 3)
    assert image.max() > 0


def test_volume_slice_is_available_on_both_fury_generations():
    data = np.arange(8 * 8 * 8, dtype=np.float32).reshape((8, 8, 8))
    assert volume_slice(data, np.eye(4), (0, data.max())) is not None


def test_parallel_camera_tightly_fits_wide_scene():
    scene = window.Scene()
    mask = _cube_mask()
    for column in range(6):
        scene.add(
            plot_mask(
                scene,
                mask,
                np.eye(4),
                column * 20,
                0,
                smoothing=0,
            )
        )

    renderer = SceneRenderer(scene, (384, 96), parallel=True)
    try:
        image = renderer.snapshot()
    finally:
        renderer.close()

    foreground = image.max(axis=2) > 5
    _, columns = np.where(foreground)
    assert columns.max() - columns.min() > image.shape[1] * 0.8


def test_legacy_rotating_scene_preserves_fury_record_camera(monkeypatch, tmp_path):
    if FURY_GE_2:
        return

    class FakeScene:
        def __init__(self):
            self.margin_factor = None

        def reset_camera_tight(self, margin_factor):
            self.margin_factor = margin_factor

    captured = {}

    def fake_record(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(window, "record", fake_record)
    scene = FakeScene()
    record_rotating_scene(scene, tmp_path / "frame_", (700, 900), 12, 36)

    assert scene.margin_factor == 1.02
    assert captured["reset_camera"] is True
    assert captured["path_numbering"] is True
    assert captured["n_frames"] == 12
    assert captured["az_ang"] == 36
