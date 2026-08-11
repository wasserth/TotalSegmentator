import numpy as np
from fury import window

from totalsegmentator.rendering import (
    SceneRenderer,
    plane_actors,
    plot_mask,
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
