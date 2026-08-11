"""Rendering helpers compatible with both VTK- and pygfx-based FURY releases."""

from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image
from fury import __version__ as fury_version
from fury import actor, window
from scipy import sparse
from scipy.spatial.transform import Rotation
from skimage import measure


FURY_GE_2 = int(fury_version.split(".", maxsplit=1)[0]) >= 2


def fury_display_context():
    """Provide a virtual display only for legacy FURY releases that need one."""
    if FURY_GE_2:
        return nullcontext()

    from xvfbwrapper import Xvfb

    return Xvfb()


def _smooth_vertices(vertices, faces, iterations, relaxation=0.1):
    """Apply the same uniform Laplacian smoothing used by the old VTK path."""
    if iterations <= 0 or len(vertices) == 0:
        return vertices

    edges = np.concatenate(
        (
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        )
    )
    rows = np.concatenate((edges[:, 0], edges[:, 1]))
    columns = np.concatenate((edges[:, 1], edges[:, 0]))
    adjacency = sparse.coo_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, columns)),
        shape=(len(vertices), len(vertices)),
    ).tocsr()
    adjacency.data[:] = 1
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    degree[degree == 0] = 1

    vertices = vertices.astype(np.float32, copy=True)
    for _ in range(iterations):
        neighbor_mean = adjacency @ vertices / degree[:, None]
        vertices += relaxation * (neighbor_mean - vertices)
    return vertices


def contour_from_roi_smooth(
    data,
    affine=None,
    color=np.array([1, 0, 0]),
    opacity=1,
    smoothing=0,
):
    """Create a smoothed surface actor without using a rendering-backend API."""
    if data.ndim != 3:
        raise ValueError("Only 3D arrays are currently supported.")
    if not np.any(data):
        raise ValueError("Cannot create a surface from an empty mask.")

    padded = np.pad(np.asarray(data, dtype=np.uint8) > 0, 1)
    vertices, faces, _, _ = measure.marching_cubes(padded, level=0.5)
    vertices -= 1
    vertices = _smooth_vertices(vertices, faces, int(smoothing))

    if affine is not None:
        affine = np.asarray(affine)
        vertices = (
            np.column_stack((vertices, np.ones(len(vertices)))) @ affine.T
        )[:, :3]
        if np.linalg.det(affine[:3, :3]) < 0:
            faces = faces[:, [0, 2, 1]]

    color = np.asarray(color, dtype=float)[:3]
    if FURY_GE_2:
        return actor.surface(
            vertices,
            faces,
            colors=color,
            opacity=opacity,
        )

    rgba = np.tile(np.append(color, opacity), (len(vertices), 1))
    return actor.surface(vertices, faces=faces, colors=rgba)


def plot_mask(
    renderer,
    mask_data,
    affine,
    x_current,
    y_current,
    orientation="axial",
    smoothing=10,
    brain_mask=None,
    color=(1, 0.27, 0.18),
    opacity=1,
):
    """Create a positioned surface actor for a binary mask."""
    del renderer, brain_mask
    mask = mask_data.transpose(0, 2, 1)[::-1, :, :]
    if orientation == "sagittal":
        mask = mask.transpose(2, 1, 0)[::-1, :, :]

    contour_actor = contour_from_roi_smooth(
        mask,
        affine=affine,
        color=color,
        opacity=opacity,
        smoothing=smoothing,
    )
    set_actor_position(contour_actor, (x_current, y_current, 0))
    return contour_actor


def set_actor_position(render_actor, position):
    """Set an actor position across FURY's VTK and pygfx backends."""
    if FURY_GE_2:
        render_actor.local.position = np.asarray(position, dtype=float)
    else:
        render_actor.SetPosition(*position)


def volume_slice(data, affine, value_range):
    """Create the single volume slice used in segmentation previews."""
    if FURY_GE_2:
        return actor.volume_slicer(
            data,
            affine=affine,
            value_range=value_range,
            visibility=(False, False, True),
        )
    return actor.slicer(data=data, affine=affine, value_range=value_range)


def text(text_value, position, scale, color):
    """Create 3D text across FURY versions."""
    if FURY_GE_2:
        return actor.text(
            text=str(text_value),
            position=position,
            font_size=float(max(scale)),
            colors=color,
        )
    return actor.vector_text(
        text=str(text_value),
        pos=position,
        scale=scale,
        color=color,
    )


def plane_actors(center, normal, radius):
    """Create the filled disk and outline used for CPR measurement planes."""
    centers = np.asarray([center], dtype=float)
    directions = np.asarray([normal], dtype=float)
    fill_color = (1.0, 0.6, 0.0)
    outline_color = (1.0, 0.5, 0.0)

    if FURY_GE_2:
        return [
            actor.disk(
                centers,
                directions=directions,
                radii=radius,
                sectors=64,
                colors=fill_color,
                opacity=0.85,
            ),
            actor.ring(
                centers,
                directions=directions,
                inner_radius=radius * 0.88,
                outer_radius=radius,
                circumferential_segments=64,
                colors=outline_color,
                opacity=1.0,
            ),
        ]

    return [
        actor.disk(
            centers,
            directions,
            np.asarray([(*fill_color, 0.85)]),
            rinner=0.0,
            router=radius,
            cresolution=64,
        ),
        actor.disk(
            centers,
            directions,
            np.asarray([(*outline_color, 1.0)]),
            rinner=radius * 0.88,
            router=radius,
            cresolution=64,
        ),
    ]


def remove_actors(scene, render_actors):
    """Remove multiple actors from a scene."""
    if FURY_GE_2:
        scene.remove(*render_actors)
    else:
        for render_actor in render_actors:
            scene.rm(render_actor)


class SceneRenderer:
    """Reusable offscreen renderer hiding FURY's backend-specific window API."""

    def __init__(self, scene, size, parallel=False, reset_camera=True):
        self.scene = scene
        self.size = tuple(size)
        self._scene_camera = None

        if FURY_GE_2:
            camera = None
            if parallel:
                from fury.lib import DirectionalLight, OrthographicCamera

                camera = OrthographicCamera()
                camera.add(DirectionalLight())
                scene.add(camera)
                self._scene_camera = camera
            self.show_manager = window.ShowManager(
                scene=scene,
                camera=camera,
                size=self.size,
                window_type="offscreen",
                pixel_ratio=1,
                enable_events=False,
            )
            self.screen = self.show_manager.screens[0]
            self.show_manager.render()
            self.show_manager.window.draw()
            if reset_camera:
                self.screen.camera.show_object(
                    scene.main_scene,
                    view_dir=(0, 0, -1),
                    up=(0, 1, 0),
                    scale=1.02,
                    match_aspect=True,
                )
            self.show_manager.render()
            self.show_manager.window.draw()
            self._target = self._scene_center()
        else:
            self.show_manager = window.ShowManager(
                scene=scene,
                size=self.size,
                reset_camera=False,
            )
            self.show_manager.initialize()
            if parallel:
                scene.projection(proj_type="parallel")
            if reset_camera:
                scene.reset_camera_tight(margin_factor=1.02)
            self.show_manager.render()

    def _scene_center(self):
        bounds = self.scene.main_scene.get_world_bounding_box()
        if bounds is None:
            return np.zeros(3)
        return np.asarray(bounds, dtype=float).mean(axis=0)

    def snapshot(self):
        if FURY_GE_2:
            self.show_manager.render()
            self.show_manager.window.draw()
            image = self.show_manager.snapshot()
        else:
            self.show_manager.render()
            image = window.snapshot(
                self.scene,
                size=self.size,
                render_window=self.show_manager.window,
            )
        return np.asarray(image)[..., :3].copy()

    def azimuth(self, angle):
        if FURY_GE_2:
            camera = self.screen.camera
            offset = np.asarray(camera.world.position) - self._target
            offset = Rotation.from_euler("y", angle, degrees=True).apply(offset)
            camera.world.position = self._target + offset
            camera.world.reference_up = (0, 1, 0)
            camera.look_at(self._target)
        else:
            self.scene.azimuth(angle)

    def close(self):
        if FURY_GE_2:
            self.show_manager.close()
            if self._scene_camera is not None:
                self.scene.remove(self._scene_camera)
        else:
            self.show_manager.exit()


def save_scene(scene, output_path, size, parallel=False, reset_camera=True):
    """Render one scene image to disk."""
    renderer = SceneRenderer(
        scene,
        size,
        parallel=parallel,
        reset_camera=reset_camera,
    )
    try:
        Image.fromarray(renderer.snapshot()).save(output_path)
    finally:
        renderer.close()


def record_rotating_scene(
    scene,
    output_prefix,
    size,
    nr_frames,
    azimuth_angle,
):
    """Render a numbered sequence while rotating the camera."""
    renderer = SceneRenderer(scene, size, reset_camera=True)
    try:
        for frame_index in range(nr_frames):
            frame_path = Path(f"{output_prefix}{frame_index:06d}.png")
            Image.fromarray(renderer.snapshot()).save(frame_path)
            renderer.azimuth(azimuth_angle)
    finally:
        renderer.close()
