import gc
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from totalsegmentator.aorta_report.centerline import resample_points_by_arc_length
from totalsegmentator.aorta_report.geometry import get_region_diameters_pd
from totalsegmentator.aorta_report.nifti import (
    _allocate_rgb_nifti_buffer,
    _save_rgb_nifti_buffer,
    _write_rgb_frame,
)
from totalsegmentator.aorta_report.plotting import (
    _fit_curve_positions_to_display,
    _get_landmark_positions,
    _iter_smoothed_contours,
    _plot_diameter_curves,
    _plot_mask_contours,
    _style_diameter_profile,
)
from totalsegmentator.aorta_report.report_layout import compose_report_image, get_report_layout_base


def _resample_animation_centerline(cl, affine, sample_positions_mm):
    """Recreate the CPR centerline at the supplied physical positions."""
    points_vox = np.asarray([vertex.point for vertex in cl], dtype=float)
    points_mm = nib.affines.apply_affine(affine, points_vox)
    resampled_mm = resample_points_by_arc_length(points_mm, sample_positions_mm)
    if len(resampled_mm) >= 5:
        resampled_mm = gaussian_filter1d(
            resampled_mm, sigma=2.0, axis=0, mode="nearest"
        )
        resampled_mm[0] = points_mm[0]
        resampled_mm[-1] = points_mm[-1]
    return points_vox, nib.affines.apply_affine(np.linalg.inv(affine), resampled_mm)


def _centerline_tangents(points):
    tangents = np.zeros_like(points)
    if len(points) >= 2:
        tangents[0] = points[1] - points[0]
        tangents[-1] = points[-1] - points[-2]
    if len(points) > 2:
        tangents[1:-1] = points[2:] - points[:-2]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    return tangents / norms


def _voxel_to_render(point, volume_shape):
    size_x, size_y, _ = volume_shape
    return np.array([size_y - 1 - point[1], point[2], size_x - 1 - point[0]])


def _direction_to_render(direction):
    return np.array([-direction[1], direction[2], -direction[0]])


def _create_scene(aorta, smoothing, plane_radius, centers, normals):
    from fury import window
    from totalsegmentator.rendering import (
        SceneRenderer,
        plane_actors,
        plot_mask,
        remove_actors,
    )

    scene = window.Scene()
    aorta_actor = plot_mask(
        scene, aorta, np.eye(4), 0, 0, smoothing=smoothing,
        color=[0.7, 0.7, 0.7], opacity=0.3, orientation="sagittal",
    )
    scene.add(aorta_actor)

    bounds_actors = []
    for center, normal in (
        (centers[0], normals[0]),
        (centers[-1], normals[-1]),
    ):
        actors = plane_actors(center, normal, plane_radius)
        scene.add(*actors)
        bounds_actors.extend(actors)

    renderer = SceneRenderer(scene, (350, 700), parallel=True, reset_camera=True)
    remove_actors(scene, bounds_actors)
    return scene, renderer


def _content_crop_bounds(image, padding=12):
    foreground = image.max(axis=2) > 5
    rows = np.any(foreground, axis=1)
    columns = np.any(foreground, axis=0)
    if not rows.any() or not columns.any():
        return 0, image.shape[0], 0, image.shape[1]
    row_indices = np.where(rows)[0]
    column_indices = np.where(columns)[0]
    return (
        max(0, int(row_indices[0]) - padding),
        min(image.shape[0], int(row_indices[-1]) + padding + 1),
        max(0, int(column_indices[0]) - padding),
        min(image.shape[1], int(column_indices[-1]) + padding + 1),
    )


def _figure_rgb(figure):
    figure.canvas.draw()
    height, width = figure.canvas.get_width_height()[::-1]
    return np.frombuffer(
        figure.canvas.buffer_rgba(), dtype=np.uint8
    ).reshape(height, width, 4)[:, :, :3].copy()


def _render_cpr_panel_base(image, mask, nr_frames, width, height, vmin, vmax):
    dpi = 100
    figure = plt.figure(
        figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="black"
    )
    try:
        axis = figure.add_axes([0, 0, 1, 1])
        axis.imshow(
            image, cmap="gray", vmin=vmin, vmax=vmax,
            interpolation="bicubic", aspect="auto",
        )
        _plot_mask_contours(axis, mask, color="#ff4d4d", smooth=35, linewidth=1.5)
        axis.axis("off")
        panel = _figure_rgb(figure)
        pixel_y = np.empty(nr_frames, dtype=int)
        for frame_index in range(nr_frames):
            pixel = axis.transData.transform([0, nr_frames - 1 - frame_index])
            pixel_y[frame_index] = int(panel.shape[0] - pixel[1])
    finally:
        plt.close(figure)

    if panel.shape[0] != height:
        scale = height / panel.shape[0]
        panel = np.asarray(
            Image.fromarray(panel).resize((panel.shape[1], height), Image.LANCZOS)
        )
        pixel_y = np.clip((pixel_y * scale).astype(int), 0, height - 1)
    return panel, pixel_y


def _render_profile_panel_base(
    height, axis_bottom, axis_height, curve_y_mm, display_y_max,
    aorta_curve, true_curve, false_curve, landmark_positions,
):
    width = 400
    dpi = 100
    figure = plt.figure(
        figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="black"
    )
    try:
        axis = figure.add_axes([0.22, axis_bottom, 0.52, axis_height])
        plotted, finite_max = _plot_diameter_curves(
            axis, curve_y_mm, aorta_curve, true_curve, false_curve
        )
        _style_diameter_profile(
            axis, display_y_max, finite_max, landmark_positions, labelsize=7
        )
        if plotted:
            axis.legend(
                frameon=False, loc="upper center",
                bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=7,
            )
        panel = _figure_rgb(figure)
    finally:
        plt.close(figure)
    if panel.shape[0] != height:
        panel = np.asarray(
            Image.fromarray(panel).resize((panel.shape[1], height), Image.LANCZOS)
        )
    return panel


def _panel_with_position_line(base, pixel_y, color=(255, 140, 0)):
    panel = base.copy()
    y = int(np.clip(pixel_y, 1, panel.shape[0] - 2))
    panel[y - 1:y + 2, :] = color
    return panel


def _render_cross_section_panel(
    ct_slice, mask_slice, spacing_mm, panel_width, panel_height, vmin, vmax, font
):
    normalized = ((ct_slice.astype(float) - vmin) / (vmax - vmin) * 255)
    image_view = normalized.clip(0, 255).astype(np.uint8).T[::-1]
    mask_view = mask_slice.T[::-1]
    display_size = min(panel_width - 20, panel_height // 2 - 10)
    section = Image.fromarray(image_view, mode="L").convert("RGB")
    section = section.resize((display_size, display_size), Image.LANCZOS)

    if mask_view.max() > 0:
        draw = ImageDraw.Draw(section)
        scale = display_size / mask_view.shape[0]
        for x_values, y_values in _iter_smoothed_contours(mask_view, smooth=20):
            points = [
                (int(x * scale), int(y * scale))
                for x, y in zip(x_values, y_values)
            ]
            draw.line(points + [points[0]], fill=(0, 255, 0), width=2)

    area_cm2 = mask_slice.sum() * spacing_mm ** 2 / 100
    try:
        diameter, diameter_pd, _, _ = get_region_diameters_pd(
            mask_slice.astype(np.uint8), [spacing_mm] * 2, np.eye(4), z=0
        )
    except Exception:
        diameter, diameter_pd = 0.0, 0.0

    panel = Image.new("RGB", (panel_width, panel_height), (0, 0, 0))
    paste_x = (panel_width - display_size) // 2
    panel.paste(section, (paste_x, 10))
    if area_cm2 > 0.05:
        draw = ImageDraw.Draw(panel)
        text_y = 10 + display_size + 14
        draw.text(
            (paste_x, text_y), f"\u25A8: {area_cm2:.1f} cm\u00B2",
            fill=(0, 255, 0), font=font,
        )
        draw.text(
            (paste_x, text_y + 22),
            f"\u00D8: {diameter / 10:.1f} x {diameter_pd / 10:.1f} cm",
            fill=(0, 255, 0), font=font,
        )
    return np.asarray(panel)


def _compose_frame(panels, titles, title_height, title_font):
    total_width = sum(panel.shape[1] for panel in panels)
    panel_height = panels[0].shape[0]
    image = Image.new("RGB", (total_width, panel_height + title_height), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    x_offset = 0
    for title, panel in zip(titles, panels):
        try:
            text_width = draw.textlength(title, font=title_font)
        except AttributeError:
            text_width = len(title) * 8
        draw.text(
            (x_offset + (panel.shape[1] - text_width) / 2, 6),
            title, fill="white", font=title_font,
        )
        image.paste(Image.fromarray(panel), (x_offset, title_height))
        x_offset += panel.shape[1]
    return np.asarray(image)


def generate_animated_cpr_nifti(
    aorta, cl, affine,
    res_ct_img, res_seg_img,
    true_lumen_cpr_img, false_lumen_cpr_img,
    cpr_info,
    curve_positions_mm, aorta_curve_mm, true_curve_mm, false_curve_mm,
    landmarks,
    max_dia_all,
    output_path, tmp_dir,
    smoothing=100,
    vmin=-700, vmax=1000,
    logger=None,
    metadata=None,
):
    """Generate an animated CPR as a scrollable nifti stack.

    Each slice shows 4 panels side-by-side:
      1. 3D aorta rendering with an orange cross-sectional plane
      2. Coronal CPR with an orange position line
      3. Diameter profile with an orange position line
      4. 2D cross-sectional view with area / diameter overlay
    """
    from tqdm import tqdm

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cpr_nr_frames = cpr_info["nr_slices"]
    sp_mm = cpr_info["sample_positions_mm"]
    step_mm = cpr_info["centerline_step_mm"]

    TARGET_H = 800
    TITLE_H = 28

    cl_pts, rsp_vox = _resample_animation_centerline(cl, affine, sp_mm)
    tan = _centerline_tangents(rsp_vox)
    centers = np.asarray([_voxel_to_render(point, aorta.shape) for point in rsp_vox])
    normals = np.asarray([_direction_to_render(direction) for direction in tan])

    ct_cpr = res_ct_img.get_fdata()
    seg_cpr = (res_seg_img.get_fdata() > 0.5).astype(np.uint8)
    in_plane_sp = float(res_ct_img.header.get_zooms()[0])

    img_cor = ct_cpr[:, ct_cpr.shape[1] // 2, :].T[::-1, ::-1]
    mask_cor = seg_cpr[:, seg_cpr.shape[1] // 2, :].T[::-1, ::-1]

    cpr_width = max(int(img_cor.shape[1] * TARGET_H / img_cor.shape[0]), 150)
    with plt.style.context("dark_background"):
        cpr_base, cpr_pixel_y = _render_cpr_panel_base(
            img_cor, mask_cor, cpr_nr_frames, cpr_width, TARGET_H, vmin, vmax
        )
    display_y_mm = np.arange(cpr_nr_frames)[::-1] * step_mm
    if curve_positions_mm is not None and len(curve_positions_mm) > 0:
        frame_positions_mm = _fit_curve_positions_to_display(
            curve_positions_mm, display_y_mm
        )
        frame_indices = np.asarray([
            int(np.argmin(np.abs(display_y_mm - position)))
            for position in frame_positions_mm
        ])
    else:
        frame_indices = np.arange(cpr_nr_frames)
    nr_frames = len(frame_indices)
    cpr_pixel_y = cpr_pixel_y[frame_indices]

    if logger:
        logger.info(f"Generating animated CPR ({nr_frames} frames) ...")

    data_top = int(np.min(cpr_pixel_y))
    data_bottom = int(np.max(cpr_pixel_y))
    axis_bottom = max(0.01, 1.0 - data_bottom / TARGET_H)
    axis_height = min(
        (data_bottom - data_top) / TARGET_H, 1.0 - axis_bottom - 0.01
    )
    curve_y_mm = _fit_curve_positions_to_display(curve_positions_mm, display_y_mm)
    landmark_positions = _get_landmark_positions(
        landmarks, cl_pts, affine, sp_mm, cpr_nr_frames, step_mm
    )
    with plt.style.context("dark_background"):
        diameter_base = _render_profile_panel_base(
            TARGET_H, axis_bottom, axis_height, curve_y_mm, float(display_y_mm.max()),
            aorta_curve_mm, true_curve_mm, false_curve_mm, landmark_positions,
        )

    width_3d = 200
    width_cross_section = 280
    total_width = (
        width_3d + cpr_base.shape[1] + diameter_base.shape[1] + width_cross_section
    )
    try:
        title_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13
        )
        small_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12
        )
    except Exception:
        title_font = ImageFont.load_default()
        small_font = title_font

    layout_base = get_report_layout_base(
        total_width, TARGET_H + TITLE_H, metadata=metadata, tmp_dir=tmp_dir
    )
    nifti_buffer = _allocate_rgb_nifti_buffer(
        layout_base[0].width, layout_base[0].height, nr_frames
    )

    from totalsegmentator.rendering import plane_actors, remove_actors

    scene = renderer = None
    try:
        scene, renderer = _create_scene(
            aorta, smoothing, max_dia_all * 0.75, centers, normals
        )
        crop_bounds = _content_crop_bounds(renderer.snapshot())
        row_start, row_end, column_start, column_end = crop_bounds

        for frame_number in tqdm(range(nr_frames), desc="Animated CPR"):
            cpr_index = frame_indices[frame_number]
            current_plane_actors = plane_actors(
                centers[cpr_index],
                normals[cpr_index],
                max_dia_all * 0.75,
            )
            scene.add(*current_plane_actors)
            snapshot = renderer.snapshot()
            remove_actors(scene, current_plane_actors)
            snapshot = snapshot[row_start:row_end, column_start:column_end]
            panel_3d = np.asarray(
                Image.fromarray(snapshot).resize((width_3d, TARGET_H), Image.LANCZOS)
            )
            cpr_panel = _panel_with_position_line(
                cpr_base, cpr_pixel_y[frame_number]
            )
            diameter_panel = _panel_with_position_line(
                diameter_base, cpr_pixel_y[frame_number]
            )
            cross_section_panel = _render_cross_section_panel(
                ct_cpr[:, :, cpr_index],
                seg_cpr[:, :, cpr_index],
                in_plane_sp,
                width_cross_section,
                TARGET_H,
                vmin,
                vmax,
                small_font,
            )
            frame = _compose_frame(
                (panel_3d, cpr_panel, diameter_panel, cross_section_panel),
                ("3D", "Coronal CPR", "Diameter Profile", "Crossectional view"),
                TITLE_H,
                title_font,
            )
            layout_frame = compose_report_image(frame, layout_base=layout_base)
            _write_rgb_frame(nifti_buffer, np.asarray(layout_frame), frame_number)
    finally:
        try:
            if renderer is not None:
                renderer.close()
            if scene is not None:
                scene.clear()
        finally:
            scene = renderer = None
            gc.collect()

    _save_rgb_nifti_buffer(nifti_buffer, output_path)

    if logger:
        logger.info(f"Animated CPR saved to {output_path}")
