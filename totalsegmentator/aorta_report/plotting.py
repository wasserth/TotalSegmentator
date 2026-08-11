from pathlib import Path

from PIL import Image
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage import measure
from scipy import interpolate

from totalsegmentator.aorta_report.geometry import get_region_diameters_pd
from totalsegmentator.aorta_report.report_layout import compose_report_image


colors = {
    "green": [0, 1, 0],
    "red_light": [1, .27, .18],
    "blue": [0, 0, 1],
    "green_light": [.27, 1, .18],
    "red_dark": [0.6, 0, 0],
    "green_dark": [0, 0.6, 0],
    "blue_light": [.27, .18, 1],
    "brown_light": [0.8, 0.5, 0],
    "brown": [0.3, 0, 0],
    "gray": [0.5, 0.5, 0.5],
    "purple": [0.4, 0, 0.7],
    "blue_dark": [0, 0, 0.6],
    "orange": [0.9, 0.3, 0],
    "yellow": [0.9, 1.0, 0.1],
    "pink": [1, 0.5, 0.7],
    "white": [1, 1, 1],
    "red": [1, 0, 0],
    "gray_light": [0.7, 0.7, 0.7],
    "gray_dark": [0.3, 0.3, 0.3],
}

def smooth_contours(contour, s=20):
    """
    s: 10: no real smoothing
       20: medium smoothing, still close to original
       50: medium smoothing, already a bit too far
      100: stronger smoothing (too for from original)
    """
    x, y = contour[:, 1], contour[:, 0]
    if s == 0:
        return x, y
    tck, u = interpolate.splprep([x, y], s=s)  
    x_new, y_new = interpolate.splev(np.linspace(0, 1, 1000), tck)
    return x_new, y_new
    

def plot_img(img, mask, mask_2, diameter_points, vmin=None, vmax=None, cmap="gray", smooth=20):

    fig = plt.figure()

    # plot img
    plt.imshow(img, cmap=cmap, interpolation="bilinear", origin='lower', vmin=vmin, vmax=vmax)

    # plot contours 1
    for x, y in _iter_smoothed_contours(mask, smooth=smooth):
        plt.plot(x, y, linewidth=2, color="green")

    # plot contours 2
    if mask_2 is not None:
        for x, y in _iter_smoothed_contours(mask_2, smooth=smooth):
            plt.plot(x, y, linewidth=2, color="red")

    # plot diameter
    if diameter_points is not None:
        start, end, start_pd, end_pd = diameter_points
        plt.plot([start[1], end[1]], [start[0], end[0]], 'g-') 
        if start_pd is not None:
            plt.plot([start_pd[1], end_pd[1]], [start_pd[0], end_pd[0]], 'g-')

    plt.axis('off')
    return fig


def plot_sagittal_and_coronal_slice(img, mask, output_path, vmin, vmax):
    """
    Plot sagittal and coronal slice of nifti image and nifti binary mask.
    Saves png image with of slices next to each other in one image.

    img: nifti image
    mask: nifti binary mask
    output_path: path to save image
    vmin, vmax: min and max values for image
    """
    image_data = img.get_fdata()
    mask_data = mask.get_fdata()

    img_sagittal = image_data[image_data.shape[0] // 2, :, :].T[::-1, ::-1]
    img_coronal = image_data[:, image_data.shape[1] // 2, :].T[::-1, ::-1]

    mask_sagittal = mask_data[mask_data.shape[0] // 2, :, :].T[::-1, ::-1]
    mask_coronal = mask_data[:, mask_data.shape[1] // 2, :].T[::-1, ::-1]

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3, 13))

    ax1.imshow(img_sagittal, cmap="gray", vmin=vmin, vmax=vmax)
    ax2.imshow(img_coronal, cmap="gray", vmin=vmin, vmax=vmax)


    # plot mask
    _plot_mask_contours(ax1, mask_sagittal, color="red", smooth=0, linewidth=1)
    _plot_mask_contours(ax2, mask_coronal, color="red", smooth=0, linewidth=1)

    ax1.axis("off")
    ax2.axis("off")

    fig.savefig(output_path, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)


def _safe_smooth_contour(contour, smooth=20):
    if smooth == 0 or contour.shape[0] < 4:
        return contour[:, 1], contour[:, 0]
    try:
        return smooth_contours(contour, s=smooth)
    except ValueError:
        return contour[:, 1], contour[:, 0]


def _iter_smoothed_contours(mask_2d, smooth=20):
    for contour in measure.find_contours(mask_2d, 0.5):
        if len(contour) >= 3:
            yield _safe_smooth_contour(contour, smooth=smooth)


def _plot_mask_contours(ax, mask_2d, color, smooth=20, linewidth=1.6):
    for x, y in _iter_smoothed_contours(mask_2d, smooth=smooth):
        ax.plot(x, y, linewidth=linewidth, color=color)


def _plot_diameter_curves(ax, curve_y_mm, aorta_curve, true_curve=None, false_curve=None):
    """Plot all available diameter curves and return handles and x-axis maximum."""
    plotted_curves = []
    finite_curves = []
    curve_specs = (
        (aorta_curve, "#f7c66a", 1.6, 0.85, "Aorta"),
        (true_curve, "#7CFC70", 2.2, 1.0, "True lumen"),
        (false_curve, "#ff6b6b", 2.2, 1.0, "False lumen"),
    )
    for curve, color, width, alpha, label in curve_specs:
        if curve is None:
            continue
        curve = np.asarray(curve)
        finite = np.isfinite(curve)
        if not finite.any():
            continue
        plotted_curves.append(
            ax.plot(
                curve, curve_y_mm, color=color, linewidth=width, alpha=alpha, label=label
            )[0]
        )
        finite_curves.append(curve[finite])
    finite_max = max((float(np.max(curve)) for curve in finite_curves), default=10.0)
    return plotted_curves, max(finite_max, 10.0)


def _style_diameter_profile(
    ax, display_y_max, max_curve_mm, landmark_positions=(), labelsize=None
):
    for landmark in landmark_positions:
        ax.axhline(landmark["distance_mm"], color="white", linewidth=0.8, alpha=0.25)
        ax.text(
            max_curve_mm * 1.05,
            landmark["distance_mm"],
            landmark["label"],
            color="white",
            fontsize=8 if labelsize is None else labelsize,
            va="center",
            ha="left",
        )
    ax.set_xlabel("Diameter [mm]", color="white", fontsize=labelsize)
    ax.set_ylabel("Along aorta centerline [mm]", color="white", fontsize=labelsize)
    ax.set_xlim(0, max_curve_mm * 1.45)
    ax.set_ylim(display_y_max, 0)
    ax.tick_params(colors="white", labelsize=labelsize)
    ax.grid(axis="x", color="white", alpha=0.15, linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_color("white")
        spine.set_alpha(0.4)


def _fit_curve_positions_to_display(curve_positions_mm, display_y_mm):
    """Stretch profile y-positions to the visible CPR extent.

    The diameter profile is sometimes sampled on a slightly shorter centerline
    than the CPR stack, so we map its finite y-range onto the CPR display
    range to keep both panels synchronized at the distal end.
    """
    if curve_positions_mm is None:
        return display_y_mm

    curve_y_mm = np.asarray(curve_positions_mm, dtype=float)
    display_y_mm = np.asarray(display_y_mm, dtype=float)
    if curve_y_mm.size == 0 or display_y_mm.size == 0:
        return curve_y_mm

    finite = np.isfinite(curve_y_mm)
    if finite.sum() < 2:
        return curve_y_mm

    curve_min = float(np.nanmin(curve_y_mm))
    curve_max = float(np.nanmax(curve_y_mm))
    display_min = float(np.nanmin(display_y_mm))
    display_max = float(np.nanmax(display_y_mm))
    if abs(curve_max - curve_min) < 1e-8:
        return np.full_like(curve_y_mm, display_min, dtype=float)

    scale = (display_max - display_min) / (curve_max - curve_min)
    return display_min + (curve_y_mm - curve_min) * scale


def _get_cpr_longitudinal_views(image_data, mask_data):
    img_sagittal = image_data[image_data.shape[0] // 2, :, :].T[::-1, ::-1]
    img_coronal = image_data[:, image_data.shape[1] // 2, :].T[::-1, ::-1]
    mask_sagittal = mask_data[mask_data.shape[0] // 2, :, :].T[::-1, ::-1]
    mask_coronal = mask_data[:, mask_data.shape[1] // 2, :].T[::-1, ::-1]
    return img_sagittal, img_coronal, mask_sagittal, mask_coronal


def _get_cpr_diameter_curve(mask_img):
    mask_data = mask_img.get_fdata() > 0.5
    spacing = np.asarray(mask_img.header.get_zooms()[:2], dtype=float)
    nr_slices = mask_data.shape[2]
    diameters_mm = np.full(nr_slices, np.nan, dtype=float)

    for slice_idx in range(nr_slices):
        mask_slice = mask_data[:, :, slice_idx].astype(np.uint8)
        if mask_slice.sum() < 5:
            continue
        diameter_mm, _, _, _ = get_region_diameters_pd(mask_slice, spacing, np.eye(4), z=0)
        if diameter_mm > 0:
            diameters_mm[slice_idx] = float(diameter_mm)

    return diameters_mm


def _get_landmark_positions(landmarks, centerline_points_vox, centerline_affine, sample_positions_mm, nr_slices, step_mm):
    if landmarks is None or centerline_points_vox is None or centerline_affine is None or sample_positions_mm is None:
        return []

    centerline_points_vox = np.asarray(centerline_points_vox, dtype=float)
    if centerline_points_vox.shape[0] == 0:
        return []

    centerline_points_mm = nib.affines.apply_affine(centerline_affine, centerline_points_vox)
    segment_lengths = np.linalg.norm(np.diff(centerline_points_mm, axis=0), axis=1)
    cumulative_length_mm = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    landmark_positions = []

    for lm_nr, lm_dict in landmarks.items():
        if lm_dict.get("empty") or lm_dict.get("cl_idx") is None:
            continue

        cl_idx = int(np.clip(lm_dict["cl_idx"], 0, len(cumulative_length_mm) - 1))
        landmark_distance_mm = cumulative_length_mm[cl_idx]
        cpr_slice_idx = int(np.argmin(np.abs(sample_positions_mm - landmark_distance_mm)))
        display_slice_idx = nr_slices - 1 - cpr_slice_idx
        label = f"{lm_nr} {lm_dict.get('name', '')}".strip()
        landmark_positions.append({
            "label": label,
            "slice_idx": display_slice_idx,
            "distance_mm": display_slice_idx * step_mm,
        })

    return landmark_positions


def plot_cpr_overview(img, mask, output_path, vmin, vmax, true_lumen_mask=None, false_lumen_mask=None,
                      landmarks=None, centerline_points_vox=None, centerline_affine=None, sample_positions_mm=None,
                      curve_positions_mm=None, aorta_curve_mm=None, true_curve_mm=None, false_curve_mm=None,
                      metadata=None, tmp_dir=None):
    """
    Create a CPR overview PNG with two longitudinal views and a diameter profile.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_data = img.get_fdata()
    mask_data = mask.get_fdata() > 0.5
    img_sagittal, img_coronal, mask_sagittal, mask_coronal = _get_cpr_longitudinal_views(image_data, mask_data)

    nr_slices = image_data.shape[2]
    step_mm = float(img.header.get_zooms()[2])
    display_y_mm = np.arange(nr_slices)[::-1] * step_mm

    aorta_curve = aorta_curve_mm if aorta_curve_mm is not None else _get_cpr_diameter_curve(mask)
    true_curve = true_curve_mm if true_curve_mm is not None else (
        _get_cpr_diameter_curve(true_lumen_mask) if true_lumen_mask is not None else None
    )
    false_curve = false_curve_mm if false_curve_mm is not None else (
        _get_cpr_diameter_curve(false_lumen_mask) if false_lumen_mask is not None else None
    )
    curve_y_mm = _fit_curve_positions_to_display(curve_positions_mm, display_y_mm)
    landmark_positions = _get_landmark_positions(
        landmarks,
        centerline_points_vox,
        centerline_affine,
        sample_positions_mm,
        nr_slices,
        step_mm,
    )

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(13.5, 8.5), facecolor="black")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.2], wspace=0.16)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    for ax, title, curr_img, curr_mask in (
        (ax1, "Sagittal CPR", img_sagittal, mask_sagittal),
        (ax2, "Coronal CPR", img_coronal, mask_coronal),
    ):
        ax.imshow(curr_img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="bicubic")
        _plot_mask_contours(ax, curr_mask, color="#ff4d4d", smooth=35, linewidth=1.8)
        for landmark in landmark_positions:
            ax.axhline(landmark["slice_idx"], color="white", linewidth=0.8, alpha=0.25)
        ax.set_title(title, color="white", fontsize=12, pad=10)
        ax.axis("off")

    plotted_curves, max_curve_mm = _plot_diameter_curves(
        ax3, curve_y_mm, aorta_curve, true_curve, false_curve
    )
    _style_diameter_profile(
        ax3, float(display_y_mm.max()), max_curve_mm, landmark_positions
    )

    ax3.set_title("Diameter Profile", color="white", fontsize=12, pad=10)
    if plotted_curves:
        ax3.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9)

    fig.canvas.draw()
    fig_h, fig_w = fig.canvas.get_width_height()[::-1]
    fig_rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(fig_h, fig_w, 4)
    content_img = Image.fromarray(fig_rgba[:, :, :3].copy())
    plt.close(fig)
    final_img = compose_report_image(content_img, metadata=metadata, tmp_dir=tmp_dir)
    final_img.save(output_path)


def _record_rotating_scene(scene, output_prefix, window_size, nr_frames):
    from totalsegmentator.rendering import record_rotating_scene

    azimuth_angle = int(360 / nr_frames * 1.2)
    record_rotating_scene(
        scene,
        output_prefix,
        window_size,
        nr_frames,
        azimuth_angle,
    )
    for frame_index in range(nr_frames):
        frame_path = f"{output_prefix}{frame_index:06d}.png"
        with Image.open(frame_path) as frame_image:
            cropped = np.asarray(frame_image)[100:-100, 150:-150].copy()
        Image.fromarray(cropped).save(frame_path)


def plot_masks_3d(masks, output_path, file_prefix, smoothing=20, nr_frames=12, debug=False, colors_subset=None):
    """
    todo: check that all masks are not empty -> otherwise error

    masks: list of binary mask ([ndarray])
    """
    from fury import window
    from totalsegmentator.rendering import plot_mask

    window_size = (700, 900)
    scene = window.Scene()
    try:
        for idx, mask in enumerate(masks):
            color = list(colors.keys())[idx] if colors_subset is None else colors_subset[idx]
            scene.add(plot_mask(
                scene, mask, np.eye(4), 0, 0, smoothing=smoothing,
                color=colors[color], opacity=1.0, orientation="sagittal",
            ))
        _record_rotating_scene(
            scene, Path(output_path) / file_prefix, window_size, nr_frames
        )
    finally:
        scene.clear()
    
    
def plot_aorta_3d(aorta, true_lumen, false_lumen, all_vessels, centerline, landmarks, output_path, smoothing=20, nr_frames=12, debug=False):
    """
    todo: check that all masks are not empty -> otherwise error

    aorta: binary mask (ndarray)
    all_vessels: binary mask (ndarray)
    landsmakrs: dict of all landsmarks. each landmark has keys cl_idx, roi, diameter
    """
    from fury import window
    from totalsegmentator.rendering import plot_mask, text

    for lm_nr, lm_dict in landmarks.items():
        lm_dict["color"] = list(colors.values())[lm_nr-1]

    landmarks[1]["txt_offset"] = np.array([-50,-10,0])
    landmarks[2]["txt_offset"] = np.array([-50,-5,0])
    landmarks[3]["txt_offset"] = np.array([-50,0,0])
    landmarks[4]["txt_offset"] = np.array([-50,0,0])
    landmarks[5]["txt_offset"] = np.array([-40,0,0])
    landmarks[6]["txt_offset"] = np.array([-10,20,0])
    landmarks[7]["txt_offset"] = np.array([10,0,0])
    landmarks[8]["txt_offset"] = np.array([20,0,0])
    landmarks[9]["txt_offset"] = np.array([20,0,0])
    landmarks[10]["txt_offset"] = np.array([20,0,0])
    landmarks[11]["txt_offset"] = np.array([20,0,0])

    window_size = (700, 900)
    scene = window.Scene()
    try:
        scene.add(plot_mask(
            scene, aorta, np.eye(4), 0, 0, smoothing=smoothing,
            color=colors["gray_light"], opacity=.3, orientation="sagittal",
        ))
        scene.add(plot_mask(
            scene, false_lumen, np.eye(4), 0, 0, smoothing=smoothing,
            color=colors["red"], opacity=.1, orientation="sagittal",
        ))
        scene.add(plot_mask(
            scene, all_vessels, np.eye(4), 0, 0, smoothing=smoothing,
            color=colors["gray_dark"], opacity=1.0, orientation="sagittal",
        ))
        scene.add(plot_mask(
            scene, centerline, np.eye(4), 0, 0, smoothing=0,
            color=colors["red"], opacity=1.0, orientation="sagittal",
        ))

        for lm_nr, lm_dict in landmarks.items():
            if lm_dict["empty"]:
                continue
            scene.add(plot_mask(
                scene, lm_dict["roi"], np.eye(4), 0, 0,
                color=lm_dict["color"], orientation="sagittal", smoothing=smoothing,
            ))
            size = all_vessels.shape
            x, y, z = lm_dict["cl_point"]
            x = size[0] - x
            x, y, z = y, z, x
            x = size[1] - x
            position = np.array([x, y, z]) + lm_dict["txt_offset"]
            scene.add(text(lm_nr, position, (8, 8, 8), colors["white"]))

        _record_rotating_scene(
            scene,
            Path(output_path) / "preview_3d_rotating_",
            window_size,
            nr_frames,
        )
    finally:
        scene.clear()
