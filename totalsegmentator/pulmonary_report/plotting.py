from pathlib import Path

import numpy as np
from PIL import Image

from totalsegmentator.aorta_report.plotting import colors


def plot_pulmonary_seg(
    pulmonary_artery,
    landmarks,
    output_path,
    smoothing=20,
    nr_frames=12,
    debug=False,
):
    """Render the pulmonary artery and its five measurement planes."""
    from fury import window
    from totalsegmentator.rendering import plot_mask, record_rotating_scene, text

    del debug
    for landmark_index, landmark in landmarks.items():
        landmark["color"] = list(colors.values())[landmark_index - 1]
        landmark["txt_offset"] = np.array([-20, 0, 0])

    scene = window.Scene()
    try:
        scene.add(
            plot_mask(
                scene,
                pulmonary_artery,
                np.eye(4),
                0,
                0,
                smoothing=smoothing,
                color=colors["gray_light"],
                opacity=0.3,
                orientation="sagittal",
            )
        )
        for landmark_index, landmark in landmarks.items():
            if landmark["empty"]:
                continue
            scene.add(
                plot_mask(
                    scene,
                    landmark["roi"],
                    np.eye(4),
                    0,
                    0,
                    color=landmark["color"],
                    orientation="sagittal",
                    smoothing=smoothing,
                )
            )
            size = pulmonary_artery.shape
            x, y, z = landmark["cl_point"]
            x = size[0] - x
            x, y, z = y, z, x
            x = size[1] - x
            position = np.array([x, y, z]) + landmark["txt_offset"]
            scene.add(text(landmark_index, position, (5, 5, 5), colors["white"]))

        azimuth_angle = int(360 / nr_frames * 1.2)
        prefix = Path(output_path) / "preview_3d_rotating_"
        record_rotating_scene(
            scene,
            prefix,
            (700, 700),
            nr_frames,
            azimuth_angle,
        )
        for frame_index in range(nr_frames):
            frame_path = f"{prefix}{frame_index:06d}.png"
            with Image.open(frame_path) as image:
                cropped = np.asarray(image)[100:-100, 50:-50].copy()
            Image.fromarray(cropped).save(frame_path)
    finally:
        scene.clear()
