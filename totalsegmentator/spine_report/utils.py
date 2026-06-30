import json
import time

import numpy as np


def get_erosion_struct_elem(nifti_img, erosion_mm=2):
    """Create an anisotropic 3D erosion kernel in voxel space."""
    voxel_spacings = nifti_img.header.get_zooms()
    radii = [erosion_mm / spacing for spacing in voxel_spacings]

    x, y, z = np.ogrid[
        -radii[0]:radii[0] + 1,
        -radii[1]:radii[1] + 1,
        -radii[2]:radii[2] + 1,
    ]

    return x * x / (radii[0] ** 2) + y * y / (radii[1] ** 2) + z * z / (radii[2] ** 2) <= 1


def save_runtime(start_time, report_name, nodeinfo_path):
    runtime = time.time() - start_time
    runtime_file = nodeinfo_path.parent / "META" / "runtime.json"
    runtime_file.parent.mkdir(parents=True, exist_ok=True)
    runtime_data = json.load(open(runtime_file)) if runtime_file.exists() else {}
    runtime_data[report_name] = round(runtime, 2)
    json.dump(runtime_data, open(runtime_file, "w"), indent=4)
