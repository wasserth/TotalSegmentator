import sys
from pathlib import Path
import time
import argparse
import json
from importlib import resources
import tempfile
import io
import subprocess

import nibabel as nib
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.ndimage import binary_dilation

from totalsegmentator.resampling import change_spacing
from totalsegmentator.postprocessing import keep_largest_blob, remove_small_blobs
from totalsegmentator.registration import calc_transform, apply_transform
from totalsegmentator.python_api import totalsegmentator
from totalsegmentator.nifti_ext_header import load_multilabel_nifti
from totalsegmentator.serialization_utils import decompress_and_deserialize, filestream_to_nifti, serialize_and_compress, convert_to_serializable, nib_load_eager, NumpyJsonEncoder
from totalsegmentator.dicom_io import dcm_to_nifti
from totalsegmentator.config import send_usage_stats_application


# run models in python_api but does not work properly in e.g. streamlit or modal
def run_models(ct_img, verbose=False):
    brain_skull = totalsegmentator(ct_img, None, roi_subset=["brain", "skull"], ml=True, nr_thr_saving=1, quiet=not verbose)
    yield brain_skull
    ventricle_parts = totalsegmentator(ct_img, None, task="ventricle_parts", ml=True, nr_thr_saving=1, quiet=not verbose)
    yield ventricle_parts


# Required if calling from e.g. streamlit; calling python_api not working there
def run_models_shell(ct_img, verbose=False):
    quiet = "--quiet" if not verbose else ""
    with tempfile.TemporaryDirectory(prefix="totalseg_tmp_2_") as tmp_folder:
        tmp_dir = Path(tmp_folder)
        ct_img_path = tmp_dir / "ct.nii.gz"
        nib.save(ct_img, ct_img_path)
        subprocess.call(f"TotalSegmentator -i {ct_img_path} -o {tmp_dir / 'brain_skull.nii.gz'} --roi_subset brain skull --ml --nr_thr_saving 1 {quiet}", shell=True)
        brain_skull_img = nib_load_eager(tmp_dir / "brain_skull.nii.gz")  # eager loading required here
        yield brain_skull_img
        subprocess.call(f"TotalSegmentator -i {ct_img_path} -o {tmp_dir / 'ventricle_parts.nii.gz'} --task ventricle_parts --ml --nr_thr_saving 1 {quiet}", shell=True)
        ventricle_parts_img = nib_load_eager(tmp_dir / "ventricle_parts.nii.gz")
        yield ventricle_parts_img


def extract_brain(brain_mask, ct_img):
    mask = brain_mask.get_fdata() > 0.5
    data = ct_img.get_fdata()
    data[~mask] = 0
    return nib.Nifti1Image(data, ct_img.affine)


def max_diameter_x(mask):
    diameters = []
    for z in range(mask.shape[2]):
        slice_2d = mask[:, :, z]
        max_diameter = 0
        start = (0, z)
        end = (0, z)
        for y in range(slice_2d.shape[1]):
            x_indices = np.where(slice_2d[:, y])[0]
            if len(x_indices) > 0:
                diameter = x_indices[-1] - x_indices[0]
                if diameter > max_diameter:
                    max_diameter = diameter
                    start = [x_indices[0], y, z]
                    end = [x_indices[-1], y, z]
        diameters.append( (max_diameter, (start, end)) )
    
    # sort diameters for printing
    diameters = sorted(diameters, key=lambda x: x[0], reverse=True)
    # print([d[0] for d in diameters[:20]])
        
    # from diameters get the max diameter
    max_diameter_all = max(diameters, key=lambda x: x[0])
    return max_diameter_all


def plot_slice_with_diameters(brain, start_b, end_b, start_v, end_v, evans_index, brain_vol, vent_vol, vol_ratio):
    z = start_v[2]
    slice_2d_b = brain[:, :, z]
    slice_2d_b = slice_2d_b.transpose(1,0)  # x and y axis are flipped in imshow
    
    # Create figure with extra space at bottom for disclaimer
    plt.figure(figsize=(8, 9.0))
    
    # Main plot
    plt.subplot(111)
    plt.imshow(slice_2d_b, cmap="gray", origin="lower", interpolation="gaussian")
    # brain diameter
    plt.scatter([start_b[0], end_b[0]], [start_b[1], end_b[1]], color="red", marker="x", s=200)
    plt.plot([start_b[0], end_b[0]], [start_b[1], end_b[1]], color="green", linewidth=3)
    # ventricle diameter
    plt.scatter([start_v[0], end_v[0]], [start_v[1], end_v[1]], color="red", marker="x", s=200)
    plt.plot([start_v[0], end_v[0]], [start_v[1], end_v[1]], color="green", linewidth=3)
    plt.title(f"Evans index: {evans_index:.3f}\n".upper() + 
              f"brain volume: {brain_vol:.1f}ml*\n" +
              f"ventricle volume: {vent_vol:.1f}ml\n" +
              f"ventricle/brain ratio: {vol_ratio:.3f}", 
              fontweight='normal')
    plt.axis("off")
    plt.gca().invert_xaxis()

    # Add disclaimer text at bottom with adjusted position
    disclaimer = "* Volume of brain + cranial cavity (area inside of skull)\n\n" + \
                 "This is a research prototype and not designed for diagnosis of any medical complaint.\n" + \
                 "Created by AI Lab, Department of Radiology, University Hospital Basel"
    plt.figtext(0.5, -0.01, disclaimer, ha='center', va='bottom', fontsize=8, wrap=True)

    plt.tight_layout()
    
    # Save to bytes buffer instead of file (to be able to return it from modal function)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=200, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    buf.seek(0)
    return buf.getvalue()


def plot_empty_result():
    """Create an image with only text for cases where segmentation is empty"""
    plt.figure(figsize=(8, 9.0))
    
    # Set black background
    ax = plt.gca()
    ax.set_facecolor('black')
    plt.gcf().patch.set_facecolor('black')
    
    # Add main text in white
    plt.text(0.5, 0.5, "No calculation possible because the segmentation is empty.\nDoes your image contain the full brain?",
             ha='center', va='center', wrap=True, color='white')
    plt.axis('off')
    
    # Add disclaimer text at bottom in white
    disclaimer = "This is a research prototype and not designed for diagnosis of any medical complaint.\n" + \
                "Created by AI Lab, Department of Radiology, University Hospital Basel"
    plt.figtext(0.5, 0.01, disclaimer, ha='center', va='bottom', fontsize=8, wrap=True, color='white')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=200, bbox_inches='tight', pad_inches=0.1, facecolor='black')
    plt.close()
    buf.seek(0)
    return buf.getvalue()


def evans_index(ct_bytes, f_type, verbose=False):
    """
    ct_bytes: file path to nifti file | 
              Nifti1Image object |
              filestream of zip file (result of streamlit.file_uploader) |
              filestream of nifti file (result of streamlit.file_uploader)
    f_type: "niigz" or "nii" or "dicom" 
    """
    st = time.time()
    yield {"id": 1, "progress": 2, "status": "Loading data"}

    if isinstance(ct_bytes, Path) and f_type != "dicom":  # for local usage
        ct_img = nib.load(ct_bytes)
    elif isinstance(ct_bytes, nib.Nifti1Image):  # for local usage
        ct_img = ct_bytes
    elif f_type == "dicom":  # for online zip file bytes
        print("Converting dicom to nifti...")
        with tempfile.TemporaryDirectory(prefix="totalseg_tmp_") as tmp_folder:
            ct_tmp_path = Path(tmp_folder) / "ct.nii.gz"
            dcm_to_nifti(ct_bytes, ct_tmp_path, tmp_dir=Path(tmp_folder), verbose=True)
            ct_img = nib.load(ct_tmp_path)
            ct_img = nib.Nifti1Image(np.asanyarray(ct_img.dataobj), ct_img.affine, ct_img.header)  # eager loading into memory
    elif f_type == "niigz":  # for online nifti bytes
        ct_img = filestream_to_nifti(ct_bytes, gzipped=True)
    else:  # for online nifti bytes
        ct_img = filestream_to_nifti(ct_bytes, gzipped=False)

    resolution = 1.0
    
    # Run models
    yield {"id": 2, "progress": 10, "status": "Run brain and skull model"}
    # brain_skull_img, ventricle_img_orig = run_models(ct_img, verbose=verbose)
    model_results = run_models_shell(ct_img, verbose=verbose)
    brain_skull_img = next(model_results)
    yield {"id": 3, "progress": 40, "status": "Run ventricle model"}
    ventricle_img_orig = next(model_results)

    # Load atlas
    yield {"id": 4, "progress": 70, "status": "Process segmentations"}
    with resources.files('totalsegmentator').joinpath('resources/ct_brain_atlas_1mm.nii.gz').open('rb') as f:
        brain_atlas = f.name
    atlas_img = nib.load(brain_atlas)
    
    # Load images
    brain_skull_img, label_map = load_multilabel_nifti(brain_skull_img)
    label_map_inv = {v: k for k, v in label_map.items()}
    brain_skull_data = brain_skull_img.get_fdata()
    brain = (brain_skull_data == label_map_inv["brain"]).astype(np.uint8)
    skull = (brain_skull_data == label_map_inv["skull"]).astype(np.uint8)
    brain_img_orig = nib.Nifti1Image(brain, brain_skull_img.affine)
    skull_img_orig = nib.Nifti1Image(skull, brain_skull_img.affine)
    ventricle = ventricle_img_orig.get_fdata()
    frontal_horn = np.where((ventricle == 1) | (ventricle == 6), 1, 0)
    
    # Check if segmentation is empty
    if np.sum(brain) == 0 or np.sum(skull) == 0 or np.sum(frontal_horn) == 0:
        evans_index = {
            "evans_index": None,
            "brain_volume_ml": None,
            "ventricle_volume_ml": None, 
            "ventricle_brain_ratio": None
        }
        report_png_bytes = plot_empty_result()
    else:
        # Canonical
        brain_img = nib.as_closest_canonical(brain_img_orig)
        skull_img = nib.as_closest_canonical(skull_img_orig)
        ventricle_img = nib.as_closest_canonical(ventricle_img_orig)
        ct_img = nib.as_closest_canonical(ct_img)

        # Resampling
        yield {"id": 5, "progress": 75, "status": "Resampling"}
        brain_img = change_spacing(brain_img, resolution, dtype=np.uint8, order=0)  # order=1 leads to holes in seg
        skull_img = change_spacing(skull_img, resolution, dtype=np.uint8, order=0)
        ventricle_img = change_spacing(ventricle_img, resolution, dtype=np.uint8, order=0)
        ct_img = change_spacing(ct_img, resolution, dtype=np.uint8, order=1)
        
        # Editing
        ct_img = extract_brain(brain_img, ct_img)

        # Registration
        yield {"id": 6, "progress": 80, "status": "Registration"}
        transform = calc_transform(ct_img, atlas_img, transform_type="Rigid", resample=None, verbose=False)
        brain_img = apply_transform(brain_img, atlas_img, transform, resample=None, dtype=np.uint8, order=0)
        skull_img = apply_transform(skull_img, atlas_img, transform, resample=None, dtype=np.uint8, order=0, interp="genericLabel")
        ventricle_img = apply_transform(ventricle_img, atlas_img, transform, resample=None, dtype=np.uint8, order=0, interp="genericLabel")

        # Get data
        brain_data = brain_img.get_fdata()
        skull_data = skull_img.get_fdata()
        ventricle_data = ventricle_img.get_fdata()
        ventricle_all = (ventricle_data > 0).astype(np.uint8)

        
        # Increase brain to fill entire space inside of skull.
        # But do not increase too much because otherwise areas without skull around will get too big.
        # (fast if doing here in 1mm space, in orig space can take 10s if high img resolution)
        brain_data = binary_dilation(brain_data, iterations=2).astype(np.uint8) 
        brain_data[skull_data > 0] = 0  # Remove skull from brain mask
        brain_data = keep_largest_blob(brain_data)
        
        # Calculate volumes
        voxel_vol_mm3 = np.prod(brain_img.header.get_zooms())
        brain_volume_ml = np.sum(brain_data) * voxel_vol_mm3 * 0.001
        ventricle_volume_ml = np.sum(ventricle_all) * voxel_vol_mm3 * 0.001
        ventricle_brain_ratio = ventricle_volume_ml / brain_volume_ml

        # select only frontal horn
        ventricle_data = np.where((ventricle_data == 1) | (ventricle_data == 6), 1, 0)

        # postprocessing for robustness
        yield {"id": 7, "progress": 90, "status": "Postprocessing"}
        brain_data = remove_small_blobs(brain_data, [200, 1e10])  # fast
        ventricle_data = remove_small_blobs(ventricle_data, [10, 1e10])  # fast

        # Get diameters
        max_dia_vent, (start_vent, end_vent) = max_diameter_x(ventricle_data)
        brain_data_slice = brain_data[:,:,start_vent[2]:end_vent[2]+1]  # select same slice as ventricles
        max_dia_brain, (start_brain, end_brain) = max_diameter_x(brain_data_slice)
        
        # Make 2mm wider to correct for skull-brain gap
        # addon_mm = 1 / resolution
        # max_dia_brain += 2 * addon_mm
        # start_brain[0] = start_brain[0] - addon_mm
        # end_brain[0] = end_brain[0] + addon_mm
        
        # Calc index
        evans_index = max_dia_vent / max_dia_brain
        
        # Plot
        yield {"id": 8, "progress": 95, "status": "Plotting"}
        skull_data[ventricle_all > 0] = 1  # combine skull and ventricle masks
        report_png_bytes = plot_slice_with_diameters(skull_data, start_brain, end_brain, 
                                                    start_vent, end_vent, evans_index,
                                                    brain_volume_ml, ventricle_volume_ml,
                                                    ventricle_brain_ratio)
    
        evans_index = {
            "evans_index": round(evans_index, 3),
            "brain_volume_ml": round(brain_volume_ml, 1),
            "ventricle_volume_ml": round(ventricle_volume_ml, 1),
            "ventricle_brain_ratio": round(ventricle_brain_ratio, 3)
        }

    masks_output = {
        "brain_mask": brain_img_orig,
        "skull_mask": skull_img_orig,
        "ventricle_mask": ventricle_img_orig
    }

    if verbose:
        print(f"took: {time.time()-st:.2f}s")

    yield {"id": 9, "progress": 100, "status": "Done",
            "report_json": evans_index, 
            "masks": serialize_and_compress(masks_output),
            "report_png": report_png_bytes}


"""
cd /mnt/nvme/data/test_data/evans_index/31103170_rot_large
python ~/dev/TotalSegmentator/totalsegmentator/bin/totalseg_evans_index.py -i ct_sm.nii.gz -o evans_index.json -p evans_index.png
python ~/dev/TotalSegmentator/totalsegmentator/bin/totalseg_evans_index.py -i ct_nonbrain.nii.gz -o evans_index_nonbrain.json -p evans_index_nonbrain.png
"""
if __name__ == "__main__":
    """
    For more documentation see resources/evans_index.md

    Requires:
    pip install antspyx blosc
    """
    parser = argparse.ArgumentParser(description="Calculate evans index.")
    parser.add_argument("-i", "--ct_img", type=lambda p: Path(p).resolve(), required=True,
                        help="Path to ct_img.")
    parser.add_argument("-o", "--output_file", type=lambda p: Path(p).resolve(), required=True,
                        help="json output file.")
    parser.add_argument("-p", "--preview_file", type=lambda p: Path(p).resolve(), required=True,
                        help="Preview PNG file of evans index to check if calculation is correct.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print detailed progress information")
    args = parser.parse_args()

    # Check if additional dependencies are installed
    for pkg, name in [("ants", "antspyx"), ("blosc", "blosc")]:
        try:
            __import__(pkg)
        except ImportError:
            print(f"Error: {name} package not installed. Please install it using:")
            print(f"pip install {name}")
            sys.exit(1)

    if str(args.ct_img).endswith(".nii.gz"):
        f_type = "niigz"
    elif str(args.ct_img).endswith(".zip"):
        f_type = "dicom"
    else:
        f_type = "nii"

    # Passing a file path
    res = evans_index(args.ct_img, f_type, verbose=args.verbose)

    # Passing a Nifti1Image object (does not work with zip files)
    # ct_img = nib.load(args.ct_img)
    # ct_img = nib.Nifti1Image(np.asanyarray(ct_img.dataobj), ct_img.affine, ct_img.header)  # eager loading into memory
    # res = evans_index(ct_img, f_type, verbose=args.verbose)

    for r in res:
        print(f"progress: {r['progress']}%: {r['status']}")
        if r["progress"] == 100:
            final_result = r

    # Save report json
    json.dump(final_result["report_json"], open(args.output_file, "w"), indent=4, cls=NumpyJsonEncoder)

    # Save report png
    with open(args.preview_file, "wb") as f:
        f.write(final_result["report_png"])

    # Save masks for debugging
    # masks = decompress_and_deserialize(final_result["masks"])
    # nib.save(masks["brain_mask"], str(args.output_file).replace(".json", "_brain_mask.nii.gz"))
    # nib.save(masks["skull_mask"], str(args.output_file).replace(".json", "_skull_mask.nii.gz"))
    # nib.save(masks["ventricle_mask"], str(args.output_file).replace(".json", "_ventricle_mask.nii.gz"))

    send_usage_stats_application("evans_index")
