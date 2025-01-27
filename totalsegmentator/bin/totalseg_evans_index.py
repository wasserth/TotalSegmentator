import sys
from pathlib import Path
import time
import argparse
import json
import pkg_resources

import nibabel as nib
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from totalsegmentator.resampling import change_spacing
from totalsegmentator.postprocessing import keep_largest_blob, remove_small_blobs
from totalsegmentator.registration import calc_transform, apply_transform



def run_models_consecutive(ct_path, tmp_dir, logger):
    """
    """
    st = time.time()

    yield "Running TotalSegmentator - brain and skull"
    if (tmp_dir / 'brain.nii.gz').exists():
        logger.info("  Skipping TotalSeg brain and skull (already exists)")
    else:
        rois = ["brain", "skull"]
        rois_str = " ".join(rois)
        subprocess.call(f"TotalSegmentator -i {ct_path} -o {tmp_dir} -ns 1 -rs {rois_str}", shell=True)

    yield "Running TotalSegmentator - ventricle parts"
    if (tmp_dir / 'ventricle_parts.nii.gz').exists():
        logger.info("  Skipping TotalSeg ventricle parts (already exists)")
    else:
        subprocess.call(f"TotalSegmentator -i {ct_path} -o {tmp_dir / 'ventricle_parts.nii.gz'} -ns 1 -ta ventricle_parts -ml 1", shell=True)

    print(f"Models done (took: {time.time()-st:.2f}s)")
    yield "Models done"



async def run_totalsegmentator_async(ct_img, file_out, args):
    tmp_dir = Path(tempfile.mkdtemp())
    ct_path = tmp_dir / "ct.nii.gz"
    file_out = tmp_dir / f"{file_out}.nii.gz"
    nib.save(ct_img, ct_path)
    # subprocess.call(f"TotalSegmentator -i {ct_path} -o {file_out} {args}", shell=True)
    command = f"TotalSegmentator -i {ct_path} -o {file_out} {args}"
    _ = await asyncio.to_thread(subprocess.call, command, shell=True)
    return nib.load(file_out)


async def run_models_parallel(ct_path, tmp_dir, logger, host="local"):
    """
    host: local | modal
    """

    print("Running Models - ASYNC")

    rois = ["brain", "skull"]
    rois_str = " ".join(rois)

    ct_img = nib.load(ct_path)
    ct_img = nib.Nifti1Image(ct_img.get_fdata(), ct_img.affine)  # copy image to be able to pass it as parameter to modal
    st = time.time()

    if host == "local":
        img1, img2 = await asyncio.gather(   # this needs to be inside of asnyc func with await
            run_totalsegmentator_async(ct_img, "totalseg_body.nii.gz", f"-ml -ns 1 -rs {rois_str}"), 
            run_totalsegmentator_async(ct_img, "totalseg_ventricle_parts.nii.gz", f"-ml -ns 1 -ta ventricle_parts"), 
        )
    elif host == "modal":
        import modal 
        run_ts = modal.Function.lookup("totalsegmentator", "run_totalsegmentator")
        img1, img2 = await asyncio.gather(
            run_ts.remote.aio(ct_img, {"roi_subset": rois, "ml": True, "nr_thr_saving": 1}), 
            run_ts.remote.aio(ct_img, {"task": "ventricle_parts", "ml": True, "nr_thr_saving": 1}), 
        )

    print(f"Models ASYNC done (took: {time.time()-st:.2f}s)")
    # todo: add this function to totalsegmentator
    map_to_binary_custom(img1, tmp_dir, "header")
    map_to_binary_custom(img2, tmp_dir, "header")


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


def plot_slice_with_diameters(brain, start_b, end_b, start_v, end_v, evans_index, output_file):
    z = start_v[2]
    slice_2d_b = brain[:, :, z]
    slice_2d_b = slice_2d_b.transpose(1,0)  # x and y axis are flipped in imshow
    
    # Create figure with extra space at bottom for disclaimer
    plt.figure(figsize=(8, 8.5))
    
    # Main plot
    plt.subplot(111)
    plt.imshow(slice_2d_b, cmap="gray", origin="lower", interpolation="gaussian")
    # brain diameter
    plt.scatter([start_b[0], end_b[0]], [start_b[1], end_b[1]], color="red", marker="x", s=200)
    plt.plot([start_b[0], end_b[0]], [start_b[1], end_b[1]], color="green", linewidth=3)
    # ventricle diameter
    plt.scatter([start_v[0], end_v[0]], [start_v[1], end_v[1]], color="red", marker="x", s=200)
    plt.plot([start_v[0], end_v[0]], [start_v[1], end_v[1]], color="green", linewidth=3)
    plt.title(f"Evans index: {evans_index:.2f}", fontweight='bold')  # (slice {z})
    plt.axis("off")
    plt.gca().invert_xaxis()

    # Add disclaimer text at bottom with adjusted position
    disclaimer = "This is a research prototype and not designed for diagnosis of any medical complaint.\n" + \
                "Created by AI Lab, Department of Radiology, University Hospital Basel"
    plt.figtext(0.5, 0.01, disclaimer, ha='center', va='bottom', fontsize=8, wrap=True)

    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches='tight', pad_inches=0.1)  # Reduced padding
    plt.close()


def evans_index(brain_mask, ventricle_masks, skull_mask, ct_img, output_file, preview_file):

    resolution = 1.0  # 8s; good enough
    # resolution = 0.7  # 25s    

    st = time.time()

    brain_atlas = pkg_resources.resource_filename('totalsegmentator', 'resources/ct_brain_atlas_1mm.nii.gz')
    atlas_img = nib.load(brain_atlas)
    
    # Loading
    brain_img = nib.load(brain_mask)
    skull_img = nib.load(skull_mask)
    ventricle_img = nib.load(ventricle_masks)
    ct_img = nib.load(ct_img)
    
    # Canonical
    brain_img = nib.as_closest_canonical(brain_img)
    skull_img = nib.as_closest_canonical(skull_img)
    ventricle_img = nib.as_closest_canonical(ventricle_img)
    ct_img = nib.as_closest_canonical(ct_img)

    # Resampling
    brain_img = change_spacing(brain_img, resolution, dtype=np.uint8, order=0)  # order=1 leads to holes in seg
    skull_img = change_spacing(skull_img, resolution, dtype=np.uint8, order=0)
    ventricle_img = change_spacing(ventricle_img, resolution, dtype=np.uint8, order=0)
    ct_img = change_spacing(ct_img, resolution, dtype=np.uint8, order=1)
    
    # Editing
    ct_img = extract_brain(brain_img, ct_img)
    ventricle_all_img = nib.Nifti1Image((ventricle_img.get_fdata() > 0).astype(np.uint8), ventricle_img.affine)

    # Registration
    # transform = calc_transform(brain_img, atlas_img, transform_type="Rigid", resample=None, verbose=False)  # not working robustly
    # transform = calc_transform(ventricle_all_img, atlas_img, transform_type="Rigid", resample=None, verbose=False)  # not working robustly
    # transform = calc_transform(skull_img, atlas_img, transform_type="Rigid", resample=None, verbose=False)  # not working robustly
    transform = calc_transform(ct_img, atlas_img, transform_type="Rigid", resample=None, verbose=False)

    brain_img = apply_transform(brain_img, atlas_img, transform, resample=None, dtype=np.uint8, order=0)
    skull_img = apply_transform(skull_img, atlas_img, transform, resample=None, dtype=np.uint8, order=0, interp="genericLabel")
    ventricle_img = apply_transform(ventricle_img, atlas_img, transform, resample=None, dtype=np.uint8, order=0, interp="genericLabel")

    # Get data
    brain_data = brain_img.get_fdata()
    skull_data = skull_img.get_fdata()
    ventricle_data = ventricle_img.get_fdata()
    ventricle_all = (ventricle_data > 0).astype(np.uint8)

    # select only frontal horn
    ventricle_data = np.where((ventricle_data == 1) | (ventricle_data == 6), 1, 0)

    # postprocessing for robustness
    brain_data = remove_small_blobs(brain_data, [200, 1e10])  # fast
    ventricle_data = remove_small_blobs(ventricle_data, [10, 1e10])  # fast

    # Get diameters
    max_dia_vent, (start_vent, end_vent) = max_diameter_x(ventricle_data)
    brain_data_slice = brain_data[:,:,start_vent[2]:end_vent[2]+1]  # select same slice as ventricles
    max_dia_brain, (start_brain, end_brain) = max_diameter_x(brain_data_slice)
    
    # Make 2mm wider to correct for skull-brain gap
    addon_mm = 1 / resolution
    max_dia_brain += 2 * addon_mm
    start_brain[0] = start_brain[0] - addon_mm
    end_brain[0] = end_brain[0] + addon_mm
    
    # Calc index
    evans_index = max_dia_vent / max_dia_brain
    
    # Plot
    skull_data[ventricle_all > 0] = 1  # combine skull and ventricle masks
    plot_slice_with_diameters(skull_data, start_brain, end_brain, 
                              start_vent, end_vent, evans_index, preview_file)
    
    # save to json
    evans_index = {"evans_index": round(evans_index, 3)}
    json.dump(evans_index, open(output_file, "w"), indent=4)

    print(f"took: {time.time()-st:.2f}s")


"""
# pip install antspyx
cd ~/Downloads/evans_index/31103170_rot_large
python ~/dev/TotalSegmentator/totalsegmentator/bin/totalseg_evans_index.py -b roi/brain.nii.gz -s roi/skull.nii.gz -v predicted/T552_ventricle_parts.nii.gz -c NR_01_SCHAEDEL/Sch_del_Syngo_1_0_Hr38_3_s003.nii -a /mnt/nvme/data/ventricle_vol_nathan/atlas/atlas_ct_07.nii.gz -o evans_index_TEST.json -p evans_index_TEST.png
"""
if __name__ == "__main__":
    """
    For more documentation see resources/evans_index.md
    """
    parser = argparse.ArgumentParser(description="Calc evans index.")
    parser.add_argument("-b", "--brain_mask", type=lambda p: Path(p).resolve(), required=True,
                        help="Path to brain mask file.")
    parser.add_argument("-v", "--ventricle_masks", type=lambda p: Path(p).resolve(), required=True,
                        help="Path to ventricle part masks.")
    parser.add_argument("-s", "--skull_mask", type=lambda p: Path(p).resolve(), required=True,
                        help="Path to skull masks.")
    parser.add_argument("-c", "--ct_img", type=lambda p: Path(p).resolve(), required=True,
                        help="Path to ct_img.")
    parser.add_argument("-o", "--output_file", type=lambda p: Path(p).resolve(), required=True,
                        help="json output file.")
    parser.add_argument("-p", "--preview_file", type=lambda p: Path(p).resolve(), required=True,
                        help="Preview file of evans index to check if correct.")
    args = parser.parse_args()

    try:
        import ants
    except ImportError:
        print("Error: antspyx package not installed. Please install it using:")
        print("pip install antspyx")
        sys.exit(1)
    
    evans_index(
        brain_mask=args.brain_mask,
        ventricle_masks=args.ventricle_masks,
        skull_mask=args.skull_mask,
        ct_img=args.ct_img,
        output_file=args.output_file,
        preview_file=args.preview_file
    )