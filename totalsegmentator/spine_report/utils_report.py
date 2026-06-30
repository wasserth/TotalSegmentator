import time
import json
import subprocess
import tempfile
from pathlib import Path
import asyncio

import nibabel as nib

from totalsegmentator.spine_report.phase import get_phase



async def run_totalsegmentator_async(ct_img, file_out, args):
    tmp_dir = Path(tempfile.mkdtemp())
    ct_path = tmp_dir / "ct.nii.gz"
    file_out = tmp_dir / f"{file_out}.nii.gz"
    nib.save(ct_img, ct_path)
    # subprocess.call(f"TotalSegmentator -i {ct_path} -o {file_out} {args}", shell=True)
    command = f"TotalSegmentator -i {ct_path} -o {file_out} {args}"
    _ = await asyncio.to_thread(subprocess.call, command, shell=True)
    return nib.load(file_out)


def run_totalsegmentator(ct_img, file_out, args):
    tmp_dir = Path(tempfile.mkdtemp())
    ct_path = tmp_dir / "ct.nii.gz"
    file_out = tmp_dir / f"{file_out}.nii.gz"
    nib.save(ct_img, ct_path)
    subprocess.call(f"TotalSegmentator -i {ct_path} -o {file_out} {args}", shell=True)
    return nib.load(file_out)


async def run_models_abdomen_parallel(ct_path, tmp_dir, logger, host="local"):
    """
    
    host: local | modal
    """

    print("Running TotalSegmentator - ASYNC")

    ct_img = nib.load(ct_path)
    ct_img = nib.Nifti1Image(ct_img.get_fdata(), ct_img.affine)  # copy image to be able to pass it as parameter to modal
    st = time.time()

    if host == "local":
        img = await run_totalsegmentator_async(
            ct_img,
            "totalseg_vertebrae_pp_refined",
            "-ta vertebrae_pp_refined -ml -ns 1",
        )
    elif host == "modal":
        import modal 
        run_ts = modal.Function.from_name("totalsegmentator", "run_totalsegmentator")
        img = await run_ts.remote.aio(
            ct_img,
            {"task": "vertebrae_pp_refined", "ml": True, "nr_thr_saving": 1},
        )

    print(f"TotalSegmentator ASYNC done (took: {time.time()-st:.2f}s)")

    nib.save(img, tmp_dir / "totalseg_vertebrae_pp_refined.nii.gz")
    

def run_models_abdomen(ct_path, tmp_dir, logger):
    """
    Call totalsegmentator from shell to avoid interference between multiple calls.

    gc.collect() + torch.cuda.empty_cache() do not seem to solve the issue for python api
    """
    st = time.time()

    yield "Running TotalSegmentator - vertebrae_pp_refined"
    if (tmp_dir / 'totalseg_vertebrae_pp_refined.nii.gz').exists():
        logger.info("  Skipping TotalSeg vertebrae_pp_refined (already exists)")
    else:
        subprocess.call(f"TotalSegmentator -i {ct_path} -o {tmp_dir / 'totalseg_vertebrae_pp_refined.nii.gz'} -ta vertebrae_pp_refined -ml -ns 1 -ro 1", shell=True)

    print(f"TotalSegmentator done (took: {time.time()-st:.2f}s)")
    yield "TotalSegmentator done"


def get_contrast_phase(ct_path, tmp_dir, logger, host="local", original_nifti_path=None):
    """
    Call totalseg_get_phase from shell to get the contrast phase.
    """
    st = time.time()
    if (tmp_dir / "contrast_phase.json").exists():
        logger.info("  Skipping totalseg_get_phase (already exists)")
        d = json.load(open(tmp_dir / "contrast_phase.json"))
    else:
        if host == "local":
            if original_nifti_path is None:
                print("Predict contrast phase")
                subprocess.call(f"totalseg_get_phase -i {ct_path} -o {tmp_dir / 'contrast_phase.json'} -q", shell=True)
            else:
                print("Use dicom header from Nora project")
                # Here we have to phase the original Nifti file path, otherwise we can not find the corresponding DICOM file in Nora
                phase, pi_time = get_phase(original_nifti_path, "header", tmp_dir)
                json.dump({"phase": phase, "pi_time": pi_time}, open(tmp_dir / "contrast_phase.json", "w"), indent=4)

            d = json.load(open(tmp_dir / "contrast_phase.json"))
        else:
            import modal
            ct_img = nib.load(ct_path)  # using mmap=False does not work with modal; have to recreate nifti image
            ct_img = nib.Nifti1Image(ct_img.get_fdata(), ct_img.affine)  # copy image to be able to pass it as parameter to modal
            run_get_phase = modal.Function.from_name("totalsegmentator", "run_get_phase")
            d = run_get_phase.remote(ct_img)
    print(f"Contrast phase done (took: {time.time()-st:.2f}s)")
    return d["phase"]


# def replace_none_with_string(d):
#     for k, v in d.items():
#         if v is None:
#             d[k] = "None"
#         elif isinstance(v, dict):
#             replace_none_with_string(v)
#     return d
