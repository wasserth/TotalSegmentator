import os
import sys
import random
import string
import time
import shutil
import subprocess
from pathlib import Path
from os.path import join
import numpy as np
import nibabel as nib
from functools import partial
from p_tqdm import p_map
from multiprocessing import Pool
import tempfile

from totalsegmentator.libs import nostdout

with nostdout():
    from nnunet.inference.predict import predict_from_folder
    from nnunet.paths import default_plans_identifier, network_training_output_dir, default_trainer

from totalsegmentator.map_to_binary import class_map, class_map_5_parts, map_taskid_to_partname
from totalsegmentator.alignment import as_closest_canonical_nifti, undo_canonical_nifti
from totalsegmentator.alignment import as_closest_canonical, undo_canonical
from totalsegmentator.resampling import change_spacing
from totalsegmentator.preview import generate_preview
from totalsegmentator.libs import combine_masks, compress_nifti, check_if_shape_and_affine_identical
from totalsegmentator.cropping import crop_to_mask_nifti, undo_crop_nifti
from totalsegmentator.cropping import crop_to_mask, undo_crop
from totalsegmentator.postprocessing import remove_outside_of_mask


def _get_full_task_name(task_id: int, src: str="raw"):
    if src == "raw":
        base = Path(os.environ['nnUNet_raw_data_base']) / "nnUNet_raw_data"
    elif src == "preprocessed":
        base = Path(os.environ['nnUNet_preprocessed'])
    elif src == "results":
        base = Path(os.environ['RESULTS_FOLDER']) / "nnUNet" / "3d_fullres"
    dirs = [str(dir).split("/")[-1] for dir in base.glob("*")]
    for dir in dirs:
        if f"Task{task_id:03d}" in dir:
            return dir

    # If not found in 3d_fullres, search in 2d
    if src == "results":
        base = Path(os.environ['RESULTS_FOLDER']) / "nnUNet" / "2d"
        dirs = [str(dir).split("/")[-1] for dir in base.glob("*")]
        for dir in dirs:
            if f"Task{task_id:03d}" in dir:
                return dir

    raise ValueError(f"task_id {task_id} not found")


def contains_empty_img(imgs):
    """
    imgs: List of image pathes
    """
    is_empty = True
    for img in imgs:
        this_is_empty = len(np.unique(nib.load(img).get_fdata())) == 1
        is_empty = is_empty and this_is_empty
    return is_empty


def nnUNet_predict(dir_in, dir_out, task_id, model="3d_fullres", folds=None,
                   trainer="nnUNetTrainerV2", tta=False):
    """
    Identical to bash function nnUNet_predict

    folds:  folds to use for prediction. Default is None which means that folds will be detected 
            automatically in the model output folder.
            for all folds: None
            for only fold 0: [0]
    """
    save_npz = False
    num_threads_preprocessing = 6
    num_threads_nifti_save = 2
    # num_threads_preprocessing = 1
    # num_threads_nifti_save = 1
    lowres_segmentations = None
    part_id = 0
    num_parts = 1
    disable_tta = not tta
    overwrite_existing = False
    mode = "normal" if model == "2d" else "fastest"
    all_in_gpu = None
    step_size = 0.5
    chk = "model_final_checkpoint"
    disable_mixed_precision = False

    task_id = int(task_id)
    task_name = _get_full_task_name(task_id, src="results")

    # trainer_class_name = default_trainer
    # trainer = trainer_class_name
    plans_identifier = default_plans_identifier

    model_folder_name = join(network_training_output_dir, model, task_name, trainer + "__" + plans_identifier)
    print("using model stored in ", model_folder_name)

    predict_from_folder(model_folder_name, dir_in, dir_out, folds, save_npz, num_threads_preprocessing,
                        num_threads_nifti_save, lowres_segmentations, part_id, num_parts, not disable_tta,
                        overwrite_existing=overwrite_existing, mode=mode, overwrite_all_in_gpu=all_in_gpu,
                        mixed_precision=not disable_mixed_precision,
                        step_size=step_size, checkpoint_name=chk)


def save_segmentation_nifti(class_map_item, tmp_dir=None, file_out=None, nora_tag=None):
    k, v = class_map_item
    # Have to load img inside of each thread. If passing it as argument a lot slower.
    img = nib.load(tmp_dir / "s01.nii.gz")
    img_data = img.get_fdata()
    binary_img = img_data == k
    output_path = str(file_out / f"{v}.nii.gz")
    nib.save(nib.Nifti1Image(binary_img.astype(np.uint8), img.affine, img.header), output_path)
    if nora_tag != "None":
        subprocess.call(f"/opt/nora/src/node/nora -p {nora_tag} --add {output_path} --addtag mask", shell=True)


def nnUNet_predict_image(file_in, file_out, task_id, model="3d_fullres", folds=None,
                         trainer="nnUNetTrainerV2", tta=False, multilabel_image=True, 
                         resample=None, crop=None, crop_path=None, task_name="total", nora_tag=None, preview=False, 
                         nr_threads_resampling=1, nr_threads_saving=6, quiet=False, 
                         verbose=False, test=0):
    """
    resample: None or float  (target spacing for all dimensions)
    """
    file_in, file_out = Path(file_in), Path(file_out)
    multimodel = type(task_id) is list

    # for debugging
    # tmp_dir = file_in.parent / ("nnunet_tmp_" + ''.join(random.Random().choices(string.ascii_uppercase + string.digits, k=8)))
    # (tmp_dir).mkdir(exist_ok=True)
    # with tmp_dir as tmp_folder:
    with tempfile.TemporaryDirectory(prefix="nnunet_tmp_") as tmp_folder:
        tmp_dir = Path(tmp_folder)
        if verbose: print(f"tmp_dir: {tmp_dir}")

        img_in_orig = nib.load(file_in)
        img_in = nib.Nifti1Image(img_in_orig.get_fdata(), img_in_orig.affine)  # copy img_in_orig

        if crop is not None:
            if crop == "lung" or crop == "pelvis" or crop == "heart":
                combine_masks(crop_path, crop_path / f"{crop}.nii.gz", crop)
            crop_mask_img = nib.load(crop_path / f"{crop}.nii.gz")
            img_in, bbox = crop_to_mask(img_in, crop_mask_img, addon=[5, 5, 5], dtype=np.int32,
                                      verbose=verbose)
            if verbose:
                print(f"  cropping from {crop_mask_img.shape} to {img_in.shape}")

        img_in = as_closest_canonical(img_in)

        if resample is not None:
            if not quiet: print(f"Resampling...")
            st = time.time()
            img_in_shape = img_in.shape
            img_in_zooms = img_in.header.get_zooms()
            img_in_rsp = change_spacing(img_in, [resample, resample, resample],
                                        order=3, dtype=np.int32, nr_cpus=nr_threads_resampling)  # 4 cpus instead of 1 makes it a bit slower
            if verbose:
                print(f"  from shape {img_in.shape} to shape {img_in_rsp.shape}")
            if not quiet: print(f"  Resampled in {time.time() - st:.2f}s")
        else:
            img_in_rsp = img_in

        nib.save(img_in_rsp, tmp_dir / "s01_0000.nii.gz")

        # nr_voxels_thr = 512*512*900
        nr_voxels_thr = 256*256*900
        img_parts = ["s01"]
        ss = img_in_rsp.shape
        # If image to big then split into 3 parts along z axis. Also make sure that z-axis is at least 200px otherwise
        # splitting along it does not really make sense.
        if np.prod(ss) > nr_voxels_thr and ss[2] > 200 and multimodel:
            if not quiet: print(f"Splitting into subparts...")
            img_parts = ["s01", "s02", "s03"]
            third = img_in_rsp.shape[2] // 3
            margin = 20  # set margin with fixed values to avoid rounding problem if using percentage of third
            img_in_rsp_data = img_in_rsp.get_fdata()
            nib.save(nib.Nifti1Image(img_in_rsp_data[:, :, :third+margin], img_in_rsp.affine),
                    tmp_dir / "s01_0000.nii.gz")
            nib.save(nib.Nifti1Image(img_in_rsp_data[:, :, third+1-margin:third*2+margin], img_in_rsp.affine),
                    tmp_dir / "s02_0000.nii.gz")
            nib.save(nib.Nifti1Image(img_in_rsp_data[:, :, third*2+1-margin:], img_in_rsp.affine),
                    tmp_dir / "s03_0000.nii.gz")

        st = time.time()
        if multimodel:  # if running multiple models 
            if test == 0:
                class_map_inv = {v: k for k, v in class_map[task_name].items()}
                (tmp_dir / "parts").mkdir(exist_ok=True)
                seg_combined = {}
                # iterate over subparts of image
                for img_part in img_parts:
                    img_shape = nib.load(tmp_dir / f"{img_part}_0000.nii.gz").shape
                    seg_combined[img_part] = np.zeros(img_shape, dtype=np.uint8)
                # Run several tasks and combine results into one segmentation
                for idx, tid in enumerate(task_id):
                    print(f"Predicting part {idx} of 5 ...")
                    with nostdout(verbose):
                        nnUNet_predict(tmp_dir, tmp_dir, tid, model, folds, trainer, tta)
                    # iterate over models (different sets of classes)
                    for img_part in img_parts:
                        (tmp_dir / f"{img_part}.nii.gz").rename(tmp_dir / "parts" / f"{img_part}_{tid}.nii.gz")
                        seg = nib.load(tmp_dir / "parts" / f"{img_part}_{tid}.nii.gz").get_fdata()
                        for jdx, class_name in class_map_5_parts[map_taskid_to_partname[tid]].items():
                            seg_combined[img_part][seg == jdx] = class_map_inv[class_name]
                # iterate over subparts of image
                for img_part in img_parts:
                    nib.save(nib.Nifti1Image(seg_combined[img_part], img_in_rsp.affine), tmp_dir / f"{img_part}.nii.gz")
            elif test == 1:
                print("WARNING: Using reference seg instead of prediction for testing.")
                shutil.copy(Path("tests") / "reference_files" / "example_seg.nii.gz", tmp_dir / f"s01.nii.gz")
        else:
            if not quiet: print(f"Predicting...")
            if test == 0:
                with nostdout(verbose):
                    nnUNet_predict(tmp_dir, tmp_dir, task_id, model, folds, trainer, tta)
            elif test == 2:
                print("WARNING: Using reference seg instead of prediction for testing.")
                shutil.copy(Path("tests") / "reference_files" / "example_seg_fast.nii.gz", tmp_dir / f"s01.nii.gz")
            elif test == 3:
                print("WARNING: Using reference seg instead of prediction for testing.")
                shutil.copy(Path("tests") / "reference_files" / "example_seg_lung_vessels.nii.gz", tmp_dir / f"s01.nii.gz")
        if not quiet: print("  Predicted in {:.2f}s".format(time.time() - st))

        # Combine image subparts back to one image
        if np.prod(ss) > nr_voxels_thr and ss[2] > 200 and multimodel:
            combined_img = np.zeros(img_in_rsp.shape, dtype=np.uint8)
            combined_img[:,:,:third] = nib.load(tmp_dir / "s01.nii.gz").get_fdata()[:,:,:-margin]
            combined_img[:,:,third:third*2] = nib.load(tmp_dir / "s02.nii.gz").get_fdata()[:,:,margin-1:-margin]
            combined_img[:,:,third*2:] = nib.load(tmp_dir / "s03.nii.gz").get_fdata()[:,:,margin-1:]
            nib.save(nib.Nifti1Image(combined_img, img_in_rsp.affine), tmp_dir / "s01.nii.gz")

        st_a = time.time()

        img_pred = nib.load(tmp_dir / "s01.nii.gz")
        if preview:
            # Generate preview before upsampling so it is faster and still in canonical space 
            # for better orientation.
            if not quiet: print("Generating preview...")
            st = time.time()
            smoothing = 20
            generate_preview(img_in_rsp, file_out / f"preview_{task_name}.png", img_pred.get_fdata(), smoothing, task_name)
            if not quiet: print("  Generated in {:.2f}s".format(time.time() - st))

        if resample is not None:
            if not quiet: print("Resampling...")
            if verbose: print(f"  back to original shape: {img_in_shape}")    
            img_pred = change_spacing(img_pred, [resample, resample, resample], img_in_shape,
                                        order=0, dtype=np.uint8, nr_cpus=nr_threads_resampling)
            # Sometimes the output spacing does not completely match the input spacing but it is off
            # by a very small amount (reason might be the resampling: we resample back to original 
            # shape which might result in slightly different spacing). Here we just overwrite the 
            # output affine by the input affine to make it match exactly. It must match exactly 
            # because otherwise in undo_canonical_nifti it will result in differences also in offest.
            img_pred = nib.Nifti1Image(img_pred.get_fdata(), img_in.affine)

        img_pred = undo_canonical(img_pred, img_in_orig)

        if crop is not None:
            img_pred = undo_crop(img_pred, img_in_orig, bbox)

        check_if_shape_and_affine_identical(img_in_orig, img_pred)

        if not quiet: print("Saving segmentations...")
        st = time.time()
        img_data = img_pred.get_fdata()
        if multilabel_image:
            nib.save(img_pred.astype(np.uint8), file_out)
            if nora_tag != "None":
                subprocess.call(f"/opt/nora/src/node/nora -p {nora_tag} --add {file_out} --addtag atlas", shell=True)
        else:  # save each class as a separate binary image
            file_out.mkdir(exist_ok=True, parents=True)
            # Code for single threaded execution  (runtime:24s)
            if nr_threads_saving == 1:
                for k, v in class_map[task_name].items():
                    binary_img = img_data == k
                    output_path = str(file_out / f"{v}.nii.gz")
                    nib.save(nib.Nifti1Image(binary_img.astype(np.uint8), img_pred.affine, img_pred.header), output_path)
                    # nib.save(nib.Nifti1Image(binary_img.astype(np.uint8), img.affine, img.header), output_path)
                    if nora_tag != "None":
                        subprocess.call(f"/opt/nora/src/node/nora -p {nora_tag} --add {output_path} --addtag mask", shell=True)
            else:
                # Code for multithreaded execution
                #   Speed with different number of threads:
                #   1: 46s, 2: 24s, 6: 11s, 10: 8s, 14: 8s
                nib.save(img_pred, tmp_dir / "s01.nii.gz")
                _ = p_map(partial(save_segmentation_nifti, tmp_dir=tmp_dir, file_out=file_out, nora_tag=nora_tag),
                        class_map[task_name].items(), num_cpus=nr_threads_saving, disable=quiet)

                # Multihreaded saving with same functions as in nnUNet -> same speed as p_map
                # pool = Pool(nr_threads_saving)
                # results = []
                # for k, v in class_map[task_name].items():
                #     results.append(pool.starmap_async(save_segmentation_nifti, ((k, v, tmp_dir, file_out, nora_tag),) ))
                # _ = [i.get() for i in results]  # this actually starts the execution of the async functions
                # pool.close()
                # pool.join()
        if not quiet: print(f"  Saved in {time.time() - st:.2f}s")

        # Postprocessing
        if task_name == "lung_vessels":
            remove_outside_of_mask(file_out / "lung_vessels.nii.gz", file_out / "lung.nii.gz")

        print(f"  Combining took {time.time() - st_a:.2f}s")

    return img_data
