import os
import sys
import random
import string
import time
import platform
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
import torch

from totalsegmentator.libs import nostdout

# todo important: change
# with nostdout():
# from nnunetv2.inference.predict_from_raw_data import predict_from_raw_data
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.file_path_utilities import get_output_folder

from totalsegmentator.map_to_binary import class_map, class_map_5_parts, map_taskid_to_partname
from totalsegmentator.alignment import as_closest_canonical_nifti, undo_canonical_nifti
from totalsegmentator.alignment import as_closest_canonical, undo_canonical
from totalsegmentator.resampling import change_spacing
from totalsegmentator.libs import combine_masks, compress_nifti, check_if_shape_and_affine_identical, reorder_multilabel_like_v1
from totalsegmentator.dicom_io import dcm_to_nifti, save_mask_as_rtstruct
from totalsegmentator.cropping import crop_to_mask_nifti, undo_crop_nifti
from totalsegmentator.cropping import crop_to_mask, undo_crop
from totalsegmentator.postprocessing import remove_outside_of_mask, extract_skin, remove_auxiliary_labels
from totalsegmentator.postprocessing import keep_largest_blob_multilabel, remove_small_blobs_multilabel
from totalsegmentator.nifti_ext_header import save_multilabel_nifti
from totalsegmentator.statistics import get_basic_statistics


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

    # If not found in 3d_fullres, search in 3d_lowres
    if src == "results":
        base = Path(os.environ['RESULTS_FOLDER']) / "nnUNet" / "3d_lowres"
        dirs = [str(dir).split("/")[-1] for dir in base.glob("*")]
        for dir in dirs:
            if f"Task{task_id:03d}" in dir:
                return dir

    # If not found in 3d_lowres, search in 2d
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
                   trainer="nnUNetTrainerV2", tta=False,
                   num_threads_preprocessing=6, num_threads_nifti_save=2):
    """
    Identical to bash function nnUNet_predict

    folds:  folds to use for prediction. Default is None which means that folds will be detected 
            automatically in the model output folder.
            for all folds: None
            for only fold 0: [0]
    """
    with nostdout():
        from nnunet.inference.predict import predict_from_folder
        from nnunet.paths import default_plans_identifier, network_training_output_dir, default_trainer

    save_npz = False
    # num_threads_preprocessing = 6
    # num_threads_nifti_save = 2
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


def nnUNetv2_predict(dir_in, dir_out, task_id, model="3d_fullres", folds=None,
                     trainer="nnUNetTrainer", tta=False,
                     num_threads_preprocessing=3, num_threads_nifti_save=2,
                     plans="nnUNetPlans", device="cuda", quiet=False):
    """
    Identical to bash function nnUNetv2_predict

    folds:  folds to use for prediction. Default is None which means that folds will be detected 
            automatically in the model output folder.
            for all folds: None
            for only fold 0: [0]
    """
    dir_in = str(dir_in)
    dir_out = str(dir_out)

    model_folder = get_output_folder(task_id, trainer, plans, model)

    assert device in ['cpu', 'cuda',
                           'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {device}.'
    if device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        # torch.set_num_interop_threads(1)  # throws error if setting the second time
        device = torch.device('cuda')
    else:
        device = torch.device('mps')
    step_size = 0.5
    disable_tta = not tta
    verbose = False
    save_probabilities = False
    continue_prediction = False
    chk = "checkpoint_final.pth"
    npp = num_threads_preprocessing
    nps = num_threads_nifti_save
    prev_stage_predictions = None
    num_parts = 1
    part_id = 0
    allow_tqdm = not quiet

    # predict_from_raw_data(dir_in,
    #                       dir_out,
    #                       model_folder,
    #                       folds,
    #                       step_size,
    #                       use_gaussian=True,
    #                       use_mirroring=not disable_tta,
    #                       perform_everything_on_gpu=True,
    #                       verbose=verbose,
    #                       save_probabilities=save_probabilities,
    #                       overwrite=not continue_prediction,
    #                       checkpoint_name=chk,
    #                       num_processes_preprocessing=npp,
    #                       num_processes_segmentation_export=nps,
    #                       folder_with_segs_from_prev_stage=prev_stage_predictions,
    #                       num_parts=num_parts,
    #                       part_id=part_id,
    #                       device=device)

    
    predictor = nnUNetPredictor(
        tile_step_size=step_size,
        use_gaussian=True,
        use_mirroring=not disable_tta,
        perform_everything_on_gpu=True,
        device=device,
        verbose=verbose,
        verbose_preprocessing=verbose,
        allow_tqdm=allow_tqdm
    )
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=folds,
        checkpoint_name=chk,
    )
    predictor.predict_from_files(dir_in, dir_out,
                                 save_probabilities=save_probabilities, overwrite=not continue_prediction,
                                 num_processes_preprocessing=npp, num_processes_segmentation_export=nps,
                                 folder_with_segs_from_prev_stage=prev_stage_predictions, 
                                 num_parts=num_parts, part_id=part_id)


def save_segmentation_nifti(class_map_item, tmp_dir=None, file_out=None, nora_tag=None, header=None, task_name=None, quiet=None):
    k, v = class_map_item
    # Have to load img inside of each thread. If passing it as argument a lot slower.
    if not task_name.startswith("total") and not quiet:
        print(f"Creating {v}.nii.gz")
    img = nib.load(tmp_dir / "s01.nii.gz")
    img_data = img.get_fdata()
    binary_img = img_data == k
    output_path = str(file_out / f"{v}.nii.gz")
    nib.save(nib.Nifti1Image(binary_img.astype(np.uint8), img.affine, header), output_path)
    if nora_tag != "None":
        subprocess.call(f"/opt/nora/src/node/nora -p {nora_tag} --add {output_path} --addtag mask", shell=True)


def nnUNet_predict_image(file_in, file_out, task_id, model="3d_fullres", folds=None,
                         trainer="nnUNetTrainerV2", tta=False, multilabel_image=True, 
                         resample=None, crop=None, crop_path=None, task_name="total", nora_tag="None", preview=False, 
                         save_binary=False, nr_threads_resampling=1, nr_threads_saving=6, force_split=False,
                         crop_addon=[3,3,3], roi_subset=None, output_type="nifti", 
                         statistics=False, quiet=False, verbose=False, test=0, skip_saving=False,
                         device="cuda", exclude_masks_at_border=True, no_derived_masks=False,
                         v1_order=False):
    """
    crop: string or a nibabel image
    resample: None or float  (target spacing for all dimensions)
    """
    file_in = Path(file_in)
    if file_out is not None:
        file_out = Path(file_out)
    if not file_in.exists():
        sys.exit("ERROR: The input file or directory does not exist.")
    multimodel = type(task_id) is list

    img_type = "nifti" if str(file_in).endswith(".nii") or str(file_in).endswith(".nii.gz") else "dicom"

    if img_type == "nifti" and output_type == "dicom":
        raise ValueError("To use output type dicom you also have to use a Dicom image as input.")

    # for debugging
    # tmp_dir = file_in.parent / ("nnunet_tmp_" + ''.join(random.Random().choices(string.ascii_uppercase + string.digits, k=8)))
    # (tmp_dir).mkdir(exist_ok=True)
    # with tmp_dir as tmp_folder:
    with tempfile.TemporaryDirectory(prefix="nnunet_tmp_") as tmp_folder:
        tmp_dir = Path(tmp_folder)
        if verbose: print(f"tmp_dir: {tmp_dir}")

        if img_type == "dicom":
            if not quiet: print("Converting dicom to nifti...")
            (tmp_dir / "dcm").mkdir()  # make subdir otherwise this file would be included by nnUNet_predict
            dcm_to_nifti(file_in, tmp_dir / "dcm" / "converted_dcm.nii.gz", verbose=verbose)
            file_in_dcm = file_in
            file_in = tmp_dir / "dcm" / "converted_dcm.nii.gz"

            # Workaround to be able to access file_in on windows (see issue #106)
            # if platform.system() == "Windows":
            #     file_in = file_in.NamedTemporaryFile(delete = False)
            #     file_in.close() 

            # if not multilabel_image:
            #     shutil.copy(file_in, file_out / "input_file.nii.gz")
            if not quiet: print(f"  found image with shape {nib.load(file_in).shape}")

        img_in_orig = nib.load(file_in)
        if len(img_in_orig.shape) == 2:
            raise ValueError("TotalSegmentator does not work for 2D images. Use a 3D image.")
        if len(img_in_orig.shape) > 3:
            print(f"WARNING: Input image has {len(img_in_orig.shape)} dimensions. Only using first three dimensions.")
            img_in_orig = nib.Nifti1Image(img_in_orig.get_fdata()[:,:,:,0], img_in_orig.affine)
        
        # takes ~0.9s for medium image
        img_in = nib.Nifti1Image(img_in_orig.get_fdata(), img_in_orig.affine)  # copy img_in_orig

        if crop is not None:
            if type(crop) is str:
                if crop == "lung" or crop == "pelvis":
                    crop_mask_img = combine_masks(crop_path, crop)
                else:
                    crop_mask_img = nib.load(crop_path / f"{crop}.nii.gz")
            else:
                crop_mask_img = crop
            img_in, bbox = crop_to_mask(img_in, crop_mask_img, addon=crop_addon, dtype=np.int32,
                                      verbose=verbose)
            if not quiet:
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
        do_triple_split = np.prod(ss) > nr_voxels_thr and ss[2] > 200 and multimodel
        if force_split:
            do_triple_split = True
        if do_triple_split:
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

            # only compute model parts containing the roi subset
            if roi_subset is not None:
                part_names = []
                new_task_id = []
                for part_name, part_map in class_map_5_parts.items():
                    if any(organ in roi_subset for organ in part_map.values()):
                        # get taskid associated to model part_name
                        map_partname_to_taskid = {v:k for k,v in map_taskid_to_partname.items()}
                        new_task_id.append(map_partname_to_taskid[part_name])
                        part_names.append(part_name)
                task_id = new_task_id
                if verbose:
                    print(f"Computing parts: {part_names} based on the provided roi_subset")

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
                    print(f"Predicting part {idx+1} of {len(task_id)} ...")
                    with nostdout(verbose):
                        # nnUNet_predict(tmp_dir, tmp_dir, tid, model, folds, trainer, tta,
                        #                nr_threads_resampling, nr_threads_saving)
                        nnUNetv2_predict(tmp_dir, tmp_dir, tid, model, folds, trainer, tta,
                                         nr_threads_resampling, nr_threads_saving, device=device, quiet=quiet)
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
                    # nnUNet_predict(tmp_dir, tmp_dir, task_id, model, folds, trainer, tta,
                    #                nr_threads_resampling, nr_threads_saving)
                    nnUNetv2_predict(tmp_dir, tmp_dir, task_id, model, folds, trainer, tta,
                                     nr_threads_resampling, nr_threads_saving, device=device, quiet=quiet)
            # elif test == 2:
            #     print("WARNING: Using reference seg instead of prediction for testing.")
            #     shutil.copy(Path("tests") / "reference_files" / "example_seg_fast.nii.gz", tmp_dir / f"s01.nii.gz")
            elif test == 3:
                print("WARNING: Using reference seg instead of prediction for testing.")
                shutil.copy(Path("tests") / "reference_files" / "example_seg_lung_vessels.nii.gz", tmp_dir / f"s01.nii.gz")
        if not quiet: print("  Predicted in {:.2f}s".format(time.time() - st))

        # Combine image subparts back to one image
        if do_triple_split:
            combined_img = np.zeros(img_in_rsp.shape, dtype=np.uint8)
            combined_img[:,:,:third] = nib.load(tmp_dir / "s01.nii.gz").get_fdata()[:,:,:-margin]
            combined_img[:,:,third:third*2] = nib.load(tmp_dir / "s02.nii.gz").get_fdata()[:,:,margin-1:-margin]
            combined_img[:,:,third*2:] = nib.load(tmp_dir / "s03.nii.gz").get_fdata()[:,:,margin-1:]
            nib.save(nib.Nifti1Image(combined_img, img_in_rsp.affine), tmp_dir / "s01.nii.gz")

        img_pred = nib.load(tmp_dir / "s01.nii.gz")

        # Currently only relevant for T304 (appendicular bones)
        img_pred = remove_auxiliary_labels(img_pred, task_name)

        # Postprocessing multilabel (run here on lower resolution)
        if task_name == "body":
            img_pred_pp = keep_largest_blob_multilabel(img_pred.get_fdata().astype(np.uint8),
                                                       class_map[task_name], ["body_trunc"], debug=False)
            img_pred = nib.Nifti1Image(img_pred_pp, img_pred.affine)

        if task_name == "body":
            vox_vol = np.prod(img_pred.header.get_zooms())
            size_thr_mm3 = 50000 / vox_vol
            img_pred_pp = remove_small_blobs_multilabel(img_pred.get_fdata().astype(np.uint8),
                                                        class_map[task_name], ["body_extremities"],
                                                        interval=[size_thr_mm3, 1e10], debug=False)
            img_pred = nib.Nifti1Image(img_pred_pp, img_pred.affine)

        if preview:
            from totalsegmentator.preview import generate_preview
            # Generate preview before upsampling so it is faster and still in canonical space 
            # for better orientation.
            if not quiet: print("Generating preview...")
            st = time.time()
            smoothing = 20
            preview_dir = file_out.parent if multilabel_image else file_out
            generate_preview(img_in_rsp, preview_dir / f"preview_{task_name}.png", img_pred.get_fdata(), smoothing, task_name)
            if not quiet: print("  Generated in {:.2f}s".format(time.time() - st))

        # Statistics calculated on the 3mm downsampled image are very similar to statistics
        # calculated on the original image. Volume often completely identical. For intensity
        # some more change but still minor.
        #
        # Speed: 
        # stats on 1.5mm: 37s
        # stats on 3.0mm: 4s    -> great improvement
        if statistics:
            if not quiet: print("Calculating statistics fast...")
            st = time.time()
            stats_dir = file_out.parent if multilabel_image else file_out
            stats_dir.mkdir(exist_ok=True)
            get_basic_statistics(img_pred.get_fdata(), img_in_rsp, stats_dir / "statistics.json", quiet, task_name,
                                 exclude_masks_at_border)
            if not quiet: print(f"  calculated in {time.time()-st:.2f}s")

        if resample is not None:
            if not quiet: print("Resampling...")
            if verbose: print(f"  back to original shape: {img_in_shape}")    
            # Use force_affine otherwise output affine sometimes slightly off (which then is even increased
            # by undo_canonical)
            img_pred = change_spacing(img_pred, [resample, resample, resample], img_in_shape,
                                        order=0, dtype=np.uint8, nr_cpus=nr_threads_resampling, 
                                        force_affine=img_in.affine)

        if verbose: print("Undoing canonical...")
        img_pred = undo_canonical(img_pred, img_in_orig)

        if crop is not None:
            if verbose: print("Undoing cropping...")
            img_pred = undo_crop(img_pred, img_in_orig, bbox)

        check_if_shape_and_affine_identical(img_in_orig, img_pred)

        img_data = img_pred.get_fdata().astype(np.uint8)
        if save_binary:
            img_data = (img_data > 0).astype(np.uint8)

        if file_out is not None and skip_saving is False:
            if not quiet: print("Saving segmentations...")

            # Select subset of classes if required
            selected_classes = class_map[task_name]
            if roi_subset is not None:
                selected_classes = {k:v for k, v in selected_classes.items() if v in roi_subset}

            if output_type == "dicom":
                file_out.mkdir(exist_ok=True, parents=True)
                save_mask_as_rtstruct(img_data, selected_classes, file_in_dcm, file_out / "segmentations.dcm")
            else:
                # Copy header to make output header exactly the same as input. But change dtype otherwise it will be 
                # float or int and therefore the masks will need a lot more space.
                # (infos on header: https://nipy.org/nibabel/nifti_images.html)
                new_header = img_in_orig.header.copy()
                new_header.set_data_dtype(np.uint8)

                st = time.time()
                if multilabel_image:
                    file_out.parent.mkdir(exist_ok=True, parents=True)
                else:
                    file_out.mkdir(exist_ok=True, parents=True)
                if multilabel_image:
                    if v1_order and task_name == "total":
                        img_data = reorder_multilabel_like_v1(img_data, class_map["total"], class_map["total_v1"])
                        label_map = class_map["total_v1"]
                    else:
                        label_map = class_map[task_name]
                    # Keep only voxel values corresponding to the roi_subset
                    if roi_subset is not None:
                        label_map = {k: v for k, v in label_map.items() if v in roi_subset}
                        img_data *= np.isin(img_data, list(label_map.keys()))
                    img_out = nib.Nifti1Image(img_data, img_pred.affine, new_header)
                    save_multilabel_nifti(img_out, file_out, label_map)
                    if nora_tag != "None":
                        subprocess.call(f"/opt/nora/src/node/nora -p {nora_tag} --add {file_out} --addtag atlas", shell=True)
                else:  # save each class as a separate binary image
                    file_out.mkdir(exist_ok=True, parents=True)

                    if np.prod(img_data.shape) > 512*512*1000:
                        print(f"Shape of output image is very big. Setting nr_threads_saving=1 to save memory.")
                        nr_threads_saving = 1

                    # Code for single threaded execution  (runtime:24s)
                    if nr_threads_saving == 1:
                        for k, v in selected_classes.items():
                            binary_img = img_data == k
                            output_path = str(file_out / f"{v}.nii.gz")
                            nib.save(nib.Nifti1Image(binary_img.astype(np.uint8), img_pred.affine, new_header), output_path)
                            if nora_tag != "None":
                                subprocess.call(f"/opt/nora/src/node/nora -p {nora_tag} --add {output_path} --addtag mask", shell=True)
                    else:
                        # Code for multithreaded execution
                        #   Speed with different number of threads:
                        #   1: 46s, 2: 24s, 6: 11s, 10: 8s, 14: 8s
                        nib.save(img_pred, tmp_dir / "s01.nii.gz")
                        _ = p_map(partial(save_segmentation_nifti, tmp_dir=tmp_dir, file_out=file_out, nora_tag=nora_tag, header=new_header, task_name=task_name, quiet=quiet),
                                selected_classes.items(), num_cpus=nr_threads_saving, disable=quiet)

                        # Multihreaded saving with same functions as in nnUNet -> same speed as p_map
                        # pool = Pool(nr_threads_saving)
                        # results = []
                        # for k, v in selected_classes.items():
                        #     results.append(pool.starmap_async(save_segmentation_nifti, ((k, v, tmp_dir, file_out, nora_tag),) ))
                        # _ = [i.get() for i in results]  # this actually starts the execution of the async functions
                        # pool.close()
                        # pool.join()
            if not quiet: print(f"  Saved in {time.time() - st:.2f}s")

            # Postprocessing single files
            #    (these not directly transferable to multilabel)

            # Lung mask does not exist since I use 6mm model. Would have to save lung mask from 6mm seg.
            # if task_name == "lung_vessels":
            #     remove_outside_of_mask(file_out / "lung_vessels.nii.gz", file_out / "lung.nii.gz")

            # if task_name == "heartchambers_test":
            #     remove_outside_of_mask(file_out / "heart_myocardium.nii.gz", file_out / "heart.nii.gz", addon=5)
            #     remove_outside_of_mask(file_out / "heart_atrium_left.nii.gz", file_out / "heart.nii.gz", addon=5)
            #     remove_outside_of_mask(file_out / "heart_ventricle_left.nii.gz", file_out / "heart.nii.gz", addon=5)
            #     remove_outside_of_mask(file_out / "heart_atrium_right.nii.gz", file_out / "heart.nii.gz", addon=5)
            #     remove_outside_of_mask(file_out / "heart_ventricle_right.nii.gz", file_out / "heart.nii.gz", addon=5)
            #     remove_outside_of_mask(file_out / "aorta.nii.gz", file_out / "heart.nii.gz", addon=5)
            #     remove_outside_of_mask(file_out / "pulmonary_artery.nii.gz", file_out / "heart.nii.gz", addon=5)

            if task_name == "body" and not multilabel_image and not no_derived_masks:
                if not quiet: print("Creating body.nii.gz")
                body_img = combine_masks(file_out, "body")
                nib.save(body_img, file_out / "body.nii.gz")
                if not quiet: print("Creating skin.nii.gz")
                skin = extract_skin(img_in_orig, nib.load(file_out / "body.nii.gz"))
                nib.save(skin, file_out / "skin.nii.gz")

    return nib.Nifti1Image(img_data, img_pred.affine), img_in_orig
