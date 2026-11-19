"""
TotalSegmentator's inference pipeline. Every task runs through it, every model on nnU-Net's streaming predictor.

    task = { models: [m1 .. mN],  roi: { task, classes, margin } or none,  label rules }

    segment(task, image):
        box  = bbox(segment(task.roi.task, image), classes) + margin      # only if the task has a roi
        maps = [ stream(m, image[box]) for m in task.models ]             # one nnU-Net streaming pass per model
        seg  = merge(maps) -> label rules -> paste into the full grid

An image is read once, by nnU-Net's reader (canonical orientation, float32); segment() works on that array, and the
label map goes back to the voxel grid of the input at the end.

nnU-Net (nnunetv2.inference.predict_streaming) does everything between an image and one model's label map on the grid
of that image: resampling to the model spacing, sliding window, exact trilinear resampling of the logits back, argmax -
in constant memory, so there is nothing to split, chunk or work around here. This module adds what is
TotalSegmentator's: the tasks (map_tasks_config.TASK_CONFIGS) and their weights, the region a model runs on, merging the
models of a multi-model task, label rules, input types (NIfTI path, in-memory Nifti1Image, DICOM), output formats,
statistics.
"""
import difflib
import itertools
import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from contextlib import ExitStack, contextmanager
from functools import partial
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from collections.abc import Sequence

import nibabel as nib
import numpy as np
import torch
from nibabel.nifti1 import Nifti1Image

from totalsegmentator.config import has_valid_license_offline, set_license_number, setup_nnunet, setup_totalseg
from totalsegmentator.libs import download_pretrained_weights, reorder_multilabel_like_v1, save_binary_niftis
from totalsegmentator.map_tasks_config import DEFAULT_CONFIG, TASK_CONFIGS
from totalsegmentator.map_to_binary import (class_map, class_map_5_parts, class_map_5_parts_total_v2,
                                            class_map_parts_headneck_muscles, class_map_parts_mr,
                                            map_taskid_to_partname_ct, map_taskid_to_partname_headneck_muscles,
                                            map_taskid_to_partname_mr)
from totalsegmentator.nifti_ext_header import add_label_map_to_nifti
from totalsegmentator.postprocessing import (extract_skin, keep_largest_blob_multilabel, postprocess_vertebrae_pp,
                                             refine_vertebrae_pp_with_body_mask, remove_outside_of_mask,
                                             remove_small_blobs_multilabel)
from totalsegmentator.resampling import resample_img_torch
from totalsegmentator.statistics import get_basic_statistics

# images read and preprocessed ahead of the GPU by predict()'s loader thread (see there)
READ_AHEAD = 2

TOTAL_TASKS = ("total", "total_5parts", "total_v2", "total_mr", "total_v2_mr")

# Multi-model tasks: ({part: {label id in that part's model: class}}, {model: part}). The label maps of the parts are
# merged in the order of the task's "task_id" list; where two parts disagree the later one wins.
PART_MAPS = {
    "total_5parts": (class_map_5_parts, map_taskid_to_partname_ct),
    "total_v2": (class_map_5_parts_total_v2, map_taskid_to_partname_ct),
    "total_highres": (class_map_5_parts, map_taskid_to_partname_ct),
    "total_mr": (class_map_parts_mr, map_taskid_to_partname_mr),
    "total_v2_mr": (class_map_parts_mr, map_taskid_to_partname_mr),
    "headneck_muscles": (class_map_parts_headneck_muscles, map_taskid_to_partname_headneck_muscles),
}


# ------------------------------------------------------------------ the registry, as the pipeline sees it ----------
def show_license_info():
    status, message = has_valid_license_offline()
    if status == "missing_license":
        print(textwrap.dedent("""\
              In contrast to the other tasks this task is not openly available.
              It requires a license. For non-commercial usage a free license can be
              acquired here:
              https://backend.totalsegmentator.com/license-academic/

              For commercial usage contact: jakob.wasserthal@usb.ch
              """))
        sys.exit(1)
    elif status in ("invalid_license", "missing_config_file"):
        print(message)
        sys.exit(1)


def get_task_config(task: str, mode: str = "default", quiet: bool = False, model_size: str = "big") -> dict:
    """The configuration of a task in a mode ('default', 'fast' or 'fastest'): DEFAULT_CONFIG updated with
    TASK_CONFIGS[task]. model_size 'small': the variant of the models with fewer filters (exists for 'total_5parts'
    and the --fast model of 'total')."""
    if task not in TASK_CONFIGS:
        raise ValueError(f"Unknown task: {task}")
    config = DEFAULT_CONFIG.copy()
    raw_task = TASK_CONFIGS[task]
    if "sub_modes" in raw_task:
        if mode == "fastest" and "fastest" not in raw_task["sub_modes"]:      # the body tasks only have 'fast'
            mode = "fast"
        config.update(raw_task["sub_modes"][mode])
        if not quiet and mode != "default":
            print(f"Using '{mode}' option: a model with lower resolution")
    else:
        if raw_task.get("disallow_fast") and mode == "fast":
            raise ValueError(f"task {task} does not work with option --fast")
        config.update({k: v for k, v in raw_task.items() if k != "info_msg"})
    config.setdefault("step_size", 0.8 if task in TOTAL_TASKS else 0.5)
    if model_size not in ("big", "small"):
        raise ValueError("model_size must be 'big' or 'small'.")
    if model_size == "small":
        if not (task == "total_5parts" or (task == "total" and mode != "default")):
            raise ValueError("model_size='small' is currently only supported for task 'total_5parts' and for task "
                             "'total' with --fast.")
        config["plans"] = "nnUNetResEncUNetLPlans_8"
    if "info_msg" in raw_task and not quiet:
        print(raw_task["info_msg"])
    if raw_task.get("commercial"):
        show_license_info()
    return config


def get_roi(task: str, config: dict, roi_subset: Optional[list[str]] = None, robust: bool = False,
            mode: str = "default") -> Optional[dict]:
    """
    The region the models of a task run on: the bounding box of `classes` in the label map of another (cheap) task,
    plus `margin` mm. None: the whole image. A task has a region if its config names one ("crop"), or if the caller
    only wants some classes of a total task (roi_subset) in the default mode. The --fast and --fastest models are
    single low resolution models; a crop barely makes them faster.
    """
    classes = config["crop"] or roi_subset
    if classes is None or (config["crop"] is None and mode != "default"):
        return None
    if config["crop_model"] is not None:
        return {"task": config["crop_model"], "mode": "default", "classes": classes, "margin": config["crop_addon"]}
    if "body_trunc" in classes or "body_extremities" in classes:
        provider, mode = "body", "fast"
    else:
        provider = task if task in ("total_v2", "total_v2_mr") else "total_mr" if task.endswith("_mr") else "total"
        # 6 mm model; the 3 mm model if asked for and for MR, where 6 mm is too coarse
        mode = "fast" if robust or config["robust_crop"] or provider.endswith("_mr") else "fastest"
    return {"task": provider, "mode": mode, "classes": classes, "margin": config["crop_addon"]}


def class_ids(task: str, classes: Sequence[str]) -> list[int]:
    return [i for i, name in class_map[task].items() if name in classes]


def label_isin(seg: np.ndarray, ids: Sequence[int]) -> np.ndarray:
    """np.isin(seg, ids) for a label map: a uint8 / uint16 map goes through a lookup table, which is several times as
    fast as np.isin, needs no full-size temporaries and keeps the memory layout of seg."""
    if seg.dtype not in (np.uint8, np.uint16):
        return np.isin(seg, ids)
    lut = np.zeros(np.iinfo(seg.dtype).max + 1, dtype=bool)
    ids = np.asarray(list(ids), dtype=np.int64)
    lut[ids[(ids >= 0) & (ids < lut.size)]] = True
    return lut[seg]


def validate_roi_subset(roi_subset, task):
    """
    Raise an error if roi_subset contains class names which the task does not have.

    Without this check a wrong class name (a typo, or a class of the CT task used with
    total_mr) only fails with a KeyError after the weights were downloaded and the
    cropping model was run.
    """
    if not isinstance(roi_subset, list) or task not in class_map:
        return  # nothing to check here; wrong types and tasks are reported further down
    valid_names = list(class_map[task].values())
    unknown = []
    for roi in roi_subset:
        if roi not in valid_names:
            close = difflib.get_close_matches(str(roi), valid_names, n=1)
            suggestion = f" (did you mean '{close[0]}'?)" if close else ""
            unknown.append(f"'{roi}'{suggestion}")
    if unknown:
        raise ValueError(f"roi_subset contains class names which are not part of task '{task}': {', '.join(unknown)}. "
                         f"Run `totalseg_info --classes -ta {task}` to see all class names of this task.")


def bounding_box(mask: np.ndarray, margin_voxels) -> Optional[tuple[slice, ...]]:
    """The bounding box of a mask plus a margin per axis, clipped to the array, as slices. None for an empty mask."""
    if not mask.any():
        return None
    box = []
    for axis, margin in enumerate(margin_voxels):
        along = np.flatnonzero(mask.any(axis=tuple(a for a in range(mask.ndim) if a != axis)))
        box.append(slice(max(0, int(along[0]) - int(margin)), min(mask.shape[axis], int(along[-1]) + 1 + int(margin))))
    return tuple(box)


def merge_parts(parts: Sequence[tuple[np.ndarray, Optional[np.ndarray]]]) -> np.ndarray:
    """Merge the label maps of a task's models: [(label map, lookup table to the task's label ids or None)]. A later
    model overwrites an earlier one wherever it predicts foreground."""
    merged = None
    for seg, lut in parts:
        seg = seg if lut is None else lut[seg]
        if merged is None:
            merged = seg
        else:
            np.copyto(merged, seg, where=seg != 0)
    return merged


def apply_label_rules(seg: np.ndarray, task: str, spacing, quiet: bool = True) -> np.ndarray:
    """The rules that are part of a task's definition, applied to its label map at the resolution of the input image.
    seg and spacing (mm) have the axis order of nnU-Net's readers: (z, y, x) of the canonical (RAS) orientation."""
    labels = class_map[task]
    auxiliary = class_map.get(task + "_auxiliary")
    if auxiliary:
        seg[label_isin(seg, list(auxiliary))] = 0
    if task == "body":
        seg = keep_largest_blob_multilabel(seg, labels, ["body_trunc"], quiet=quiet)
        seg = remove_small_blobs_multilabel(seg, labels, ["body_extremities"],
                                            interval=[50000 / np.prod(spacing), 1e10], quiet=quiet)
    if task in ("vertebrae_pp", "vertebrae_pp_refined"):        # counts the vertebrae along z of (x, y, z)
        seg = postprocess_vertebrae_pp(np.ascontiguousarray(seg.transpose(2, 1, 0)), labels,
                                       voxel_volume=np.prod(spacing), voxel_spacing=spacing[::-1]).transpose(2, 1, 0)
    return seg


# ------------------------------------------------------------------ one nnU-Net model ------------------------------
def resample_cubic(data: np.ndarray, new_shape, current_spacing=None, new_spacing=None, device="cpu") -> np.ndarray:
    """
    Input resampling for the models whose training images were resampled to the model spacing with a cubic spline
    on all three axes before nnU-Net saw them ("input_resampling": "cubic"), on the inference device.
    The grid is center-aligned (the image keeps its extent), which is the rule nnU-Net maps the predicted logits back
    with. Has the signature of an nnU-Net resampling function: data is (c, x, y, z).
    """
    return np.stack([resample_img_torch(channel, new_shape, device=device, order=3, grid_mode=True)
                     for channel in data])


class Model:
    """One nnU-Net model: preprocess an image for it, predict the label map on the image's grid."""

    def __init__(self, task_id: int, config: dict, device: torch.device, compile, verbose: bool, show_progress: bool):
        from nnunetv2.inference.predict_streaming import nnUNetPredictorStreaming
        from nnunetv2.utilities.file_path_utilities import get_output_folder
        from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager

        download_pretrained_weights(task_id)
        cubic = config["input_resampling"] == "cubic"
        self.nnunet = nnUNetPredictorStreaming(
            tile_step_size=config["step_size"], use_gaussian=True, use_mirroring=False,
            perform_everything_on_device=True, device=device, verbose=verbose, verbose_preprocessing=verbose,
            allow_tqdm=show_progress, compile=compile, quiet=not verbose, force_trilinear=cubic)
        # the network is built from the plans; the weights load in the background, while the first image is read
        self.nnunet.initialize_from_folder_base_trainer(
            get_output_folder(task_id, config["trainer"], config["plans"], config["model"]), config["folds"],
            load_weights_in_background=True)
        if cubic:
            class CubicInput(ConfigurationManager):
                resampling_fn_data = property(lambda _: partial(resample_cubic, device=device))
            self.nnunet.configuration_manager = CubicInput(self.nnunet.configuration_manager.configuration)
        self.preprocessor = self.nnunet.configuration_manager.preprocessor_class(verbose=verbose)
        self.cubic = cubic

    def preprocess(self, data: np.ndarray, properties: dict):
        """nnU-Net's preprocessing of an image as its reader returns it. Returns (data, properties) for predict()."""
        from nnunetv2.inference.streaming_pipeline import load_and_preprocess, load_npy_case
        # resampling_threads can return a plain ConfigurationManager built from the plans, which would drop
        # CubicInput: not for the cubic models (their resampling runs on the inference device anyway)
        return load_and_preprocess(load_npy_case, (data, properties), self.nnunet.plans_manager,
                                   self.nnunet.configuration_manager, self.nnunet.dataset_json, self.preprocessor,
                                   resampling_threads=None if self.cubic else os.cpu_count())

    def predict(self, data: np.ndarray, properties: dict, num_cases: int = 1) -> np.ndarray:
        """The label map on the grid of the image that was preprocessed (in the reader's axis order). num_cases: how
        many images the caller is going to predict (nnU-Net compiles the network for larger batches)."""
        item = {"data": data, "data_properties": properties, "ofile": None}
        return self.nnunet.predict_from_data_iterator_streaming([item], num_cases=num_cases)[0]


# ------------------------------------------------------------------ input ------------------------------------------
def load_image(input: Union[str, Path, Nifti1Image], reader, keep_image: bool = False, verbose: bool = False):
    """
    One input (NIfTI path, DICOM folder / zip, or Nifti1Image), read once: (data, properties) as nnU-Net's reader
    returns a 3D image (float32, canonical orientation, (1, z, y, x)), plus what export_case needs to know about the
    input. keep_image: keep an input that is not a 3D NIfTI file for statistics / preview (a file is read again there).
    """
    dicom_dir = None
    with ExitStack() as stack:
        if isinstance(input, Nifti1Image):
            img = input
        else:
            input = Path(input)
            if not input.exists():
                raise FileNotFoundError(f"The input file or directory does not exist: {input}")
            if str(input).endswith((".nii", ".nii.gz")):
                img = nib.load(input)
            else:
                from totalsegmentator.dicom_io import dcm_to_nifti
                dicom_dir = input
                tmp = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="totalseg_dcm_")))
                dcm_to_nifti(input, tmp / "converted_dcm.nii.gz", tmp, verbose=verbose)
                img = nib.load(tmp / "converted_dcm.nii.gz")
        if len(img.shape) == 2:
            raise ValueError("TotalSegmentator does not work for 2D images. Use a 3D image.")
        if img.get_data_dtype().fields is not None:
            raise TypeError(f"Invalid dtype {img.get_data_dtype()}. Expected a simple dtype, not a structured one.")
        is_file = isinstance(input, Path) and dicom_dir is None and len(img.shape) == 3
        if len(img.shape) > 3:
            print(f"WARNING: Input image has {len(img.shape)} dimensions. Only using first three dimensions.")
            img = img.slicer[(slice(None),) * 3 + (0,) * (len(img.shape) - 3)]
        if keep_image and dicom_dir is not None:    # the converted file is gone when this function returns
            img = Nifti1Image(np.asanyarray(img.dataobj), img.affine, img.header)
        data, properties = reader.read_nibabel_images([img])
    source = {"affine": img.affine, "header": img.header.copy(), "nifti_path": str(input) if is_file else None,
              "dicom_dir": dicom_dir, "image": img if keep_image and not is_file else None}
    return data, properties, source


def nifti_cases(folder: Union[str, Path]) -> Optional[list[tuple[str, Path]]]:
    """The NIfTI images directly in a folder as [(case name, file)], sorted by name (case name = file name without
    .nii / .nii.gz). None if the folder holds no NIfTI image: then it is one DICOM series."""
    files = sorted(p for p in Path(folder).iterdir()
                   if p.is_file() and not p.name.startswith(".") and p.name.endswith((".nii", ".nii.gz")))
    if not files:
        return None
    names = [p.name[:-len(".nii.gz")] if p.name.endswith(".nii.gz") else p.name[:-len(".nii")] for p in files]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Two images with the same name in {folder}: {duplicates}")
    return list(zip(names, files))


def output_file(output: Optional[Path], ml: bool, output_types: Sequence[str]) -> Optional[Path]:
    """Where a result is written: a multilabel NIfTI (ml, output type nifti only) gets .nii.gz appended when its name has
    no .nii / .nii.gz ending (nibabel would write an uncompressed .nii); every other output as given."""
    if output is None or not ml or list(output_types) != ["nifti"] or output.name.endswith((".nii", ".nii.gz")):
        return output
    return output.with_name(output.name + ".nii.gz")


def _input_image(source: dict) -> Nifti1Image:
    return source["image"] if source["image"] is not None else nib.load(source["nifti_path"])


# ------------------------------------------------------------------ output -----------------------------------------
def export_case(data: np.ndarray, source: dict, output: Optional[Path], task: str, label_map: dict, options: dict):
    """
    What the caller asked for beyond the label map of the task (data, on the input's grid): optional post-processing,
    preview, statistics, class subset, saving. Returns (image or None, statistics or None).
    """
    quiet = options["quiet"]
    zooms = source["header"].get_zooms()[:3]
    if options["remove_small_blobs"]:
        if not quiet: print("Removing small blobs...")
        size_thr_mm3 = 200 if options["remove_small_blobs"] is True else options["remove_small_blobs"]
        data = remove_small_blobs_multilabel(data, class_map[task], list(class_map[task].values()),
                                             interval=[size_thr_mm3 / float(np.prod(zooms)), 1e10], quiet=quiet)

    out_dir = None if output is None else (output.parent if options["ml"] or options["dicom_output"] else output)
    # several images in one call, each written as a file: its statistics and preview are named after that file
    prefix = f"{output.name.split('.')[0]}_" if options["batch"] and out_dir is not None and out_dir != output else ""
    if options["preview"]:
        if isinstance(options["preview"], (str, Path)):
            preview_file = Path(options["preview"]).absolute()
        elif out_dir is not None:
            preview_file = out_dir / f"{prefix}preview_{task}.png"
        else:
            preview_file = None
        if preview_file is None:
            print("WARNING: No output directory specified. Skipping preview generation.")
        else:
            from totalsegmentator.preview import generate_preview
            from totalsegmentator.resampling import change_spacing
            if not quiet: print("Generating preview...")
            # canonical orientation, 1.5 mm (rendering at input resolution is slower)
            ct = change_spacing(nib.as_closest_canonical(_input_image(source)), 1.5, order=1, dtype=np.int32)
            seg = change_spacing(nib.as_closest_canonical(Nifti1Image(data, source["affine"])), 1.5,
                                 target_shape=ct.shape, order=0, dtype=np.uint8)
            preview_file.parent.mkdir(exist_ok=True, parents=True)
            generate_preview(ct, preview_file, np.asanyarray(seg.dataobj), 20, task)

    stats = None
    if options["statistics"]:
        if not quiet: print("Calculating statistics...")
        if isinstance(options["statistics"], (str, Path)):
            stats_file = Path(options["statistics"]).absolute()
        elif out_dir is not None:
            out_dir.mkdir(exist_ok=True, parents=True)
            stats_file = out_dir / f"{prefix}statistics.json"
        else:
            stats_file = None
        stats = get_basic_statistics(data, _input_image(source), stats_file, quiet, task,
                                     options["statistics_exclude_masks_at_border"], options["roi_subset"],
                                     metric=options["stats_aggregation"],
                                     normalized_intensities=options["statistics_normalized_intensities"],
                                     extra_metrics=options["statistics_extra"])

    if options["v1_order"]:
        data = reorder_multilabel_like_v1(data, class_map[task], class_map["total_v1"])
    if options["roi_subset"] is not None:       # label_map: the ids that are saved (v1 ids with v1_order)
        data *= label_isin(data, list(label_map.keys()))

    header = source["header"].copy()
    header.set_data_dtype(np.uint8)
    img_out = add_label_map_to_nifti(Nifti1Image(data, source["affine"], header), label_map)
    if output is not None and not options["skip_saving"]:
        _save(img_out, header, output, task, label_map, source, options)
    return (img_out if options["return_images"] else None), stats


def _save(img_out: Nifti1Image, header, output: Path, task: str, label_map: dict, source: dict, options: dict):
    """img_out: the multilabel image with the label table in its header; header: the same header without the table,
    for the files that hold one class."""
    from totalsegmentator.dicom_io import save_mask_as_dicomseg, save_mask_as_rtstruct
    output_types, nora_tag, data = options["output_types"], options["nora_tag"], np.asanyarray(img_out.dataobj)
    if not options["quiet"]: print("Saving segmentations...")
    st = time.time()
    if len(output_types) > 1:       # several files next to each other, named after the output
        base_dir = output if output.is_dir() else output.parent
        base_name = f"{task}_segmentation" if output.is_dir() else output.name.split(".")[0]
        base_dir.mkdir(exist_ok=True, parents=True)
        files = {"nifti": base_dir / f"{base_name}.nii.gz", "dicom_rtstruct": base_dir / f"{base_name}_rtstruct.dcm",
                 "dicom_seg": base_dir / f"{base_name}_seg.dcm"}
    else:
        files = {output_types[0]: output}
    for output_type in output_types:
        file = files[output_type]
        if output_type == "dicom_rtstruct":
            save_mask_as_rtstruct(data, label_map, source["dicom_dir"], file)
        elif output_type == "dicom_seg":
            save_mask_as_dicomseg(data, label_map, source["dicom_dir"], file, img_out.affine,
                                  use_gpu=options["use_gpu"])
        elif options["ml"] or len(output_types) > 1:
            file.parent.mkdir(exist_ok=True, parents=True)
            nib.save(img_out, file)
            if nora_tag != "None" and len(output_types) == 1:
                subprocess.call(f"/opt/nora/src/node/nora -p {nora_tag} --add {file} --addtag atlas", shell=True)
        else:
            save_binary_niftis(data, img_out.affine, header, label_map, file, options["nr_thr_saving"], nora_tag)
            if task == "body" and not options["no_derived_masks"]:
                body = Nifti1Image((data > 0).astype(np.uint8), img_out.affine, header)
                nib.save(body, file / "body.nii.gz")
                nib.save(extract_skin(_input_image(source), body), file / "skin.nii.gz")
    if not options["quiet"]: print(f"  Saved in {time.time() - st:.2f}s")


# ------------------------------------------------------------------ the predictor ----------------------------------
class TotalSegmentatorPredictor:
    """
    A persistent TotalSegmentator predictor for one task. Its models are loaded when they are first needed and stay
    loaded; call predict() as often as needed. One image runs in this process; a list runs as a batch in which reading
    and preprocessing the next image, the GPU and saving the previous images overlap. close() (or a with block) shuts
    the export workers down.

        with TotalSegmentatorPredictor("total", device="gpu") as predictor:
            seg = predictor.predict(ct_image)                                   # Nifti1Image in, Nifti1Image out
            predictor.predict(["a.nii.gz", "b.nii.gz"], ["a_seg", "b_seg"])      # one folder of masks per image

    compile: 'auto' compiles the networks for batches of more than 4 images, never for a single one; True / False
    force it. nr_thr_export: background processes that save the results of a batch.
    """

    def __init__(self, task: str = "total", fast: bool = False, fastest: bool = False, device="gpu",
                 quiet: bool = False, verbose: bool = False, compile: Union[bool, str] = "auto",
                 robust_crop: bool = False, model_size: str = "big", nr_thr_export: int = 3,
                 license_number: Optional[str] = None):
        from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
        from totalsegmentator.python_api import select_device, validate_device_type_api
        if isinstance(device, str):
            validate_device_type_api(device)
            device = select_device(device)
        self.device = torch.device(device)
        self.task, self.quiet, self.verbose, self.compile = task, quiet, verbose, compile
        self.robust_crop, self.nr_thr_export = robust_crop, nr_thr_export
        setup_nnunet()
        setup_totalseg()
        if license_number is not None:
            set_license_number(license_number)
        self.mode = "fast" if fast else "fastest" if fastest else "default"
        self._config = get_task_config(task, self.mode, quiet, model_size)   # prints the task's messages
        # every model gets the image in canonical (RAS) orientation, in the axis order of nnU-Net's readers
        self.reader = NibabelIOWithReorient()
        self._models: dict[tuple, Model] = {}
        self._models_lock = threading.Lock()
        self._export_pool = None
        # threads do not help nnU-Net on the GPU; nnU-Net switches cudnn.benchmark off. close() restores both
        self._torch_settings_before = (torch.get_num_threads(), torch.backends.cudnn.benchmark)
        torch.set_num_threads(os.cpu_count() if self.device.type == "cpu" else 1)

    def close(self):
        """Shut down the export workers (they are started again when needed). The models stay loaded."""
        if self._export_pool is not None:
            self._export_pool.shutdown()
            self._export_pool = None
        for model in self._models.values():
            model.nnunet.close()
        torch.set_num_threads(self._torch_settings_before[0])
        torch.backends.cudnn.benchmark = self._torch_settings_before[1]

    @contextmanager
    def _dropped_if_broken(self):
        """A process pool whose worker died (e.g. killed for its memory) stays broken: drop it, so that the next batch
        starts a new one."""
        try:
            yield
        except BrokenProcessPool:
            self._export_pool.shutdown(wait=False, cancel_futures=True)
            self._export_pool = None
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # ---------------------------------------------------------------- models ---------------------------------------
    def models(self, task: str, config: dict, roi_subset: Optional[list[str]] = None):
        """The models of a task as [(Model, lookup table from its label ids to the task's, or None)]. With a
        roi_subset, only the models of a multi-model task that predict one of its classes."""
        def model(task_id):
            key = (task_id, *(repr(config[k]) for k in ("trainer", "plans", "model", "folds", "input_resampling",
                                                        "step_size")))
            with self._models_lock:
                if key not in self._models:
                    self._models[key] = Model(task_id, config, self.device, self.compile, self.verbose,
                                              show_progress=not self.quiet)
                return self._models[key]

        if not isinstance(config["task_id"], list):
            return [(model(config["task_id"]), None)]
        parts, part_of_model = PART_MAPS[task]
        ids = {name: i for i, name in class_map[task].items()}
        selected = []
        for task_id in config["task_id"]:
            part = parts[part_of_model[task_id]]
            if roi_subset is None or any(name in roi_subset for name in part.values()):
                lut = np.zeros(256, dtype=np.uint8)
                for local_id, name in part.items():
                    lut[local_id] = ids[name]
                selected.append((model(task_id), lut))
        return selected

    def _task_config(self, task: str, mode: str) -> dict:
        return self._config if (task, mode) == (self.task, self.mode) else get_task_config(task, mode, quiet=True)

    # ---------------------------------------------------------------- the pipeline ---------------------------------
    @torch.inference_mode()
    def segment(self, data: Optional[np.ndarray], properties: dict, task: str, mode: str = "default",
                roi_subset: Optional[list[str]] = None, prepared: Optional[list] = None,
                num_cases: int = 1) -> np.ndarray:
        """
        The label map of a task (ids of class_map[task]) for an image as nnU-Net's reader returns it: data (1, z, y, x)
        and its properties; the label map is (z, y, x). See the module docstring. prepared: the image preprocessed for
        the models that run on all of it (_load); each entry is taken out of the list once its model runs. data is only
        needed where a model runs on a region of the image.
        """
        config = self._task_config(task, mode)
        spacing = np.array(properties["spacing"])
        models = self.models(task, config, roi_subset)

        roi = get_roi(task, config, roi_subset, self.robust_crop, mode)
        if roi is not None:
            rough = self.segment(data, properties, roi["task"], roi["mode"], prepared=prepared, num_cases=num_cases)
            box = bounding_box(label_isin(rough, class_ids(roi["task"], roi["classes"])),
                               np.array(roi["margin"])[::-1] / spacing)
            if box is None:
                if not self.quiet: print(f"INFO: None of {roi['classes']} found. Returning empty segmentation.")
                return np.zeros(data.shape[1:], dtype=np.uint8)
            crop = data[(slice(None), *box)]
            inputs = (model.preprocess(crop, properties) for model, _ in models)
        elif prepared is not None:
            inputs = (prepared.pop(0) for _ in models)
        else:
            inputs = (model.preprocess(data, properties) for model, _ in models)

        # a model's input is dropped once the model has run (a crop is preprocessed only when its model is next)
        seg = merge_parts((model.predict(*model_input, num_cases), lut)
                          for (model, lut), model_input in zip(models, inputs))
        seg = apply_label_rules(seg, task, spacing, self.quiet)
        if roi is not None:
            cropped, seg = seg, np.zeros(data.shape[1:], dtype=np.uint8)
            seg[box] = cropped

        if config["remove_outside"] is not None:
            keep = label_isin(rough, class_ids(roi["task"], config["remove_outside"]))
            seg = remove_outside_of_mask(seg, keep, addon=int(config["remove_outside_dilation"] / spacing.mean()))
        return seg

    def _to_input_grid(self, seg: np.ndarray, properties: dict) -> np.ndarray:
        """A label map of segment() on the voxel grid of the input, in NIfTI's memory layout (no transpose when it
        is saved)."""
        return np.asfortranarray(self.reader.seg_to_nibabel(seg, properties).dataobj)

    def _whole_image_models(self, roi_subset) -> tuple[tuple, bool]:
        """Which models run on the whole image - the task's own, or those at the end of its chain of rois - as
        arguments of models(), and whether segment() needs the image itself as well (for a roi, or for the second
        task of vertebrae_pp_refined)."""
        task, mode, config, needs_image = self.task, self.mode, self._config, self.task == "vertebrae_pp_refined"
        while (roi := get_roi(task, config, roi_subset, self.robust_crop, mode)) is not None:
            task, mode, roi_subset, needs_image = roi["task"], roi["mode"], None, True
            config = self._task_config(task, roi["mode"])
        return (task, config, roi_subset), needs_image

    def _load(self, input, keep_image: bool, whole_image_models: tuple, needs_image: bool):
        """Runs in a background thread, ahead of the GPU: read the image and preprocess it for the models that run on
        all of it (_whole_image_models)."""
        data, properties, source = load_image(input, self.reader, keep_image, self.verbose)
        prepared = [model.preprocess(data, properties) for model, _ in self.models(*whole_image_models)]
        return (data if needs_image else None), properties, source, prepared

    def predict(self, input: Union[str, Path, Nifti1Image, Sequence[Union[str, Path, Nifti1Image]]],
                output: Union[str, Path, None, Sequence[Union[str, Path, None]]] = None, ml: bool = False,
                roi_subset: Optional[list[str]] = None, output_type: Union[str, list[str]] = "nifti",
                statistics: Union[bool, str, Path] = False, statistics_extra: bool = False,
                stats_aggregation: str = "mean", statistics_exclude_masks_at_border: bool = True,
                statistics_normalized_intensities: bool = False, preview: Union[bool, str, Path] = False,
                remove_small_blobs: Union[bool, float] = False, v1_order: bool = False,
                no_derived_masks: bool = False, nr_thr_saving: int = 6, skip_saving: bool = False,
                nora_tag: str = "None", return_images: Optional[bool] = None):
        """
        Segment one image or a list of images. output: for each image a file (ml=True, or a DICOM output type) or a
        folder that gets one file per class, or None to only return the result. The other arguments are those of
        totalsegmentator(). Returns, per image, the multilabel Nifti1Image (or (image, statistics) with statistics);
        for a list with outputs the images are not sent back from the export workers unless return_images=True.
        For a list, the statistics and preview of an image whose output is a file are named after that file
        (<name>_statistics.json, <name>_preview_<task>.png), so that images sharing a folder keep their own.
        """
        single = isinstance(input, (str, Path, Nifti1Image))
        if not single and isinstance(output, (str, Path)):
            raise ValueError("Give one output per input (a list), not a single path for a list of inputs.")
        inputs = [input] if single else list(input)
        outputs = [output] if single else (list(output) if output is not None else [None] * len(inputs))
        if len(outputs) != len(inputs):
            raise ValueError("Give one output per input.")
        outputs = [None if o is None else Path(o) for o in outputs]
        if return_images is None:
            return_images = single or all(o is None for o in outputs)
        if not single and isinstance(statistics, (str, Path)):
            raise ValueError("statistics: give True for several images (each gets its own statistics file), not a path.")
        if not single and isinstance(preview, (str, Path)):
            raise ValueError("preview: give True for several images (each gets its own preview file), not a path.")
        if roi_subset is not None and not isinstance(roi_subset, list):
            raise ValueError("roi_subset must be a list of strings")
        if roi_subset is not None and not self.task.startswith("total"):
            raise ValueError("roi_subset only works with the total tasks")
        validate_roi_subset(roi_subset, self.task)      # before any weights are downloaded or models are run
        output_types = [output_type] if isinstance(output_type, str) else list(output_type)
        outputs = [output_file(o, ml, output_types) for o in outputs]
        v1_order = v1_order and self.task in ("total", "total_5parts", "total_v2")
        label_map = class_map["total_v1"] if v1_order else class_map[self.task]
        if roi_subset is not None:
            label_map = {k: v for k, v in label_map.items() if v in roi_subset}
        options = {"quiet": self.quiet, "ml": ml, "roi_subset": roi_subset, "output_types": output_types,
                   "dicom_output": any(t.startswith("dicom") for t in output_types), "statistics": statistics,
                   "statistics_extra": statistics_extra, "stats_aggregation": stats_aggregation,
                   "statistics_exclude_masks_at_border": statistics_exclude_masks_at_border,
                   "statistics_normalized_intensities": statistics_normalized_intensities, "preview": preview,
                   "remove_small_blobs": remove_small_blobs, "v1_order": v1_order,
                   "no_derived_masks": no_derived_masks, "nr_thr_saving": nr_thr_saving, "skip_saving": skip_saving,
                   "nora_tag": nora_tag, "return_images": return_images, "use_gpu": self.device.type == "cuda",
                   "batch": not single}
        keep_image = bool(statistics or preview or self.task == "body")
        if options["dicom_output"] and any(isinstance(i, Nifti1Image) or str(i).endswith((".nii", ".nii.gz"))
                                           for i in inputs):
            raise ValueError("To use output type dicom_rtstruct or dicom_seg you also have to use a Dicom image as "
                             "input.")

        st = time.time()
        if not self.quiet: print("Predicting...")
        if not single and self._export_pool is None:
            self._export_pool = ProcessPoolExecutor(self.nr_thr_export, mp_context=get_context("spawn"))
        results = []
        # One thread reads and preprocesses ahead of the GPU: two images where only the preprocessed image waits (a
        # large file takes longer to read than the GPU pass of a small image before it), one where the image is
        # kept as well (statistics, a roi, the refinement).
        whole_image_models, needs_image = self._whole_image_models(roi_subset)
        todo = iter(inputs)
        with ThreadPoolExecutor(1, thread_name_prefix="totalseg_load") as loader:
            def load(number):
                return [loader.submit(self._load, input, keep_image, whole_image_models, needs_image)
                        for input in itertools.islice(todo, number)]
            loading = deque(load(1 if keep_image or needs_image else READ_AHEAD))
            self.models(self.task, self._config, roi_subset)        # load the models while the first image is read
            for out in outputs:
                data, properties, source, prepared = loading.popleft().result()
                loading.extend(load(1))                 # read on while this image is predicted; nothing else holds it
                seg = self.segment(data, properties, self.task, self.mode, roi_subset, prepared, len(inputs))
                body = self.segment(data, properties, "vertebrae_body", num_cases=len(inputs)) \
                    if self.task == "vertebrae_pp_refined" else None
                del data, prepared
                seg = self._to_input_grid(seg, properties)
                if body is not None:
                    # on the input's own grid: the dilation kernel is not symmetric, so the result depends on the
                    # direction of the axes, and this is the grid the refinement has always run on. C-ordered
                    # copies: its searches per vertebra are several times slower on NIfTI's memory layout
                    body = np.ascontiguousarray(self._to_input_grid(body, properties))
                    seg = np.asfortranarray(refine_vertebrae_pp_with_body_mask(
                        np.ascontiguousarray(seg), body, class_map[self.task], dilation_mm=2,
                        voxel_spacing=source["header"].get_zooms()[:3]))
                if single:
                    results.append(export_case(seg, source, out, self.task, label_map, options))
                else:
                    # at most nr_thr_export + 2 results wait to be saved
                    while sum(not r.done() for r in results) >= self.nr_thr_export + 2:
                        wait([r for r in results if not r.done()], return_when=FIRST_COMPLETED)
                    with self._dropped_if_broken():
                        results.append(self._export_pool.submit(export_case, seg, source, out, self.task, label_map,
                                                                options))
        if not single:
            with self._dropped_if_broken():
                results = [r.result() for r in results]
        if not self.quiet: print(f"  Predicted {len(inputs)} image(s) in {time.time() - st:.2f}s")
        results = [r if statistics else r[0] for r in results]
        return results[0] if single else results
