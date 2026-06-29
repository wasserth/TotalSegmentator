import inspect
import os
import warnings
from copy import deepcopy
from time import sleep
from typing import Callable, List, Tuple, Union

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
from batchgenerators.utilities.file_and_folder_operations import load_json, save_pickle
from scipy.ndimage import map_coordinates

from nnunetv2.configuration import default_num_processes
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager


CROPPED_LOGITS_RESAMPLING_MARGIN = 3
_PATCH_MARKER = "_totalseg_cropped_logits_resampling_patch"


def _supports_keyword_argument(func, keyword: str) -> bool:
    return keyword in inspect.signature(func).parameters


def _as_numpy(predicted_logits: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(predicted_logits, torch.Tensor):
        return predicted_logits.detach().cpu().numpy()
    return predicted_logits


def _get_bbox(mask: np.ndarray, margin: int) -> Union[Tuple[np.ndarray, np.ndarray], None]:
    foreground = np.where(mask)
    if len(foreground) == 0 or len(foreground[0]) == 0:
        return None

    shape = np.array(mask.shape)
    bbox_start = np.maximum(np.array([axis.min() for axis in foreground]) - margin, 0)
    bbox_end = np.minimum(np.array([axis.max() + 1 for axis in foreground]) + margin, shape)
    return bbox_start, bbox_end


def _resample_channel_crop(channel: np.ndarray,
                           new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
                           crop_start: np.ndarray,
                           crop_end: np.ndarray,
                           order: int = 1) -> Tuple[Union[Tuple[slice, ...], None], Union[np.ndarray, None]]:
    old_shape = np.array(channel.shape)
    new_shape = np.array(new_shape).astype(int)
    scale = new_shape / old_shape

    target_start = np.maximum(np.floor(crop_start * scale).astype(int), 0)
    target_end = np.minimum(np.ceil(crop_end * scale).astype(int), new_shape)
    if np.any(target_end <= target_start):
        return None, None

    crop_slices = tuple(slice(start, end) for start, end in zip(crop_start, crop_end))
    target_slices = tuple(slice(start, end) for start, end in zip(target_start, target_end))
    channel_crop = channel[crop_slices].astype(float, copy=False)

    coordinates = []
    for start, end, old_dim, new_dim, offset in zip(target_start, target_end, old_shape, new_shape, crop_start):
        coordinates.append((np.arange(start, end, dtype=np.float32) + 0.5) * old_dim / new_dim - 0.5 - offset)
    coordinate_map = np.array(np.meshgrid(*coordinates, indexing="ij"), dtype=np.float32)
    resampled_crop = map_coordinates(channel_crop, coordinate_map, order=order, mode="nearest")
    return target_slices, resampled_crop


def _convert_logits_to_segmentation_with_cropped_resampling(
        predicted_logits: Union[torch.Tensor, np.ndarray],
        label_manager: LabelManager,
        new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
        current_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
        target_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
        resampling_fn_probabilities: Callable) -> np.ndarray:
    """
    Memory-light approximation of nnU-Net's default export.

    Default export resamples all class logits to the target shape and only then
    converts them to labels. This keeps the same order of operations but only
    resamples foreground/region channels around their low-resolution prediction
    bounding boxes. For regular multiclass models, background is resampled as a
    full single channel and foreground classes compete against that score volume.
    """
    predicted_logits = _as_numpy(predicted_logits)
    old_shape = np.array(predicted_logits.shape[1:])
    new_shape = np.array(new_shape).astype(int)
    margin = CROPPED_LOGITS_RESAMPLING_MARGIN

    if np.array_equal(old_shape, new_shape):
        segmentation = label_manager.convert_logits_to_segmentation(predicted_logits)
        if isinstance(segmentation, torch.Tensor):
            segmentation = segmentation.cpu().numpy()
        return segmentation

    segmentation_dtype = np.uint8 if len(label_manager.foreground_labels) < 255 else np.uint16

    if label_manager.has_regions:
        segmentation = np.zeros(new_shape, dtype=segmentation_dtype)
        for channel_idx, class_id in enumerate(label_manager.regions_class_order):
            lowres_region = predicted_logits[channel_idx] > 0
            bbox = _get_bbox(lowres_region, margin)
            if bbox is None:
                continue

            target_slices, resampled_logits = _resample_channel_crop(
                predicted_logits[channel_idx], new_shape, bbox[0], bbox[1], order=1
            )
            if target_slices is not None:
                segmentation[target_slices][resampled_logits > 0] = class_id
        return segmentation

    background_logits = resampling_fn_probabilities(
        predicted_logits[:1], new_shape, current_spacing, target_spacing
    )
    best_score = np.asarray(background_logits[0])
    segmentation = np.zeros(new_shape, dtype=segmentation_dtype)

    lowres_segmentation = np.argmax(predicted_logits, axis=0)
    for class_id in np.unique(lowres_segmentation):
        class_id = int(class_id)
        if class_id == 0 or class_id >= predicted_logits.shape[0]:
            continue

        bbox = _get_bbox(lowres_segmentation == class_id, margin)
        if bbox is None:
            continue

        target_slices, resampled_logits = _resample_channel_crop(
            predicted_logits[class_id], new_shape, bbox[0], bbox[1], order=1
        )
        if target_slices is None:
            continue

        update_mask = resampled_logits > best_score[target_slices]
        segmentation[target_slices][update_mask] = class_id
        best_score[target_slices][update_mask] = resampled_logits[update_mask]

    return segmentation


def convert_predicted_logits_to_segmentation_with_correct_shape(
        predicted_logits: Union[torch.Tensor, np.ndarray],
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        label_manager: LabelManager,
        properties_dict: dict,
        return_probabilities: bool = False,
        num_threads_torch: int = default_num_processes,
        use_cropped_logits_resampling: bool = False):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    spacing_transposed = [properties_dict["spacing"][i] for i in plans_manager.transpose_forward]
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == len(properties_dict["shape_after_cropping_and_before_resampling"]) else \
        [spacing_transposed[0], *configuration_manager.spacing]
    target_shape = properties_dict["shape_after_cropping_and_before_resampling"]
    target_spacing = [properties_dict["spacing"][i] for i in plans_manager.transpose_forward]

    try:
        if use_cropped_logits_resampling and not return_probabilities:
            segmentation = _convert_logits_to_segmentation_with_cropped_resampling(
                predicted_logits, label_manager, target_shape, current_spacing, target_spacing,
                configuration_manager.resampling_fn_probabilities
            )
        else:
            if use_cropped_logits_resampling and return_probabilities:
                warnings.warn(
                    "Cropped logits resampling is ignored when return_probabilities/save_probabilities is enabled "
                    "because exporting probabilities requires the full resampled probability tensor.",
                    RuntimeWarning
                )
            predicted_logits = configuration_manager.resampling_fn_probabilities(
                predicted_logits, target_shape, current_spacing, target_spacing
            )
            if not return_probabilities:
                segmentation = label_manager.convert_logits_to_segmentation(predicted_logits)
            else:
                predicted_probabilities = label_manager.apply_inference_nonlin(predicted_logits)
                segmentation = label_manager.convert_probabilities_to_segmentation(predicted_probabilities)
        del predicted_logits

        segmentation_reverted_cropping = np.zeros(
            properties_dict["shape_before_cropping"],
            dtype=np.uint8 if len(label_manager.foreground_labels) < 255 else np.uint16
        )
        segmentation_reverted_cropping = insert_crop_into_image(
            segmentation_reverted_cropping, segmentation, properties_dict["bbox_used_for_cropping"]
        )
        del segmentation

        if isinstance(segmentation_reverted_cropping, torch.Tensor):
            segmentation_reverted_cropping = segmentation_reverted_cropping.cpu().numpy()

        segmentation_reverted_cropping = segmentation_reverted_cropping.transpose(plans_manager.transpose_backward)
        if return_probabilities:
            predicted_probabilities = label_manager.revert_cropping_on_probabilities(
                predicted_probabilities,
                properties_dict["bbox_used_for_cropping"],
                properties_dict["shape_before_cropping"]
            )
            predicted_probabilities = predicted_probabilities.cpu().numpy()
            predicted_probabilities = predicted_probabilities.transpose(
                [0] + [i + 1 for i in plans_manager.transpose_backward]
            )
            return segmentation_reverted_cropping, predicted_probabilities
        return segmentation_reverted_cropping
    finally:
        torch.set_num_threads(old_threads)


def export_prediction_from_logits(predicted_array_or_file: Union[np.ndarray, torch.Tensor],
                                  properties_dict: dict,
                                  configuration_manager: ConfigurationManager,
                                  plans_manager: PlansManager,
                                  dataset_json_dict_or_file: Union[dict, str],
                                  output_file_truncated: str,
                                  save_probabilities: bool = False,
                                  num_threads_torch: int = default_num_processes,
                                  use_cropped_logits_resampling: bool = False):
    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    ret = convert_predicted_logits_to_segmentation_with_correct_shape(
        predicted_array_or_file, plans_manager, configuration_manager, label_manager, properties_dict,
        return_probabilities=save_probabilities, num_threads_torch=num_threads_torch,
        use_cropped_logits_resampling=use_cropped_logits_resampling
    )
    del predicted_array_or_file

    if save_probabilities:
        segmentation_final, probabilities_final = ret
        np.savez_compressed(output_file_truncated + ".npz", probabilities=probabilities_final)
        save_pickle(properties_dict, output_file_truncated + ".pkl")
        del probabilities_final, ret
    else:
        segmentation_final = ret
        del ret

    rw = plans_manager.image_reader_writer_class()
    rw.write_seg(segmentation_final, output_file_truncated + dataset_json_dict_or_file["file_ending"], properties_dict)


def predict_from_files(self,
                       list_of_lists_or_source_folder: Union[str, List[List[str]]],
                       output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
                       save_probabilities: bool = False,
                       overwrite: bool = True,
                       num_processes_preprocessing: int = default_num_processes,
                       num_processes_segmentation_export: int = default_num_processes,
                       folder_with_segs_from_prev_stage: str = None,
                       num_parts: int = 1,
                       part_id: int = 0,
                       use_cropped_logits_resampling: bool = False):
    import nnunetv2.inference.predict_from_raw_data as prd

    assert part_id <= num_parts, ("Part ID must be smaller than num_parts. Remember that we start counting with 0. "
                                  "So if there are 3 parts then valid part IDs are 0, 1, 2")
    if isinstance(output_folder_or_list_of_truncated_output_files, str):
        output_folder = output_folder_or_list_of_truncated_output_files
    elif isinstance(output_folder_or_list_of_truncated_output_files, list):
        output_folder = os.path.dirname(output_folder_or_list_of_truncated_output_files[0])
    else:
        output_folder = None

    if output_folder is not None:
        my_init_kwargs = {}
        for k in inspect.signature(self.predict_from_files).parameters.keys():
            my_init_kwargs[k] = locals()[k]
        my_init_kwargs = deepcopy(my_init_kwargs)
        prd.recursive_fix_for_json_export(my_init_kwargs)
        prd.maybe_mkdir_p(output_folder)
        prd.save_json(my_init_kwargs, prd.join(output_folder, "predict_from_raw_data_args.json"))
        prd.save_json(self.dataset_json, prd.join(output_folder, "dataset.json"), sort_keys=False)
        prd.save_json(self.plans_manager.plans, prd.join(output_folder, "plans.json"), sort_keys=False)

    if self.configuration_manager.previous_stage_name is not None:
        assert folder_with_segs_from_prev_stage is not None, \
            f"The requested configuration is a cascaded network. It requires the segmentations of the previous " \
            f"stage ({self.configuration_manager.previous_stage_name}) as input. Please provide the folder where" \
            f" they are located via folder_with_segs_from_prev_stage"

    list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files = \
        self._manage_input_and_output_lists(
            list_of_lists_or_source_folder,
            output_folder_or_list_of_truncated_output_files,
            folder_with_segs_from_prev_stage, overwrite, part_id, num_parts,
            save_probabilities
        )
    if len(list_of_lists_or_source_folder) == 0:
        return

    data_iterator = self._internal_get_data_iterator_from_lists_of_filenames(
        list_of_lists_or_source_folder,
        seg_from_prev_stage_files,
        output_filename_truncated,
        num_processes_preprocessing
    )

    return self.predict_from_data_iterator(
        data_iterator, save_probabilities, num_processes_segmentation_export,
        use_cropped_logits_resampling
    )


def predict_from_data_iterator(self,
                               data_iterator,
                               save_probabilities: bool = False,
                               num_processes_segmentation_export: int = default_num_processes,
                               use_cropped_logits_resampling: bool = False):
    import nnunetv2.inference.predict_from_raw_data as prd

    with prd.multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
        worker_list = [i for i in export_pool._pool]
        r = []
        for preprocessed in data_iterator:
            data = preprocessed["data"]
            if isinstance(data, str):
                delfile = data
                data = torch.from_numpy(np.load(data))
                os.remove(delfile)

            ofile = preprocessed["ofile"]
            if ofile is not None:
                print(f"\nPredicting {os.path.basename(ofile)}:")
            else:
                print(f"\nPredicting image of shape {data.shape}:")

            print(f"perform_everything_on_device: {self.perform_everything_on_device}")
            properties = preprocessed["data_properties"]

            proceed = not prd.check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)
            while not proceed:
                sleep(0.1)
                proceed = not prd.check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)

            prediction = self.predict_logits_from_preprocessed_data(data).cpu().detach().numpy()

            if ofile is not None:
                print("sending off prediction to background worker for resampling and export")
                r.append(
                    export_pool.apply_async(
                        export_prediction_from_logits,
                        (prediction, properties, self.configuration_manager, self.plans_manager,
                         self.dataset_json, ofile, save_probabilities, default_num_processes,
                         use_cropped_logits_resampling)
                    )
                )
            else:
                print("sending off prediction to background worker for resampling")
                r.append(
                    export_pool.apply_async(
                        convert_predicted_logits_to_segmentation_with_correct_shape,
                        (prediction, self.plans_manager, self.configuration_manager, self.label_manager,
                         properties, save_probabilities, default_num_processes, use_cropped_logits_resampling)
                    )
                )
            if ofile is not None:
                print(f"done with {os.path.basename(ofile)}")
            else:
                print(f"\nDone with image of shape {data.shape}:")

        print("GPU prediction completed. Waiting for remaining segmentation exports to finish...")
        ret = [None] * len(r)
        with prd.tqdm(desc="Collecting results", total=len(r), disable=not self.allow_tqdm) as pbar:
            for i, result in enumerate(r):
                while True:
                    all_alive = all([j.is_alive() for j in worker_list])
                    if not all_alive:
                        raise RuntimeError("Segmentation export worker died. It was likely killed by "
                                           "your OS because of insufficient available CPU RAM.")
                    try:
                        ret[i] = result.get(timeout=0.1)
                        break
                    except prd.multiprocessing.TimeoutError:
                        pass
                pbar.update()
        print("Segmentation export complete.")

    if isinstance(data_iterator, prd.MultiThreadedAugmenter):
        data_iterator._finish()

    prd.compute_gaussian.cache_clear()
    prd.empty_cache(self.device)
    return ret


def predict_from_list_of_npy_arrays(self,
                                    image_or_list_of_images: Union[np.ndarray, List[np.ndarray]],
                                    segs_from_prev_stage_or_list_of_segs_from_prev_stage,
                                    properties_or_list_of_properties: Union[dict, List[dict]],
                                    truncated_ofname: Union[str, List[str], None],
                                    num_processes: int = 3,
                                    save_probabilities: bool = False,
                                    num_processes_segmentation_export: int = default_num_processes,
                                    use_cropped_logits_resampling: bool = False):
    iterator = self.get_data_iterator_from_raw_npy_data(
        image_or_list_of_images,
        segs_from_prev_stage_or_list_of_segs_from_prev_stage,
        properties_or_list_of_properties,
        truncated_ofname,
        num_processes
    )
    return self.predict_from_data_iterator(
        iterator, save_probabilities, num_processes_segmentation_export,
        use_cropped_logits_resampling
    )


def predict_single_npy_array(self,
                             input_image: np.ndarray,
                             image_properties: dict,
                             segmentation_previous_stage: np.ndarray = None,
                             output_file_truncated: str = None,
                             save_or_return_probabilities: bool = False,
                             use_cropped_logits_resampling: bool = False):
    import nnunetv2.inference.predict_from_raw_data as prd

    ppa = prd.PreprocessAdapterFromNpy(
        [input_image], [segmentation_previous_stage], [image_properties],
        [output_file_truncated],
        self.plans_manager, self.dataset_json, self.configuration_manager,
        num_threads_in_multithreaded=1, verbose=self.verbose
    )
    if self.verbose:
        print("preprocessing")
    dct = next(ppa)

    if self.verbose:
        print("predicting")
    predicted_logits = self.predict_logits_from_preprocessed_data(dct["data"]).cpu()

    if self.verbose:
        print("resampling to original shape")
    if output_file_truncated is not None:
        export_prediction_from_logits(
            predicted_logits, dct["data_properties"], self.configuration_manager,
            self.plans_manager, self.dataset_json, output_file_truncated,
            save_or_return_probabilities,
            use_cropped_logits_resampling=use_cropped_logits_resampling
        )
    else:
        ret = convert_predicted_logits_to_segmentation_with_correct_shape(
            predicted_logits, self.plans_manager, self.configuration_manager,
            self.label_manager, dct["data_properties"],
            return_probabilities=save_or_return_probabilities,
            use_cropped_logits_resampling=use_cropped_logits_resampling
        )
        if save_or_return_probabilities:
            return ret[0], ret[1]
        return ret


def predict_from_files_sequential(self,
                                  list_of_lists_or_source_folder: Union[str, List[List[str]]],
                                  output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
                                  save_probabilities: bool = False,
                                  overwrite: bool = True,
                                  folder_with_segs_from_prev_stage: str = None,
                                  use_cropped_logits_resampling: bool = False):
    import nnunetv2.inference.predict_from_raw_data as prd

    if isinstance(output_folder_or_list_of_truncated_output_files, str):
        output_folder = output_folder_or_list_of_truncated_output_files
    elif isinstance(output_folder_or_list_of_truncated_output_files, list):
        output_folder = os.path.dirname(output_folder_or_list_of_truncated_output_files[0])
        if len(output_folder) == 0:
            output_folder = os.path.curdir
    else:
        output_folder = None

    if output_folder is not None:
        my_init_kwargs = {}
        for k in inspect.signature(self.predict_from_files_sequential).parameters.keys():
            my_init_kwargs[k] = locals()[k]
        my_init_kwargs = deepcopy(my_init_kwargs)
        prd.recursive_fix_for_json_export(my_init_kwargs)
        prd.save_json(my_init_kwargs, prd.join(output_folder, "predict_from_raw_data_args.json"))
        prd.save_json(self.dataset_json, prd.join(output_folder, "dataset.json"), sort_keys=False)
        prd.save_json(self.plans_manager.plans, prd.join(output_folder, "plans.json"), sort_keys=False)

    if self.configuration_manager.previous_stage_name is not None:
        assert folder_with_segs_from_prev_stage is not None, \
            f"The requested configuration is a cascaded network. It requires the segmentations of the previous " \
            f"stage ({self.configuration_manager.previous_stage_name}) as input. Please provide the folder where" \
            f" they are located via folder_with_segs_from_prev_stage"

    list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files = \
        self._manage_input_and_output_lists(
            list_of_lists_or_source_folder,
            output_folder_or_list_of_truncated_output_files,
            folder_with_segs_from_prev_stage, overwrite, 0, 1,
            save_probabilities
        )
    if len(list_of_lists_or_source_folder) == 0:
        return

    label_manager = self.plans_manager.get_label_manager(self.dataset_json)
    preprocessor = self.configuration_manager.preprocessor_class(verbose=self.verbose)

    if output_filename_truncated is None:
        output_filename_truncated = [None] * len(list_of_lists_or_source_folder)
    if seg_from_prev_stage_files is None:
        seg_from_prev_stage_files = [None] * len(list_of_lists_or_source_folder)

    ret = []
    for li, of, sps in zip(list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files):
        data, seg, data_properties = preprocessor.run_case(
            li,
            sps,
            self.plans_manager,
            self.configuration_manager,
            self.dataset_json
        )
        if folder_with_segs_from_prev_stage is not None:
            seg_onehot = prd.convert_labelmap_to_one_hot(seg[0], label_manager.foreground_labels, data.dtype)
            data = np.vstack((data, seg_onehot))

        print(f"perform_everything_on_device: {self.perform_everything_on_device}")

        prediction = self.predict_logits_from_preprocessed_data(torch.from_numpy(data)).cpu()

        if of is not None:
            export_prediction_from_logits(
                prediction, data_properties, self.configuration_manager, self.plans_manager,
                self.dataset_json, of, save_probabilities,
                use_cropped_logits_resampling=use_cropped_logits_resampling
            )
        else:
            ret.append(
                convert_predicted_logits_to_segmentation_with_correct_shape(
                    prediction, self.plans_manager, self.configuration_manager, self.label_manager,
                    data_properties, save_probabilities,
                    use_cropped_logits_resampling=use_cropped_logits_resampling
                )
            )

    prd.compute_gaussian.cache_clear()
    prd.empty_cache(self.device)
    return ret


def patch_nnunet_cropped_logits_resampling() -> bool:
    """
    Add TotalSegmentator's cropped-logits export path to upstream nnU-Net at runtime.

    Returns True when this function installed the patch. If the installed nnU-Net
    already provides the keyword natively, no patch is applied and False is returned.
    """
    import nnunetv2.inference.export_prediction as export_module
    import nnunetv2.inference.predict_from_raw_data as predict_module

    predictor_cls = predict_module.nnUNetPredictor
    if getattr(predictor_cls, _PATCH_MARKER, False):
        return False

    if _supports_keyword_argument(predictor_cls.predict_from_files, "use_cropped_logits_resampling"):
        return False

    export_module.convert_predicted_logits_to_segmentation_with_correct_shape = \
        convert_predicted_logits_to_segmentation_with_correct_shape
    export_module.export_prediction_from_logits = export_prediction_from_logits
    predict_module.convert_predicted_logits_to_segmentation_with_correct_shape = \
        convert_predicted_logits_to_segmentation_with_correct_shape
    predict_module.export_prediction_from_logits = export_prediction_from_logits

    predictor_cls.predict_from_files = predict_from_files
    predictor_cls.predict_from_data_iterator = predict_from_data_iterator
    predictor_cls.predict_from_list_of_npy_arrays = predict_from_list_of_npy_arrays
    predictor_cls.predict_single_npy_array = predict_single_npy_array
    predictor_cls.predict_from_files_sequential = predict_from_files_sequential
    setattr(predictor_cls, _PATCH_MARKER, True)
    return True
