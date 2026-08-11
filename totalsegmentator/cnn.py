import pickle
import warnings
from collections.abc import Mapping
from pathlib import Path

import nibabel as nib
import numpy as np

from totalsegmentator.config import get_weights_dir
from totalsegmentator.libs import download_pretrained_weights
from totalsegmentator.resampling import change_spacing


DEFAULT_BODY_STATS_CNN_ROOT_DIR = get_weights_dir() / "lightning_models"
DEFAULT_BODY_STATS_CNN_DIRS = {
    "mr": DEFAULT_BODY_STATS_CNN_ROOT_DIR / "mr_all_ext_splitOrig_noise_can1_huber",
    "ct": DEFAULT_BODY_STATS_CNN_ROOT_DIR / "ct_all_ext_splitOrig_noise_can1_huber",
}
BODY_STATS_CNN_DOWNLOAD_TASKS = {
    "mr": "body_stats_mr",
    "ct": "body_stats_ct",
}

CNN_FALLBACK_CROP_SIZE = {
    "ct": {2: (240, 240), 3: (240, 240, 240)},
    "mr": {2: (210, 210), 3: (210, 210, 150)},
}
CNN_TARGET_SPACING_MM = 2.0
CNN_TTA_FLIP_AXES = (0, 1, 2)
MANUFACTURER_MAP = {
    "siemens": 0,
    "philips": 1,
    "ge": 2,
    "toshiba": 3,
    "canon": 4,
    "other": 5,
}
MR_SEQUENCE_MAP = {
    "t1": 0,
    "pd": 1,
    "t2": 2,
    "flair": 3,
    "stir": 4,
    "t2star": 5,
    "swi": 6,
    "dwi": 7,
    "mra": 8,
    "other": 9,
}
VERTEBRA_NAMES = (
    [f"C{i}" for i in range(1, 8)]
    + [f"T{i}" for i in range(1, 13)]
    + [f"L{i}" for i in range(1, 6)]
)
VERTEBRA_MAP = {
    f"vertebrae_{name}": position
    for position, name in enumerate(VERTEBRA_NAMES, start=1)
}
CNN_MODEL_TARGET_ORDER = ["PatientWeight", "PatientSize", "PatientAge", "PatientSex_01"]
CNN_MODEL_TARGET_ORDERS = {
    "mr": [
        *CNN_MODEL_TARGET_ORDER,
        "Manufacturer_int",
        "contrast",
        "verte_upper",
        "verte_lower",
        "noise_p75",
        "mr_sequence_int",
    ],
    "ct": [
        *CNN_MODEL_TARGET_ORDER,
        "Manufacturer_int",
        "KVP",
        "XRayTubeCurrent",
        "ConvolutionKernel",
        "contrast",
        "pi_time",
        "verte_upper",
        "verte_lower",
        "noise_p75",
    ],
}
CNN_TARGET_SPECS = {
    "weight": {"training_name": "PatientWeight", "unit": "kg"},
    "size": {"training_name": "PatientSize", "unit": "cm"},
    "age": {"training_name": "PatientAge", "unit": "years"},
    "sex": {"training_name": "PatientSex_01", "unit": None},
    "manufacturer": {"training_name": "Manufacturer_int", "unit": None},
    "contrast": {"training_name": "contrast", "unit": None},
    "verte_upper": {"training_name": "verte_upper", "unit": None},
    "verte_lower": {"training_name": "verte_lower", "unit": None},
    "noise": {"training_name": "noise_p75", "unit": None},
    "mr_sequence": {"training_name": "mr_sequence_int", "unit": None},
    "kvp": {"training_name": "KVP", "unit": "kV"},
    "xray_tube_current": {"training_name": "XRayTubeCurrent", "unit": "mA"},
    "convolution_kernel": {"training_name": "ConvolutionKernel", "unit": None},
    "pi_time": {"training_name": "pi_time", "unit": "seconds"},
}
CNN_TRAINING_NAME_TO_TARGET = {
    spec["training_name"]: target for target, spec in CNN_TARGET_SPECS.items()
}


def _get_slice_indices(mid_idx: int, nr_slices: int, offset: int, size: int) -> list[int]:
    if nr_slices < 1:
        raise ValueError(f"nr_slices must be >= 1, got {nr_slices}")

    if nr_slices == 1:
        slice_indices = [mid_idx]
    else:
        slice_indices = np.round(
            np.linspace(mid_idx - offset, mid_idx + offset, nr_slices)
        ).astype(int).tolist()
    return np.clip(slice_indices, 0, size - 1).astype(int).tolist()


def _get_even_slice_indices(
    nr_slices: int, size: int, rand_int: int = 0, edge_fraction: float = 0.1
) -> list[int]:
    if nr_slices < 1:
        raise ValueError(f"nr_slices must be >= 1, got {nr_slices}")
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}")

    axis_max = size - 1
    lower = int(round(axis_max * edge_fraction))
    upper = int(round(axis_max * (1 - edge_fraction)))

    if nr_slices == 1:
        slice_indices = [int(round(axis_max / 2))]
    else:
        slice_indices = np.round(np.linspace(lower, upper, nr_slices)).astype(int).tolist()

    return np.clip(np.array(slice_indices) + rand_int, lower, upper).astype(int).tolist()


def _extract_multi_orientation_slices(
    img_data: np.ndarray, nr_slices: int, offset: int, sample_evenly: bool = False
) -> list[np.ndarray]:
    """Mirror multi_orientation=True from the deterministic validation dataset."""
    axis_offsets = np.array(
        [0 if axis_size <= 1 else max(1, axis_size // 8) for axis_size in img_data.shape],
        dtype=int,
    )
    axis_offsets = np.minimum(axis_offsets, offset)
    mid = (np.array(img_data.shape) / 2).astype(int)
    if sample_evenly:
        x_indices = _get_even_slice_indices(nr_slices, img_data.shape[0])
        y_indices = _get_even_slice_indices(nr_slices, img_data.shape[1])
        z_indices = _get_even_slice_indices(nr_slices, img_data.shape[2])
    else:
        x_indices = _get_slice_indices(mid[0], nr_slices, axis_offsets[0], img_data.shape[0])
        y_indices = _get_slice_indices(mid[1], nr_slices, axis_offsets[1], img_data.shape[1])
        z_indices = _get_slice_indices(mid[2], nr_slices, axis_offsets[2], img_data.shape[2])

    x_slices = img_data[x_indices, :, :]
    y_slices = img_data[:, y_indices, :].transpose(1, 0, 2)
    z_slices = img_data[:, :, z_indices].transpose(2, 0, 1)

    return [*x_slices, *y_slices, *z_slices]


def _extract_single_orientation_slices(
    img_data: np.ndarray,
    nr_slices: int,
    offset: int,
    orientation: str,
    sample_evenly: bool = False,
) -> list[np.ndarray]:
    mid = (np.array(img_data.shape) / 2).astype(int)

    if orientation == "x":
        slice_indices = (
            _get_even_slice_indices(nr_slices, img_data.shape[0])
            if sample_evenly
            else _get_slice_indices(mid[0], nr_slices, offset, img_data.shape[0])
        )
        return list(img_data[slice_indices, :, :])
    if orientation == "y":
        slice_indices = (
            _get_even_slice_indices(nr_slices, img_data.shape[1])
            if sample_evenly
            else _get_slice_indices(mid[1], nr_slices, offset, img_data.shape[1])
        )
        return list(img_data[:, slice_indices, :].transpose(1, 0, 2))
    if orientation == "z":
        slice_indices = (
            _get_even_slice_indices(nr_slices, img_data.shape[2])
            if sample_evenly
            else _get_slice_indices(mid[2], nr_slices, offset, img_data.shape[2])
        )
        return list(img_data[:, :, slice_indices].transpose(2, 0, 1))
    raise ValueError(f"Unsupported slice orientation: {orientation}")


def _require_hparam(hparams: dict | None, key: str):
    if not hparams or key not in hparams:
        raise KeyError(f"Checkpoint is missing required hyperparameter: {key}")
    return hparams[key]


def _extract_slices(img_data: np.ndarray, hparams: dict | None) -> list[np.ndarray]:
    nr_slices = int(_require_hparam(hparams, "nr_slices"))
    multi_orientation = bool(_require_hparam(hparams, "multi_orientation"))
    orientation = hparams.get("slice_orientation", "z") if hparams else "z"
    sample_evenly = bool(hparams.get("sample_evenly", False)) if hparams else False
    orientation_to_axis = {"x": 0, "y": 1, "z": 2}
    if orientation not in orientation_to_axis:
        raise ValueError(f"Unsupported slice orientation: {orientation}")

    offset = int(img_data.shape[orientation_to_axis[orientation]] / 8)
    if multi_orientation:
        return _extract_multi_orientation_slices(
            img_data, nr_slices, offset, sample_evenly
        )
    return _extract_single_orientation_slices(
        img_data, nr_slices, offset, orientation, sample_evenly
    )


def _apply_training_unpack_slice_stride(
    img_data: np.ndarray, hparams: dict | None
) -> np.ndarray:
    """
    Mirror training-time .npy unpacking when it saved only every nth slice.

    Some 2D models were trained with unpack_to_npy=1 and unpack_slice_stride>1,
    which means the dataset loader saw a volume with fewer slices along the
    configured orientation. During inference we start from the full NIfTI volume,
    so we need to apply the same stride before choosing center slices. Models
    trained with unpack_to_npy=0 are left unchanged.
    """
    if not hparams or not hparams.get("unpack_to_npy", False):
        return img_data
    if int(hparams.get("dim", 2)) != 2 or hparams.get("slice_subset", False):
        return img_data

    slice_stride = int(hparams.get("unpack_slice_stride", 1))
    if slice_stride <= 1:
        return img_data

    orientation = "z" if hparams.get("tiles", False) else hparams.get("slice_orientation", "z")
    if orientation == "x":
        return img_data[::slice_stride, :, :]
    if orientation == "y":
        return img_data[:, ::slice_stride, :]
    if orientation == "z":
        return img_data[:, :, ::slice_stride]
    raise ValueError(f"Unsupported slice orientation: {orientation}")


def _center_pad_or_crop_nd(img_data: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    target_shape = tuple(int(size) for size in target_shape)
    if img_data.ndim != len(target_shape):
        raise ValueError(
            f"Can not crop/pad array with shape {img_data.shape} to {target_shape}."
        )

    src_slices = []
    dst_slices = []
    for src_size, target_size in zip(img_data.shape, target_shape):
        src_start = max((src_size - target_size) // 2, 0)
        src_end = src_start + min(src_size, target_size)
        dst_start = max((target_size - (src_end - src_start)) // 2, 0)
        dst_end = dst_start + (src_end - src_start)
        src_slices.append(slice(src_start, src_end))
        dst_slices.append(slice(dst_start, dst_end))

    out = np.zeros(target_shape, dtype=img_data.dtype)
    out[tuple(dst_slices)] = img_data[tuple(src_slices)]
    return out


def _normalize_per_channel(img_stack: np.ndarray, nonzero: bool = False) -> np.ndarray:
    img_stack = img_stack.astype(np.float32, copy=False)
    normalized = np.empty_like(img_stack, dtype=np.float32)

    for channel_idx in range(img_stack.shape[0]):
        channel = img_stack[channel_idx]
        mask = channel != 0 if nonzero else np.ones(channel.shape, dtype=bool)
        if not np.any(mask):
            normalized[channel_idx] = channel
            continue

        mean = float(channel[mask].mean())
        std = float(channel[mask].std())
        channel_normalized = channel.copy()
        if std < 1e-8:
            channel_normalized[mask] = channel[mask] - mean
        else:
            channel_normalized[mask] = (channel[mask] - mean) / std
        normalized[channel_idx] = channel_normalized

    return normalized


def _scale_intensity_to_minus_one_one(img_stack: np.ndarray) -> np.ndarray:
    img_stack = img_stack.astype(np.float32, copy=False)
    min_value = float(img_stack.min())
    max_value = float(img_stack.max())
    if max_value - min_value < 1e-8:
        return np.zeros_like(img_stack, dtype=np.float32)
    return ((img_stack - min_value) / (max_value - min_value) * 2.0 - 1.0).astype(np.float32)


def _normalize_with_training_hparams(img_stack: np.ndarray, hparams: dict | None) -> np.ndarray:
    img_stack = img_stack.astype(np.float32, copy=False)
    if not hparams or not hparams.get("normalize", True):
        return img_stack

    if hparams.get("norm_global", False):
        global_mean = _as_channel_stats(
            hparams["global_mean"], img_stack.shape[0], img_stack.ndim, "global_mean"
        )
        global_std = _as_channel_stats(
            hparams["global_std"], img_stack.shape[0], img_stack.ndim, "global_std"
        )
        global_std = np.where(np.abs(global_std) < 1e-8, 1.0, global_std)
        return (img_stack - global_mean) / global_std

    if hparams.get("norm_channel_wise", True):
        return _normalize_per_channel(
            img_stack, nonzero=bool(hparams.get("norm_ignore_zero", False))
        )

    mean = float(img_stack.mean())
    std = float(img_stack.std())
    if std < 1e-8:
        return img_stack - mean
    return (img_stack - mean) / std


def _as_channel_stats(
    values, nr_channels: int, img_stack_ndim: int, name: str
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 1 and nr_channels != 1:
        values = np.repeat(values, nr_channels)
    if values.size != nr_channels:
        raise ValueError(
            f"Expected {nr_channels} values for {name}, got {values.size}: {values.tolist()}"
        )
    return values.reshape((nr_channels,) + (1,) * (img_stack_ndim - 1))


def _apply_validation_intensity_transforms(
    img_stack: np.ndarray, hparams: dict | None
) -> np.ndarray:
    if hparams and hparams.get("clip", False):
        img_stack = np.clip(img_stack, hparams["clip_low"], hparams["clip_high"])
    if hparams and hparams.get("daug_scale_itensity", False):
        img_stack = _scale_intensity_to_minus_one_one(img_stack)
    return _normalize_with_training_hparams(img_stack, hparams)


def _get_crop_size(hparams: dict | None, modality: str) -> tuple[int, ...]:
    dim = int(hparams.get("dim", 2)) if hparams else 2
    if hparams and hparams.get("crop_size") is not None:
        crop_size = tuple(int(size) for size in hparams["crop_size"])
    else:
        crop_size = CNN_FALLBACK_CROP_SIZE[modality][dim]

    if len(crop_size) != dim:
        raise ValueError(f"Expected {dim}D crop_size, got {crop_size}.")
    return crop_size


def _prepare_image_tensor(
    img: nib.Nifti1Image, modality: str, hparams: dict | None = None,
    skip_canonical: bool = False
):
    if not skip_canonical:
        img = nib.as_closest_canonical(img)
    img = change_spacing(img, CNN_TARGET_SPACING_MM, dtype=np.float32, order=1)
    img_data = np.asarray(img.dataobj, dtype=np.float32)
    dim = int(hparams.get("dim", 2)) if hparams else 2
    crop_size = _get_crop_size(hparams, modality)

    if dim == 3:
        img_stack = img_data[None, ...]
        img_stack = _apply_validation_intensity_transforms(img_stack, hparams)
        img_stack = np.stack(
            [_center_pad_or_crop_nd(channel, crop_size) for channel in img_stack],
            axis=0,
        )
    elif dim == 2:
        img_data = _apply_training_unpack_slice_stride(img_data, hparams)
        slices = _extract_slices(img_data, hparams)
        img_stack = np.stack(
            [_center_pad_or_crop_nd(slice_2d, crop_size) for slice_2d in slices],
            axis=0,
        )
        img_stack = _apply_validation_intensity_transforms(img_stack, hparams)
        img_stack = np.stack(
            [_center_pad_or_crop_nd(slice_2d, crop_size) for slice_2d in img_stack],
            axis=0,
        )
    else:
        raise ValueError(f"Unsupported CNN input dimensionality: {dim}")

    img_stack = img_stack.astype(np.float32, copy=False)

    try:
        import torch
    except ImportError as exc:
        raise ImportError("CNN body-stats inference requires PyTorch to be installed.") from exc

    return torch.from_numpy(img_stack[None, ...])


def _flip_nifti_data(img: nib.Nifti1Image, axis: int) -> nib.Nifti1Image:
    img_data = np.asanyarray(img.dataobj)
    flipped_data = np.flip(img_data, axis=axis).copy()
    return nib.Nifti1Image(flipped_data, img.affine, img.header)


def _get_inference_image_variants(
    img: nib.Nifti1Image,
    skip_canonical: bool = False,
    test_time_augmentation: bool = False,
) -> list[tuple[nib.Nifti1Image, bool]]:
    if not test_time_augmentation:
        return [(img, skip_canonical)]

    variants = []
    for axis in CNN_TTA_FLIP_AXES:
        flipped_img = _flip_nifti_data(img, axis)
        variants.append((flipped_img, False))
        variants.append((flipped_img, True))
    return variants


def _prepare_image_tensors(
    img: nib.Nifti1Image,
    modality: str,
    hparams: dict | None = None,
    skip_canonical: bool = False,
    test_time_augmentation: bool = False,
):
    img_tensors = [
        _prepare_image_tensor(
            variant_img, modality=modality, hparams=hparams, skip_canonical=variant_skip_canonical
        )
        for variant_img, variant_skip_canonical in _get_inference_image_variants(
            img, skip_canonical=skip_canonical, test_time_augmentation=test_time_augmentation
        )
    ]

    if len(img_tensors) == 1:
        return img_tensors[0]

    try:
        import torch
    except ImportError as exc:
        raise ImportError("CNN body-stats inference requires PyTorch to be installed.") from exc

    return torch.cat(img_tensors, dim=0)


def _resolve_device(device):
    try:
        import torch
    except ImportError as exc:
        raise ImportError("CNN body-stats inference requires PyTorch to be installed.") from exc

    if isinstance(device, torch.device):
        return device
    if device == "gpu":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str) and device.startswith("gpu:"):
        gpu_idx = int(device.split(":", maxsplit=1)[1])
        if torch.cuda.is_available() and gpu_idx < torch.cuda.device_count():
            return torch.device(f"cuda:{gpu_idx}")
        return torch.device("cpu")
    if device == "mps":
        return torch.device("mps")
    return torch.device("cpu")


def _find_fold_checkpoint(model_dir: Path, fold_idx: int) -> Path:
    ckpt_dir = model_dir / f"version_{fold_idx}" / "checkpoints"
    ckpt_files = sorted(ckpt_dir.glob("epoch*.ckpt"))
    if len(ckpt_files) == 1:
        return ckpt_files[0]
    if len(ckpt_files) > 1:
        raise FileNotFoundError(
            f"Expected exactly one checkpoint in {ckpt_dir}, found {len(ckpt_files)}."
        )

    last_ckpt = ckpt_dir / "last.ckpt"
    if last_ckpt.exists():
        return last_ckpt

    raise FileNotFoundError(
        f"Expected exactly one checkpoint in {ckpt_dir}, found 0."
    )


def _load_monai_meta_tensor():
    try:
        from monai.data.meta_tensor import MetaTensor
    except ImportError as exc:
        raise ImportError(
            "CNN body-stats inference requires monai to load the pretrained checkpoints. "
            "Install it with 'pip install monai'."
        ) from exc
    return MetaTensor


def _load_torch_checkpoint(ckpt_file: Path, weights_only: bool):
    import torch

    def _load(**kwargs):
        try:
            return torch.load(ckpt_file, map_location="cpu", **kwargs)
        except ModuleNotFoundError as exc:
            if exc.name == "monai":
                raise ImportError(
                    "CNN body-stats inference requires monai to load the pretrained checkpoints. "
                    "Install it with 'pip install monai'."
                ) from exc
            raise

    try:
        return _load(weights_only=weights_only)
    except pickle.UnpicklingError:
        if not weights_only:
            raise

        # Some Lightning checkpoints include MONAI objects such as MetaTensor in metadata.
        # Allowlist them so we can keep using the safer weights_only=True path when possible.
        MetaTensor = _load_monai_meta_tensor()
        try:
            with torch.serialization.safe_globals([MetaTensor]):
                return _load(weights_only=True)
        except Exception:
            # Fall back to the legacy loading mode for trusted local checkpoints.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="You are using `torch.load` with `weights_only=False`",
                    category=FutureWarning,
                )
                return _load(weights_only=False)
    except TypeError:
        # Support older PyTorch versions which do not expose weights_only yet.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="You are using `torch.load` with `weights_only=False`",
                category=FutureWarning,
            )
            return _load()


class _MultiHeadOutput:
    def __new__(cls, in_features: int, nr_heads: int, dropout: float):
        import torch

        class MultiHeadOutput(torch.nn.Module):
            def __init__(self):
                super().__init__()
                hidden_features = min(128, in_features)
                self.heads = torch.nn.ModuleList([
                    torch.nn.Sequential(
                        torch.nn.Linear(in_features, hidden_features),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Dropout(dropout),
                        torch.nn.Linear(hidden_features, 1),
                    )
                    for _ in range(nr_heads)
                ])

            def forward(self, x):
                return torch.cat([head(x) for head in self.heads], dim=1)

        return MultiHeadOutput()


def _get_checkpoint_hparams(checkpoint: dict, ckpt_file: Path) -> dict:
    if "hyper_parameters" not in checkpoint:
        raise KeyError(
            f"Checkpoint {ckpt_file} does not contain a 'hyper_parameters' entry."
        )

    hparams = checkpoint["hyper_parameters"]
    if not isinstance(hparams, Mapping):
        raise TypeError(
            f"Checkpoint {ckpt_file} contains invalid 'hyper_parameters': "
            f"expected a mapping, got {type(hparams).__name__}."
        )
    return dict(hparams)


def _load_fold_checkpoint_and_hparams(model_dir: Path, fold_idx: int) -> tuple[dict, dict]:
    ckpt_file = _find_fold_checkpoint(model_dir, fold_idx)
    checkpoint = _load_torch_checkpoint(ckpt_file, weights_only=True)
    if "state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint {ckpt_file} does not contain a 'state_dict' entry.")

    hparams = _get_checkpoint_hparams(checkpoint, ckpt_file)
    return checkpoint, hparams


def _get_nr_channels(hparams: dict) -> int:
    nr_channels = hparams.get("nr_channels")
    if nr_channels is not None:
        return int(nr_channels)

    dim = int(hparams.get("dim", 2))
    img_files = hparams.get("img_files", ["ct.nii.gz"])
    nr_img_files = len(img_files) if isinstance(img_files, list) else 1
    if dim == 3:
        return nr_img_files

    nr_slices = int(_require_hparam(hparams, "nr_slices"))
    multi_orientation = bool(_require_hparam(hparams, "multi_orientation"))
    return nr_img_files * nr_slices * (3 if multi_orientation else 1)


def _get_nr_classes(hparams: dict) -> int:
    target_names = hparams.get("reg_target_names") or CNN_MODEL_TARGET_ORDER
    nr_classes = int(hparams.get("nr_classes", len(target_names)))
    if hparams.get("loss") == "bce":
        nr_classes -= 1
    return nr_classes


def _create_resnet3d_model(hparams: dict):
    try:
        from monai.networks.nets import resnet10, resnet18, resnet50
    except ImportError as exc:
        raise ImportError("CNN body-stats inference requires monai to be installed.") from exc

    import torch

    model_name = hparams.get("model")
    resnet3d_map = {
        "resnet3d_10": (resnet10, 512, "B", False, 1.0, False),
        "resnet3d_10_w05": (resnet10, 256, "B", False, 0.5, False),
        "resnet3d_10_multihead": (resnet10, 512, "B", False, 1.0, True),
        "resnet3d_18": (resnet18, 512, "A", True, 1.0, False),
        "resnet3d_50": (resnet50, 2048, "B", False, 1.0, False),
    }
    if model_name not in resnet3d_map:
        raise ValueError(f"Unsupported 3D ResNet model: {model_name}")

    resnet_fn, feat_dim, shortcut_type, bias_downsample, widen_factor, multihead = resnet3d_map[model_name]
    model = resnet_fn(
        pretrained=False,
        spatial_dims=int(hparams.get("dim", 3)),
        n_input_channels=_get_nr_channels(hparams),
        feed_forward=False,
        shortcut_type=shortcut_type,
        bias_downsample=bias_downsample,
        conv1_t_stride=int(hparams.get("resnet3d_stride", 2)),
        widen_factor=widen_factor,
    )

    nr_classes = _get_nr_classes(hparams)
    dropout = float(hparams.get("dropout", 0.0))
    if multihead:
        model.fc = _MultiHeadOutput(feat_dim, nr_classes, dropout)
    else:
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(feat_dim, nr_classes),
        )
    return model


def _create_model_from_hparams(hparams: dict):
    model_name = hparams.get("model", "tf_efficientnet_b0_ns")
    if model_name.startswith("resnet3d"):
        return _create_resnet3d_model(hparams)

    try:
        import timm
    except ImportError as exc:
        raise ImportError("CNN body-stats inference requires timm to be installed.") from exc

    if model_name == "tf_efficientnet_b0_ns":
        model_name = "tf_efficientnet_b0.ns_jft_in1k"
    return timm.create_model(
        model_name,
        pretrained=False,
        num_classes=_get_nr_classes(hparams),
        in_chans=_get_nr_channels(hparams),
    )


def _extract_backbone_state_dict(state_dict: dict) -> dict:
    if any(key.startswith("backbone.") for key in state_dict):
        return {
            key.removeprefix("backbone."): value
            for key, value in state_dict.items()
            if key.startswith("backbone.")
        }
    return state_dict


def _load_fold_model(model_dir: Path, fold_idx: int, device):
    checkpoint, hparams = _load_fold_checkpoint_and_hparams(model_dir, fold_idx)
    model = _create_model_from_hparams(hparams)
    model.load_state_dict(_extract_backbone_state_dict(checkpoint["state_dict"]), strict=True)
    model.to(device)
    model.eval()
    return model, hparams


def _load_fold_hparams(model_dir: Path, fold_idx: int) -> dict:
    _, hparams = _load_fold_checkpoint_and_hparams(model_dir, fold_idx)
    return hparams


def _get_fold_indices(fold: int | str | None = "all") -> list[int]:
    if fold is None or fold == "all":
        return list(range(5))
    fold = int(fold)
    if fold not in range(5):
        raise ValueError(f"Fold must be in [0, 4] or 'all', got {fold}.")
    return [fold]


def _validate_modality(modality: str) -> None:
    if modality not in DEFAULT_BODY_STATS_CNN_DIRS:
        supported_modalities = ", ".join(sorted(DEFAULT_BODY_STATS_CNN_DIRS))
        raise ValueError(f"Unsupported CNN modality: {modality}. Supported: {supported_modalities}")


def _validate_target(target: str) -> None:
    if target not in CNN_TARGET_SPECS:
        raise ValueError(f"Unsupported CNN target: {target}")


def _validate_modality_and_target(modality: str, target: str) -> None:
    _validate_modality(modality)
    _validate_target(target)
    supported_targets = [
        CNN_TRAINING_NAME_TO_TARGET[training_name]
        for training_name in CNN_MODEL_TARGET_ORDERS[modality]
    ]
    if target not in supported_targets:
        raise ValueError(
            f"Unsupported CNN target '{target}' for modality '{modality}'. "
            f"Supported: {', '.join(supported_targets)}"
        )


def _resolve_body_stats_model_dir(
    modality: str = "mr", model_dir: Path | str | None = None
) -> Path:
    _validate_modality(modality)

    if model_dir is None:
        resolved_dir = DEFAULT_BODY_STATS_CNN_DIRS[modality]
        if not resolved_dir.exists():
            download_pretrained_weights(BODY_STATS_CNN_DOWNLOAD_TASKS[modality])
    else:
        candidate = Path(model_dir).expanduser()
        if (candidate / "version_0").exists():
            resolved_dir = candidate
        else:
            modality_candidate = candidate / DEFAULT_BODY_STATS_CNN_DIRS[modality].name
            if modality_candidate.exists():
                resolved_dir = modality_candidate
            else:
                resolved_dir = candidate

    if not resolved_dir.exists():
        raise FileNotFoundError(f"CNN model directory does not exist: {resolved_dir}")
    return resolved_dir


def _get_model_target_names(
    hparams: dict, output_count: int, modality: str | None = None
) -> list[str]:
    target_names = hparams.get("reg_target_names") or []
    if target_names:
        target_names = list(target_names)
        if len(target_names) != output_count:
            raise ValueError(
                f"Checkpoint defines {len(target_names)} target names, "
                f"but the model produced {output_count} outputs."
            )
        return target_names
    if modality is not None:
        modality_target_order = CNN_MODEL_TARGET_ORDERS[modality]
        if output_count == len(modality_target_order):
            return modality_target_order
    if output_count == len(CNN_MODEL_TARGET_ORDER):
        return CNN_MODEL_TARGET_ORDER
    return []


def _get_target_output_index(
    hparams: dict, target: str, output_count: int, modality: str | None = None
) -> int:
    target_names = _get_model_target_names(hparams, output_count, modality)
    training_name = CNN_TARGET_SPECS[target]["training_name"]
    if training_name in target_names:
        return target_names.index(training_name)
    if target in target_names:
        return target_names.index(target)
    if output_count == 1:
        return 0
    raise ValueError(
        f"Can not find output target '{training_name}' in model target names: {target_names}"
    )


def _apply_regression_target_denormalization(pred: np.ndarray, hparams: dict) -> np.ndarray:
    if not bool(hparams.get("reg_target_normalize", False)):
        return pred

    means = np.asarray(hparams.get("reg_target_mean", []), dtype=np.float32)
    stds = np.asarray(hparams.get("reg_target_std", []), dtype=np.float32)
    if means.size == 0 or stds.size == 0:
        return pred
    if means.size == 1 and pred.size != 1:
        means = np.repeat(means, pred.size)
    if stds.size == 1 and pred.size != 1:
        stds = np.repeat(stds, pred.size)
    if means.size != pred.size or stds.size != pred.size:
        raise ValueError(
            f"Regression normalization has shape mean={means.shape}, std={stds.shape}, "
            f"but prediction has shape {pred.shape}."
        )
    return pred * stds.reshape(pred.shape) + means.reshape(pred.shape)


def _format_binary_result(
    preds: np.ndarray, negative_label: str, positive_label: str
) -> dict:
    mean_score = float(np.mean(preds))
    pred_binary = int(mean_score >= 0.5)
    class_probs = np.clip(
        preds if pred_binary == 1 else 1.0 - preds, 0.0, 1.0
    )
    result = {
        "value": positive_label if pred_binary == 1 else negative_label,
        "unit": None,
    }
    if len(class_probs) > 1:
        result["probability"] = round(float(np.mean(class_probs)), 4)
        result["stddev"] = round(float(np.std(class_probs)), 4)
    return result


def _format_mapped_result(
    preds: np.ndarray,
    value_map: dict[str, int],
    fallback: str | None,
    max_out_of_range_distance: int | None = None,
) -> dict:
    mean_score = float(np.mean(preds))
    class_idx = int(np.floor(mean_score + 0.5))
    inverse_map = {value: name for name, value in value_map.items()}
    if (
        class_idx not in inverse_map
        and max_out_of_range_distance is not None
    ):
        closest_idx = min(inverse_map, key=lambda idx: abs(idx - class_idx))
        if abs(closest_idx - class_idx) <= max_out_of_range_distance:
            class_idx = closest_idx
    result = {
        "value": inverse_map.get(class_idx, fallback),
        "unit": None,
    }
    if len(preds) > 1:
        result["stddev"] = round(float(np.std(preds)), 4)
    return result


def _format_regression_result(preds: np.ndarray, target: str) -> dict:
    target_spec = CNN_TARGET_SPECS[target]
    preds = preds.reshape(-1).astype(np.float32)

    if target == "sex":
        return _format_binary_result(preds, "F", "M")
    if target == "contrast":
        return _format_binary_result(preds, "no", "yes")
    if target == "manufacturer":
        return _format_mapped_result(preds, MANUFACTURER_MAP, "other")
    if target == "mr_sequence":
        return _format_mapped_result(preds, MR_SEQUENCE_MAP, "other")
    if target in {"verte_upper", "verte_lower"}:
        result = _format_mapped_result(
            preds, VERTEBRA_MAP, None, max_out_of_range_distance=2
        )
        if result["value"] is not None:
            result["value"] = result["value"].removeprefix("vertebrae_")
        return result

    result = {
        "value": round(float(np.mean(preds)), 2),
        "unit": target_spec["unit"],
    }
    if len(preds) > 1:
        result["min"] = round(float(np.min(preds)), 2)
        result["max"] = round(float(np.max(preds)), 2)
        result["stddev"] = round(float(np.std(preds)), 4)
    return result


def _format_all_body_stats(
    preds: np.ndarray, hparams: dict, modality: str | None = None
) -> dict:
    output_count = preds.shape[1] if preds.ndim > 1 else 1
    training_names = _get_model_target_names(hparams, output_count, modality)
    result = {}
    for training_name in training_names:
        target = CNN_TRAINING_NAME_TO_TARGET.get(training_name)
        if target is None:
            raise ValueError(f"Unsupported CNN model target: {training_name}")
        if modality == "mr" and target == "manufacturer":
            continue
        target_idx = _get_target_output_index(
            hparams, target, output_count, modality
        )
        result[target] = _format_regression_result(preds[:, target_idx], target)
    return result


def predict_all_body_stats_with_cnn(
    img: nib.Nifti1Image,
    modality: str = "mr",
    model_dir: Path | str | None = None,
    fold: int | str | None = "all",
    device="gpu",
    skip_canonical: bool = False,
    test_time_augmentation: bool = False,
    debug: bool = False,
) -> dict:
    _validate_modality(modality)

    model_dir = _resolve_body_stats_model_dir(modality=modality, model_dir=model_dir)
    resolved_device = _resolve_device(device)
    fold_indices = _get_fold_indices(fold)
    hparams = _load_fold_hparams(model_dir, fold_indices[0])
    img_tensor = _prepare_image_tensors(
        img,
        modality=modality,
        hparams=hparams,
        skip_canonical=skip_canonical,
        test_time_augmentation=test_time_augmentation,
    ).to(resolved_device)
    if debug:
        print(f"DEBUG: CNN input tensor shape: {tuple(img_tensor.shape)}")

    try:
        import torch
    except ImportError as exc:
        raise ImportError("CNN body-stats inference requires PyTorch to be installed.") from exc

    preds = []
    with torch.inference_mode():
        for fold_idx in fold_indices:
            model, fold_hparams = _load_fold_model(model_dir, fold_idx, resolved_device)
            fold_preds = model(img_tensor).detach().float().cpu().numpy()
            fold_preds = np.asarray(fold_preds, dtype=np.float32)
            if fold_preds.ndim == 1:
                fold_preds = fold_preds[:, None] if img_tensor.shape[0] == fold_preds.shape[0] else fold_preds[None, :]
            else:
                fold_preds = np.atleast_2d(fold_preds)
            for pred in fold_preds:
                preds.append(_apply_regression_target_denormalization(pred, fold_hparams))

    preds = np.stack(preds, axis=0)
    return _format_all_body_stats(preds, hparams, modality)


def predict_body_stats_with_cnn(
    img: nib.Nifti1Image,
    target: str,
    modality: str = "mr",
    model_dir: Path | str | None = None,
    fold: int | str | None = "all",
    device="gpu",
    skip_canonical: bool = False,
    test_time_augmentation: bool = False,
    debug: bool = False,
) -> dict:
    _validate_modality_and_target(modality, target)
    return predict_all_body_stats_with_cnn(
        img, modality=modality, model_dir=model_dir, fold=fold, device=device,
        skip_canonical=skip_canonical, test_time_augmentation=test_time_augmentation,
        debug=debug
    )[target]


def predict_body_weight_with_cnn(
    img: nib.Nifti1Image,
    modality: str = "mr",
    model_dir: Path | str | None = None,
    fold: int | str | None = "all",
    device="gpu",
    skip_canonical: bool = False,
    test_time_augmentation: bool = False,
    debug: bool = False,
) -> dict:
    return predict_body_stats_with_cnn(
        img, target="weight", modality=modality, model_dir=model_dir, fold=fold,
        device=device, skip_canonical=skip_canonical,
        test_time_augmentation=test_time_augmentation, debug=debug
    )
