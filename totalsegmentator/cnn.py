import pickle
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np

from totalsegmentator.config import get_weights_dir
from totalsegmentator.libs import download_pretrained_weights
from totalsegmentator.resampling import change_spacing


DEFAULT_BODY_STATS_CNN_ROOT_DIR = get_weights_dir() / "lightning_models"
DEFAULT_BODY_STATS_CNN_DIRS = {
    "mr": {

        # "weight": DEFAULT_BODY_STATS_CNN_ROOT_DIR / "mr_weight_splitXGB_2d_ns5_effnetv2",
        "weight": DEFAULT_BODY_STATS_CNN_ROOT_DIR / "mr_weight_splitOrig_2d_ns5_effnetv2",
        "size": DEFAULT_BODY_STATS_CNN_ROOT_DIR / "mr_size_2mm_splitXGB_2d_ns5",
        "age": DEFAULT_BODY_STATS_CNN_ROOT_DIR / "mr_age_2mm_splitXGB_2d_ns5",
        "sex": DEFAULT_BODY_STATS_CNN_ROOT_DIR / "mr_sex_2mm_splitXGB_2d_ns5",
    },
    "ct": {
        # can not use older mo1 models, because they are based on sparse z-sampling which is not done during inference
        # "weight": DEFAULT_BODY_STATS_CNN_ROOT_DIR / "ct_weight_splitXGB_2d_ns5_effnetv2_ep40",
        "weight": DEFAULT_BODY_STATS_CNN_ROOT_DIR / "ct_weight_splitOrig_2d_ns5_effnetv2",
        "size": DEFAULT_BODY_STATS_CNN_ROOT_DIR / "ct_size_2mm_splitXGB_2d_ns5",
        "age": DEFAULT_BODY_STATS_CNN_ROOT_DIR / "ct_age_2mm_splitXGB_2d_ns5",
        "sex": DEFAULT_BODY_STATS_CNN_ROOT_DIR / "ct_sex_2mm_splitXGB_2d_ns5",
    },
}
BODY_STATS_CNN_DOWNLOAD_TASKS = {
    "mr": {
        "weight": "body_stats_cnn_mr_weight",
        "size": "body_stats_cnn_mr_size",
        "age": "body_stats_cnn_mr_age",
        "sex": "body_stats_cnn_mr_sex",
    },
    "ct": {
        "weight": "body_stats_cnn_ct_weight",
        "size": "body_stats_cnn_ct_size",
        "age": "body_stats_cnn_ct_age",
        "sex": "body_stats_cnn_ct_sex",
    },
}

CNN_CROP_SIZE = {
    "ct": (240, 240),
    "mr": (210, 210),
}
CNN_TARGET_SPACING_MM = 2.0
CNN_TARGET_SPECS = {
    "weight": {"loss": "mse", "num_classes": 1, "unit": "kg"},
    "size": {"loss": "mse", "num_classes": 1, "unit": "cm"},
    "age": {"loss": "mse", "num_classes": 1, "unit": "years"},
    "sex": {"loss": "ce", "num_classes": 2, "unit": None},
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


def _extract_multi_orientation_slices(
    img_data: np.ndarray, nr_slices: int, offset: int
) -> list[np.ndarray]:
    """Mirror multi_orientation=True from the deterministic validation dataset."""
    axis_offsets = np.array(
        [0 if axis_size <= 1 else max(1, axis_size // 8) for axis_size in img_data.shape],
        dtype=int,
    )
    axis_offsets = np.minimum(axis_offsets, offset)
    mid = (np.array(img_data.shape) / 2).astype(int)

    x_slices = img_data[
        _get_slice_indices(mid[0], nr_slices, axis_offsets[0], img_data.shape[0]), :, :
    ]
    y_slices = img_data[
        :, _get_slice_indices(mid[1], nr_slices, axis_offsets[1], img_data.shape[1]), :
    ].transpose(1, 0, 2)
    z_slices = img_data[
        :, :, _get_slice_indices(mid[2], nr_slices, axis_offsets[2], img_data.shape[2])
    ].transpose(2, 0, 1)

    return [*x_slices, *y_slices, *z_slices]


def _extract_single_orientation_slices(
    img_data: np.ndarray, nr_slices: int, offset: int, orientation: str
) -> list[np.ndarray]:
    mid = (np.array(img_data.shape) / 2).astype(int)

    if orientation == "x":
        return list(
            img_data[_get_slice_indices(mid[0], nr_slices, offset, img_data.shape[0]), :, :]
        )
    if orientation == "y":
        return list(
            img_data[
                :, _get_slice_indices(mid[1], nr_slices, offset, img_data.shape[1]), :
            ].transpose(1, 0, 2)
        )
    if orientation == "z":
        return list(
            img_data[
                :, :, _get_slice_indices(mid[2], nr_slices, offset, img_data.shape[2])
            ].transpose(2, 0, 1)
        )
    raise ValueError(f"Unsupported slice orientation: {orientation}")


def _require_hparam(hparams: dict | None, key: str):
    if not hparams or key not in hparams:
        raise KeyError(f"Checkpoint is missing required hyperparameter: {key}")
    return hparams[key]


def _extract_slices(img_data: np.ndarray, hparams: dict | None) -> list[np.ndarray]:
    nr_slices = int(_require_hparam(hparams, "nr_slices"))
    multi_orientation = bool(_require_hparam(hparams, "multi_orientation"))
    orientation = hparams.get("slice_orientation", "z") if hparams else "z"
    orientation_to_axis = {"x": 0, "y": 1, "z": 2}
    if orientation not in orientation_to_axis:
        raise ValueError(f"Unsupported slice orientation: {orientation}")

    offset = int(img_data.shape[orientation_to_axis[orientation]] / 8)
    if multi_orientation:
        return _extract_multi_orientation_slices(img_data, nr_slices, offset)
    return _extract_single_orientation_slices(img_data, nr_slices, offset, orientation)


def _center_pad_or_crop_2d(img_2d: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_shape
    src_h, src_w = img_2d.shape

    src_h_start = max((src_h - target_h) // 2, 0)
    src_w_start = max((src_w - target_w) // 2, 0)
    src_h_end = src_h_start + min(src_h, target_h)
    src_w_end = src_w_start + min(src_w, target_w)

    cropped = img_2d[src_h_start:src_h_end, src_w_start:src_w_end]
    out = np.zeros((target_h, target_w), dtype=img_2d.dtype)

    dst_h_start = max((target_h - cropped.shape[0]) // 2, 0)
    dst_w_start = max((target_w - cropped.shape[1]) // 2, 0)
    dst_h_end = dst_h_start + cropped.shape[0]
    dst_w_end = dst_w_start + cropped.shape[1]
    out[dst_h_start:dst_h_end, dst_w_start:dst_w_end] = cropped
    return out


def _normalize_per_channel(img_stack: np.ndarray) -> np.ndarray:
    img_stack = img_stack.astype(np.float32, copy=False)
    normalized = np.empty_like(img_stack, dtype=np.float32)

    for channel_idx in range(img_stack.shape[0]):
        channel = img_stack[channel_idx]
        mean = float(channel.mean())
        std = float(channel.std())
        if std < 1e-8:
            normalized[channel_idx] = channel - mean
        else:
            normalized[channel_idx] = (channel - mean) / std

    return normalized


def _normalize_with_training_hparams(img_stack: np.ndarray, hparams: dict | None) -> np.ndarray:
    img_stack = img_stack.astype(np.float32, copy=False)
    if not hparams or not hparams.get("normalize", True):
        return img_stack

    if hparams.get("norm_global", False):
        global_mean = np.asarray(hparams["global_mean"], dtype=np.float32)
        global_std = np.asarray(hparams["global_std"], dtype=np.float32)
        return (img_stack - global_mean[:, None, None]) / global_std[:, None, None]

    if hparams.get("norm_channel_wise", True):
        return _normalize_per_channel(img_stack)

    mean = float(img_stack.mean())
    std = float(img_stack.std())
    if std < 1e-8:
        return img_stack - mean
    return (img_stack - mean) / std


def _prepare_image_tensor(
    img: nib.Nifti1Image, crop_size: tuple[int, int], hparams: dict | None = None
):
    img = nib.as_closest_canonical(img)
    img = change_spacing(img, CNN_TARGET_SPACING_MM, dtype=np.float32, order=1)
    img_data = np.asarray(img.dataobj, dtype=np.float32)

    if hparams and hparams.get("clip", False):
        img_data = np.clip(img_data, hparams["clip_low"], hparams["clip_high"])

    crop_size = tuple(hparams.get("crop_size", crop_size)) if hparams else crop_size
    slices = _extract_slices(img_data, hparams)
    slices = np.stack(
        [_center_pad_or_crop_2d(slice_2d, crop_size) for slice_2d in slices],
        axis=0,
    )
    slices = _normalize_with_training_hparams(slices, hparams)

    try:
        import torch
    except ImportError as exc:
        raise ImportError("CNN body-stats inference requires PyTorch to be installed.") from exc

    return torch.from_numpy(slices[None, ...])


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


def _load_fold_model(model_dir: Path, fold_idx: int, device, target: str):
    import torch 
    try:
        import timm
    except ImportError as exc:
        raise ImportError("CNN body-stats inference requires timm to be installed.") from exc

    ckpt_dir = model_dir / f"version_{fold_idx}" / "checkpoints"
    ckpt_files = sorted(ckpt_dir.glob("epoch*.ckpt"))
    if len(ckpt_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one checkpoint in {ckpt_dir}, found {len(ckpt_files)}."
        )

    try:
        checkpoint = torch.load(ckpt_files[0], map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        # Some Lightning checkpoints include MONAI objects such as MetaTensor in metadata.
        # Allowlist them so we can keep using the safer weights_only=True path when possible.
        try:
            from monai.data.meta_tensor import MetaTensor

            with torch.serialization.safe_globals([MetaTensor]):
                checkpoint = torch.load(ckpt_files[0], map_location="cpu", weights_only=True)
        except Exception:
            # Fall back to the legacy loading mode for trusted local checkpoints.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="You are using `torch.load` with `weights_only=False`",
                    category=FutureWarning,
                )
                checkpoint = torch.load(ckpt_files[0], map_location="cpu", weights_only=False)
    except TypeError:
        # Support older PyTorch versions which do not expose weights_only yet.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="You are using `torch.load` with `weights_only=False`",
                category=FutureWarning,
            )
            checkpoint = torch.load(ckpt_files[0], map_location="cpu")
    if "state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint {ckpt_files[0]} does not contain a 'state_dict' entry.")

    model_name = checkpoint.get("hyper_parameters", {}).get("model", "tf_efficientnet_b0_ns")
    if model_name == "tf_efficientnet_b0_ns":
        model_name = "tf_efficientnet_b0.ns_jft_in1k"

    hparams = checkpoint.get("hyper_parameters", {})
    nr_channels = hparams.get("nr_channels")
    if nr_channels is None:
        nr_slices = int(_require_hparam(hparams, "nr_slices"))
        multi_orientation = bool(_require_hparam(hparams, "multi_orientation"))
        nr_channels = nr_slices * (3 if multi_orientation else 1)

    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=CNN_TARGET_SPECS[target]["num_classes"],
        in_chans=int(nr_channels),
    )
    state_dict = {
        key.removeprefix("backbone."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("backbone.")
    }
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def _load_fold_hparams(model_dir: Path, fold_idx: int) -> dict:
    import torch

    ckpt_dir = model_dir / f"version_{fold_idx}" / "checkpoints"
    ckpt_files = sorted(ckpt_dir.glob("epoch*.ckpt"))
    if len(ckpt_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one checkpoint in {ckpt_dir}, found {len(ckpt_files)}."
        )

    try:
        checkpoint = torch.load(ckpt_files[0], map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_files[0], map_location="cpu")
    return dict(checkpoint.get("hyper_parameters", {}))


def _get_fold_indices(fold: int | None) -> list[int]:
    if fold is None:
        return list(range(5))
    if fold not in range(5):
        raise ValueError(f"Fold must be in [0, 4], got {fold}.")
    return [fold]


def _validate_modality_and_target(modality: str, target: str) -> None:
    if modality not in DEFAULT_BODY_STATS_CNN_DIRS:
        supported_modalities = ", ".join(sorted(DEFAULT_BODY_STATS_CNN_DIRS))
        raise ValueError(f"Unsupported CNN modality: {modality}. Supported: {supported_modalities}")
    if target not in CNN_TARGET_SPECS:
        raise ValueError(f"Unsupported CNN target: {target}")
    if target not in DEFAULT_BODY_STATS_CNN_DIRS[modality]:
        supported_targets = ", ".join(sorted(DEFAULT_BODY_STATS_CNN_DIRS[modality]))
        raise ValueError(
            f"Unsupported CNN target for {modality.upper()}: {target}. "
            f"Supported: {supported_targets}"
        )


def _resolve_target_model_dir(
    target: str, modality: str = "mr", model_dir: Path | str | None = None
) -> Path:
    _validate_modality_and_target(modality, target)

    if model_dir is None:
        resolved_dir = DEFAULT_BODY_STATS_CNN_DIRS[modality][target]
        if not resolved_dir.exists():
            download_pretrained_weights(BODY_STATS_CNN_DOWNLOAD_TASKS[modality][target])
    else:
        candidate = Path(model_dir).expanduser()
        if (candidate / "version_0").exists():
            resolved_dir = candidate
        else:
            target_candidate = candidate / DEFAULT_BODY_STATS_CNN_DIRS[modality][target].name
            if target_candidate.exists():
                resolved_dir = target_candidate
            else:
                resolved_dir = candidate

    if not resolved_dir.exists():
        raise FileNotFoundError(f"CNN model directory does not exist: {resolved_dir}")
    return resolved_dir


def predict_body_stats_with_cnn(
    img: nib.Nifti1Image,
    target: str,
    modality: str = "mr",
    model_dir: Path | str | None = None,
    fold: int | None = None,
    device="gpu",
) -> dict:
    _validate_modality_and_target(modality, target)

    model_dir = _resolve_target_model_dir(target, modality=modality, model_dir=model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"CNN model directory does not exist: {model_dir}")

    resolved_device = _resolve_device(device)
    fold_indices = _get_fold_indices(fold)
    hparams = _load_fold_hparams(model_dir, fold_indices[0])
    img_tensor = _prepare_image_tensor(
        img, crop_size=CNN_CROP_SIZE[modality], hparams=hparams
    ).to(resolved_device)

    try:
        import torch
    except ImportError as exc:
        raise ImportError("CNN body-stats inference requires PyTorch to be installed.") from exc

    preds = []
    with torch.inference_mode():
        for fold_idx in fold_indices:
            model = _load_fold_model(model_dir, fold_idx, resolved_device, target)
            pred = model(img_tensor).detach().float().cpu().numpy()[0]
            preds.append(pred)

    preds = np.array(preds)
    target_spec = CNN_TARGET_SPECS[target]

    if target_spec["loss"] == "mse":
        preds = preds.reshape(-1)
        return {
            "value": round(float(np.mean(preds)), 2),
            "min": round(float(np.min(preds)), 2),
            "max": round(float(np.max(preds)), 2),
            "stddev": round(float(np.std(preds)), 4),
            "unit": target_spec["unit"],
        }

    if target_spec["loss"] == "ce":
        probs = preds - preds.max(axis=1, keepdims=True)
        probs = np.exp(probs) / np.exp(probs).sum(axis=1, keepdims=True)
        mean_probs = probs.mean(axis=0)
        pred_binary = int(np.argmax(mean_probs))
        pred = "M" if pred_binary == 1 else "F"
        reported_prob = float(mean_probs[pred_binary])
        pred_std = round(float(np.std(probs[:, pred_binary])), 4)
        return {
            "value": pred,
            "probability": round(reported_prob, 4),
            "stddev": pred_std,
            "unit": None,
        }

    raise ValueError(f"Unsupported CNN loss type: {target_spec['loss']}")


def predict_body_weight_with_cnn(
    img: nib.Nifti1Image,
    modality: str = "mr",
    model_dir: Path | str | None = None,
    fold: int | None = None,
    device="gpu",
) -> dict:
    return predict_body_stats_with_cnn(
        img, target="weight", modality=modality, model_dir=model_dir, fold=fold, device=device
    )
