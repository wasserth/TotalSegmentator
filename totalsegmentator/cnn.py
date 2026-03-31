from pathlib import Path
import pickle
import warnings

import nibabel as nib
import numpy as np

from totalsegmentator.resampling import change_spacing


DEFAULT_BODY_STATS_CNN_DIR = (
    Path("~/.totalsegmentator/nnunet/results/lightning_models/2mm_splitXGB_2d_ns5")
    .expanduser()
)

CNN_CROP_SIZE = (210, 210)
CNN_NR_SLICES = 5
CNN_TARGET_SPACING_MM = 2.0


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


def _extract_axial_slices(img_data: np.ndarray) -> np.ndarray:
    """Mirror the deterministic validation-time slice extraction for this model."""
    mid_idx = int(img_data.shape[2] / 2)
    offset = int(img_data.shape[2] / 8)
    slice_indices = _get_slice_indices(mid_idx, CNN_NR_SLICES, offset, img_data.shape[2])
    return img_data[:, :, slice_indices].transpose(2, 0, 1)


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


def _prepare_image_tensor(img: nib.Nifti1Image):
    img = nib.as_closest_canonical(img)
    img = change_spacing(img, CNN_TARGET_SPACING_MM, dtype=np.float32, order=1)
    img_data = np.asarray(img.dataobj, dtype=np.float32)
    slices = _extract_axial_slices(img_data)
    slices = np.stack(
        [_center_pad_or_crop_2d(slice_2d, CNN_CROP_SIZE) for slice_2d in slices],
        axis=0,
    )
    slices = _normalize_per_channel(slices)

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


def _load_fold_model(model_dir: Path, fold_idx: int, device):
    try:
        import torch
    except ImportError as exc:
        raise ImportError("CNN body-stats inference requires PyTorch to be installed.") from exc

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

    model = timm.create_model(
        "tf_efficientnet_b0_ns",
        pretrained=False,
        num_classes=1,
        in_chans=CNN_NR_SLICES,
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


def _get_fold_indices(fold: int | None) -> list[int]:
    if fold is None:
        return list(range(5))
    if fold not in range(5):
        raise ValueError(f"Fold must be in [0, 4], got {fold}.")
    return [fold]


def predict_body_weight_with_cnn(
    img: nib.Nifti1Image,
    model_dir: Path | str | None = None,
    fold: int | None = None,
    device="gpu",
) -> dict:
    model_dir = Path(model_dir or DEFAULT_BODY_STATS_CNN_DIR).expanduser()
    if not model_dir.exists():
        raise FileNotFoundError(f"CNN model directory does not exist: {model_dir}")

    resolved_device = _resolve_device(device)
    img_tensor = _prepare_image_tensor(img).to(resolved_device)

    try:
        import torch
    except ImportError as exc:
        raise ImportError("CNN body-stats inference requires PyTorch to be installed.") from exc

    preds = []
    with torch.inference_mode():
        for fold_idx in _get_fold_indices(fold):
            model = _load_fold_model(model_dir, fold_idx, resolved_device)
            pred = model(img_tensor).detach().float().cpu().numpy().reshape(-1)[0]
            preds.append(float(pred))

    preds = np.array(preds, dtype=np.float32)
    return {
        "value": round(float(np.mean(preds)), 2),
        "min": round(float(np.min(preds)), 2),
        "max": round(float(np.max(preds)), 2),
        "stddev": round(float(np.std(preds)), 4),
        "unit": "kg",
    }
