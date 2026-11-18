"""In-memory inference for the whole-body CT models on their isotropic model grid."""

from contextlib import nullcontext
import math
import platform
from itertools import product
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from totalsegmentator.channels_last import (
    prepare_channels_last,
    prepare_metal_upsampling,
    prepare_apple_cpu_convolutions,
)


def model_grid(image, spacing, device):
    """Resample canonical XYZ data to the model grid, returning ZYX voxels and the new affine."""
    if torch.device(device).type == "cuda":
        # Use the stock interpolation and integer conversion before normalization.
        # Tiny HU differences can flip a marginal voxel on the coarse label grid.
        from totalsegmentator.resampling import change_spacing

        grid = change_spacing(image, spacing, order=1, dtype=np.int32, nr_cpus=1, use_gpu=True)
        values = np.asarray(grid.dataobj, dtype=np.float32)
        return torch.from_numpy(np.ascontiguousarray(values.T)).to(device), grid.affine
    zoom = np.asarray(image.header.get_zooms(), dtype=np.float64) / spacing
    shape = tuple(int(n) for n in np.rint(np.asarray(image.shape) * zoom))
    # Metal has no float64. Keep corner interpolation in double before the CT integer cast.
    interpolation_device = "cpu" if torch.device(device).type == "mps" else device
    data = torch.from_numpy(np.ascontiguousarray(image.get_fdata(dtype=np.float64).T)).to(
        interpolation_device
    )
    if np.any(zoom != 1):
        data = F.interpolate(data[None, None], size=shape[::-1], mode="trilinear", align_corners=True)[
            0, 0
        ].trunc_()
    affine = image.affine.copy()
    affine[:3, :3] /= zoom
    return data.to(device=device, dtype=torch.float32), affine


def restore_labels(labels, shape):
    """Nearest-neighbor zoom using scipy.ndimage.zoom's corner-aligned sample positions."""
    for axis, size in enumerate(shape):
        old = labels.shape[axis]
        if size != old:
            scale = (old - 1) / max(size - 1, 1)
            indices = (torch.arange(size, dtype=torch.float64) * scale + 0.5).floor().long()
            labels = labels.index_select(axis, indices.to(labels.device))
    return labels


def normalized_labels(scores, weights):
    scores.div_(weights)
    lo, hi = scores.amin(), scores.amax()
    if not (lo.isfinite() & hi.isfinite()).item():
        raise RuntimeError("Non-finite sliding-window logits after Gaussian normalization.")
    return scores.argmax(0).byte()


def window_labels(image, net, patch, steps, importance, heads):
    """Batch sliding windows and finalize each depth slab after its last overlap."""
    depth, height, width = image.shape
    dz, dy, dx = patch
    dtype = next(net.parameters()).dtype
    layout = torch.contiguous_format if image.is_cuda else torch.channels_last_3d
    if tuple(image.shape) == tuple(patch):
        # A single tile needs neither an overlap buffer nor a second copy of its logits.
        batch = image[None, None].to(dtype=dtype, memory_format=layout)
        weighted = net(batch)[0].mul_(importance).to(importance.dtype)
        return normalized_labels(weighted, importance)
    scores = image.new_zeros((heads, dz, height, width), dtype=importance.dtype)
    weights = torch.zeros_like(scores[0])
    labels = image.new_empty(image.shape, dtype=torch.uint8)
    # CUDA matches the stock batch-one forward and bounds decoder activations.
    # The largest Metal tiles also need to run individually.
    batch_size = 1 if image.is_cuda or (image.device.type == "mps" and math.prod(patch) > 128**3) else 2
    plane = list(product(steps[1], steps[2]))
    for row, z in enumerate(steps[0]):
        for i in range(0, len(plane), batch_size):
            locations = plane[i : i + batch_size]
            batch = torch.stack([image[z : z + dz, y : y + dy, x : x + dx] for y, x in locations])[:, None]
            logits = net(batch.to(dtype=dtype, memory_format=layout)).mul_(importance)
            for tile, (y, x) in zip(logits, locations):
                scores[:, :, y : y + dy, x : x + dx] += tile
                weights[:, y : y + dy, x : x + dx] += importance
            del tile, logits
        end = steps[0][row + 1] if row + 1 < len(steps[0]) else depth
        count = end - z
        labels[z:end] = normalized_labels(scores[:, :count], weights[:count])
        if end < depth:
            for buf in (scores, weights.unsqueeze(0)):
                buf[:, : dz - count] = buf[:, count:].clone()
                buf[:, dz - count :].zero_()
    return labels


def predict_model(ct, net, plans, config, heads, step):
    """Crop, normalize, pad, segment and undo preprocessing on the model grid."""
    from nnunetv2.inference.sliding_window_prediction import (
        compute_gaussian,
        compute_steps_for_sliding_window,
    )

    axes = plans.transpose_forward
    image = ct.permute(*axes)
    nonzero = image != 0
    box = []
    for axis in range(3):
        hits = nonzero.any(dim=tuple(i for i in range(3) if i != axis)).nonzero().flatten()
        box.append(slice(int(hits[0]), int(hits[-1]) + 1) if hits.numel() else slice(None))
    cropped = image[tuple(box)]
    props = plans.foreground_intensity_properties_per_channel["0"]
    data = (cropped.clamp(props["percentile_00_5"], props["percentile_99_5"]) - props["mean"]) / max(
        props["std"], 1e-8
    )
    patch = tuple(config.patch_size)
    missing = [max(k - n, 0) for n, k in zip(data.shape, patch)]
    padding = [(n // 2, n - n // 2) for n in missing]
    data = F.pad(data, [v for pair in reversed(padding) for v in pair])
    steps = compute_steps_for_sliding_window(data.shape, patch, step)
    gaussian = compute_gaussian(patch, sigma_scale=1 / 8, value_scaling_factor=10, device=ct.device).to(
        torch.float16
    )
    # Match nnU-Net's CUDA autocast and FP16 Gaussian accumulation, including
    # division before argmax: rounding ties can change the winning label.
    amp = torch.autocast("cuda") if ct.is_cuda else nullcontext()
    with amp:
        labels = window_labels(data, net, patch, steps, gaussian, heads)
    keep = tuple(slice(lo, lo + n) for (lo, _), n in zip(padding, cropped.shape))
    out = torch.zeros_like(image, dtype=torch.uint8)
    out[tuple(box)] = labels[keep]
    return out.permute(*plans.transpose_backward)


class CTSession:
    """Reuse loaded fold-zero networks across serial calls to totalsegmentator."""

    def __init__(self):
        self._models = {}
        self._resident_cuda = None

    def _load(self, folder, device):
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        key = (Path(folder).resolve(), device)
        if key not in self._models:
            predictor = nnUNetPredictor(device=device, use_mirroring=False, allow_tqdm=False)
            predictor.initialize_from_trained_model_folder(
                str(folder), use_folds=(0,), checkpoint_name="checkpoint_final.pth"
            )
            net = getattr(predictor.network, "_orig_mod", predictor.network)
            if (
                not isinstance(net, torch.nn.Module)
                or predictor.label_manager is None
                or not predictor.list_of_parameters
            ):
                raise RuntimeError("nnU-Net did not initialize the network and fold-zero checkpoint.")
            net.load_state_dict(predictor.list_of_parameters[0])
            net.eval().requires_grad_(False)
            if device.type != "cuda":
                dtype = torch.float32 if device.type == "cpu" else torch.float16
                net.to(device=device, dtype=dtype)
                prepare_channels_last(net)
            if device.type == "mps":
                prepare_metal_upsampling(net)
            elif device.type == "cpu" and platform.system() == "Darwin" and platform.machine() == "arm64":
                prepare_apple_cpu_convolutions(net)
            self._models[key] = (
                net,
                predictor.plans_manager,
                predictor.configuration_manager,
                predictor.label_manager.num_segmentation_heads,
            )
        return self._models[key]

    @torch.inference_mode()
    def segment(self, ct, spacing, models, step, split):
        """Run each selected part and merge foreground labels on the selected device."""
        depth = ct.shape[0]
        third, margin = depth // 3, 20
        slabs = [(slice(None), slice(None))]
        if split:
            # Match the stock three-way split and its overlap at both boundaries.
            if third < margin:
                raise ValueError("Three-way splitting requires at least 60 model-grid slices.")
            slabs = [
                (slice(0, third + margin), slice(0, third)),
                (slice(third + 1 - margin, 2 * third + margin), slice(margin - 1, margin - 1 + third)),
                (slice(2 * third + 1 - margin, None), slice(margin - 1, None)),
            ]
        results = []
        for region, keep in slabs:
            chunk = ct[region]
            merged = torch.zeros_like(chunk, dtype=torch.uint8)
            for folder, mapping in models:
                net, plans, config, heads = self._load(folder, ct.device)
                if config.normalization_schemes != ["CTNormalization"] or not np.allclose(
                    config.spacing, spacing
                ):
                    raise ValueError(
                        "Optimized CT requires CT normalization and input on the trained isotropic grid."
                    )
                if ct.is_cuda:
                    if self._resident_cuda is not None and self._resident_cuda is not net:
                        self._resident_cuda.cpu()
                        self._resident_cuda = None
                    net.to(ct.device)
                    self._resident_cuda = net
                try:
                    labels = predict_model(chunk, net, plans, config, heads, step)
                finally:
                    if ct.is_cuda and len(models) > 1:
                        # Retain the initialized model for the next scan in host memory.
                        net.cpu()
                        self._resident_cuda = None
                if mapping is None:
                    merged = labels
                else:
                    lut = torch.as_tensor(mapping, dtype=torch.uint8, device=ct.device)
                    mapped = lut[labels.long()]
                    merged = torch.where(mapped != 0, mapped, merged)
            results.append(merged[keep])
        return torch.cat(results) if split else results[0]


def nifti_zyx(data, affine):
    return nib.Nifti1Image(data.cpu().numpy().T, affine)
