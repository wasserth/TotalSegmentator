"""Channels-last layout for the whole-body CT inference networks."""

from typing import Optional

import torch
from torch import nn


def prepare_channels_last(network):
    """Keep convolution and per-instance normalization in a compatible layout."""
    nn.utils.convert_conv3d_weight_memory_format(network, torch.channels_last_3d)

    # One group per channel uses the same spatial statistics as InstanceNorm3d
    # without running stats. GroupNorm can consume channels-last activations.
    # nnU-Net registers some layers twice, individually and in a Sequential.
    replacements = {}
    for path, layer in list(network.named_modules(remove_duplicate=False)):
        if type(layer) is not nn.InstanceNorm3d or layer.track_running_stats:
            continue
        if layer not in replacements:
            norm = nn.GroupNorm(layer.num_features, layer.num_features, eps=layer.eps, affine=layer.affine)
            norm.weight = layer.weight
            norm.bias = layer.bias
            replacements[layer] = norm.train(layer.training)
        parent, _, name = path.rpartition(".")
        setattr(network.get_submodule(parent), name, replacements[layer])


class NonOverlappingUpsample(nn.Module):
    """Project each input voxel to its own output block for stride-equals-kernel upsampling."""

    projection: torch.Tensor
    offset: Optional[torch.Tensor]

    def __init__(self, conv):
        super().__init__()
        self.block = tuple(conv.kernel_size)
        self.channels = conv.out_channels
        # Linear output order is (block z, block y, block x, output channel).
        weights = conv.weight.permute(2, 3, 4, 1, 0).reshape(-1, conv.in_channels)
        self.register_buffer("projection", weights.contiguous())
        bias = conv.bias.repeat(weights.shape[0] // self.channels) if conv.bias is not None else None
        self.register_buffer("offset", bias)

    def forward(self, x):
        n, _, d, h, w = x.shape
        dz, dy, dx = self.block
        blocks = torch.nn.functional.linear(x.permute(0, 2, 3, 4, 1), self.projection, self.offset)
        grid = blocks.reshape(n, d, h, w, dz, dy, dx, self.channels).permute(0, 1, 4, 2, 5, 3, 6, 7)
        return grid.reshape(n, d * dz, h * dy, w * dx, self.channels).permute(0, 4, 1, 2, 3)


def prepare_metal_upsampling(net):
    """Metal supports the FP16 linear projection used by these decoder layers."""
    for path, layer in list(net.named_modules()):
        if isinstance(layer, nn.ConvTranspose3d):
            if (
                layer.groups != 1
                or layer.stride != layer.kernel_size
                or any(layer.padding)
                or any(layer.output_padding)
                or layer.dilation != (1, 1, 1)
            ):
                raise ValueError("The Metal CT decoder requires non-overlapping transposed convolutions.")
            parent, _, name = path.rpartition(".")
            setattr(net.get_submodule(parent), name, NonOverlappingUpsample(layer))


class DepthBlockedConv(nn.Module):
    """Apply a 3D CPU convolution as 2D convolutions over small groups of depth slices."""

    filters: torch.Tensor
    offset: Optional[torch.Tensor]

    def __init__(self, conv):
        super().__init__()
        self.kernel = tuple(conv.kernel_size)
        self.stride = tuple(conv.stride)
        self.padding = tuple(conv.padding)
        out_channels, channels, kz, ky, kx = conv.weight.shape
        weight = conv.weight.transpose(1, 2).reshape(out_channels, kz * channels, ky, kx)
        self.register_buffer("filters", weight.contiguous(memory_format=torch.channels_last))
        self.register_buffer("offset", conv.bias)

    def forward(self, x):
        n, channels, depth, height, width = x.shape
        kz, ky, kx = self.kernel
        sz, sy, sx = self.stride
        pz, py, px = self.padding
        out_depth = (depth + 2 * pz - kz) // sz + 1
        out_height = (height + 2 * py - ky) // sy + 1
        out_width = (width + 2 * px - kx) // sx + 1
        out_channels = self.filters.shape[0]
        out = torch.empty(
            (n, out_channels, out_depth, out_height, out_width),
            dtype=x.dtype,
            device=x.device,
            memory_format=torch.channels_last_3d,
        )
        # Bound the packed input to about 16 MiB per convolution on this CPU.
        block = max(1, (4 * 1024**2) // (n * channels * kz * height * width))
        for start in range(0, out_depth, block):
            count = min(block, out_depth - start)
            lo = start * sz - pz
            hi = lo + (count - 1) * sz + kz
            slices = x[:, :, max(0, lo) : min(depth, hi)]
            if lo < 0 or hi > depth:
                slices = torch.nn.functional.pad(slices, (0, 0, 0, 0, max(0, -lo), max(0, hi - depth)))
            packed = slices.unfold(2, kz, sz).permute(0, 2, 3, 4, 5, 1)
            packed = packed.reshape(n * count, height, width, kz * channels).permute(0, 3, 1, 2)
            y = torch.nn.functional.conv2d(
                packed, self.filters, self.offset, stride=(sy, sx), padding=(py, px)
            )
            out[:, :, start : start + count] = y.reshape(
                n, count, out_channels, out_height, out_width
            ).transpose(1, 2)
        return out


def prepare_apple_cpu_convolutions(net):
    """Pack plain 3D convolution weights once for the Apple CPU's 2D convolution path."""
    replacements = {}
    for path, layer in list(net.named_modules(remove_duplicate=False)):
        if (
            isinstance(layer, nn.Conv3d)
            and layer.kernel_size == (3, 3, 3)
            and layer.groups == 1
            and layer.dilation == (1, 1, 1)
            and layer.padding_mode == "zeros"
        ):
            if layer not in replacements:
                replacements[layer] = DepthBlockedConv(layer)
            parent, _, name = path.rpartition(".")
            setattr(net.get_submodule(parent), name, replacements[layer])
