from copy import deepcopy

import pytest
import torch
from torch import nn

from totalsegmentator.channels_last import prepare_channels_last


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")),
    ],
)
@pytest.mark.parametrize("affine", [True, False])
def test_channels_last_preserves_logits_and_checkpoint_keys(affine, device):
    torch.manual_seed(41)
    ref = (
        nn.Sequential(
            nn.Conv3d(2, 4, 3, padding=1),
            nn.InstanceNorm3d(4, affine=affine),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(4, 2, 2),
        )
        .to(device)
        .eval()
    )
    net = deepcopy(ref)
    # nnU-Net exposes the same layer directly and via its sequential container.
    wrapper = nn.Module()
    wrapper.block = net
    wrapper.norm = net[1]
    prepare_channels_last(wrapper)
    assert wrapper.norm is net[1]
    assert isinstance(net[0], nn.Conv3d)
    assert net[0].weight.is_contiguous(memory_format=torch.channels_last_3d)
    net.load_state_dict(ref.state_dict(), strict=True)
    x = torch.randn(2, 2, 7, 5, 3, device=device)
    with torch.inference_mode(), torch.autocast(device, enabled=device == "cuda"):
        tolerance = 2e-3 if device == "cuda" else 1e-5
        torch.testing.assert_close(net(x), ref(x), atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("stride", [(1, 1, 1), (2, 2, 2), (1, 2, 1)])
def test_depth_blocked_conv(stride):
    from totalsegmentator.channels_last import DepthBlockedConv

    torch.manual_seed(4)
    conv = nn.Conv3d(4, 7, 3, padding=1, stride=stride)
    x = torch.randn(2, 4, 9, 11, 13)
    torch.testing.assert_close(DepthBlockedConv(conv)(x), conv(x), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("kernel", [(2, 2, 2), (1, 2, 3)])
def test_projection_upsampling(kernel):
    from totalsegmentator.channels_last import NonOverlappingUpsample

    torch.manual_seed(8)
    conv = nn.ConvTranspose3d(8, 4, kernel, stride=kernel)
    x = torch.randn(2, 8, 5, 6, 7)
    torch.testing.assert_close(NonOverlappingUpsample(conv)(x), conv(x))
