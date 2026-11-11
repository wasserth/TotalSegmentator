from unittest import mock

import nibabel as nib
import numpy as np
import pytest
import torch

from totalsegmentator import python_api
from totalsegmentator.python_api import convert_device_to_string, select_device, totalsegmentator


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_string_devices_are_kept(device):
    # select_device() returns these as strings. They were converted to None.
    assert convert_device_to_string(select_device(device)) == device


@pytest.mark.parametrize("device, expected", [
    (torch.device("cuda"), "gpu"),
    (torch.device("cuda:0"), "gpu:0"),
    (torch.device("cuda:1"), "gpu:1"),  # was converted to "gpu", so the crop model ran on GPU 0
    (torch.device("cpu"), "cpu"),
    (torch.device("mps"), "mps"),
])
def test_torch_devices_are_converted(device, expected):
    assert convert_device_to_string(device) == expected


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_task_with_crop_model_runs_without_cuda(device, tmp_path, monkeypatch):
    # teeth crops with the craniofacial_structures task: totalsegmentator() calls itself
    # with the converted device. On cpu and mps this raised
    # "TypeError: expected string or bytes-like object, got 'NoneType'".
    monkeypatch.setenv("TOTALSEG_HOME_DIR", str(tmp_path))
    img = nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.int16), np.eye(4))
    devices_used = []

    def fake_predict(file_in, file_out, task_id, **kwargs):
        devices_used.append(str(kwargs["device"]))
        return nib.Nifti1Image(np.ones((8, 8, 8), dtype=np.uint8), np.eye(4)), None, None

    with mock.patch.object(python_api, "download_pretrained_weights", lambda task_id: None), \
         mock.patch("totalsegmentator.nnunet.nnUNet_predict_image", fake_predict), \
         mock.patch.object(python_api, "send_usage_stats", lambda *args, **kwargs: None):
        totalsegmentator(img, None, task="teeth", quiet=True, device=device)

    # crop model of craniofacial_structures, craniofacial_structures, teeth
    assert devices_used == [device, device, device]
