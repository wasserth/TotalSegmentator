import json
from unittest import mock

import nibabel as nib
import numpy as np
import pytest

from totalsegmentator import config as config_module
from totalsegmentator import python_api
from totalsegmentator.bin.totalseg_download_weights import TASK_TO_ID
from totalsegmentator.map_tasks_config import TASK_CONFIGS
from totalsegmentator.python_api import totalsegmentator


@pytest.fixture
def weights_requested_by_run(tmp_path, monkeypatch):
    """Run totalsegmentator() without weights and return the task ids it wanted to download."""
    monkeypatch.setenv("TOTALSEG_HOME_DIR", str(tmp_path))
    config_module.setup_totalseg()
    config_file = tmp_path / "config.json"
    config = json.loads(config_file.read_text())
    config["license_number"] = "aca_12345678901234"  # the licensed tasks only check that one is set
    config_file.write_text(json.dumps(config))

    img = nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.int16), np.eye(4))

    def fake_predict(file_in, file_out, task_id, **kwargs):
        return nib.Nifti1Image(np.ones((8, 8, 8), dtype=np.uint8), np.eye(4)), None, None

    def _run(task, **kwargs):
        requested = []
        with mock.patch.object(python_api, "download_pretrained_weights", requested.append), \
             mock.patch("totalsegmentator.nnunet.nnUNet_predict_image", fake_predict), \
             mock.patch.object(python_api, "send_usage_stats", lambda *args, **kwargs: None), \
             mock.patch.object(python_api, "convert_device_to_string", lambda device: "cpu"):
            totalsegmentator(img, None, task=task, quiet=True, device="cpu", **kwargs)
        return requested

    return _run


@pytest.mark.parametrize("task", sorted(task for task in TASK_TO_ID if task in TASK_CONFIGS))
def test_download_fetches_all_weights_a_run_needs(task, weights_requested_by_run):
    # `totalseg_download_weights -t <task>` is how the weights get onto a machine without
    # internet, so it also has to fetch the model which crops the image for that task.
    requested = weights_requested_by_run(task)
    if task.startswith("total"):
        requested += weights_requested_by_run(task, roi_subset=["liver"])

    missing = sorted(set(requested) - set(TASK_TO_ID[task]))
    assert not missing, f"totalseg_download_weights -t {task} does not download {missing}"
