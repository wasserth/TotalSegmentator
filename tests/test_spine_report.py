import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import nibabel as nib
import numpy as np

from totalsegmentator.bin.totalseg_spine_report import create_spine_report


REFERENCE_DIR = Path(__file__).parent / "reference_files" / "spine_report"
EXAMPLE_CT = REFERENCE_DIR / "example_ct.nii.gz"
EXAMPLE_NODEINFO = REFERENCE_DIR / "example_nodeinfo.json"
REFERENCE_JSON = REFERENCE_DIR / "example_ct_report" / "spine_report.json"


def dicts_almost_equal(d1, d2):
    """
    Check if two nested dicts are equal.
    For integers, allow a difference of 1.
    """
    if d1.keys() != d2.keys():
        print(f"keys not equal: {d1.keys()}, {d2.keys()}")
        return False

    for key in d1:
        if isinstance(d1[key], dict) and isinstance(d2[key], dict):
            if not dicts_almost_equal(d1[key], d2[key]):
                return False
        elif isinstance(d1[key], int) and isinstance(d2[key], int):
            if abs(d1[key] - d2[key]) > 1:
                print(f"mismatch for key: {key}, d1: {d1[key]}, d2: {d2[key]} -> diff more than 1")
                return False
        elif d1[key] != d2[key]:
            print(f"mismatch for key: {key}, d1: {d1[key]}, d2: {d2[key]} -> not equal")
            return False

    return True


def test_spine_report_imports():
    assert callable(create_spine_report)


def test_spine_report_reference_json():
    with open(REFERENCE_JSON) as f:
        data = json.load(f)

    assert "results" in data
    assert "metadata" in data
    assert "LWS_max" in data["results"]


@pytest.mark.skipif(not EXAMPLE_CT.exists(), reason="large spine report CT fixture is not available")
def test_spine_report_end_to_end(tmp_path):
    output_nifti = tmp_path / "spine_report.nii.gz"
    output_json = tmp_path / "spine_report.json"
    output_log = tmp_path / "spine_report.log"

    # Reuse cached intermediate files when a developer provides them next to the fixture.
    cached_contrast = REFERENCE_DIR / "test_result" / "contrast_phase.json"
    if cached_contrast.exists():
        (tmp_path / "contrast_phase.json").write_text(cached_contrast.read_text())

    subprocess.run(
        [
            sys.executable,
            "-m",
            "totalsegmentator.bin.totalseg_spine_report",
            "-i",
            str(EXAMPLE_CT),
            "-n",
            str(EXAMPLE_NODEINFO),
            "-o",
            str(output_nifti),
            "-j",
            str(output_json),
            "-l",
            str(output_log),
            "-tmp",
            str(tmp_path),
            "-sr",
            "lumbar",
            "--debug",
        ],
        check=True,
    )

    assert os.path.isfile(output_nifti)
    assert os.path.isfile(output_json)
    assert os.path.isfile(output_log)

    with open(REFERENCE_JSON) as f:
        data_ref = json.load(f)
    with open(output_json) as f:
        data_new = json.load(f)
    assert dicts_almost_equal(data_ref, data_new)

    logs = output_log.read_text()
    assert "Getting metadata..." in logs
    assert "Creating report..." in logs
