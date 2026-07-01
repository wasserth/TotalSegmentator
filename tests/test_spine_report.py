import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import nibabel as nib
import numpy as np

from totalsegmentator.bin.totalseg_spine_report import create_spine_report
from totalsegmentator.spine_report.measure_verte_height import get_verte_height


REFERENCE_DIR = Path(__file__).parent / "reference_files" / "spine_report"
EXAMPLE_CT = Path(__file__).parent / "reference_files" / "example_ct.nii.gz"
EXAMPLE_NODEINFO = REFERENCE_DIR / "example_nodeinfo.json"
REFERENCE_JSON = REFERENCE_DIR / "example_ct_report" / "spine_report.json"
CACHED_RESULT_DIR = REFERENCE_DIR / "test_result"


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


def test_empty_vertebra_measurements_write_report_outputs(tmp_path):
    shape = (8, 8, 8)
    ct_img = nib.Nifti1Image(np.zeros(shape, dtype=np.float32), np.eye(4))
    empty_vertebra_img = nib.Nifti1Image(np.zeros(shape, dtype=np.uint8), np.eye(4))
    output_json = tmp_path / "spine_report_heights.json"
    preview_file = tmp_path / "spine_report_preview.png"
    combined_preview_file = tmp_path / "spine_report_combined_preview.png"

    get_verte_height(
        ct_img,
        empty_vertebra_img,
        {1: "vertebrae_L1"},
        empty_vertebra_img,
        {},
        preview_file,
        combined_preview_file,
        output_json,
        debug=False,
    )

    assert json.loads(output_json.read_text()) == {"LWS_max": None}
    assert preview_file.exists()
    assert combined_preview_file.exists()


@pytest.fixture(scope="module")
def spine_report_result(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("spine_report")
    output_nifti = tmp_path / "spine_report.nii.gz"
    output_json = tmp_path / "spine_report.json"
    output_log = tmp_path / "spine_report.log"

    assert EXAMPLE_CT.exists(), f"Missing CT fixture: {EXAMPLE_CT}"
    assert (CACHED_RESULT_DIR / "totalseg_vertebrae_pp_refined.nii.gz").exists()

    # Reuse cached model outputs so the test does not run TotalSegmentator.
    for cached_file in CACHED_RESULT_DIR.iterdir():
        if cached_file.suffix == ".gz" or cached_file.name == "contrast_phase.json":
            shutil.copy2(cached_file, tmp_path / cached_file.name)

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

    return {
        "output_nifti": output_nifti,
        "output_json": output_json,
        "output_log": output_log,
    }


def test_spine_report_logs(spine_report_result):
    logs = spine_report_result["output_log"].read_text()

    required_contents = [
        "Skipping TotalSeg vertebrae_pp_refined (already exists)",
    ]

    for content in required_contents:
        assert content in logs, f"required content not found in logs: {content}"

    assert re.search(r"width: 1201, height: \d+", logs)


def test_spine_report_json(spine_report_result):
    output_json = spine_report_result["output_json"]

    with open(REFERENCE_JSON) as f:
        data_ref = json.load(f)
    with open(output_json) as f:
        data_new = json.load(f)
    assert dicts_almost_equal(data_ref, data_new)


def test_spine_report_files_exist(spine_report_result):
    assert os.path.isfile(spine_report_result["output_nifti"])
