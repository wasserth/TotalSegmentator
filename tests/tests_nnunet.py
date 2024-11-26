import pytest
import os
import sys
import glob
import shutil
import subprocess
from pathlib import Path

import nibabel as nib

from totalsegmentator.python_api import totalsegmentator
from totalsegmentator.libs import download_pretrained_weights

"""
To run these tests do
python tests/tests_nnunet.py
"""

def run_tests_and_exit_on_failure():

    download_pretrained_weights(297)  # total 3mm

    weights_dir = Path.home() / ".totalsegmentator" / "nnunet" / "results"
    os.environ["nnUNet_results"] = str(weights_dir)
    print(f"Using weights directory: {weights_dir}")

    os.makedirs("tests/nnunet_input_files", exist_ok=True)
    shutil.copy("tests/reference_files/example_ct_sm.nii.gz", "tests/nnunet_input_files/example_ct_sm_0000.nii.gz")

    subprocess.call(f"nnUNetv2_predict -i tests/nnunet_input_files -o tests/nnunet_input_files -d 297 -tr nnUNetTrainer_4000epochs_NoMirroring -c 3d_fullres -f 0 -device cpu", shell=True)

    r = pytest.main(["-v", "tests/test_end_to_end.py::test_end_to_end::test_nnunet_prediction"])
    shutil.rmtree("tests/nnunet_input_files")
    if r != 0: sys.exit("Test failed: test_nnunet_prediction")


if __name__ == "__main__":
    run_tests_and_exit_on_failure()