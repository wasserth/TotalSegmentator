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

    # Download weights
    # download_pretrained_weights(297)  # total 3mm
    download_pretrained_weights(300)  # body 6mm

    # Set nnUNet_results env var
    weights_dir = Path.home() / ".totalsegmentator" / "nnunet" / "results"
    os.environ["nnUNet_results"] = str(weights_dir)
    print(f"Using weights directory: {weights_dir}")

    # Copy example file
    os.makedirs("tests/nnunet_input_files", exist_ok=True)
    shutil.copy("tests/reference_files/example_ct_sm.nii.gz", "tests/nnunet_input_files/example_ct_sm_0000.nii.gz")

    # Run nnunet
    # Task 297
    # subprocess.call("nnUNetv2_predict -i tests/nnunet_input_files -o tests/nnunet_input_files -d 297 -tr nnUNetTrainer_4000epochs_NoMirroring -c 3d_fullres -f 0 -device cpu", shell=True)
    # Task 300
    subprocess.call("nnUNetv2_predict -i tests/nnunet_input_files -o tests/nnunet_input_files -d 300 -tr nnUNetTrainer -c 3d_fullres -f 0 -device cpu", shell=True)

    # Check if output file exists
    assert os.path.exists("tests/nnunet_input_files/example_ct_sm.nii.gz"), "A nnunet output file was not generated."

    # Clean up
    shutil.rmtree("tests/nnunet_input_files")


if __name__ == "__main__":
    run_tests_and_exit_on_failure()