import pytest
import os
import sys
import glob
import shutil
import subprocess

import nibabel as nib

from totalsegmentator.python_api import totalsegmentator


def run_tests_and_exit_on_failure():

    # Test python api
    # Test organ predictions - fast
    totalsegmentator("tests/reference_files/example_ct_sm.nii.gz", "tests/unittest_prediction_fast", fast=True, device="cpu")
    r = pytest.main(["-v", "tests/test_end_to_end.py::test_end_to_end::test_prediction_fast"])
    shutil.rmtree("tests/unittest_prediction_fast")
    if r != 0: sys.exit("Test failed: test_prediction_fast")

    # Test python api 1 - nifti input, filepath output
    input_img = nib.load("tests/reference_files/example_ct_sm.nii.gz")
    totalsegmentator(input_img, "tests/unittest_prediction_fast", fast=True, device="cpu")
    r = pytest.main(["-v", "tests/test_end_to_end.py::test_end_to_end::test_prediction_fast"])
    shutil.rmtree("tests/unittest_prediction_fast")
    if r != 0: sys.exit("Test failed: test_prediction_fast with Nifti input")

    # Test python api 2 - nifti input, nifti output
    input_img = nib.load("tests/reference_files/example_ct_sm.nii.gz")
    output_img = totalsegmentator(input_img, None, fast=True, device="cpu")
    nib.save(output_img, "tests/unittest_prediction_fast.nii.gz")
    r = pytest.main(["-v", "tests/test_end_to_end.py::test_end_to_end::test_prediction_multilabel_fast"])
    os.remove("tests/unittest_prediction_fast.nii.gz")
    if r != 0: sys.exit("Test failed: test_prediction_fast with Nifti input and output")

    # Test python api 3 - nifti input, nifti output, ml=True (has no effect here), roi_subset=['liver', 'brain']
    input_img = nib.load("tests/reference_files/example_ct_sm.nii.gz")
    output_img = totalsegmentator(input_img, None, device="cpu", ml=True, roi_subset=['liver', 'brain'])
    nib.save(output_img, "tests/unittest_prediction_roi_subset.nii.gz")
    r = pytest.main(["-v", "tests/test_end_to_end.py::test_end_to_end::test_prediction_liver_roi_subset"])
    os.remove("tests/unittest_prediction_roi_subset.nii.gz")
    if r != 0: sys.exit("Test failed: test_prediction_fast with Nifti input and output")

    # Test terminal
    # Test organ predictions - fast - multilabel
    # makes correct path for windows and linux. Only required for terminal call. Within python
    # I can always use / and it will correctly be interpreted on windows
    file_in = os.path.join("tests", "reference_files", "example_ct_sm.nii.gz")
    file_out = os.path.join("tests", "unittest_prediction_fast.nii.gz")
    subprocess.call(f"TotalSegmentator -i {file_in} -o {file_out} --fast --ml -d cpu", shell=True)
    r = pytest.main(["-v", "tests/test_end_to_end.py::test_end_to_end::test_prediction_multilabel_fast"])
    os.remove(file_out)
    if r != 0: sys.exit("Test failed: test_prediction_multilabel_fast")

    # Test Dicom input
    totalsegmentator("tests/reference_files/example_ct_dicom", "tests/unittest_prediction_dicom.nii.gz", fast=True, ml=True, device="cpu")
    r = pytest.main(["-v", "tests/test_end_to_end.py::test_end_to_end::test_prediction_dicom"])
    os.remove("tests/unittest_prediction_dicom.nii.gz")
    if r != 0: sys.exit("Test failed: test_prediction_dicom")


if __name__ == "__main__":
    run_tests_and_exit_on_failure()