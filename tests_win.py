import pytest
import os
import sys
import glob
import shutil
import subprocess

if __name__ == "__main__":
    # Test organ predictions - fast - multilabel
    file_in = os.path.join("tests", "reference_files", "example_ct_sm.nii.gz")  # makes correct path for windows and linux
    file_out = os.path.join("tests", "unittest_prediction_fast.nii.gz")
    subprocess.call(f"TotalSegmentator -i {file_in} -o {file_out} --fast --ml -d cpu", shell=True)
    pytest.main(['-v', 'tests/test_end_to_end.py::test_end_to_end::test_prediction_multilabel_fast'])
    os.remove(file_out)