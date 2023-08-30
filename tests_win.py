import pytest
import os
import sys
import glob
import shutil
import subprocess


if __name__ == "__main__":

    # Test python api
    # Test organ predictions - fast - statistics
    totalsegmentator('tests/reference_files/example_ct_sm.nii.gz', 'tests/unittest_prediction_fast', fast=True, device="cpu")
    pytest.main(['-v', 'tests/test_end_to_end.py::test_end_to_end::test_prediction_fast'])

    # Test terminal
    # Test organ predictions - fast - multilabel
    # makes correct path for windows and linux. Only required for terminal call. Within python
    # I can always / and i will correctly be interpreted on windows
    file_in = os.path.join("tests", "reference_files", "example_ct_sm.nii.gz")  
    print("File_in path:")
    print(file_in)
    file_out = os.path.join("tests", "unittest_prediction_fast.nii.gz")
    subprocess.call(f"TotalSegmentator -i {file_in} -o {file_out} --fast --ml -d cpu", shell=True)
    pytest.main(['-v', 'tests/test_end_to_end.py::test_end_to_end::test_prediction_multilabel_fast'])
    os.remove(file_out)
