import pytest
import os
import sys
import glob
import shutil
import subprocess

from totalsegmentator.python_api import totalsegmentator

"""
Run a complete prediction locally with GPU and evaluate Dice score.
This is not possible on github actions due to missing GPU.
"""

if __name__ == "__main__":

    # Todo: add test files as release to github and download each time this is run

    # Test python api
    # Test organ predictions - fast - statistics
    totalsegmentator('tests/reference_files/example_ct_sm.nii.gz', 'tests/unittest_prediction_fast', fast=True, device="cpu")
    pytest.main(['-v', 'tests/test_end_to_end.py::test_end_to_end::test_prediction_fast'])
    shutil.rmtree('tests/unittest_prediction_fast')

    # todo: also track runtime and memory consumption!!
