import pytest
import os
import sys
import glob
import shutil

from totalsegmentator.python_api import totalsegmentator


def cleanup():
    files_to_remove = glob.glob('tests/unittest_prediction*')
    # files_to_remove.append('tests/statistics.json')

    for f in files_to_remove:
        if os.path.isdir(f):
            shutil.rmtree(f)
        else:
            os.remove(f)


if __name__ == "__main__":

    license_number = sys.argv[1]

    # Test multilabel prediction
    totalsegmentator('tests/reference_files/example_ct_sm.nii.gz', 'tests/unittest_prediction.nii.gz', ml=True, body_seg=True, device="cpu")
    pytest.main(['-v', 'tests/test_end_to_end.py::test_end_to_end::test_prediction_multilabel'])

    # Test organ prediction - roi subset
    # 2 cpus:
    #   example_ct_sm.nii.gz: 34s, 3.0GB
    #   example_ct.nii.gz: 36s, 3.0GB
    totalsegmentator('tests/reference_files/example_ct_sm.nii.gz', 'tests/unittest_prediction_roi_subset', roi_subset=['liver', 'brain'], device="cpu")
    pytest.main(['-v', 'tests/test_end_to_end.py::test_end_to_end::test_prediction_liver_roi_subset'])

    # Test organ predictions - fast - statistics
    # 2 cpus: (statistics <1s)
    #   example_ct_sm.nii.gz: 13s, 4.1GB
    #   example_ct.nii.gz: 16s, 4.1GB
    totalsegmentator('tests/reference_files/example_ct_sm.nii.gz', 'tests/unittest_prediction_fast', fast=True, statistics=True, preview=True, device="cpu")
    pytest.main(['-v', 'tests/test_end_to_end.py::test_end_to_end::test_prediction_fast',
                 'tests/test_end_to_end.py::test_end_to_end::test_statistics',
                 'tests/test_end_to_end.py::test_end_to_end::test_preview'])

    # Test organ predictions - fast - multilabel
    totalsegmentator('tests/reference_files/example_ct_sm.nii.gz', 'tests/unittest_prediction_fast.nii.gz', ml=True, fast=True, device="cpu")
    pytest.main(['-v', 'tests/test_end_to_end.py::test_end_to_end::test_prediction_multilabel_fast'])

    # Test organ predictions - fast - multilabel - force split
    totalsegmentator('tests/reference_files/example_ct.nii.gz', 'tests/unittest_prediction_fast_force_split.nii.gz', ml=True, fast=True, force_split=True, device="cpu")
    pytest.main(['-v', 'tests/test_end_to_end.py::test_end_to_end::test_prediction_multilabel_fast_force_split'])

    # Test organ predictions - fast - multilabel - body_seg
    totalsegmentator('tests/reference_files/example_ct_sm.nii.gz', 'tests/unittest_prediction_fast_body_seg.nii.gz', ml=True, fast=True, body_seg=True, device="cpu")
    pytest.main(['-v', 'tests/test_end_to_end.py::test_end_to_end::test_prediction_multilabel_fast_body_seg'])

    # Test vessel predictions
    totalsegmentator('tests/reference_files/example_ct_sm.nii.gz', 'tests/unittest_prediction', task="lung_vessels", device="cpu")
    pytest.main(['-v', 'tests/test_end_to_end.py::test_end_to_end::test_lung_vessels'])

    # Test tissue types (with license) + body_seg
    totalsegmentator('tests/reference_files/example_ct_sm.nii.gz', 'tests/unittest_prediction', task="tissue_types", device="cpu", license_number=license_number)
    pytest.main(['-v', 'tests/test_end_to_end.py::test_end_to_end::test_tissue_types'])

    cleanup()
