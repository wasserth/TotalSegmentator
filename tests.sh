set -e

# Test organ predictions
TotalSegmentator -i tests/reference_files/example_ct.nii.gz -o tests/unittest_prediction --test 1
pytest -v tests/test_end_to_end.py::test_end_to_end::test_prediction_liver
pytest -v tests/test_end_to_end.py::test_end_to_end::test_prediction_vertebrae

# Test organ predictions - fast
TotalSegmentator -i tests/reference_files/example_ct.nii.gz -o tests/unittest_prediction_fast --fast --test 2
pytest -v tests/test_end_to_end.py::test_end_to_end::test_prediction_liver_fast
pytest -v tests/test_end_to_end.py::test_end_to_end::test_prediction_vertebrae_fast
