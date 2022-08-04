set -e

# Test organ predictions
TotalSegmentator -i tests/reference_files/example_ct.nii.gz -o tests/unittest_prediction --fast --test 1
pytest -v tests/test_end_to_end.py::test_end_to_end::test_prediction_liver
pytest -v tests/test_end_to_end.py::test_end_to_end::test_prediction_vertebrae
