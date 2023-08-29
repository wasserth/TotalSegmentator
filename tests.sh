set -e

echo "License key: $1"

# # Test organ predictions
# TotalSegmentator -i tests/reference_files/example_ct.nii.gz -o tests/unittest_prediction --test 1
# pytest -v tests/test_end_to_end.py::test_end_to_end::test_prediction_liver
# pytest -v tests/test_end_to_end.py::test_end_to_end::test_prediction_vertebrae

# # Test multilabel prediction
# TotalSegmentator -i tests/reference_files/example_ct.nii.gz -o tests/unittest_prediction.nii.gz --ml --test 1
# pytest -v tests/test_end_to_end.py::test_end_to_end::test_prediction_multilabel

# Test organ prediction - roi subset
# 20 cpu
#   example_ct_sm.nii.gz: 20s, 3.1GB
#   example_ct.nii.gz: 22s, 3.1GB
# 2 cpus:
#   example_ct_sm.nii.gz: 34s, 3.0GB
#   example_ct.nii.gz: 36s, 3.0GB
TotalSegmentator -i tests/reference_files/example_ct_sm.nii.gz -o tests/unittest_prediction_roi_subset -rs liver brain -d cpu
pytest -v tests/test_end_to_end.py::test_end_to_end::test_prediction_liver_roi_subset

# Test organ predictions - fast
# 2 cpus: (w/o statistics)
#   example_ct_sm.nii.gz: 13s, 4.1GB
#   example_ct.nii.gz: 16s, 4.1GB
# TotalSegmentator -i tests/reference_files/example_ct_sm.nii.gz -o tests/unittest_prediction_fast --fast --statistics -d cpu
# pytest -v tests/test_end_to_end.py::test_end_to_end::test_prediction_liver_fast
# pytest -v tests/test_end_to_end.py::test_end_to_end::test_prediction_vertebrae_fast
# pytest -v tests/test_end_to_end.py::test_end_to_end::test_statistics

# # Test vessel predictions
# TotalSegmentator -i tests/reference_files/example_ct.nii.gz -o tests/unittest_prediction -ta lung_vessels --test 3 -d cpu
# pytest -v tests/test_end_to_end.py::test_end_to_end::test_lung_vessels