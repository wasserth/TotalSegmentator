set -e

# To run these tests do
# ./tests/tests.sh

# Test device type selection function
pytest -v tests/test_device_type.py

# Test - multilabel prediction
TotalSegmentator -i tests/reference_files/example_ct_sm.nii.gz -o tests/unittest_prediction.nii.gz -bs --ml -d cpu
pytest -v tests/test_end_to_end.py::test_end_to_end::test_prediction_multilabel

# Test - roi subset
# 2 cpus:
#   example_ct_sm.nii.gz: 34s, 3.0GB
#   example_ct.nii.gz: 36s, 3.0GB
TotalSegmentator -i tests/reference_files/example_ct_sm.nii.gz -o tests/unittest_prediction_roi_subset.nii.gz --ml -rs liver brain -d cpu
pytest -v tests/test_end_to_end.py::test_end_to_end::test_prediction_liver_roi_subset

# Test - roi subset - MR
TotalSegmentator -i tests/reference_files/example_mr_sm.nii.gz -o tests/unittest_prediction_roi_subset_mr.nii.gz -ta total_mr --ml -rs liver brain -d cpu
pytest -v tests/test_end_to_end.py::test_end_to_end::test_prediction_liver_roi_subset_mr

# Test - fast - statistics
# 2 cpus: (statistics <1s)
#   example_ct_sm.nii.gz: 13s, 4.1GB
#   example_ct.nii.gz: 16s, 4.1GB
TotalSegmentator -i tests/reference_files/example_ct_sm.nii.gz -o tests/unittest_prediction_fast --fast --statistics -sii -p -d cpu
pytest -v tests/test_end_to_end.py::test_end_to_end::test_prediction_fast
pytest -v tests/test_end_to_end.py::test_end_to_end::test_statistics
pytest -v tests/test_end_to_end.py::test_end_to_end::test_preview

# Test - fast - multilabel
TotalSegmentator -i tests/reference_files/example_ct_sm.nii.gz -o tests/unittest_prediction_fast.nii.gz --fast --ml -d cpu
pytest -v tests/test_end_to_end.py::test_end_to_end::test_prediction_multilabel_fast

# Test - fast - multilabel - force split
TotalSegmentator -i tests/reference_files/example_ct.nii.gz -o tests/unittest_prediction_fast_force_split.nii.gz --fast --ml -fs -d cpu
pytest -v tests/test_end_to_end.py::test_end_to_end::test_prediction_multilabel_fast_force_split

# Test - fast - multilabel - body_seg
TotalSegmentator -i tests/reference_files/example_ct_sm.nii.gz -o tests/unittest_prediction_fast_body_seg.nii.gz --fast --ml -bs -d cpu
pytest -v tests/test_end_to_end.py::test_end_to_end::test_prediction_multilabel_fast_body_seg

# Test phase prediction
totalseg_get_phase -i tests/reference_files/example_ct_sm.nii.gz -o tests/unittest_phase_prediction.json
pytest -v tests/test_end_to_end.py::test_end_to_end::test_phase_prediction

# Cleanup generated files and directories
rm -rf tests/unittest_prediction_roi_subset
rm -rf tests/unittest_prediction_fast
rm tests/unittest_prediction.nii.gz
rm tests/unittest_prediction_fast.nii.gz
rm tests/unittest_prediction_fast_force_split.nii.gz
rm tests/unittest_prediction_fast_body_seg.nii.gz
rm tests/unittest_phase_prediction.json
# rm tests/statistics.json
