set -e

# To run these tests simply do 
# ./tests.sh <license_key>


# Test multilabel prediction
TotalSegmentator -i tests/reference_files/example_ct_sm.nii.gz -o tests/unittest_prediction.nii.gz -bs --ml -d cpu
pytest -v tests/test_end_to_end.py::test_end_to_end::test_prediction_multilabel

# Test organ prediction - roi subset
# 2 cpus:
#   example_ct_sm.nii.gz: 34s, 3.0GB
#   example_ct.nii.gz: 36s, 3.0GB
TotalSegmentator -i tests/reference_files/example_ct_sm.nii.gz -o tests/unittest_prediction_roi_subset.nii.gz --ml -rs liver brain -d cpu
pytest -v tests/test_end_to_end.py::test_end_to_end::test_prediction_liver_roi_subset

# Test organ predictions - fast - statistics
# 2 cpus: (statistics <1s)
#   example_ct_sm.nii.gz: 13s, 4.1GB
#   example_ct.nii.gz: 16s, 4.1GB
TotalSegmentator -i tests/reference_files/example_ct_sm.nii.gz -o tests/unittest_prediction_fast --fast --statistics -sii -p -d cpu
pytest -v tests/test_end_to_end.py::test_end_to_end::test_prediction_fast
pytest -v tests/test_end_to_end.py::test_end_to_end::test_statistics
pytest -v tests/test_end_to_end.py::test_end_to_end::test_preview

# Test organ predictions - fast - multilabel
TotalSegmentator -i tests/reference_files/example_ct_sm.nii.gz -o tests/unittest_prediction_fast.nii.gz --fast --ml -d cpu
pytest -v tests/test_end_to_end.py::test_end_to_end::test_prediction_multilabel_fast

# Test organ predictions - fast - multilabel - force split
TotalSegmentator -i tests/reference_files/example_ct.nii.gz -o tests/unittest_prediction_fast_force_split.nii.gz --fast --ml -fs -d cpu
pytest -v tests/test_end_to_end.py::test_end_to_end::test_prediction_multilabel_fast_force_split

# Test organ predictions - fast - multilabel - body_seg
TotalSegmentator -i tests/reference_files/example_ct_sm.nii.gz -o tests/unittest_prediction_fast_body_seg.nii.gz --fast --ml -bs -d cpu
pytest -v tests/test_end_to_end.py::test_end_to_end::test_prediction_multilabel_fast_body_seg

# Test vessel predictions
TotalSegmentator -i tests/reference_files/example_ct_sm.nii.gz -o tests/unittest_prediction -ta lung_vessels -d cpu  # ~1min
pytest -v tests/test_end_to_end.py::test_end_to_end::test_lung_vessels

# Test tissue types (without license)
TotalSegmentator -i tests/reference_files/example_ct_sm.nii.gz -o tests/unittest_no_license.nii.gz -ta tissue_types -d cpu --ml
pytest -v tests/test_end_to_end.py::test_end_to_end::test_tissue_types_wo_license

# Test tissue types (wrong license)
TotalSegmentator -i tests/reference_files/example_ct_sm.nii.gz -o tests/unittest_wrong_license.nii.gz -ta tissue_types -d cpu --ml -l aca_123456789
pytest -v tests/test_end_to_end.py::test_end_to_end::test_tissue_types_wrong_license

# Test tissue types (with license)
TotalSegmentator -i tests/reference_files/example_ct_sm.nii.gz -o tests/unittest_prediction -ta tissue_types -d cpu -l $1
pytest -v tests/test_end_to_end.py::test_end_to_end::test_tissue_types

# Test appendicular bones (with license)
TotalSegmentator -i tests/reference_files/example_ct_sm.nii.gz -o tests/unittest_prediction -ta appendicular_bones -d cpu
pytest -v tests/test_end_to_end.py::test_end_to_end::test_appendicular_bones

# Cleanup generated files and directories
rm -rf tests/unittest_prediction_roi_subset
rm -rf tests/unittest_prediction_fast
rm -rf tests/unittest_prediction
rm tests/unittest_prediction.nii.gz
rm tests/unittest_prediction_fast.nii.gz
rm tests/unittest_prediction_fast_force_split.nii.gz
rm tests/unittest_prediction_fast_body_seg.nii.gz
# rm tests/statistics.json
