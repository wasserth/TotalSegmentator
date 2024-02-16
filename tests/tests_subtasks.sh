set -e

# To run these tests simply do
# ./tests_subtasks.sh <license_key>


# Test vessel predictions
TotalSegmentator -i tests/reference_files/example_ct_sm.nii.gz -o tests/unittest_prediction -ta lung_vessels -d cpu  # ~1min
pytest -v tests/test_end_to_end.py::test_end_to_end::test_lung_vessels

# Test tissue types (without license)
#   (use "|| true" to not abort if this command returns exit code 1, which it should do)
TotalSegmentator -i tests/reference_files/example_ct_sm.nii.gz -o tests/unittest_no_license.nii.gz -ta tissue_types -d cpu --ml || true
pytest -v tests/test_end_to_end.py::test_end_to_end::test_tissue_types_wo_license

# Test tissue types (wrong license)
TotalSegmentator -i tests/reference_files/example_ct_sm.nii.gz -o tests/unittest_wrong_license.nii.gz -ta tissue_types -d cpu --ml -l aca_123456789  || true
pytest -v tests/test_end_to_end.py::test_end_to_end::test_tissue_types_wrong_license

# Test tissue types (with license)
TotalSegmentator -i tests/reference_files/example_ct_sm.nii.gz -o tests/unittest_prediction -ta tissue_types -d cpu -l $1
pytest -v tests/test_end_to_end.py::test_end_to_end::test_tissue_types

# Test appendicular bones (with license)
TotalSegmentator -i tests/reference_files/example_ct_sm.nii.gz -o tests/unittest_prediction -ta appendicular_bones -d cpu
pytest -v tests/test_end_to_end.py::test_end_to_end::test_appendicular_bones

# Cleanup generated files and directories
rm -rf tests/unittest_prediction
