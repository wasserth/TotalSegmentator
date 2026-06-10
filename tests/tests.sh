set -e

# To run these tests do
# ./tests/tests.sh <license_key>

# Test device type selection function
pytest -v tests/test_device_type.py

# Test config helpers
pytest -v tests/test_config.py

# Test task registry + totalseg_info command (no GPU/model needed)
pytest -v tests/test_registry.py

# Test batch command helpers (no GPU/model needed)
pytest -v tests/test_batch.py

# Smoke test the introspection commands
totalseg_info --list-tasks > /dev/null
totalseg_info --classes -ta total --json > /dev/null
TotalSegmentator --list-tasks > /dev/null

# Test - multilabel prediction (also exercises --report manifest)
TotalSegmentator -i tests/reference_files/example_ct_sm.nii.gz -o tests/unittest_prediction.nii.gz -bs --ml -d cpu --report tests/unittest_run_report.json
python -c "import json; r=json.load(open('tests/unittest_run_report.json')); assert r['task']=='total' and r['num_classes']>0, r"
pytest -v tests/test_end_to_end.py::test_end_to_end::test_prediction_multilabel

# Test - batch processing. Verifies that the cached-predictor (batch) path produces a
# segmentation identical to the normal single-image path above (same options, same image).
mkdir -p tests/unittest_batch_in
cp tests/reference_files/example_ct_sm.nii.gz tests/unittest_batch_in/
totalseg_batch -i tests/unittest_batch_in -o tests/unittest_batch_out -bs --ml -d cpu
python -c "import nibabel as nib, numpy as np; a=nib.load('tests/unittest_prediction.nii.gz').get_fdata(); b=nib.load('tests/unittest_batch_out/example_ct_sm/segmentation.nii.gz').get_fdata(); assert np.array_equal(a, b), 'batch output differs from single-image output'"

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

# Research utilities: cohort statistics aggregation + extra metrics (no GPU needed)
pytest -v tests/test_research_utils.py
# Aggregate the statistics.json just produced into a cohort table
totalseg_aggregate_stats -i tests/unittest_prediction_fast -o tests/unittest_cohort_stats.csv
python -c "import csv; rows=list(csv.DictReader(open('tests/unittest_cohort_stats.csv'))); assert len(rows)>0, 'empty cohort table'"

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

# Test body stats prediction
totalseg_get_body_stats -i tests/reference_files/example_ct_sm.nii.gz -m ct -o tests/unittest_body_stats_prediction.json -d cpu -l $1
pytest -v tests/test_end_to_end.py::test_end_to_end::test_body_stats_prediction

# Cleanup generated files and directories
rm -rf tests/unittest_prediction_roi_subset
rm -rf tests/unittest_prediction_fast
rm tests/unittest_prediction.nii.gz
rm tests/unittest_prediction_fast.nii.gz
rm tests/unittest_prediction_fast_force_split.nii.gz
rm tests/unittest_prediction_fast_body_seg.nii.gz
rm tests/unittest_phase_prediction.json
rm tests/unittest_body_stats_prediction.json
rm tests/unittest_run_report.json
rm tests/unittest_cohort_stats.csv
rm -rf tests/unittest_batch_in
rm -rf tests/unittest_batch_out
# rm tests/statistics.json
