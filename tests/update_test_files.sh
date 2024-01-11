set -e

# Run this to regenerate the reference files for the tests.
#
# Usage:
# ./tests/update_test_files.sh <license_key>

TotalSegmentator -i tests/reference_files/example_ct_sm.nii.gz -o tests/reference_files/example_seg.nii.gz -bs --ml -d cpu
TotalSegmentator -i tests/reference_files/example_ct_sm.nii.gz -o tests/reference_files/example_seg_roi_subset.nii.gz --ml -rs liver brain -d cpu
TotalSegmentator -i tests/reference_files/example_ct_sm.nii.gz -o tests/reference_files/example_seg_fast --fast --statistics -sii -p -d cpu
TotalSegmentator -i tests/reference_files/example_ct_sm.nii.gz -o tests/reference_files/example_seg_fast.nii.gz --fast --ml -d cpu
TotalSegmentator -i tests/reference_files/example_ct.nii.gz -o tests/reference_files/example_seg_fast_force_split.nii.gz --fast --ml -fs -d cpu
TotalSegmentator -i tests/reference_files/example_ct_sm.nii.gz -o tests/reference_files/example_seg_fast_body_seg.nii.gz --fast --ml -bs -d cpu
TotalSegmentator -i tests/reference_files/example_ct_sm.nii.gz -o tests/reference_files/example_seg_lung_vessels -ta lung_vessels -d cpu
TotalSegmentator -i tests/reference_files/example_ct_sm.nii.gz -o tests/reference_files/example_seg_tissue_types -ta tissue_types -d cpu -l $1
TotalSegmentator -i tests/reference_files/example_ct_sm.nii.gz -o tests/reference_files/example_seg_appendicular_bones -ta appendicular_bones -d cpu
TotalSegmentator -i tests/reference_files/example_ct_dicom -o tests/reference_files/example_seg_dicom.nii.gz --fast --ml -d cpu

# Manually check if segmentations in tests/reference_files/example_seg_fast_force_split.nii.gz look correct
# (all others have too small FOV for manual inspection)