#!/bin/bash

base_path=$(realpath ../..)
script_path="$base_path/totalsegmentator/bin/TotalSegmentator.py"

python3 "$script_path" -i "$base_path/tests/reference_files/example_ct_sm.nii.gz" -o "$base_path/tests/unittest_prediction.nii.gz" -bs --ml -d 1
pytest -v "$base_path/tests/test_end_to_end.py"::test_end_to_end::test_prediction_multilabel