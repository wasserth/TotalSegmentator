set -e

# todo
TotalSegmentator --help
pytest -v tests/test_end_to_end.py::test_end_to_end::test_prediction
