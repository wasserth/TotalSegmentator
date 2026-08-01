import unittest

from totalsegmentator_mcp.utils import validate_segment_request


class TestMCPValidation(unittest.TestCase):
    def test_valid_total_request(self):
        validate_segment_request("total", "standard", ["spleen"])

    def test_unknown_roi_raises(self):
        with self.assertRaises(ValueError):
            validate_segment_request("total", "standard", ["not_a_real_roi"])


if __name__ == "__main__":
    unittest.main()
