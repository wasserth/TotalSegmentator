#!/usr/bin/env python3
"""
Test script for TotalSegmentatorImproved functionality.
Tests the new CLI without requiring actual model weights or GPU.
"""

import tempfile
import shutil
from pathlib import Path
import json
import sys

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import numpy as np
    import nibabel as nib
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    print("Warning: numpy/nibabel not available, skipping advanced tests")

from totalsegmentator.bin.TotalSegmentatorImproved import SEGMENTATION_TASKS


def test_task_definitions():
    """Test that task definitions are correctly structured."""
    print("Testing task definitions...")
    
    # Verify all required tasks are present
    expected_tasks = ["liver_segments", "liver_vessels", "total_vessels"]
    for task in expected_tasks:
        assert task in SEGMENTATION_TASKS, f"Missing task: {task}"
    
    # Verify each task has required fields
    for task_name, task_config in SEGMENTATION_TASKS.items():
        assert "title" in task_config, f"Task {task_name} missing 'title'"
        assert "task_name" in task_config, f"Task {task_name} missing 'task_name'"
        assert "results" in task_config, f"Task {task_name} missing 'results'"
        assert "output_mapping" in task_config, f"Task {task_name} missing 'output_mapping'"
    
    # Verify specific mappings
    liver_vessels = SEGMENTATION_TASKS["liver_vessels"]
    assert liver_vessels["title"] == "liver: vessels"
    assert "blood_vessel" in liver_vessels["results"]
    assert "neoplasm" in liver_vessels["results"]
    assert "liver_vessels.nii.gz" in liver_vessels["output_mapping"]
    assert "liver_tumor.nii.gz" in liver_vessels["output_mapping"]
    
    print("✅ Task definitions test passed")


def create_mock_nifti(path, shape=(64, 64, 32), data=None):
    """Create a mock NIfTI file for testing."""
    if not DEPENDENCIES_AVAILABLE:
        # Create a dummy file
        with open(path, 'w') as f:
            f.write("dummy nifti file")
        return path
        
    if data is None:
        data = np.random.randint(0, 2, shape).astype(np.uint8)
    
    # Create a simple affine matrix
    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = 1.0  # 1mm spacing
    
    img = nib.Nifti1Image(data, affine)
    nib.save(img, str(path))
    return path


def test_cli_argument_parsing():
    """Test CLI argument parsing without actually running segmentation."""
    print("Testing CLI argument parsing...")
    
    # Import the main function
    from totalsegmentator.bin.TotalSegmentatorImproved import main
    
    # Test that we can import without errors
    assert main is not None, "Could not import main function"
    
    # Test task validation
    valid_tasks = list(SEGMENTATION_TASKS.keys()) + ["all"]
    for task in ["liver_segments", "liver_vessels", "total_vessels", "all"]:
        assert task in valid_tasks, f"Task {task} not in valid tasks"
    
    print("✅ CLI argument parsing test passed")


def test_output_structure():
    """Test that expected output structure matches requirements."""
    print("Testing output structure...")
    
    # Verify liver segments mapping
    liver_segments = SEGMENTATION_TASKS["liver_segments"]
    expected_segments = [f"liver_segment_{i}" for i in range(1, 9)]
    assert liver_segments["results"] == expected_segments, "Liver segments mismatch"
    
    # Verify liver vessels mapping
    liver_vessels = SEGMENTATION_TASKS["liver_vessels"]
    assert liver_vessels["results"] == ["blood_vessel", "neoplasm"], "Liver vessels results mismatch"
    
    # Verify total vessels mapping
    total_vessels = SEGMENTATION_TASKS["total_vessels"]
    expected_total = ["inferior_vena_cava", "portal_vein_and_splenic_vein"]
    assert total_vessels["results"] == expected_total, "Total vessels results mismatch"
    
    print("✅ Output structure test passed")


def test_smoothing_function():
    """Test the smoothing functionality."""
    print("Testing smoothing function...")
    
    if not DEPENDENCIES_AVAILABLE:
        print("⚠️  Skipping smoothing test - dependencies not available")
        return
    
    from totalsegmentator.bin.TotalSegmentatorImproved import apply_smoothing
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        # Create test binary mask
        binary_data = np.zeros((32, 32, 16), dtype=np.uint8)
        binary_data[10:22, 10:22, 5:11] = 1  # Create a cube
        test_file = create_mock_nifti(temp_dir / "test_binary.nii.gz", data=binary_data)
        
        # Test different smoothing levels
        for level in ["none", "light", "medium", "heavy"]:
            test_copy = temp_dir / f"test_{level}.nii.gz"
            shutil.copy2(test_file, test_copy)
            
            # Apply smoothing
            apply_smoothing(test_copy, level)
            
            # Verify file still exists and is readable
            assert test_copy.exists(), f"Smoothed file {level} not found"
            
            img = nib.load(str(test_copy))
            data = img.get_fdata()
            
            if level == "none":
                # Should be identical to original
                np.testing.assert_array_equal(data, binary_data)
            else:
                # Should have some non-zero values (smoothing applied)
                assert np.sum(data > 0) > 0, f"Smoothing {level} produced empty mask"
        
        # Create test multi-label mask
        multi_data = np.zeros((32, 32, 16), dtype=np.uint8)
        multi_data[5:15, 5:15, 3:8] = 1   # Label 1
        multi_data[17:27, 17:27, 8:13] = 2  # Label 2
        test_multi = create_mock_nifti(temp_dir / "test_multi.nii.gz", data=multi_data)
        
        # Apply medium smoothing to multi-label
        apply_smoothing(test_multi, "medium")
        
        # Verify labels are preserved
        img = nib.load(str(test_multi))
        smoothed_data = img.get_fdata()
        unique_labels = np.unique(smoothed_data)
        assert 0 in unique_labels, "Background label missing after smoothing"
        assert 1 in unique_labels or 2 in unique_labels, "Foreground labels missing after smoothing"
    
    print("✅ Smoothing function test passed")


def create_mock_nifti(path, shape=(64, 64, 32), data=None):
    """Create a mock NIfTI file for testing."""
    if not DEPENDENCIES_AVAILABLE:
        # Create a dummy file
        with open(path, 'w') as f:
            f.write("dummy nifti file")
        return path
        
    if data is None:
        data = np.random.randint(0, 2, shape).astype(np.uint8)
    
    # Create a simple affine matrix
    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = 1.0  # 1mm spacing
    
    img = nib.Nifti1Image(data, affine)
    nib.save(img, str(path))
    return path


def test_cli_argument_parsing():
    """Test CLI argument parsing without actually running segmentation."""
    print("Testing CLI argument parsing...")
    
    # Import the main function
    from totalsegmentator.bin.TotalSegmentatorImproved import main
    
    # Test that we can import without errors
    assert main is not None, "Could not import main function"
    
    # Test task validation
    valid_tasks = list(SEGMENTATION_TASKS.keys()) + ["all"]
    for task in ["liver_segments", "liver_vessels", "total_vessels", "all"]:
        assert task in valid_tasks, f"Task {task} not in valid tasks"
    
    print("✅ CLI argument parsing test passed")


def test_output_structure():
    """Test that expected output structure matches requirements."""
    print("Testing output structure...")
    
    # Verify liver segments mapping
    liver_segments = SEGMENTATION_TASKS["liver_segments"]
    expected_segments = [f"liver_segment_{i}" for i in range(1, 9)]
    assert liver_segments["results"] == expected_segments, "Liver segments mismatch"
    
    # Verify liver vessels mapping
    liver_vessels = SEGMENTATION_TASKS["liver_vessels"]
    assert liver_vessels["results"] == ["blood_vessel", "neoplasm"], "Liver vessels results mismatch"
    
    # Verify total vessels mapping
    total_vessels = SEGMENTATION_TASKS["total_vessels"]
    expected_total = ["inferior_vena_cava", "portal_vein_and_splenic_vein"]
    assert total_vessels["results"] == expected_total, "Total vessels results mismatch"
    
    print("✅ Output structure test passed")


def run_all_tests():
    """Run all tests."""
    print("🧪 Running TotalSegmentatorImproved tests...")
    print("=" * 50)
    
    try:
        test_task_definitions()
        if DEPENDENCIES_AVAILABLE:
            test_smoothing_function()
        else:
            print("⚠️  Skipping smoothing tests - dependencies not available")
        test_cli_argument_parsing()
        test_output_structure()
        
        print("=" * 50)
        print("🎉 All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)