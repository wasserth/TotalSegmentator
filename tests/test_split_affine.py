"""
Tests for the triple-split sub-volume logic extracted into
split_image_into_parts(), save_merged_predictions(), and reassemble_image()
in totalsegmentator/nnunet.py.

These call the actual production functions, not local reimplementations.
No nnU-Net or GPU required — heavy dependencies are mocked at import time.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import nibabel as nib
import pytest

from tests.mock_imports import mock_imports

# Modules to mock so we can import totalsegmentator.nnunet without
# torch, nnunetv2, or other heavy deps.
_MOCK_MODULES = [
    "torch", "torch.cuda", "torch.backends", "torch.backends.cudnn",
    "nnunetv2", "nnunetv2.utilities", "nnunetv2.utilities.find_class_by_name",
    "nnunetv2.utilities.file_path_utilities",
    "nnunetv2.inference", "nnunetv2.inference.predict_from_raw_data",
    "SimpleITK", "xvfbwrapper", "dicom2nifti", "pyarrow", "xmltodict",
    "rt_utils", "pydicom", "cv2",
    "totalsegmentator.custom_trainers",
    "totalsegmentator.libs",
    "totalsegmentator.config",
    "totalsegmentator.dicom_io",
    "totalsegmentator.postprocessing",
    "totalsegmentator.cropping",
]

with mock_imports(*_MOCK_MODULES) as _mock_ctx:
    from totalsegmentator.nnunet import (
        split_image_into_parts,
        save_merged_predictions,
        reassemble_image,
    )


def make_test_image(shape=(64, 64, 300), voxel_size=(1.5, 1.5, 1.5)):
    """Create a synthetic nifti image with a non-trivial affine."""
    affine = np.eye(4)
    affine[0, 0] = voxel_size[0]
    affine[1, 1] = voxel_size[1]
    affine[2, 2] = voxel_size[2]
    affine[:3, 3] = [-48.0, -48.0, -225.0]
    data = np.random.randint(0, 100, shape, dtype=np.int16)
    return nib.Nifti1Image(data, affine), data


def make_rotated_test_image(shape=(64, 64, 300)):
    """Create a synthetic nifti image with off-diagonal rotation in the affine."""
    angle = np.radians(15)
    affine = np.eye(4)
    affine[0, 0] = 1.5 * np.cos(angle)
    affine[0, 2] = 1.5 * np.sin(angle)
    affine[1, 1] = 1.5
    affine[2, 0] = -1.5 * np.sin(angle)
    affine[2, 2] = 1.5 * np.cos(angle)
    affine[:3, 3] = [-48.0, -48.0, -225.0]
    data = np.random.randint(0, 100, shape, dtype=np.int16)
    return nib.Nifti1Image(data, affine), data


class TestSplitImageIntoParts:
    """Tests for split_image_into_parts() — the production split function."""

    def test_s01_origin_unchanged(self):
        img, _ = make_test_image()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            _, part_affines, _, _ = split_image_into_parts(img, tmp_dir)
            np.testing.assert_array_equal(
                part_affines["s01"][:3, 3],
                img.affine[:3, 3],
                err_msg="s01 origin should match the original image origin",
            )

    def test_s02_origin_shifted(self):
        img, _ = make_test_image()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            _, part_affines, third, margin = split_image_into_parts(img, tmp_dir)
            s02_start = third + 1 - margin
            expected_origin = img.affine[:3, 3] + s02_start * img.affine[:3, 2]
            np.testing.assert_array_almost_equal(
                part_affines["s02"][:3, 3],
                expected_origin,
                err_msg="s02 origin should be shifted by s02_start voxels along z",
            )

    def test_s03_origin_shifted(self):
        img, _ = make_test_image()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            _, part_affines, third, margin = split_image_into_parts(img, tmp_dir)
            s03_start = third * 2 + 1 - margin
            expected_origin = img.affine[:3, 3] + s03_start * img.affine[:3, 2]
            np.testing.assert_array_almost_equal(
                part_affines["s03"][:3, 3],
                expected_origin,
                err_msg="s03 origin should be shifted by s03_start voxels along z",
            )

    def test_voxel_to_world_consistency(self):
        """Voxel (0,0,0) of each sub-volume should map to the world coordinate
        of its starting slice in the original image."""
        img, _ = make_test_image()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            _, part_affines, third, margin = split_image_into_parts(img, tmp_dir)

            starts = {"s01": 0, "s02": third + 1 - margin, "s03": third * 2 + 1 - margin}
            for name, start_z in starts.items():
                part_world = part_affines[name] @ np.array([0, 0, 0, 1])
                orig_world = img.affine @ np.array([0, 0, start_z, 1])
                np.testing.assert_array_almost_equal(
                    part_world[:3],
                    orig_world[:3],
                    err_msg=f"{name} voxel (0,0,0) should map to original voxel (0,0,{start_z})",
                )

    def test_rotated_affine_origins(self):
        """Affine origin correction should work with off-diagonal rotations."""
        img, _ = make_rotated_test_image()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            _, part_affines, third, margin = split_image_into_parts(img, tmp_dir)

            starts = {"s01": 0, "s02": third + 1 - margin, "s03": third * 2 + 1 - margin}
            for name, start_z in starts.items():
                part_world = part_affines[name] @ np.array([0, 0, 0, 1])
                orig_world = img.affine @ np.array([0, 0, start_z, 1])
                np.testing.assert_array_almost_equal(
                    part_world[:3],
                    orig_world[:3],
                    err_msg=f"{name} origin wrong with rotated affine",
                )

    def test_parts_saved_to_disk_have_correct_affines(self):
        """Verify nifti files on disk have the correct affines."""
        img, _ = make_test_image()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            _, part_affines, _, _ = split_image_into_parts(img, tmp_dir)

            for name in ["s01", "s02", "s03"]:
                loaded = nib.load(tmp_dir / f"{name}_0000.nii.gz")
                np.testing.assert_array_almost_equal(
                    loaded.affine,
                    part_affines[name],
                    err_msg=f"{name} affine on disk doesn't match expected",
                )


class TestSaveMergedPredictions:
    """Tests for save_merged_predictions() — the production merge function.

    This is the code path that previously overwrote corrected affines with
    the base affine, which was the actual regression.
    """

    def test_merged_outputs_have_correct_affines(self):
        """The specific regression: merged s02/s03 must retain shifted affines."""
        img, _ = make_test_image()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            img_parts, part_affines, third, margin = split_image_into_parts(img, tmp_dir)

            seg_combined = {}
            for name in img_parts:
                shape = nib.load(tmp_dir / f"{name}_0000.nii.gz").shape
                seg_combined[name] = np.ones(shape, dtype=np.uint8)

            save_merged_predictions(seg_combined, img_parts, part_affines,
                                    do_triple_split=True, base_affine=img.affine,
                                    tmp_dir=tmp_dir)

            for name in img_parts:
                loaded = nib.load(tmp_dir / f"{name}.nii.gz")
                np.testing.assert_array_almost_equal(
                    loaded.affine,
                    part_affines[name],
                    err_msg=f"Merged {name}.nii.gz has wrong affine — "
                            f"regression: merge overwrote corrected origin",
                )

    def test_non_split_uses_base_affine(self):
        """When do_triple_split is False, all parts use the base affine."""
        img, _ = make_test_image()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            seg_combined = {"s01": np.ones(img.shape, dtype=np.uint8)}

            save_merged_predictions(seg_combined, ["s01"], part_affines=None,
                                    do_triple_split=False, base_affine=img.affine,
                                    tmp_dir=tmp_dir)

            loaded = nib.load(tmp_dir / "s01.nii.gz")
            np.testing.assert_array_almost_equal(
                loaded.affine,
                img.affine,
                err_msg="Non-split path should use base affine",
            )


class TestReassembleImage:
    """Tests for reassemble_image() — the production reassembly function."""

    def test_roundtrip_data_integrity(self):
        """split → save merged → reassemble should recover original data."""
        img, original_data = make_test_image()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            img_parts, part_affines, third, margin = split_image_into_parts(img, tmp_dir)

            seg_combined = {}
            for name in img_parts:
                seg_combined[name] = nib.load(tmp_dir / f"{name}_0000.nii.gz").get_fdata().astype(np.uint8)

            save_merged_predictions(seg_combined, img_parts, part_affines,
                                    do_triple_split=True, base_affine=img.affine,
                                    tmp_dir=tmp_dir)

            reassembled = reassemble_image(tmp_dir, img.shape, img.affine, third, margin)
            np.testing.assert_array_equal(
                reassembled.get_fdata().astype(np.int16),
                original_data.astype(np.uint8).astype(np.int16),
                err_msg="Reassembled data should match the original (modulo dtype)",
            )

    def test_roundtrip_affine_preserved(self):
        img, _ = make_test_image()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            img_parts, part_affines, third, margin = split_image_into_parts(img, tmp_dir)

            seg_combined = {}
            for name in img_parts:
                seg_combined[name] = nib.load(tmp_dir / f"{name}_0000.nii.gz").get_fdata().astype(np.uint8)

            save_merged_predictions(seg_combined, img_parts, part_affines,
                                    do_triple_split=True, base_affine=img.affine,
                                    tmp_dir=tmp_dir)

            reassembled = reassemble_image(tmp_dir, img.shape, img.affine, third, margin)
            np.testing.assert_array_equal(
                reassembled.affine,
                img.affine,
                err_msg="Reassembled affine should match the original",
            )


class TestMockCleanup:
    """Verify that mock_imports() restored sys.modules properly."""

    def test_mocked_modules_not_in_sys_modules(self):
        """No mocked module should remain in sys.modules after cleanup."""
        leaked = [
            name for name in _mock_ctx.mocked_names
            if name in sys.modules and name not in _mock_ctx.snapshot
        ]
        assert leaked == [], f"Mocked modules leaked into sys.modules: {leaked}"

    def test_totalsegmentator_nnunet_not_cached(self):
        """totalsegmentator.nnunet should not remain cached under mocked deps."""
        assert "totalsegmentator.nnunet" not in sys.modules or \
               "totalsegmentator.nnunet" in _mock_ctx.snapshot, \
               "totalsegmentator.nnunet is cached in sys.modules from mocked import"
