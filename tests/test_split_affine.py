"""
Tests that triple-split sub-volumes get correct affine origins and that
reassembly recovers the original data and affine.

These tests exercise the split/merge logic from nnunet.py without
requiring nnU-Net or GPU inference.
"""

import tempfile
from pathlib import Path

import numpy as np
import nibabel as nib
import pytest


def make_test_image(shape=(64, 64, 300), voxel_size=(1.5, 1.5, 1.5)):
    """Create a synthetic nifti image with a non-trivial affine."""
    affine = np.eye(4)
    affine[0, 0] = voxel_size[0]
    affine[1, 1] = voxel_size[1]
    affine[2, 2] = voxel_size[2]
    # Non-zero origin
    affine[:3, 3] = [-48.0, -48.0, -225.0]
    data = np.random.randint(0, 100, shape, dtype=np.int16)
    return nib.Nifti1Image(data, affine), data


def make_rotated_test_image(shape=(64, 64, 300)):
    """Create a synthetic nifti image with off-diagonal rotation in the affine."""
    # ~15 degree rotation around the y-axis
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


def split_image(img, tmp_dir):
    """Replicate the triple-split logic from nnunet.py with corrected affines."""
    third = img.shape[2] // 3
    margin = 20
    data = img.get_fdata()
    base_affine = img.affine

    s02_start = third + 1 - margin
    s03_start = third * 2 + 1 - margin

    affine_s02 = np.copy(base_affine)
    affine_s02[:3, 3] = base_affine[:3, 3] + s02_start * base_affine[:3, 2]

    affine_s03 = np.copy(base_affine)
    affine_s03[:3, 3] = base_affine[:3, 3] + s03_start * base_affine[:3, 2]

    parts = {
        "s01": nib.Nifti1Image(data[:, :, :third + margin], base_affine),
        "s02": nib.Nifti1Image(data[:, :, s02_start:third * 2 + margin], affine_s02),
        "s03": nib.Nifti1Image(data[:, :, s03_start:], affine_s03),
    }

    for name, part_img in parts.items():
        nib.save(part_img, tmp_dir / f"{name}_0000.nii.gz")

    return parts, third, margin


def reassemble(tmp_dir, original_shape, original_affine, third, margin):
    """Replicate the reassembly logic from nnunet.py."""
    combined = np.zeros(original_shape, dtype=np.int16)
    combined[:, :, :third] = nib.load(tmp_dir / "s01_0000.nii.gz").get_fdata()[:, :, :-margin]
    combined[:, :, third:third * 2] = nib.load(tmp_dir / "s02_0000.nii.gz").get_fdata()[:, :, margin - 1:-margin]
    combined[:, :, third * 2:] = nib.load(tmp_dir / "s03_0000.nii.gz").get_fdata()[:, :, margin - 1:]
    return nib.Nifti1Image(combined, original_affine)


class TestSplitAffineOrigins:
    """Verify that each sub-volume's affine origin maps voxel (0,0,0) to the
    correct world coordinate."""

    def test_s01_origin_unchanged(self):
        img, _ = make_test_image()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            parts, _, _ = split_image(img, tmp_dir)
            np.testing.assert_array_equal(
                parts["s01"].affine[:3, 3],
                img.affine[:3, 3],
                err_msg="s01 origin should match the original image origin",
            )

    def test_s02_origin_shifted(self):
        img, _ = make_test_image()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            parts, third, margin = split_image(img, tmp_dir)
            s02_start = third + 1 - margin
            expected_origin = img.affine[:3, 3] + s02_start * img.affine[:3, 2]
            np.testing.assert_array_almost_equal(
                parts["s02"].affine[:3, 3],
                expected_origin,
                err_msg="s02 origin should be shifted by s02_start voxels along z",
            )

    def test_s03_origin_shifted(self):
        img, _ = make_test_image()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            parts, third, margin = split_image(img, tmp_dir)
            s03_start = third * 2 + 1 - margin
            expected_origin = img.affine[:3, 3] + s03_start * img.affine[:3, 2]
            np.testing.assert_array_almost_equal(
                parts["s03"].affine[:3, 3],
                expected_origin,
                err_msg="s03 origin should be shifted by s03_start voxels along z",
            )

    def test_voxel_to_world_consistency(self):
        """Voxel (0,0,0) of each sub-volume should map to the world coordinate
        of its starting slice in the original image."""
        img, _ = make_test_image()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            parts, third, margin = split_image(img, tmp_dir)

            starts = {"s01": 0, "s02": third + 1 - margin, "s03": third * 2 + 1 - margin}
            for name, start_z in starts.items():
                # World coord of voxel (0,0,0) in the sub-volume
                part_world = parts[name].affine @ np.array([0, 0, 0, 1])
                # World coord of voxel (0, 0, start_z) in the original
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
            parts, third, margin = split_image(img, tmp_dir)

            starts = {"s01": 0, "s02": third + 1 - margin, "s03": third * 2 + 1 - margin}
            for name, start_z in starts.items():
                part_world = parts[name].affine @ np.array([0, 0, 0, 1])
                orig_world = img.affine @ np.array([0, 0, start_z, 1])
                np.testing.assert_array_almost_equal(
                    part_world[:3],
                    orig_world[:3],
                    err_msg=f"{name} origin wrong with rotated affine",
                )


class TestSplitReassembly:
    """Verify that split + reassemble recovers the original data."""

    def test_roundtrip_data_integrity(self):
        img, original_data = make_test_image()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            _, third, margin = split_image(img, tmp_dir)
            reassembled = reassemble(tmp_dir, img.shape, img.affine, third, margin)
            np.testing.assert_array_equal(
                reassembled.get_fdata().astype(np.int16),
                original_data,
                err_msg="Reassembled data should exactly match the original",
            )

    def test_roundtrip_affine_preserved(self):
        img, _ = make_test_image()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            _, third, margin = split_image(img, tmp_dir)
            reassembled = reassemble(tmp_dir, img.shape, img.affine, third, margin)
            np.testing.assert_array_equal(
                reassembled.affine,
                img.affine,
                err_msg="Reassembled affine should match the original",
            )

    def test_parts_saved_to_disk_have_correct_affines(self):
        """Verify that the nifti files on disk (not just in-memory) have correct affines."""
        img, _ = make_test_image()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            parts, third, margin = split_image(img, tmp_dir)

            for name in ["s01", "s02", "s03"]:
                loaded = nib.load(tmp_dir / f"{name}_0000.nii.gz")
                np.testing.assert_array_almost_equal(
                    loaded.affine,
                    parts[name].affine,
                    err_msg=f"{name} affine on disk doesn't match expected",
                )
