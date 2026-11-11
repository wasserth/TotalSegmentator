import json

import nibabel as nib
import numpy as np
import pytest

from totalsegmentator.statistics import get_basic_statistics


def _liver_statistics(image_data, file_out=None, **kwargs):
    """Statistics of a 'liver' cube which does not touch the border."""
    seg = np.zeros((20, 20, 20), dtype=np.uint8)
    seg[5:15, 5:15, 5:15] = 5  # liver in task total
    img = nib.Nifti1Image(image_data, np.eye(4))
    stats = get_basic_statistics(seg, img, file_out, quiet=True, task="total", roi_subset=["liver"], **kwargs)
    return stats["liver"]


@pytest.mark.parametrize("value", [
    0.62,     # MR image normalised to [0, 1]: was reported as 0.0
    1.1,      # e.g. an ADC map in 10^-3 mm2/s: was reported as 1.0
    41000.0,  # 16 bit MR image: wrapped around to -24536.0
    350.5,
])
def test_intensity_of_images_which_do_not_fit_int16(value):
    liver = _liver_statistics(np.full((20, 20, 20), value, dtype=np.float32))

    assert liver["intensity"] == pytest.approx(value, rel=1e-5)


def test_median_and_extra_metrics_keep_fractional_intensities():
    rng = np.random.default_rng(0)
    image = rng.random((20, 20, 20))
    inside = image[5:15, 5:15, 5:15]

    liver = _liver_statistics(image, metric="median", extra_metrics=True)

    assert liver["intensity"] == pytest.approx(np.median(inside), abs=1e-5)
    assert liver["intensity_min"] == pytest.approx(inside.min())
    assert liver["intensity_max"] == pytest.approx(inside.max())
    assert liver["intensity_std"] == pytest.approx(inside.std(), abs=1e-5)


def test_normalized_intensities_of_a_zero_to_one_image():
    # After the int16 cast the image was all zeros and the normalisation divided by zero.
    image = np.zeros((20, 20, 20), dtype=np.float32)
    image[5:15, 5:15, 5:15] = 0.5
    image[0, 0, 0] = 1.0

    liver = _liver_statistics(image, normalized_intensities=True)

    assert liver["intensity"] == pytest.approx(0.5)


def test_statistics_of_a_float_image_can_be_written_as_json(tmp_path):
    file_out = tmp_path / "statistics.json"

    _liver_statistics(np.full((20, 20, 20), 0.62, dtype=np.float32), file_out=file_out)

    assert json.loads(file_out.read_text())["liver"]["intensity"] == pytest.approx(0.62, rel=1e-5)


@pytest.mark.parametrize("dtype", [np.int16, np.float32, np.float64])
def test_whole_number_images_give_the_same_statistics_as_before(dtype):
    # e.g. CT in Hounsfield units, also when it was saved as float
    rng = np.random.default_rng(0)
    image = rng.integers(-1000, 2000, (20, 20, 20)).astype(dtype)
    inside = image[5:15, 5:15, 5:15].astype(np.int16)

    liver = _liver_statistics(image, extra_metrics=True)

    assert liver["intensity"] == np.average(inside).round(5)
    assert liver["intensity_min"] == float(inside.min())
    assert liver["intensity_max"] == float(inside.max())
    assert liver["volume"] == 1000.0
