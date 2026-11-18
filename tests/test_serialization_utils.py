from io import BytesIO

import nibabel as nib
import numpy as np
import pytest

from totalsegmentator.serialization_utils import filestream_to_nifti, nifti_to_filestream


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.float32])
def test_nifti_filestream_round_trip(dtype):
    data = np.arange(120, dtype=dtype).reshape(4, 5, 6)
    if dtype == np.float32:
        data = data / 10 - 6
    affine = np.array([
        [0, -2, 0, 10],
        [3, 0, 0, -20],
        [0, 0, 4, 30],
        [0, 0, 0, 1],
    ])
    image = nib.Nifti1Image(data, affine)
    image.header["descrip"] = b"Stream round trip"
    image.header.extensions.append(nib.nifti1.Nifti1Extension(6, b"test metadata"))

    output = nifti_to_filestream(image)
    restored = filestream_to_nifti(BytesIO(output))

    assert isinstance(output, bytes)
    np.testing.assert_array_equal(np.asanyarray(restored.dataobj), data)
    np.testing.assert_array_equal(restored.affine, affine)
    assert restored.get_data_dtype() == image.get_data_dtype()
    assert restored.header["descrip"] == image.header["descrip"]
    assert [(ext.get_code(), ext.get_content()) for ext in restored.header.extensions] == [
        (6, b"test metadata")
    ]
