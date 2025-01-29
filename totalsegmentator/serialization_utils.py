import json
import io
import hashlib
import pickle
import gzip
from gzip import GzipFile
from io import BytesIO

import nibabel as nib
from nibabel import FileHolder, Nifti1Image
import numpy as np
import blosc

from totalsegmentator.nifti_ext_header import load_multilabel_nifti, add_label_map_to_nifti


class NumpyJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        return super().default(obj)


# memory optimised
def filestream_to_nifti(input_stream, gzipped=True):
    """
    input_stream: the return value from st.file_uploader
    returns: nibabel Nifti1Image object
    """
    input_stream.seek(0)  # important if stream was partially read elsewhere

    if gzipped:
        # Read the entire gzipped file into memory
        with GzipFile(fileobj=input_stream) as gz_file:
            in_memory_stream = BytesIO(gz_file.read())
        fh = FileHolder(fileobj=in_memory_stream)
        img = Nifti1Image.from_file_map({'header': fh, 'image': fh})
    else:
        fh = FileHolder(fileobj=input_stream)
        img = Nifti1Image.from_file_map({'header': fh, 'image': fh})

    return img


# caching does not work with nifti img
def nifti_to_filestream(nifti_img):
    # fast
    # bio = io.BytesIO()
    # file_map = nifti_img.make_file_map({'image': bio, 'header': bio})
    # nifti_img.to_file_map(file_map)
    # return bio.getvalue()

    # Slow?? (but with zipping; needed if want to download as .nii.gz nifti file?)
    bio = io.BytesIO()
    zz = gzip.GzipFile(fileobj=bio, mode='w')
    file_map = nifti_img.make_file_map({'image': zz, 'header': zz})
    nifti_img.to_file_map(file_map)
    return bio.getvalue()


def hash_bytes(ct_bytes):
    ct_bytes.seek(0)  # Reset the read pointer to the start of the BytesIO object
    hasher = hashlib.sha256()

    buffer = ct_bytes.read(65536)
    while len(buffer) > 0:
        hasher.update(buffer)
        buffer = ct_bytes.read(65536)

    return hasher.hexdigest()


"""
Infos on how to speedup return of large data (which is slow per default):
- cast masks to uint8
- use tobytes() (on top of numpy)
   or
  serialize with pickle + blosc compression
   (pickle can serialize nifti or numpy; both similar in speed (if casting to uint8 for masks))
  => both methods similar in speed
"""
def serialize_and_compress(data):
    serialized_data = pickle.dumps(data)
    return blosc.compress(serialized_data)


def decompress_and_deserialize(compressed_data):
    serialized_data = blosc.decompress(compressed_data)
    return pickle.loads(serialized_data)


def convert_to_serializable(d):
    """
    Recursively converts non-JSON-serializable types in a nested dictionary
    to native Python types.

    :param d: The dictionary to traverse
    :return: A new dictionary with all types converted to JSON-serializable types
    """
    if isinstance(d, dict):
        return {k: convert_to_serializable(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [convert_to_serializable(item) for item in d]
    elif isinstance(d, tuple):
        return tuple(convert_to_serializable(item) for item in d)
    elif isinstance(d, np.ndarray):
        return d.tolist()
    elif isinstance(d, (np.float32, np.float64)):
        return float(d)
    elif isinstance(d, (np.int32, np.int64)):
        return int(d)
    else:
        return d


def nib_load_eager(img_path, dtype=np.float32):
    """
    Load nifti image with data and extended header into memory.
    """
    img, label_map = load_multilabel_nifti(img_path)
    img = nib.Nifti1Image(np.asanyarray(img.dataobj), img.affine, img.header)
    img = add_label_map_to_nifti(img, label_map)
    return img