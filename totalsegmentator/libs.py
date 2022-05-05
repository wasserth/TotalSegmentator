import os
import contextlib
import sys
import shutil
import zipfile

from pathlib import Path


"""
Helpers to suppress stdout prints from nnunet
https://stackoverflow.com/questions/2828953/silence-the-stdout-of-a-function-in-python-without-trashing-sys-stdout-and-resto
"""
class DummyFile(object):
    def write(self, x): pass

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout



def download_pretrained_weights(task_id):

    config_dir = Path.home() / ".totalsegmentator/nnunet/results/nnUNet/3d_fullres"

    old_weights = [
        "Task223_my_test"
    ]

    if task_id == 224:
        weights_path = config_dir / "Task224_hello_world"
        WEIGHTS_URL = "https://zenodo.org/record/3634539/files/best_weights_ep266.npz?download=1"

    for old_weight in old_weights:
        if (config_dir / old_weight).exists():
            shutil.rmtree(config_dir / old_weight)

    if WEIGHTS_URL is not None and not weights_path.exists():
        print("Downloading pretrained weights (~140MB) ...")

        data = urlopen(WEIGHTS_URL).read()
        with open(config_dir / "tmp_download_file.zip", "wb") as weight_file:
            weight_file.write(data)

        with zipfile.ZipFile(config_dir / "tmp_download_file.zip", 'r') as zip_f:
            zip_f.extractall(weights_path)


def setup_nnunet():
    # check if environment variable totalsegmentator_config is set
    if "TOTALSEG_WEIGHTS_PATH" in os.environ:
        weights_dir = os.environ["TOTALSEG_WEIGHTS_PATH"]
    else:
        config_dir = Path.home() / ".totalsegmentator"
        (config_dir / "nnunet/results/nnUNet/3d_fullres").mkdir(exist_ok=True, parents=True)
        weights_dir = config_dir / "nnunet/results"

    # todo
    # download_pretrained_weights(224)

    # This variables will only be active during the python script execution. Therefore
    # do not have to unset them in the end.
    os.environ["nnUNet_raw_data_base"] = str(weights_dir)  # not needed, just needs to be an existing directory
    os.environ["nnUNet_preprocessed"] = str(weights_dir)  # not needed, just needs to be an existing directory
    os.environ["RESULTS_FOLDER"] = str(weights_dir)
