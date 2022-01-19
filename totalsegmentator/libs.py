import contextlib
import sys

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


def setup_nnunet():
    config_dir = Path.home() / ".totalsegmentator"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "nnunet/results/nnUNet/3d_fullres").mkdir(exist_ok=True, parents=True)
    # todo: 
    # 1. download weights
    # 2. copy to folder
    # 3. set environment variables
    # 4. after processing: unset environment variables
 