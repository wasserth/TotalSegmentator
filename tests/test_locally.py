import sys
from pathlib import Path
# p_dir = str(Path(__file__).absolute().parents[1])
# if p_dir not in sys.path: sys.path.insert(0, p_dir)

from pathlib import Path
import pytest
import os
import sys
import glob
import shutil
import subprocess
from tqdm import tqdm
import pandas as pd
import numpy as np

from totalsegmentator.python_api import totalsegmentator
from totalsegmentator.map_to_binary import class_map
from resources.evaluate import calc_metrics

"""
Run a complete prediction locally with GPU and evaluate Dice score.
This is not possible on github actions due to missing GPU.

Info:
To create the multilable groundtruth files for 3mm use 
python ~/dev/jakob_scripts/multiseg/eval/create_3mm_multilabel_file.py
"""

if __name__ == "__main__":

    # Todo: add test files as release to github and download each time this is run

    img_dir = Path("/mnt/nvme/data/multiseg/test_locally")
    gt_dir = Path("/mnt/nvme/data/multiseg/test_locally/gt")
    pred_dir = Path("/mnt/nvme/data/multiseg/test_locally/pred")

    for img_fn in tqdm(img_dir.glob("*.nii.gz")):
        totalsegmentator(img_fn, pred_dir / img_fn.name, fast=True, ml=True, device="gpu")

    print("Calc metrics...")
    subjects = [s.name.split(".")[0] for s in img_dir.glob("*.nii.gz")]
    res = [calc_metrics(s, gt_dir, pred_dir, class_map["total"]) for s in subjects]
    res = pd.DataFrame(res)
    
    print("Print results...")
    for metric in ["dice", "surface_dice_3"]:
        res_all_rois = []
        for roi_name in class_map["total"].values():
            row_wo_nan = res[f"{metric}-{roi_name}"].dropna()
            res_all_rois.append(row_wo_nan.mean())
            # print(f"{roi_name} {metric}: {row_wo_nan.mean():.3f}")  # per roi
        print(f"{metric}: {np.nanmean(res_all_rois):.3f}")  # mean over all rois

    # pytest.main(['-v', 'tests/test_end_to_end.py::test_end_to_end::test_prediction_fast'])
    
    shutil.rmtree(pred_dir)

    # todo: also track runtime and memory consumption!!
