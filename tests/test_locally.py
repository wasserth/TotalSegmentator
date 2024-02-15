import sys
from pathlib import Path
p_dir = str(Path(__file__).absolute().parents[1])
if p_dir not in sys.path: sys.path.insert(0, p_dir)

import pytest
import os
import re
import glob
import shutil
import subprocess
from collections import defaultdict
import time
import threading
import platform
import importlib.metadata

import psutil
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import nnunetv2

from totalsegmentator.python_api import totalsegmentator
from totalsegmentator.map_to_binary import class_map
from totalsegmentator.excel import set_xlsx_column_width_to_content
from resources.evaluate import calc_metrics

"""
Run a complete prediction locally with GPU and evaluate Dice score +
CPU/GPU usage + RAM/GPU memory usage + runtime.
(This is not possible on github actions due to missing GPU)

Info:
To get the CT file and create the multilable groundtruth files use
python ~/dev/jakob_scripts/multiseg/eval/get_data_for_test_locally.py

Usage:
python test_locally.py
"""

max_memory_usage = 0  # Initialize max_memory_usage for RAM as 0
max_gpu_memory_usage = 0  # Initialize max_gpu_memory_usage for GPU as 0
cpu_utilizations = []  # Initialize an empty list to store CPU utilizations
gpu_utilizations = []  # Initialize an empty list to store GPU utilizations

def get_memory_usage():
    global max_memory_usage
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    current_memory = memory_info.rss / (1024 ** 2)  # Convert to MB
    max_memory_usage = max(max_memory_usage, round(current_memory))  # Update max_memory_usage

def get_gpu_memory_usage():
    global max_gpu_memory_usage
    current_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB
    max_gpu_memory_usage = max(max_gpu_memory_usage, round(current_memory))  # Update max_gpu_memory_usage

def memory_monitor(interval=0.5):
    while True:
        get_memory_usage()
        time.sleep(interval)

def gpu_memory_monitor(interval=0.5):
    while True:
        get_gpu_memory_usage()
        time.sleep(interval)

def get_cpu_utilization():
    cpu_util = psutil.cpu_percent(interval=0.5)  # Get CPU utilization as a percentage
    cpu_utilizations.append(cpu_util)

def cpu_utilization_monitor(interval=0.5):
    while True:
        get_cpu_utilization()
        time.sleep(interval)

def get_gpu_utilization():
    try:
        sp = subprocess.Popen(["nvidia-smi", "-q", "-i", "0", "-d", "UTILIZATION"],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out_str = sp.communicate()
        out_list = str(out_str[0]).split('\\n')

        for item in out_list:
            if "Gpu" in item:
                m = re.search(r"(\d+)", item)
                if m:
                    gpu_util = int(m.group(1))
                    gpu_utilizations.append(gpu_util)
                    break
    except Exception as e:
        print(f"An error occurred: {e}")

def gpu_utilization_monitor(interval=0.5):
    while True:
        get_gpu_utilization()
        time.sleep(interval)

def reset_monitors():
    global max_memory_usage
    global max_gpu_memory_usage
    global cpu_utilizations
    global gpu_utilizations
    max_memory_usage = 0
    max_gpu_memory_usage = 0
    cpu_utilizations = []
    gpu_utilizations = []

def start_monitors():
    # Create separate threads to monitor memory usage
    memory_thread = threading.Thread(target=memory_monitor)
    memory_thread.daemon = True
    memory_thread.start()

    gpu_memory_thread = threading.Thread(target=gpu_memory_monitor)
    gpu_memory_thread.daemon = True
    gpu_memory_thread.start()

    cpu_util_thread = threading.Thread(target=cpu_utilization_monitor)
    cpu_util_thread.daemon = True
    cpu_util_thread.start()

    gpu_util_thread = threading.Thread(target=gpu_utilization_monitor)
    gpu_util_thread.daemon = True
    gpu_util_thread.start()

def are_logs_similar(last_log, new_log, cols, tolerance_percent=0.04):
    if last_log is None or new_log is None:
        print("Cannot compare logs because one of them is None.")
        return False

    # For these columns the values differ a lot between runs so we allow a larger margin
    tolerance_percent_large_diff = 0.2
    cols_large_diff = ["runtime_3mm",
                    #    "memory_gpu_15mm", "memory_gpu_3mm",
                       "cpu_utilization_15mm", "cpu_utilization_3mm",
                       "gpu_utilization_15mm", "gpu_utilization_3mm"]

    identical = True
    for old_value, new_value, col in zip(last_log, new_log, cols):
        # Check string values for equality
        if isinstance(old_value, str) and isinstance(new_value, str):
            if old_value != new_value:
                print(f"  Difference in {col}: {old_value} != {new_value}")
                identical = False
        # Check Timestamp
        elif isinstance(old_value, pd.Timestamp) and isinstance(new_value, pd.Timestamp):
            continue
        # Check numeric values for similarity within a tolerance
        elif (isinstance(old_value, (int, float)) or np.isscalar(old_value) ) and \
                (isinstance(new_value, (int, float)) or np.isscalar(new_value)):
            if old_value == 0 and new_value == 0:
                continue
            elif old_value == 0 or new_value == 0:
                print(f"  Difference in {col}: {old_value} != {new_value} (one is zero))")
                identical = False
            percent_diff = abs(old_value - new_value) / abs(old_value)
            tolerance = tolerance_percent_large_diff if col in cols_large_diff else tolerance_percent
            if percent_diff > tolerance:
                print(f"  Difference in {col}: {old_value} != {new_value} (percent_diff: {percent_diff:.2f})")
                identical = False
        else:
            # If types are neither string nor numeric, do a direct comparison
            if old_value != new_value:
                print(f"  Difference in {col}: {old_value} != {new_value} (type: {type(old_value)})))")
                identical = False
    return identical


if __name__ == "__main__":
    start_monitors()

    base_dir = Path("/mnt/nvme/data/multiseg/test_locally")

    scores = defaultdict(dict)
    times = defaultdict(list)
    memory_ram = {}
    memory_gpu = {}
    cpu_utilization = {}
    gpu_utilization = {}

    device = "gpu"  # "cpu" or "gpu"

    for resolution in ["15mm", "3mm"]:
    # for resolution in ["3mm"]:
        img_dir = base_dir / resolution / "ct"
        gt_dir = base_dir / resolution / "gt"
        pred_dir = base_dir / resolution / "pred"
        pred_dir.mkdir(parents=True, exist_ok=True)

        print("Run totalsegmentator...")
        reset_monitors()
        for img_fn in tqdm(img_dir.glob("*.nii.gz")):
            fast = resolution == "3mm"
            st = time.time()
            totalsegmentator(img_fn, pred_dir / img_fn.name, fast=fast, ml=True, device=device)
            times[resolution].append(time.time()-st)

        print("Logging...")
        times[resolution] = np.mean(times[resolution]).round(1)
        memory_ram[resolution] = max_memory_usage
        memory_gpu[resolution] = max_gpu_memory_usage
        cpu_utilization[resolution] = np.mean(cpu_utilizations).round(1)
        gpu_utilization[resolution] = np.mean(gpu_utilizations).round(1)

        print("Calc metrics...")
        subjects = [s.name.split(".")[0] for s in img_dir.glob("*.nii.gz")]
        res = [calc_metrics(s, gt_dir, pred_dir, class_map["total"]) for s in subjects]
        res = pd.DataFrame(res)

        print("Aggregate metrics...")
        for metric in ["dice", "surface_dice_3"]:
            res_all_rois = []
            for roi_name in class_map["total"].values():
                row_wo_nan = res[f"{metric}-{roi_name}"].dropna()
                res_all_rois.append(row_wo_nan.mean())
                # print(f"{roi_name} {metric}: {row_wo_nan.mean():.3f}")  # per roi
            scores[resolution][metric] = np.nanmean(res_all_rois).round(3)  # mean over all rois

    scores = dict(scores)
    times = dict(times)

    print("Saving...")
    cols = ["time", "Dice_15mm", "NSD_15mm", "Dice_3mm", "NSD_3mm",
            "runtime_15mm", "runtime_3mm",
            "memory_ram_15mm", "memory_ram_3mm",
            "memory_gpu_15mm", "memory_gpu_3mm",
            "cpu_utilization_15mm", "cpu_utilization_3mm",
            "gpu_utilization_15mm", "gpu_utilization_3mm",
            "python_version", "torch_version", "nnunet_version",
            "cuda_version", "cudnn_version",
            "gpu_name", "comment"]
    overview_file = Path(f"{base_dir}/overview.xlsx")
    if overview_file.exists():
        overview = pd.read_excel(overview_file)
    else:
        overview = pd.DataFrame(columns=cols)

    last_log = overview.iloc[-1] if len(overview) > 0 else None

    new_log = [pd.Timestamp.now(), scores["15mm"]["dice"], scores["15mm"]["surface_dice_3"],
               scores["3mm"]["dice"], scores["3mm"]["surface_dice_3"],
               times["15mm"], times["3mm"],
               memory_ram["15mm"], memory_ram["3mm"],
               memory_gpu["15mm"], memory_gpu["3mm"],
               cpu_utilization["15mm"], cpu_utilization["3mm"],
               gpu_utilization["15mm"], gpu_utilization["3mm"],
               platform.python_version(), torch.__version__,
               importlib.metadata.version("nnunetv2"),
               float(torch.version.cuda), int(torch.backends.cudnn.version()),
               torch.cuda.get_device_name(0), ""]

    print("Comparing NEW to PREVIOUS log:")
    if are_logs_similar(last_log, new_log, cols):
        print("SUCCESS: no differences")
    else:
        print("ERROR: major differences found")

    print(f"Saving to {overview_file}...")
    overview.loc[len(overview)] = new_log
    overview.to_excel(overview_file, index=False)
    set_xlsx_column_width_to_content(overview_file)

    # Clean up
    shutil.rmtree(base_dir / "15mm" / "pred")
    shutil.rmtree(base_dir / "3mm" / "pred")
