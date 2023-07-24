import os
import sys
import random
import json
import string
import time
from pathlib import Path
import pkg_resources
import platform

import requests
import torch


def setup_nnunet():
    # check if environment variable totalsegmentator_config is set
    if "TOTALSEG_WEIGHTS_PATH" in os.environ:
        weights_dir = os.environ["TOTALSEG_WEIGHTS_PATH"]
    else:
        # in docker container finding home not properly working therefore map to /tmp
        home_path = Path("/tmp") if str(Path.home()) == "/" else Path.home()
        config_dir = home_path / ".totalsegmentator"
        (config_dir / "nnunet/results/nnUNet/3d_fullres").mkdir(exist_ok=True, parents=True)
        (config_dir / "nnunet/results/nnUNet/2d").mkdir(exist_ok=True, parents=True)
        weights_dir = config_dir / "nnunet/results"

    # This variables will only be active during the python script execution. Therefore
    # we do not have to unset them in the end.
    os.environ["nnUNet_raw_data_base"] = str(weights_dir)  # not needed, just needs to be an existing directory
    os.environ["nnUNet_preprocessed"] = str(weights_dir)  # not needed, just needs to be an existing directory
    os.environ["RESULTS_FOLDER"] = str(weights_dir)


def setup_totalseg(totalseg_id=None):
    home_path = Path("/tmp") if str(Path.home()) == "/" else Path.home()
    totalseg_path = home_path / ".totalsegmentator"
    totalseg_path.mkdir(exist_ok=True)
    totalseg_config_file = totalseg_path / "config.json"

    if totalseg_config_file.exists():
        config = json.load(open(totalseg_config_file, "r"))
    else:
        if totalseg_id is None:
            totalseg_id = "totalseg_" + ''.join(random.Random().choices(string.ascii_uppercase + string.digits, k=8))
        config = {
            "totalseg_id": totalseg_id,
            "send_usage_stats": True,
            "prediction_counter": 0
        }
        json.dump(config, open(totalseg_config_file, "w"), indent=4)

    return config


def increase_prediction_counter():
    home_path = Path("/tmp") if str(Path.home()) == "/" else Path.home()
    totalseg_config_file = home_path / ".totalsegmentator" / "config.json"
    if totalseg_config_file.exists():
        config = json.load(open(totalseg_config_file, "r"))
        config["prediction_counter"] += 1
        json.dump(config, open(totalseg_config_file, "w"), indent=4)
        return config


def get_version():
    try:
        return pkg_resources.get_distribution("TotalSegmentator").version
    except pkg_resources.DistributionNotFound:
        return "unknown"


def send_usage_stats(config, params):
    if config is not None and config["send_usage_stats"]:
        
        params["roi_subset"] = "" if params["roi_subset"] is None else "-".join(params["roi_subset"])

        try:
            st = time.time()
            url = f"http://94.16.105.223:80/"
            r = requests.post(url + "log_totalseg_run",
                              json={"totalseg_id": config["totalseg_id"],
                                    "prediction_counter": config["prediction_counter"],
                                    "task": params["task"],
                                    "fast": params["fast"],
                                    "preview": params["preview"],
                                    "multilabel": params["multilabel"],
                                    "roi_subset": params["roi_subset"],
                                    "statistics": params["statistics"],
                                    "radiomics": params["radiomics"],
                                    "platform": platform.system(),
                                    "machine": platform.machine(),
                                    "version": get_version(),
                                    "python_version": sys.version,
                                    "cuda_available": torch.cuda.is_available()
                                    }, timeout=2)
            # if r.ok:
            #     print(f"status: {r.json()['status']}")
            # else:
            #     print(f"status code: {r.status_code}")
            #     print(f"message: {r.json()['message']}")
            # print(f"Request took {time.time()-st:.3f}s")
        except Exception as e:
            # print(f"An Exception occured: {e}")
            pass
