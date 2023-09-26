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


def get_totalseg_dir():
    if "TOTALSEG_HOME_DIR" in os.environ:
        totalseg_dir = Path(os.environ["TOTALSEG_HOME_DIR"])
    else:
        # in docker container finding home not properly working therefore map to /tmp
        home_path = Path("/tmp") if str(Path.home()) == "/" else Path.home()
        totalseg_dir = home_path / ".totalsegmentator"
    return totalseg_dir


def get_weights_dir():
    if "TOTALSEG_WEIGHTS_PATH" in os.environ:
        # config_dir = Path(os.environ["TOTALSEG_WEIGHTS_PATH"]) / "nnUNet"
        config_dir = Path(os.environ["TOTALSEG_WEIGHTS_PATH"])
    else:
        totalseg_dir = get_totalseg_dir()
        config_dir = totalseg_dir / "nnunet/results"
    return config_dir


def setup_nnunet():
    # check if environment variable totalsegmentator_config is set
    if "TOTALSEG_WEIGHTS_PATH" in os.environ:
        weights_dir = os.environ["TOTALSEG_WEIGHTS_PATH"]
    else:
        # in docker container finding home not properly working therefore map to /tmp
        config_dir = get_totalseg_dir()
        # (config_dir / "nnunet/results/nnUNet/3d_fullres").mkdir(exist_ok=True, parents=True)
        # (config_dir / "nnunet/results/nnUNet/2d").mkdir(exist_ok=True, parents=True)
        weights_dir = config_dir / "nnunet/results"

    # This variables will only be active during the python script execution. Therefore
    # we do not have to unset them in the end.
    os.environ["nnUNet_raw"] = str(weights_dir)  # not needed, just needs to be an existing directory
    os.environ["nnUNet_preprocessed"] = str(weights_dir)  # not needed, just needs to be an existing directory
    os.environ["nnUNet_results"] = str(weights_dir)


def setup_totalseg(totalseg_id=None):
    totalseg_dir = get_totalseg_dir()
    totalseg_dir.mkdir(exist_ok=True)
    totalseg_config_file = totalseg_dir / "config.json"

    if totalseg_config_file.exists():
        with open(totalseg_config_file, "r") as f:
            config = json.load(f)
    else:
        if totalseg_id is None:
            totalseg_id = "totalseg_" + ''.join(random.Random().choices(string.ascii_uppercase + string.digits, k=8))
        config = {
            "totalseg_id": totalseg_id,
            "send_usage_stats": True,
            "prediction_counter": 0
        }
        with open(totalseg_config_file, "w") as f:
            json.dump(config, f, indent=4)

    return config


def set_license_number(license_number):
    if not is_valid_license(license_number):
        print("ERROR: Invalid license number. Please check your license number or contact support.")
        sys.exit(0)

    totalseg_dir = get_totalseg_dir()
    totalseg_config_file = totalseg_dir / "config.json"

    if totalseg_config_file.exists():
        with open(totalseg_config_file, "r") as f:
            config = json.load(f)
        config["license_number"] = license_number
        with open(totalseg_config_file, "w") as f:
            json.dump(config, f, indent=4)
    else:
        print(f"ERROR: Could not find config file: {totalseg_config_file}")


def get_license_number():
    totalseg_dir = get_totalseg_dir()
    totalseg_config_file = totalseg_dir / "config.json"
    if totalseg_config_file.exists():
        with open(totalseg_config_file, "r") as f:
            config = json.load(f)
        license_number = config["license_number"] if "license_number" in config else ""
    else:
        license_number = ""
    return license_number


def is_valid_license(license_number):
    try:
        url = f"http://backend.totalsegmentator.com:80/"
        r = requests.post(url + "is_valid_license_number",
                          json={"license_number": license_number}, timeout=2)
        if r.ok:
            return r.json()['status'] == "valid_license"
        else:
            print(f"An internal server error occured. status code: {r.status_code}")
            print(f"message: {r.json()['message']}")
            return False
    except Exception as e:
        print(f"An Exception occured: {e}")
        return False
    

def has_valid_license():
    totalseg_dir = get_totalseg_dir()
    totalseg_config_file = totalseg_dir / "config.json"
    if totalseg_config_file.exists():
        with open(totalseg_config_file, "r") as f:
            config = json.load(f)
        if "license_number" in config:
            license_number = config["license_number"]
        else:
            return "missing_license", "ERROR: A license number has not been set so far."
    else:
        return "missing_config_file", f"ERROR: Could not find config file: {totalseg_config_file}"
    
    if is_valid_license(license_number):
        return "yes", "SUCCESS: License is valid."
    else: 
        return "invalid_license", f"ERROR: Invalid license number ({license_number}). Please check your license number or contact support."


# Online check if license number is in config; do not do web request
def has_valid_license_offline():
    totalseg_dir = get_totalseg_dir()
    totalseg_config_file = totalseg_dir / "config.json"
    if totalseg_config_file.exists():
        with open(totalseg_config_file, "r") as f:
            config = json.load(f)
        if "license_number" in config:
            license_number = config["license_number"]
        else:
            return "missing_license", "ERROR: A license number has not been set so far."
    else:
        return "missing_config_file", f"ERROR: Could not find config file: {totalseg_config_file}"
    
    if len(license_number) == 18:
        return "yes", "SUCCESS: License is valid."
    else: 
        return "invalid_license", f"ERROR: Invalid license number ({license_number}). Please check your license number or contact support."


def increase_prediction_counter():
    totalseg_dir = get_totalseg_dir()
    totalseg_config_file = totalseg_dir / "config.json"
    if totalseg_config_file.exists():
        with open(totalseg_config_file, "r") as f:
            config = json.load(f)
        config["prediction_counter"] += 1
        with open(totalseg_config_file, "w") as f:
            json.dump(config, f, indent=4)
        return config


def get_version():
    try:
        return pkg_resources.get_distribution("TotalSegmentator").version
    except pkg_resources.DistributionNotFound:
        return "unknown"


def get_config_key(key_name):
    totalseg_dir = get_totalseg_dir()
    totalseg_config_file = totalseg_dir / "config.json"
    if totalseg_config_file.exists():
        with open(totalseg_config_file, "r") as f:
            config = json.load(f)
        if key_name in config:
            return config[key_name]
    return None


def set_config_key(key_name, value):
    totalseg_dir = get_totalseg_dir()
    totalseg_config_file = totalseg_dir / "config.json"
    if totalseg_config_file.exists():
        with open(totalseg_config_file, "r") as f:
            config = json.load(f)
        config[key_name] = value
        with open(totalseg_config_file, "w") as f:
            json.dump(config, f, indent=4)
        return config
    else:
        print("WARNING: Could not set config key, because config file not found.")


def send_usage_stats(config, params):
    if config is not None and config["send_usage_stats"]:
        
        params["roi_subset"] = "" if params["roi_subset"] is None else "-".join(params["roi_subset"])
        license_number = get_license_number()

        try:
            st = time.time()
            url = f"http://backend.totalsegmentator.com:80/"
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
                                    "cuda_available": torch.cuda.is_available(),
                                    "license_number": license_number
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
