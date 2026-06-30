from pathlib import Path
import os
import json
import glob
import re
from datetime import datetime
import subprocess
import time

import numpy as np
import nibabel as nib


"""
Infos about DICOM time tags:

The participant comes to the scanner:
1. patient loaded and study starts: this is the StudyTime (0008,0030).
2. Every time a series starts (T1 scan, T2 scan, fMRI run) all subsequent images of the series use the same series time (0008,0031) 
3. In theory, acquisition time (0008,0032) refers to the start of acquisition of 2D image, e.g. for a EPI fMRI dataset this would refer to the start of acquisition for the slice. Be aware that not all manufacturers record this with precision, and for some systems it may not even reveal actual slice order. You need to work with your vendor to understand what this value means.
4. When a new DICOM file is created (for fMRI, this is after reconstruction) a Content Time stamp is generated.
https://neurostars.org/t/dicom-header-acquisition-time-study-time-series-time-content-time/23280/4

=> this order not correct for my data. SeriesTime is after ContentTime & AcquisitionTime (the last two are typically the same).
"""

def get_series_number(file_path):
    file_name = file_path.name
    # Pattern matches both "_s" and "s" at the beginning or after underscore
    pattern = r"(?:^|_)s(\d{3})"
    match = re.search(pattern, file_name)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Could not find series number in {file_name}.")


def get_series_suffix(file_path):
    file_name = file_path.name
    match = re.search(r"a0\d", file_name)
    return match.group(0) if match else ""


def get_nora_dcm_file_path(project_name, file_path, return_series_num=False):
    """
    For each nifti file nora stores the original dicom file. Here we find the path of it.

    project_name: nora project name
    file_path: path to the nifti file
    """
    from pynora import Nora
    
    nora = Nora(project=project_name)
    fileinfo = nora.get_fileinfo(project_name, str(file_path))

    psid = fileinfo[0]['patients_id'] + fileinfo[0]['studies_id']
    dcm_dir = Path(nora.get_output_location(project_name, psid, 'DICOM'))
    print(f"DICOM directory: {dcm_dir}")

    series_number = get_series_number(file_path)
    suffix = get_series_suffix(file_path)
    suffix = f"_{suffix}" if suffix != "" else ""
    
    dcmfile = dcm_dir / f"DICOMHEADER_s{series_number:03d}{suffix}.dcm"
    print(f"Constructed DICOM file path: {dcmfile}")

    if not dcmfile.exists():
        print("No DICOM file found for " + str(file_path))
        return None
    else:
        if return_series_num:
            return dcmfile, series_number
        else:
            return dcmfile


def get_nora_dcm_file_path_locally(file_path, return_series_num=False):
    """
    Assuming that the file_path is one subdirectory into the study root directory
    """
    series_number = get_series_number(file_path)
    suffix = get_series_suffix(file_path)
    suffix = f"_{suffix}" if suffix != "" else ""
    
    dcm_file = file_path.parents[1] / "DICOM" / f"DICOMHEADER_s{series_number:03d}{suffix}.dcm"
    # print(f"dcm_file: {dcm_file}")
    if os.path.exists(dcm_file):
        if return_series_num:
            return dcm_file, series_number
        else:
            return dcm_file
    else:
        # print(f"No DICOM file found for {file_path}")
        if return_series_num:
            return None, series_number
        else:
            return None


def pi_time_to_phase(pi_time: float) -> str:
    """
    Convert the pi time to a phase and get a probability for the value.

    native: 0-10
    arterial_early: 10-30
    arterial_late:  30-60
    portal_venous:  60-100
    delayed: 100+

    returns: phase, probability
    """
    if pi_time < -5:    # for xgboost model also allow small negative values to count as native
        return "error", 1.0
    elif pi_time < 5:
        return "native", 1.0
    elif pi_time < 10:
        return "native", 0.7
    elif pi_time < 20:
        return "arterial_early", 0.7
    elif pi_time < 30:
        return "arterial_early", 1.0
    elif pi_time < 50:
        return "arterial_late", 1.0
    elif pi_time < 60:
        return "arterial_late", 0.7  # in previous version: "portal_venous"
    elif pi_time < 70:
        return "portal_venous", 1.0
    elif pi_time < 90:
        return "portal_venous", 1.0
    elif pi_time < 100:
        return "portal_venous", 0.7
    else:
        return "portal_venous", 0.3
        # return "delayed", 0.7  # not enough good training data for this
    
    
def get_pi_time_by_model(file_in, file_out, stats_file=None, retries=1, retry_delay=30):
    file_out = Path(file_out)
    file_out.parent.mkdir(exist_ok=True)

    if file_out.exists():
        try:
            with open(file_out) as f:
                d = json.load(f)
            return d["pi_time"]
        except (json.JSONDecodeError, KeyError):
            print(f"Existing PI time file is invalid. Recomputing: {file_out}")
            file_out.unlink()

    cmd = ["totalseg_get_phase", "-i", str(file_in), "-o", str(file_out), "--debug"]
    if stats_file is not None:
        cmd.extend(["-s", str(stats_file)])

    last_error = None
    for attempt in range(retries + 1):
        result = subprocess.run(cmd)
        if result.returncode == 0 and file_out.exists():
            try:
                with open(file_out) as f:
                    d = json.load(f)
                return d["pi_time"]
            except (json.JSONDecodeError, KeyError) as e:
                last_error = e
                print(f"PI time output is invalid after attempt {attempt + 1}: {file_out}")
                file_out.unlink(missing_ok=True)
        else:
            last_error = RuntimeError(
                f"totalseg_get_phase failed with return code {result.returncode}; "
                f"output exists: {file_out.exists()}"
            )

        if attempt < retries:
            print(f"Retrying PI time prediction in {retry_delay}s...")
            time.sleep(retry_delay)

    raise RuntimeError(f"Failed to compute PI time for {file_in}: {last_error}")
     

def parse_series_time(series_time):
    try:
        return datetime.strptime(series_time, "%H%M%S.%f")
    except ValueError:
        return datetime.strptime(series_time, "%H%M%S")
    
    
def get_pi_time_by_header(file_in, file_out, json_field_name, project_name, use_pynora=True, verbose=False):
    """
    Search for phase in the filename of a file and save to a json file.

    For tagging files see misc/tag_phases_by_name.py and misc/tag_phases_by_header.py
    
    file_in: path to the ct nifti file
    file_out: path to the json file where the phase info is saved
    json_field_name: name of the field in the json file
    project_name: name of the nora project (can be None if not using pynora)
    """
    
    if use_pynora:
        dcm_file = get_nora_dcm_file_path(project_name, file_in)
    else:
        dcm_file = get_nora_dcm_file_path_locally(file_in)
    if verbose:
        print(f"dcm_file: {dcm_file}")

    if dcm_file is not None:
        from pydicom import dcmread
        dcminfo = dcmread(dcm_file)
        if "SeriesTime" in dcminfo and \
            "AcquisitionTime" in dcminfo and \
            "ContentTime" in dcminfo:
            acquisition_time = str(dcminfo["AcquisitionTime"].value)  # individual slice?
            content_time = str(dcminfo["ContentTime"].value)  # after all slices?
            series_time = str(dcminfo["SeriesTime"].value)  # after series ?

            if "ContrastBolusStartTime" in dcminfo:
                contrast_time = str(dcminfo["ContrastBolusStartTime"].value)

                acquisition_time = parse_series_time(acquisition_time)
                content_time = parse_series_time(content_time)
                series_time = parse_series_time(series_time)
                contrast_time = parse_series_time(contrast_time)

                acquisition_diff = abs((series_time - acquisition_time).total_seconds())
                content_diff = abs((series_time - content_time).total_seconds())

                if verbose:
                    if acquisition_diff > 10 or content_diff > 10:
                        print("times:")
                        print(acquisition_time)
                        print(content_time)
                        print(series_time)
                        print("WARNING: time diff is large")

                time_diff_secs = abs((content_time - contrast_time).total_seconds())
            else:
                if verbose:
                    print("INFO: ContrastBolusStartTime not found in DICOM file. Setting time_diff_secs to 0.")
                time_diff_secs = 0.0
        else:
            if verbose:
                print("WARNING: SeriesTime / AcquisitionTime / ContentTime not found in DICOM file")
            time_diff_secs = -100.0
    else:
        time_diff_secs = -100.0

    time_diff_secs = round(time_diff_secs, 2)
    if verbose:
        print(f"time_diff_secs: {time_diff_secs}")
    
    if file_out is not None:
        meta=json.load(open(file_out)) if os.path.exists(file_out) else {}
        meta[json_field_name] = time_diff_secs
        json.dump(meta, open(file_out, "w"))
    
    return time_diff_secs


def get_phase(file, method, meta_dir, project_name=None):
    """
    Get the phase for a file using the given method.

    file: path to the ct nifti file
    method: "header", "model", "auto"
    meta_dir: directory for meta json files

    returns: phase, pi_time
    """
    if file.name.endswith(".nii.gz"):
        file_stem = file.name.replace(".nii.gz", "")
    else:
        file_stem = file.stem
    if method == "header":
        pi_time = get_pi_time_by_header(file, None, None, project_name, use_pynora=project_name is not None)
    elif method == "model":
        pi_time = get_pi_time_by_model(file, meta_dir / f"pi_time_{file_stem}.json")
    elif method == "auto":
        pi_time = get_pi_time_by_header(file, None, None, project_name, use_pynora=project_name is not None)
        if pi_time <= 0.0:
            print("Header method unsure. Trying model method...")
            pi_time = get_pi_time_by_model(file, meta_dir / f"pi_time_{file_stem}.json")
    return pi_time_to_phase(pi_time)[0], pi_time


def filter_files(files):
    files = [f for f in files if f.exists()]
    # remove files with size <2MB
    files = [f for f in files if f.stat().st_size > 2e6]    
    # only keep 3d files
    files = [f for f in files if len(nib.load(f).shape)]
    # only keep if at least 5 slices
    files = [f for f in files if np.array(nib.load(f).shape).min() > 5]
    # remove RGB images (report images)
    files = [f for f in files if nib.load(f).get_data_dtype().fields is None]
    return files
    

if __name__ == "__main__":
    """
    Get pi time from header for a file and save to a json file.
    """
    file_in = Path(sys.argv[1])
    file_out = Path(sys.argv[2])
    json_field_name = sys.argv[3]
    method = sys.argv[4]
    project_name = sys.argv[5]
    
    
    if method == "header":
        get_pi_time_by_header(file_in, file_out, json_field_name, project_name)
    elif method == "model":
        get_pi_time_by_model(file_in, file_out)
    
"""
Infos:
Naming of phases:
https://radiopaedia.org/articles/contrast-phases

Results from AOD___LongitudinalLiver:
fru00fch art: ca 22-32s
spu00e4t art: ca 32-42s
pv: 60-90s

AOD___AgingStudy
art: 20-30s
pv: 60-80s

AOD___MegaSegmentator
art: 20-35 (with a lot of outliers) (n=29)
pv: 60-80s (with a lot of outliers)  (n=182)
"""