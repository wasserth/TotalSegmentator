import gc
import json
import shutil
from pathlib import Path
import os, glob, time, io
import traceback
from threading import Thread
import subprocess
import datetime

import numpy as np
from flask import Flask, request, jsonify, send_file, send_from_directory, after_this_request
from markupsafe import escape

# On the server /home/ubuntu/store will be mounted to "store" when starting the docker container
STORE_DIR = Path("store")


def log(text, p=False):
    if p:
        print(text)
    with open(STORE_DIR / "log.txt", "a") as f:
        f.write(f"{datetime.datetime.now()} {text}\n")


def has_valid_credentials(api_key):
    lines = open(STORE_DIR / "api_keys.txt", "r").readlines()
    for line in lines:
        username, key = line.strip().split(":")
        if key == api_key: return True
    return False


app = Flask(__name__)


@app.route('/get_server_status', methods=["POST"])
def get_server_status():
    meta = request.json
    if not has_valid_credentials(meta["api_key"]):
        return {"message": "invalid access code"}, 401

    return {"status": "happily running"}, 200


@app.route('/predict_image', methods=['POST'])
def upload_data():
    """
    available data fields:
        api_key
    """
    meta = request.form.to_dict()
    if not has_valid_credentials(meta["api_key"]):
        return {"message": "invalid access code"}, 401

    stats = "-s" if "statistics" in meta and meta["statistics"] == "1" else ""
    print(f"stats: {stats}")

    # Upload image
    img_id = str(int(time.time()))
    img_fn = f"{img_id}.nii.gz"

    img_dir = STORE_DIR / "uploaded_images"
    img_dir.mkdir(exist_ok=True)
    request.files['data_binary'].save(img_dir / img_fn)
    log(f"upload successful")
    # print("upload successful")

    # Predict image
    seg_dir = img_dir / ('seg_' + img_id)
    subprocess.call(f"TotalSegmentator -i {img_dir / img_fn} -o {seg_dir} -f -p {stats} -ns 1", shell=True)

    shutil.make_archive(seg_dir, 'zip', seg_dir)

    # delete files
    os.remove(img_dir / img_fn)
    shutil.rmtree(seg_dir)
    @after_this_request
    def remove_file(response):
        try:
            os.remove(str(seg_dir) + ".zip")
        except Exception as error:
            app.logger.error("Error removing or closing seg file handle", error)
        return response

    return send_file(str(seg_dir) + ".zip", mimetype="application/octet-stream"), 200


if __name__ == '__main__':
    # Only for debugging while developing
    app.run(host='0.0.0.0', debug=True, port=5000)
    # app.run(threaded=True, host='0.0.0.0', port='5000')  # those options should be enabled by default