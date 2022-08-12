import sys
import os
import requests
import time
from pathlib import Path

"""
Steps to test the server:

1. Create api_key file:
    touch store/api_keys.txt
    echo jakob:abc123 >> store/api_keys.txt
    
2. Start server by running `python server.py`.  

3. Then in another shell run `python test_server.py`. 
"""

# Url needs to have a trailing slash!
# url_base = 'http://localhost:5000/'
url_base = f"http://{sys.argv[1]}/"  # read from command line
api_key = sys.argv[2]

print("------------- get_server_status ------------------")

r = requests.post(url_base + "get_server_status",
                  json={"api_key": api_key})
if r.ok:
    print(f"status: {r.json()['status']}")
else:
    print(f"status code: {r.status_code}")
    print(f"message: {r.json()['message']}")


print("------------- predict_image ------------------")

st = time.time()
test_data_dir = Path("/home/jakob/Downloads/nnunet_test")
filename = test_data_dir / "ct3mm_0000.nii.gz"  # 50s
# filename = test_data_dir / "ct15mm_0000.nii.gz"  # 100s
# filename = test_data_dir / "ct_0000.nii.gz"  # 400s
r = requests.post(url_base + "predict_image",
                  files={'data_binary': open(filename, 'rb')},
                  data={"api_key": api_key, "statistics": 1})
if r.ok:
    seg = r.content  # segmentation as bytes object
    with open(test_data_dir / 'seg.zip', 'wb') as f:
        f.write(seg)
    print("Successfully received segmentation")
else:
    print(f"status code: {r.status_code}")
    print(f"message: {r.json()['message']}")
print(f"took: {time.time() - st:.2f}s")
# without preview:
# 3mm (145x145x274: 5.7M): 28s
