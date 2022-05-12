import os
import requests
from pathlib import Path

"""
Steps to test the server:

1. Create api_key file:
    touch store/api_keys.txt
    echo jakob:abc123 >> store/api_keys.txt
    
2. Start server by running `python server.py`.  

3. Then in another shell run `python test_server.py`. 
"""

# url_base = 'http://localhost:5000/'
url_base = 'http://184.72.210.75:80/'


print("------------- get_server_status ------------------")

r = requests.post(url_base + "get_server_status",
                  json={"api_key": "abc123"})
if r.ok:
    print(f"status: {r.json()['status']}")
else:
    print(f"status code: {r.status_code}")
    print(f"message: {r.json()['message']}")


# print("------------- predict_image ------------------")

# test_data_dir = Path("/home/jakob/Downloads/nnunet_test")
# filename = test_data_dir / "ct3mm_0000.nii.gz"
# r = requests.post(url_base + "predict_image",
#                   files={'data_binary': open(filename, 'rb')},
#                   data={"api_key": "abc123"})
# if r.ok:
#     seg = r.content  # segmentation as bytes object
#     with open(test_data_dir / 'seg.zip', 'wb') as f:
#         f.write(seg)
#     print("Successfully received segmentation")
# else:
#     print(f"status code: {r.status_code}")
#     print(f"message: {r.json()['message']}")
