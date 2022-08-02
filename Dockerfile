FROM nvcr.io/nvidia/pytorch:20.10-py3

RUN apt-get update
# Needed for fury vtk. ffmpeg also needed
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN apt-get install xvfb -y

# COPY requirements.txt /requirements.txt

# RUN pip install -r /requirements.txt

# Adapt when updated usage of fury
# RUN pip install xvfbwrapper dipy==1.2.0 fury==0.7.1
# RUN pip install batchgenerators==0.21                             
# RUN pip install https://github.com/wasserth/nnUNet_cust/archive/refs/heads/working_2022_03_18.zip
RUN pip install flask gunicorn

# todo: download weights and copy into container
# RUN mkdir -p /root/.totalsegmentator/nnunet/results/nnUNet/3d_fullres
# COPY tmp_weights /root/.totalsegmentator/nnunet/results/nnUNet/3d_fullres
RUN python totalsegmentator/download_pretrained_weights.py

COPY . /app
RUN pip install /app

# expose not needed if using -p
# If using only expose and not -p then will not work
# EXPOSE 80