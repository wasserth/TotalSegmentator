FROM nvcr.io/nvidia/pytorch:20.10-py3

RUN apt-get update
# Needed for fury vtk. ffmpeg also needed
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN apt-get install xvfb -y

RUN pip install flask gunicorn
RUN pip pyradiomics

COPY . /app
RUN pip install /app

RUN python /app/totalsegmentator/download_pretrained_weights.py

# expose not needed if using -p
# If using only expose and not -p then will not work
# EXPOSE 80