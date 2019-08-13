FROM tensorflow/tensorflow:latest-gpu-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \ 
    python3-pip \
    libsm6 libxext6 libxrender-dev \
    python3-tk

RUN python3 -m pip install keras h5py \
    opencv-python matplotlib seaborn

