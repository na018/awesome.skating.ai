FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

#ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update
RUN apt-get install -y --no-install-recommends \
      build-essential \
      git \
      wget\
      libgoogle-glog-dev \
      libgtest-dev \
      libiomp-dev \
      libleveldb-dev \
      liblmdb-dev \
      libopencv-dev \
      libopenmpi-dev \
      libsnappy-dev \
      libprotobuf-dev \
      openmpi-bin \
      openmpi-doc \
      libgflags-dev \
      protobuf-compiler \
      python-dev \
      python3.6-dev \
      python3\
      python3-pip

RUN pip install --upgrade pip
RUN pip install setuptools
RUN pip install --user \
      future \
      numpy \
      protobuf \
      typing \
      hypothesis \
      tensorflow-gpu \
      numpy \
      h5py \
      image \
      graphviz \
      Cython \
      opencv-python \
      matplotlib \
      Pillow \
      scikit-image \
      scipy \
      keras \
      IPython \
      wheel

RUN pip3.6 install pycocotools

RUN mkdir skatingAI

WORKDIR skatingAI

COPY ../utils .
COPY ../nets .
COPY ../run.py .
COPY ../__init__.py .
RUN mkdir Data && mkdir Data/coco && mkdir Data/coco/img && \
 cd Data/coco/img && wget http://images.cocodataset.org/zips/train2017.zip && unzip train2017 && \
 cd Data/coco && wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip && \
 unzip annotations_trainval2017.zip





