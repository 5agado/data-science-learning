FROM tensorflow/tensorflow:2.0.0b1-gpu-py3-jupyter

# Install ffmpeg
#RUN add-apt-repository ppa:jonathonf/ffmpeg-4
RUN apt-get update \
    && apt-get install -y ffmpeg

# https://github.com/NVIDIA/nvidia-docker/issues/864
RUN apt-get install -y libsm6 libxext6 libxrender-dev

ENV HOME_PATH /tf

# Install data-science-learning library
ENV DS_DIR $HOME_PATH/notebooks/data-science-learning
ADD . $DS_DIR
WORKDIR $DS_DIR
RUN python setup.py develop
RUN rm -Rf $DS_DIR

# Jupyter notebook extensions
RUN pip3 install jupyter_contrib_nbextensions

WORKDIR $HOME_PATH