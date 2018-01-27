# Use an official Python runtime as a parent image
FROM python:3.6-slim

# Set the working directory to /face_swap
WORKDIR /face_swap

# Copy the current directory contents into the container at /app
ADD . /face_swap

RUN mkdir /face_swap/faces_from
RUN mkdir /face_swap/faces_to
RUN mkdir /face_swap/faces_out
RUN mkdir /face_swap/models

# See https://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/
RUN apt-get update && apt-get install -y \
	build-essential cmake \
	libgtk-3-dev \
	libboost-all-dev \
	tk-dev # see https://stackoverflow.com/questions/5459444/tkinter-python-may-not-be-configured-for-tk

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Copy the models and picture in target folders
# Copy the util.py file in the top folder
# python utils.py -f ./faces_from -t ./faces_to -o ./faces_out -i ./models