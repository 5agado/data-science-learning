# Use an official Python runtime as a parent image
FROM python:3.6-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# See https://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/
RUN apt-get update && apt-get install -y \
	build-essential cmake \
	libgtk-3-dev \
	libboost-all-dev

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Copy the models and picture in target folders
# Copy the util.py file in the top folder
# python utils.py -f ./face_from -t ./face_to -o ./face_out -i ./models