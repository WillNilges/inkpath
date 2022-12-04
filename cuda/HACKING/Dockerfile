#FROM docker.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
FROM docker.io/nvidia/cuda:11.4.1-cudnn8-devel-ubuntu20.04

RUN touch /root/.Xauthority
RUN apt-get update

# Python3 stuff for plotting data.
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y make liblua5.3-dev build-essential pkg-config libglib2.0-dev libpng-dev cmake \
    tmux git gdb vim x11-apps tree wget unzip python3.8-venv python3-pip

# Build and install OpenCV into the container
COPY ./build-opencv.sh .
RUN bash /build-opencv.sh -j 48 -c 6.1

