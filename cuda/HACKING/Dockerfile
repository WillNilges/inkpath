FROM docker.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN touch /root/.Xauthority
RUN apt-get update

# Line 09: Inkpath
# Line 10: Xournalpp
# Line 12: debugging tools
RUN apt-get install -y make liblua5.3-dev build-essential pkg-config libglib2.0-dev libpng-dev && \
  apt-get install -y cmake libgtk-3-dev libpoppler-glib-dev portaudio19-dev libsndfile-dev \
  dvipng texlive libxml2-dev libzip-dev librsvg2-dev gettext lua-lgi \
  libgtksourceview-4-dev
RUN apt-get install -y tmux git gdb vim x11-apps tree wget unzip

# Download the cuda toolkit because for some reason it's missing from the cudnn container
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y cuda

# Build and install OpenCV into the container
COPY ./build-opencv.sh .
RUN bash /build-opencv.sh 48

