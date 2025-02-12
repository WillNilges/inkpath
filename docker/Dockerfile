FROM ubuntu:22.04

# Install basic utilities and the script to install dependencies
RUN apt update && \
    apt install -y \
    build-essential \
    cmake \
    git \
    liblua5.4-dev \
    lua-lgi \
    make \
    pkg-config \
    sudo

# Copy local files into the container
COPY ./ /inkpath

# Build project
WORKDIR /inkpath
# Clean build directory
RUN rm -rf build
RUN apt list --installed > installed_packages.txt
RUN cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DUSE_EXTERNAL_OPENCV_STATIC=ON && \
    cmake --build build -j$(nproc)
