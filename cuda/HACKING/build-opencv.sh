#!/bin/bash

set -e

cores=4

# For compiling faster
if [ -n "$1" ]; then
    cores=$1
fi

# Create exterior directories
mkdir -p ./opencv
cd ./opencv

wget -O opencv.zip https://github.com/opencv/opencv/archive/4.5.1.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/refs/tags/4.5.1.zip

unzip opencv.zip
unzip opencv_contrib.zip

# Create opencv build directory
mkdir -p build && cd build
# Configure
cmake  ../opencv-4.5.1 \
	-DWITH_CUDA=ON \
    -DOPENCV_ENABLE_NONFREE=ON \
    -DWITH-CUBLAS=1 \
    -DENABLE_FAST_MATH=1 \
    -DWITH_CUDNN=ON \
    -DOPENCV_DNN_CUDA=ON \
    -DCUDA_ARCH_BIN=3.5 \
    -DBUILD_opencv_cudacodec=OFF \
	-DOPENCV_EXTRA_MODULES_PATH='../opencv_contrib-4.5.1/modules' \
	-DOPENCV_GENERATE_PKGCONFIG=YES \
    -DWITH_GTK=OFF \
    -DBUILD_TESTS=OFF
# Build
cmake --build . -j$cores

make install

