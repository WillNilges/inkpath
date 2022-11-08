#!/bin/bash

set -e

cores=4

# For compiling faster
if [ -n "$1" ]; then
    cores=$1
fi

# Create exterior directories
mkdir -p ../opencv
cd ../opencv

#wget -O opencv.zip https://github.com/opencv/opencv/archive/4.5.1.zip
#wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/refs/tags/4.5.1.zip
#
#unzip opencv.zip
#unzip opencv_contrib.zip
# Create opencv build directory
mkdir -p build && cd build
# Configure
cmake  ../opencv-4.5.1 \
	-DWITH_CUDA=ON -DWITH_CUDNN=OFF -DOPENCV_DNN_CUDA=OFF -DBUILD_opencv_cudacodec=OFF \
	-D OPENCV_EXTRA_MODULES_PATH='../opencv_contrib-4.5.1/modules' \
	-DBUILD_SHARED_LIBS=OFF -DOPENCV_GENERATE_PKGCONFIG=YES -DWITH_GTK=OFF -D BUILD_TESTS=OFF
# Build
cmake --build . -j$cores

make install

