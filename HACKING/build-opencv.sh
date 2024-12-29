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

wget -O opencv.zip https://github.com/opencv/opencv/archive/4.5.1.zip

unzip opencv.zip
# Create opencv build directory
mkdir -p build && cd build
# Configure
cmake  ../opencv-4.5.1 -DBUILD_SHARED_LIBS=OFF -DOPENCV_GENERATE_PKGCONFIG=YES -DWITH_GTK=OFF -D BUILD_TESTS=OFF -D BUILD_TIFF=ON -D WITH_QT=OFF -D WITH_GTK=ON



# Build
cmake --build . -j$cores

make install

