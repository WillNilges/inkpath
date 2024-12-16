#!/bin/bash

set -e

cores=4


while getopts ":j:d" option; do
	case $option in
		j)
			# Compile faster
			cores=$OPTARG;;
		d)
			# Download dependencies
			download=true;
	esac
done
		
	
# Create exterior directories
mkdir -p ../opencv
cd ../opencv

if [[ download == "true" ]]; then
	wget -O opencv.zip https://github.com/opencv/opencv/archive/4.5.1.zip
	unzip opencv.zip
fi

if [ ! -f "opencv.zip" ]; then
	echo "Did not find opencv.zip. Try running with -d to download dependencies."
	exit 1
fi

# Create opencv build directory
mkdir -p build && cd build
# Configure
cmake  ../opencv-4.5.1 -DBUILD_SHARED_LIBS=OFF -DOPENCV_GENERATE_PKGCONFIG=YES -DWITH_GTK=OFF -D BUILD_TESTS=OFF
# Build
cmake --build . -j$cores

make install

