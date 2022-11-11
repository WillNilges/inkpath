CC=g++

# Warnings
WARNINGS = -Wall -Wextra -Wpedantic -Wconversion -Wformat=2 -Winit-self \
	-Wmissing-include-dirs -Wformat-nonliteral -Wnested-externs \
	-Wno-unused-parameter -Wold-style-definition -Wredundant-decls -Wshadow \
	-Wstrict-prototypes -Wwrite-strings
LIGHT_WARNINGS = -Wall
CV_PKG_CONFIG=`pkg-config --cflags --libs opencv4`
#CUDA_INCL=-I/usr/local/cuda/include

.PHONY: test clean

build: ipcv-cuda-debug

*.o: *.cpp
	$(CC) $(CFLAGS) $(CUDA_INCL) $(CV_PKG_CONFIG) -g -c $?

# For generating a CV debugging binary
ipcv-cuda-debug: *.o 
	$(CC) $< $(CUDA_INCL) $(CV_PKG_CONFIG) -g -o $@

TEST_IMAGE=msd_board.jpg

test: ipcv-cuda-debug
	./ipcv-cuda-debug -f ~/Pictures/inkpath_sample_images_vol_2/good/$(TEST_IMAGE) -o ./outputs/output.jpg

benchmark: ipcv-cuda-debug
	./ipcv-cuda-debug -f ~/Pictures/inkpath_sample_images_vol_2/good/$(TEST_IMAGE) -i 100

clean:
	rm -rf *.o
	rm -rf ipcv-cuda-debug
