CC=nvcc

# Warnings
WARNINGS = -Wall -Wextra -Wpedantic -Wconversion -Wformat=2 -Winit-self \
	-Wmissing-include-dirs -Wformat-nonliteral -Wnested-externs \
	-Wno-unused-parameter -Wold-style-definition -Wredundant-decls -Wshadow \
	-Wstrict-prototypes -Wwrite-strings
LIGHT_WARNINGS = -Wall
CV_PKG_CONFIG=`pkg-config --cflags --libs opencv4`
#CUDA_INCL=-I/usr/local/cuda/include
COMPUTE_CAPABILITY=61
CUDA_FLAGS=-gencode arch=compute_$(COMPUTE_CAPABILITY),code=compute_$(COMPUTE_CAPABILITY) -Wno-deprecated-gpu-targets

USE_DEVICE=0

SRC = $(wildcard *.cpp)
CPP_OBJ = $(patsubst %.cpp,%.o,$(SRC))
CU_SRC = $(wildcard *.cu)
CU_OBJ = $(patsubst %.cu,%.o,$(CU_SRC))

#$(info found $(SRC)...)
#$(info compiling CPP_OBJ as $(CPP_OBJ)...)
#$(info found $(CU_SRC)...)
#$(info compiling CU_OBJ as $(CU_OBJ)...)

.PHONY: test clean

build: ipcv-cuda-debug

%.o: %.cpp #$(SRC) $(CU_SRC)
	$(CC) $(CFLAGS) $(CUDA_FLAGS) $(CUDA_INCL) $(CV_PKG_CONFIG) -g -c $< -o $@

%.o: %.cu #$(SRC) $(CU_SRC)
	$(CC) $(CFLAGS) $(CUDA_FLAGS) $(CUDA_INCL) $(CV_PKG_CONFIG) -g -c $< -o $@

# For generating a CV debugging binary
ipcv-cuda-debug: $(CPP_OBJ) $(CU_OBJ)
	$(CC) $(CPP_OBJ) $(CU_OBJ) $(CUDA_INCL) $(CV_PKG_CONFIG) -g -o $@

OTSU_IMAGE=/xopp-dev/inkpath_samples/inkpath_sample_images_vol_2/good/msd_board.jpg
ADAPTIVE_IMAGE=/xopp-dev/inkpath_samples/inkpath_sample_images/physics.jpg
TEST_IMAGE=$(OTSU_IMAGE)

test: ipcv-cuda-debug
	./ipcv-cuda-debug -f $(OTSU_IMAGE) -o ./outputs/output.jpg

adaptive-test: ipcv-cuda-debug
	./ipcv-cuda-debug -f $(ADAPTIVE_IMAGE) -o ./outputs/output.jpg

benchmark: ipcv-cuda-debug
	./ipcv-cuda-debug -f $(TEST_IMAGE) -i 1000

big-benchmark: ipcv-cuda-debug
	./ipcv-cuda-debug -f $(TEST_IMAGE) -i 100 -u 3

# /xopp-dev/inkpath/cuda
DATA_IMAGE_01=./samples/msd_board.jpg
DATA_IMAGE_02=./samples/waves.jpg
DATA_IMAGE_03=./samples/circuits_cropped.jpg
DATA_IMAGE_04=./samples/mask.jpg
DATA_IMAGE_05=./samples/roland_sip.jpg

HOSTNAME=$(shell hostname)
ts := $(shell /bin/date "+%Y%m%d-%H%M%S")
OUTPUT_PATH=./data/$(ts)/full

data_mkdir:
	mkdir -p ./data/$(ts)

recorded-test: ipcv-cuda-debug data_mkdir
	#./ipcv-cuda-debug -f $(OTSU_IMAGE) -d $(USE_DEVICE) -i 10 -t $(OUTPUT_PATH)
	./ipcv-cuda-debug -f $(DATA_IMAGE_01) -d $(USE_DEVICE) -u 0 -s -i 100 -t $(OUTPUT_PATH)-img01.csv
	./ipcv-cuda-debug -f $(DATA_IMAGE_02) -d $(USE_DEVICE) -u 0 -s -i 100 -t $(OUTPUT_PATH)-img02.csv
	./ipcv-cuda-debug -f $(DATA_IMAGE_03) -d $(USE_DEVICE) -u 0 -s -i 100 -t $(OUTPUT_PATH)-img03.csv
	./ipcv-cuda-debug -f $(DATA_IMAGE_04) -d $(USE_DEVICE) -u 0 -s -i 100 -t $(OUTPUT_PATH)-img04.csv
	./ipcv-cuda-debug -f $(DATA_IMAGE_05) -d $(USE_DEVICE) -u 0 -s -i 100 -t $(OUTPUT_PATH)-img05.csv

# Can't do 4x upscaling. Not enough memory and/or not enough time to debug.
gauntlet: ipcv-cuda-debug data_mkdir
	echo $(DATA_IMAGE_01)
	./ipcv-cuda-debug -f $(DATA_IMAGE_01) -d $(USE_DEVICE) -u 0 -i 100 -t $(OUTPUT_PATH)-img01.csv
	./ipcv-cuda-debug -f $(DATA_IMAGE_01) -d $(USE_DEVICE) -u 1 -i 100 -t $(OUTPUT_PATH)-img01.csv
	./ipcv-cuda-debug -f $(DATA_IMAGE_01) -d $(USE_DEVICE) -u 2 -i 50 -t $(OUTPUT_PATH)-img01.csv
	./ipcv-cuda-debug -f $(DATA_IMAGE_01) -d $(USE_DEVICE) -u 3 -i 50 -t $(OUTPUT_PATH)-img01.csv
#	./ipcv-cuda-debug -f $(DATA_IMAGE_01) -d $(USE_DEVICE) -u 4 -i 50 -t $(OUTPUT_PATH)-img01.csv
	echo $(DATA_IMAGE_02)
	./ipcv-cuda-debug -f $(DATA_IMAGE_02) -d $(USE_DEVICE) -u 0 -i 100 -t $(OUTPUT_PATH)-img02.csv
	./ipcv-cuda-debug -f $(DATA_IMAGE_02) -d $(USE_DEVICE) -u 1 -i 100 -t $(OUTPUT_PATH)-img02.csv
	./ipcv-cuda-debug -f $(DATA_IMAGE_02) -d $(USE_DEVICE) -u 2 -i 50 -t $(OUTPUT_PATH)-img02.csv
	./ipcv-cuda-debug -f $(DATA_IMAGE_02) -d $(USE_DEVICE) -u 3 -i 50 -t $(OUTPUT_PATH)-img02.csv
#	./ipcv-cuda-debug -f $(DATA_IMAGE_02) -d $(USE_DEVICE) -u 4 -i 50 -t $(OUTPUT_PATH)-img02.csv
	echo $(DATA_IMAGE_03)
	./ipcv-cuda-debug -f $(DATA_IMAGE_03) -d $(USE_DEVICE) -u 0 -i 100 -t $(OUTPUT_PATH)-img03.csv
	./ipcv-cuda-debug -f $(DATA_IMAGE_03) -d $(USE_DEVICE) -u 1 -i 100 -t $(OUTPUT_PATH)-img03.csv
	./ipcv-cuda-debug -f $(DATA_IMAGE_03) -d $(USE_DEVICE) -u 2 -i 50 -t $(OUTPUT_PATH)-img03.csv
	./ipcv-cuda-debug -f $(DATA_IMAGE_03) -d $(USE_DEVICE) -u 3 -i 50 -t $(OUTPUT_PATH)-img03.csv
#	./ipcv-cuda-debug -f $(DATA_IMAGE_03) -d $(USE_DEVICE) -u 4 -i 50 -t $(OUTPUT_PATH)-img03.csv
	echo $(DATA_IMAGE_04)
	./ipcv-cuda-debug -f $(DATA_IMAGE_04) -d $(USE_DEVICE) -u 0 -i 100 -t $(OUTPUT_PATH)-img04.csv
	./ipcv-cuda-debug -f $(DATA_IMAGE_04) -d $(USE_DEVICE) -u 1 -i 100 -t $(OUTPUT_PATH)-img04.csv
	./ipcv-cuda-debug -f $(DATA_IMAGE_04) -d $(USE_DEVICE) -u 2 -i 50 -t $(OUTPUT_PATH)-img04.csv
	./ipcv-cuda-debug -f $(DATA_IMAGE_04) -d $(USE_DEVICE) -u 3 -i 50 -t $(OUTPUT_PATH)-img04.csv
#	./ipcv-cuda-debug -f $(DATA_IMAGE_04) -d $(USE_DEVICE) -u 4 -i 50 -t $(OUTPUT_PATH)-img03.csv
	echo $(DATA_IMAGE_05)
	./ipcv-cuda-debug -f $(DATA_IMAGE_05) -d $(USE_DEVICE) -u 0 -i 100 -t $(OUTPUT_PATH)-img05.csv
	./ipcv-cuda-debug -f $(DATA_IMAGE_05) -d $(USE_DEVICE) -u 1 -i 100 -t $(OUTPUT_PATH)-img05.csv
	./ipcv-cuda-debug -f $(DATA_IMAGE_05) -d $(USE_DEVICE) -u 2 -i 50 -t $(OUTPUT_PATH)-img05.csv
	./ipcv-cuda-debug -f $(DATA_IMAGE_05) -d $(USE_DEVICE) -u 3 -i 50 -t $(OUTPUT_PATH)-img05.csv
#	./ipcv-cuda-debug -f $(DATA_IMAGE_05) -d $(USE_DEVICE) -u 4 -i 50 -t $(OUTPUT_PATH)-img03.csv

OUTPUT_PATH_SHORT=./data/$(ts)/short
gauntlet-thresh-only: ipcv-cuda-debug data_mkdir
	echo $(DATA_IMAGE_01)
	./ipcv-cuda-debug -f $(DATA_IMAGE_01) -d $(USE_DEVICE) -u 0 -s -i 500 -t $(OUTPUT_PATH_SHORT)-img01.csv
	./ipcv-cuda-debug -f $(DATA_IMAGE_01) -d $(USE_DEVICE) -u 1 -s -i 500 -t $(OUTPUT_PATH_SHORT)-img01.csv
	./ipcv-cuda-debug -f $(DATA_IMAGE_01) -d $(USE_DEVICE) -u 2 -s -i 100 -t $(OUTPUT_PATH_SHORT)-img01.csv
	./ipcv-cuda-debug -f $(DATA_IMAGE_01) -d $(USE_DEVICE) -u 3 -s -i 50 -t $(OUTPUT_PATH_SHORT)-img01.csv
#	./ipcv-cuda-debug -f $(DATA_IMAGE_01) -d $(USE_DEVICE) -u 4 -s -i 50 -t $(OUTPUT_PATH_SHORT)-img01.csv
	echo $(DATA_IMAGE_02)                                                                            
	./ipcv-cuda-debug -f $(DATA_IMAGE_02) -d $(USE_DEVICE) -u 0 -s -i 500 -t $(OUTPUT_PATH_SHORT)-img02.csv
	./ipcv-cuda-debug -f $(DATA_IMAGE_02) -d $(USE_DEVICE) -u 1 -s -i 500 -t $(OUTPUT_PATH_SHORT)-img02.csv
	./ipcv-cuda-debug -f $(DATA_IMAGE_02) -d $(USE_DEVICE) -u 2 -s -i 100 -t $(OUTPUT_PATH_SHORT)-img02.csv
	./ipcv-cuda-debug -f $(DATA_IMAGE_02) -d $(USE_DEVICE) -u 3 -s -i 50 -t $(OUTPUT_PATH_SHORT)-img02.csv
#	./ipcv-cuda-debug -f $(DATA_IMAGE_02) -d $(USE_DEVICE) -u 4 -s -i 50 -t $(OUTPUT_PATH_SHORT)-img02.csv
	echo $(DATA_IMAGE_03)                                                                            
	./ipcv-cuda-debug -f $(DATA_IMAGE_03) -d $(USE_DEVICE) -u 0 -s -i 500 -t $(OUTPUT_PATH_SHORT)-img03.csv
	./ipcv-cuda-debug -f $(DATA_IMAGE_03) -d $(USE_DEVICE) -u 1 -s -i 500 -t $(OUTPUT_PATH_SHORT)-img03.csv
	./ipcv-cuda-debug -f $(DATA_IMAGE_03) -d $(USE_DEVICE) -u 2 -s -i 100 -t $(OUTPUT_PATH_SHORT)-img03.csv
	./ipcv-cuda-debug -f $(DATA_IMAGE_03) -d $(USE_DEVICE) -u 3 -s -i 50 -t $(OUTPUT_PATH_SHORT)-img03.csv
#	./ipcv-cuda-debug -f $(DATA_IMAGE_03) -d $(USE_DEVICE) -u 4 -s -i 50 -t $(OUTPUT_PATH_SHORT)-img03.csv
	echo $(DATA_IMAGE_04)                                                                            
	./ipcv-cuda-debug -f $(DATA_IMAGE_04) -d $(USE_DEVICE) -u 0 -s -i 500 -t $(OUTPUT_PATH_SHORT)-img04.csv
	./ipcv-cuda-debug -f $(DATA_IMAGE_04) -d $(USE_DEVICE) -u 1 -s -i 500 -t $(OUTPUT_PATH_SHORT)-img04.csv
	./ipcv-cuda-debug -f $(DATA_IMAGE_04) -d $(USE_DEVICE) -u 2 -s -i 100 -t $(OUTPUT_PATH_SHORT)-img04.csv
	./ipcv-cuda-debug -f $(DATA_IMAGE_04) -d $(USE_DEVICE) -u 3 -s -i 50 -t $(OUTPUT_PATH_SHORT)-img04.csv
#	./ipcv-cuda-debug -f $(DATA_IMAGE_04) -d $(USE_DEVICE) -u 4 -s -i 50 -t $(OUTPUT_PATH_SHORT)-img03.csv
	echo $(DATA_IMAGE_05)                                                                            
	./ipcv-cuda-debug -f $(DATA_IMAGE_05) -d $(USE_DEVICE) -u 0 -s -i 500 -t $(OUTPUT_PATH_SHORT)-img05.csv
	./ipcv-cuda-debug -f $(DATA_IMAGE_05) -d $(USE_DEVICE) -u 1 -s -i 500 -t $(OUTPUT_PATH_SHORT)-img05.csv
	./ipcv-cuda-debug -f $(DATA_IMAGE_05) -d $(USE_DEVICE) -u 2 -s -i 100 -t $(OUTPUT_PATH_SHORT)-img05.csv
	./ipcv-cuda-debug -f $(DATA_IMAGE_05) -d $(USE_DEVICE) -u 3 -s -i 50 -t $(OUTPUT_PATH_SHORT)-img05.csv
#	./ipcv-cuda-debug -f $(DATA_IMAGE_05) -d $(USE_DEVICE) -u 4 -s -i 50 -t $(OUTPUT_PATH_SHORT)-img03.csv

clean:
	rm -rf *.o
	rm -rf ipcv-cuda-debug
