#ifndef IPCV_GPU
#define IPCV_GPU
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/mat.hpp>

// OpenCV CUDA API
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafilters.hpp>

#include "diagnostics.h"

using namespace cv;
using namespace std;

#ifndef IPCV_SHAPES
#define IPCV_SHAPES
typedef struct Shapes {
    std::vector<std::vector<Point>> contours;
    vector<Vec4i> hierarchy;
} Shapes;
#endif //#ifndef IPCV_SHAPES

Mat gpu_skeletonize(Mat img_inv, std::string output_path);
Mat gpu_otsu(Mat img, std::string output_path);
Shapes gpu_find_shapes(Mat img, std::string output_path);
#endif
