#ifndef IPCV
#define IPCV
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/types_c.h>

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

Mat adaptive(Mat img, std::string output_path);
Mat otsu(Mat img, std::string output_path);
Mat skeletonize(Mat img_inv, std::string output_path);
Shapes find_shapes(Mat img, std::string output_path);
#endif
