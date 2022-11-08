#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/mat.hpp>

// OpenCV CUDA API
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>

using namespace cv;
using namespace std;

typedef struct Shapes {
    std::vector<std::vector<Point>> contours;
    vector<Vec4i> hierarchy;
} Shapes;

Mat skeletonize(Mat img_inv, std::string output_path);
Mat otsu(Mat img, std::string output_path);
Shapes find_shapes(Mat img, std::string output_path);

