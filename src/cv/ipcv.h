#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/mat.hpp>

using namespace cv;
using namespace std;

typedef struct Shapes {
    std::vector<std::vector<Point>> contours;
    vector<Vec4i> hierarchy;
} Shapes;

void find_squares(Mat& image, vector<vector<Point> >& squares, std::string path_string, std::string file_title);
Mat processColors(Mat& img, std::string output_path);
Mat hough(Mat src, std::string output_path);
Mat crop(Mat img, std::string output_path);
Mat mask_large_contours(Mat img, std::string output_path);
Mat skeletonize(Mat img_inv, std::string output_path);
Mat otsu(Mat img, std::string output_path);
Shapes find_shapes(Mat img, std::string output_path);
