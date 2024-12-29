#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

typedef struct Shapes {
    std::vector<std::vector<Point>> contours;
    vector<Vec4i> hierarchy;
} Shapes;

void find_squares(Mat &image, vector<vector<Point>> &squares);
Mat skeletonize(Mat img_inv, std::string output_path);
Mat otsu(Mat img, std::string output_path);
Shapes find_strokes(Mat img, std::string output_path);

void sort_points_clockwise(std::vector<cv::Point> &points);

Mat get_whiteboard(Mat image, std::string output_dir);

vector<vector<Point>> locate_quadrangles(cv::Mat image,
                                         std::string output_path);
