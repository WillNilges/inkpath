#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

namespace inkp {

typedef struct Shapes {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
} Shapes;

// class StrokeFinder {
// }

// ipcv.cpp
cv::Mat thresholdWithPrep(cv::Mat img, std::string output_dir);
cv::Mat skeletonize(cv::Mat img_inv, std::string output_dir);
std::vector<std::vector<cv::Point>> findStrokes(cv::Mat img, std::string output_dir);

// quadrangle.cpp
cv::Mat getWhiteboard(cv::Mat image, std::string output_dir);
std::vector<std::vector<cv::Point>> filterQuadrangles(cv::Mat image,
                                                      std::string output_path);
void findQuadrangles(cv::Mat& image,
                     std::vector<std::vector<cv::Point>>& squares);
void sortPointsClockwise(std::vector<cv::Point>& points);
double angle(cv::Point pt1, cv::Point pt2, cv::Point pt0);
void drawSquares(cv::Mat& image,
                 const std::vector<std::vector<cv::Point>>& squares,
                 cv::Scalar color);

} // namespace inkp
