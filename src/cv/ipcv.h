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


void draw_squares(cv::Mat& image, const std::vector<std::vector<cv::Point>>& squares, cv::Scalar color);
void find_squares(cv::Mat& image, std::vector<std::vector<cv::Point>>& squares);
cv::Mat skeletonize(cv::Mat img_inv, std::string output_dir);
cv::Mat otsu(cv::Mat img, std::string output_dir);
Shapes find_strokes(cv::Mat img, std::string output_dir);

void sort_points_clockwise(std::vector<cv::Point>& points);

cv::Mat get_whiteboard(cv::Mat image, std::string output_dir);

std::vector<std::vector<cv::Point>> locate_quadrangles(cv::Mat image,
                                                       std::string output_path);

}
