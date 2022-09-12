#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
using namespace cv;

Mat skeletonize(Mat img_inv, std::string output_path);
Mat otsu(Mat img, std::string output_path);
void prep_otsu(char* image_path);
