#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
using namespace cv;

void aggressiveXylophones(Mat img_inv);
Mat otsu(Mat img);
void prep_otsu(char* image_path);
