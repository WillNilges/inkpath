#include "otsu.h"

// Skeletonization algorithm. I might mess around with this
// more down the road.
void aggressiveXylophones(Mat img_inv) {
    Mat img;
    bitwise_not(img_inv, img);
    cv::Mat skel(img.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat temp;
    cv::Mat eroded;
     
    cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
     
    bool done;		
    do
    {
      cv::erode(img, eroded, element);
      cv::dilate(eroded, temp, element); // temp = open(img)
      cv::subtract(img, temp, temp);
      cv::bitwise_or(skel, temp, skel);
      eroded.copyTo(img);
     
      done = (cv::countNonZero(img) == 0);
    } while (!done);

    Mat skel_invert;
    bitwise_not(skel, skel_invert);

    Mat downsampled;
    pyrDown(skel_invert, downsampled, Size( img.cols/2, img.rows/2 ));
    imwrite("/tmp/spook.png", downsampled);
}

Mat otsu(Mat img)
{
    int k;
    // Upsample our image
    Mat upsampled;
    pyrUp(img, upsampled,  Size( img.cols*2, img.rows*2 ));

    //otsu's thresholding after gaussian filtering
    Mat gauss_thresh;
    Mat blur;
    GaussianBlur(upsampled, blur, Size(5, 5), 0, 0); 
    threshold(blur, gauss_thresh, 0, 255, THRESH_OTSU);

    imwrite("/tmp/inkpath_cv.bmp", gauss_thresh);
    printf("image has been written to /tmp\n");
    return gauss_thresh;
}

void prep_otsu(char* image_path)
{
    Mat img = imread(image_path, 0);
    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return;
    }
    otsu(img);
}
