#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
using namespace cv;

void otsu(Mat img)
{
    int k;
    // global thresholding
    
    Mat global_thresh;
    threshold(img, global_thresh, 127, 255, THRESH_BINARY);
    imshow("global.png", global_thresh);

    // otsu's thresholding
    Mat otsu_thresh;
    threshold(img, otsu_thresh, 0, 255, THRESH_BINARY | THRESH_OTSU);
    imshow("otsu.png", otsu_thresh);

    //otsu's thresholding after gaussian filtering
    Mat gauss_thresh;
    Mat blur;
    GaussianBlur(img, blur, Size(5, 5), 0, 0); 
    threshold(blur, gauss_thresh, 0, 255, THRESH_OTSU);
    imshow("gaussian.png", gauss_thresh);

    Mat adaptive_thresh;
    adaptiveThreshold(img, adaptive_thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2);
    imshow("adaptive_thresh.png", adaptive_thresh);

    // upsample BEFORE doing the thresholding
    Mat upsampled;
    pyrUp(gauss_thresh, upsampled,  Size( gauss_thresh.cols*2, gauss_thresh.rows*2 ));
    imshow("gauss_upsampled.png", upsampled);

    while (true) {
        k = waitKey(0); // Wait for a keystroke in the window
        if (k == 'q') {
            return;
        }
    }
}

int main()
{
    std::string image_path = "/xopp-dev/inkpath/build/roland.jpg";
    Mat img = imread(image_path, IMREAD_GRAYSCALE);
    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }
    otsu(img);
    /*imshow("Display window", img);
    int k = waitKey(0); // Wait for a keystroke in the window
    if(k == 's')
    {
        imwrite("output.png", img);
    }*/
    return 0;
}


