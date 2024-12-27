#include "ipcv.h"
#include <iostream>


Mat crop(Mat img, std::string output_path) {
    // Convert the image to HSV color space
    cv::Mat hsvImage;
    cv::cvtColor(img, hsvImage, cv::COLOR_BGR2HSV);

    // Split the HSV image into channels
    std::vector<cv::Mat> hsvChannels;
    cv::split(hsvImage, hsvChannels);

    // Extract the hue channel
    cv::Mat hueChannel = hsvChannels[0];

    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV);

    std::vector<cv::Mat> channels;
    split(hsv, channels);

    Mat H = channels[0];

    // Now draw a rectangle around the darkest edge
    //cv::Mat otsuH = otsu(H, "");

    //cv::Mat cannyH;
    //cv::Canny(otsuH, cannyH, 100, 50);

    // Invert and otsu for better results
    Mat burger_img_inv;
    bitwise_not(H, burger_img_inv);

    Mat otsu_burger_img_inv = otsu(burger_img_inv, "");

    // Dialate to try to smooth the guy
    cv::Mat temp;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
    for (int i = 0; i < 25; i++)
    {
        cv::dilate(otsu_burger_img_inv, temp, element); // temp = open(img)
        otsu_burger_img_inv = temp;
    }

    Mat output = otsu_burger_img_inv;
    
    if (output_path != "") {
        imwrite(output_path, output);
        std::cout << "Image has been written to " << output_path << "\n";
    }

    return output;
}

Mat hough(Mat img, std::string output_path) {
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(src, contours, hierarchy,
        RETR_TREE, CHAIN_APPROX_SIMPLE );
    
    // Remove contours that are too big.
    double maxArea = 0.2 * img.width * img.height;
    contours.erase(remove_if(contours.begin(), contours.end(),
        [minArea](const vector<Point>& contour) {
            return contourArea(contour) > maxArea;
        }), contours.end());

    Mat output = otsu_burger_img_inv;
    
    if (output_path != "") {
        imwrite(output_path, output);
        std::cout << "Image has been written to " << output_path << "\n";
    }

    return output;

}

// Skeletonization algorithm. I might mess around with this
// more down the road.
// TODO: Where the hell did I find this?
Mat skeletonize(Mat img_inv, std::string output_path) {
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
     
      done = (cv::countNonZero(ipg) == 0);
    } while (!done);

    Mat skel_invert;
    bitwise_not(skel, skel_invert);

    //Mat downsampled;
    //pyrDown(skel_invert, downsampled, Size( img.cols/2, img.rows/2 ));
    if (output_path != "") {
        imwrite(output_path, skel_invert);
        std::cout << "Image has been written to " << output_path << "\n";
    }
    return skel_invert;
}

// Apply an Otsu's thresholding to the object. I found that this was
// the best function of the ones I tried
Mat otsu(Mat img, std::string output_path)
{
    int k;
    // Upsample our image, if needed.
    Mat upsampled;
    if (img.rows < 1000 || img.cols < 1000) {
        pyrUp(img, upsampled,  Size(img.cols*2, img.rows*2));
    } else {
        upsampled = img;
    }

    //otsu's thresholding after gaussian filtering
    Mat gauss_thresh;
    Mat blur;
    GaussianBlur(upsampled, blur, Size(5, 5), 0, 0); 
    threshold(blur, gauss_thresh, 0, 255, THRESH_OTSU);

    if (output_path != "") {
        imwrite(output_path, gauss_thresh);
        std::cout << "Image has been written to " << output_path << "\n";
    }
    return gauss_thresh;
}

// Prereqs: Must be binary color image, target must be black
Shapes find_shapes(Mat img, std::string output_path) {
    Mat src;
    bitwise_not(img, src);

    Mat dst = Mat::zeros(src.rows, src.cols, CV_8UC3);
    src = src > 1;
    //namedWindow( "Source", 1 );
    //imshow( "Source", src );
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(src, contours, hierarchy,
        RETR_TREE, CHAIN_APPROX_SIMPLE );
    
    // Remove contours that are too small in order to "de-noise" the image 
    // a little
    double minArea = 2.0;
    contours.erase(remove_if(contours.begin(), contours.end(),
        [minArea](const vector<Point>& contour) {
            return contourArea(contour) < minArea;
        }), contours.end());
    
    if (output_path != "") {
        // iterate through all the top-level contours,
        // draw each connected component with its own random color
        for (size_t i = 0; i < contours.size(); i++) {
            Scalar color( rand()&255, rand()&255, rand()&255 ); // Random color
            drawContours(dst, contours, (int)i, color, 2, LINE_8, hierarchy, 0);
        }   

        imwrite(output_path, dst);
        std::cout << "Image has been written to " << output_path << "\n";
    }

        return Shapes{contours, hierarchy};
    }
