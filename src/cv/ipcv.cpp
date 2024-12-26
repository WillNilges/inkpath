#include "ipcv.h"
#include <iostream>


Mat burger(Mat img, std::string output_path) {
    // Convert the image to HSV color space
    cv::Mat hsvImage;
    cv::cvtColor(img, hsvImage, cv::COLOR_BGR2HSV);

    // Split the HSV image into channels
    std::vector<cv::Mat> hsvChannels;
    cv::split(hsvImage, hsvChannels);

    // Extract the hue channel
    cv::Mat hueChannel = hsvChannels[0];
    

    // Get the number of channels
    int c = img.channels();
    std::cout << "Number of channels in the image: " << c << std::endl;


    Mat hsv;
    cvtColor(img,hsv,COLOR_BGR2HSV);

    std::vector<cv::Mat> channels;
    split(hsv, channels);

    Mat H = channels[0];
    Mat S = channels[1];
    Mat V = channels[2];
    
    //Mat downsampled;
    //pyrDown(skel_invert, downsampled, Size( img.cols/2, img.rows/2 ));
    if (output_path != "") {
        imwrite(output_path, H);
        std::cout << "Image has been written to " << output_path << "\n";
    }

    return H;
}

// Skeletonization algorithm. I might mess around with this
// more down the road.
// TODO: Where the hell did I find this?
Mat skeletonize(Mat img_inv, std::string output_path) {
    Mat img = img_inv;
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
    findContours( src, contours, hierarchy,
        RETR_TREE, CHAIN_APPROX_SIMPLE );

    // Try to connect contours
    vector<Rect> boundRect( contours.size() );
    for (int i = 0; i >= 0; i = hierarchy[i][0])
    {
        boundRect[i] = boundingRect( contours[i] );
    }
    
    // Try to group contours

    // Remove contours that are too small.
    /*int min_points=2; // area threshold
    for(int i = 0; i< contours.size(); i++) // iterate through each contour.
    {
        // double area=contourArea(contours[i],false); // Find the area of contour
        // if(area < min_area)
        //     contours.erase(contours.begin() + i);

        // This doesn't work
        //int points = contours[i].size();
        //if (points < min_points)
        //    cout << "Found short contour. Removing...\n";
        //    contours.erase(contours.begin() + i);
        //    hierarchy.erase(hierarchy.begin() + i);
    }*/

    if (output_path != "") {
        // iterate through all the top-level contours,
        // draw each connected component with its own random color
        int idx = 0;
        for( ; idx >= 0; idx = hierarchy[idx][0] )
        {
            Scalar color( rand()&255, rand()&255, rand()&255 );
            drawContours( dst, contours, idx, color, FILLED, 8, hierarchy );
            //rectangle( dst, boundRect[idx].tl(), boundRect[idx].br(), color, 2 );
        }

        imwrite(output_path, dst);
        std::cout << "Image has been written to " << output_path << "\n";
    }

    return Shapes{contours, hierarchy};
}

// FIXME: This shouldn't be necessary when I'm done >:)
void prep_otsu(char* image_path)
{
    Mat img = imread(image_path, 0);
    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return;
    }
    otsu(img, image_path);
}
