#include "ipcv.h"

namespace inkp {

// Skeletonization algorithm. I might mess around with this
// more down the road.
// TODO: Where did I find this?
cv::Mat skeletonize(cv::Mat img_inv, std::string output_dir) {
    cv::Mat img;
    bitwise_not(img_inv, img);
    cv::Mat skel(img.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat temp;
    cv::Mat eroded;

    cv::Mat element =
        cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));

    bool done;
    do {
        cv::erode(img, eroded, element);
        cv::dilate(eroded, temp, element); // temp = open(img)
        cv::subtract(img, temp, temp);
        cv::bitwise_or(skel, temp, skel);
        eroded.copyTo(img);

        done = (cv::countNonZero(img) == 0);
    } while (!done);

    cv::Mat skel_invert;
    bitwise_not(skel, skel_invert);

// cv::Mat downsampled;
// pyrDown(skel_invert, downsampled, Size( img.cols/2, img.rows/2 ));
#ifdef INKPATH_DEBUG
    if (output_dir != "") {
        imwrite(output_dir + "skel.jpg", skel_invert);
        std::cout << "Image has been written to " << output_dir << "\n";
    }
#endif // INKPATH_DEBUG

    return skel_invert;
}

// Grayscale and apply a threshold to the provided image
cv::Mat thresholdWithPrep(cv::Mat img, std::string output_dir) {
    // Convert to grayscale for thresholding
    cv::Mat thresh_input;
    cvtColor(img, thresh_input, cv::COLOR_BGR2GRAY);

    // Upsample our image, if needed.
    // XXX (wdn): Maybe I should normalize the resolution of the images
    // to make scaling the storkes in xournalpp easier.
    int k;
    cv::Mat upsampled;
    if (thresh_input.rows < 1000 || thresh_input.cols < 1000) {
        pyrUp(thresh_input, upsampled,
              cv::Size(thresh_input.cols * 2, thresh_input.rows * 2));
    } else {
        upsampled = thresh_input;
    }

    // Blur improves results
    cv::Mat blur;
    GaussianBlur(upsampled, blur, cv::Size(5, 5), 0, 0);

    // adaptive thresholding after gaussian filtering
    cv::Mat gauss_thresh;

    // https://stackoverflow.com/questions/65891315/opencv-adaptive-thresholding-effective-noise-reduction
    // I was getting a lot of Rice Krispies in the image. This SO post told me
    // to increase C, and that made it mostly acceptable.
    adaptiveThreshold(blur, gauss_thresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                      cv::THRESH_BINARY, 11, 12);

#ifdef INKPATH_DEBUG
    if (output_dir != "") {
        std::string opath = output_dir + "otsu.jpg";
        cv::imwrite(opath, gauss_thresh);
        std::cout << "Image has been written to " << opath << "\n";
    }
#endif // INKPATH_DEBUG

    return gauss_thresh;
}

// Prereqs: Must be binary color image, target must be black
Shapes findStrokes(cv::Mat img, std::string output_dir) {
    // XXX (wdn): Does this bitwise not matter?
    cv::Mat src;
    bitwise_not(img, src);

    cv::Mat dst = cv::Mat::zeros(src.rows, src.cols, CV_8UC3);
    src = src > 1;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    findContours(src, contours, hierarchy, cv::RETR_TREE,
                 cv::CHAIN_APPROX_SIMPLE);

    // Remove strokes that are too small in order to "de-noise" the image
    // a little
    double minArea = 2.0;
    contours.erase(
        std::remove_if(contours.begin(), contours.end(),
                       [minArea](const std::vector<cv::Point>& contour) {
                           return cv::contourArea(contour) < minArea;
                       }),
        contours.end());

#ifdef INKPATH_DEBUG
    if (output_dir != "") {
        // iterate through all the top-level contours,
        // draw each connected component with its own random color
        for (size_t i = 0; i < contours.size(); i++) {
            cv::Scalar color(rand() & 255, rand() & 255,
                             rand() & 255); // Random color
            drawContours(dst, contours, (int)i, color, 2, cv::LINE_8, hierarchy,
                         0);
        }
        std::string opath = output_dir + "strokes.jpg";

        imwrite(opath, dst);
        std::cout << "Image has been written to " << opath << "\n";
    }
#endif

    return Shapes{contours, hierarchy};
}
} // namespace inkp
