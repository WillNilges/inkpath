#include "ipcv.h"
#include <iostream>


void draw_squares(Mat& image, const vector<vector<Point>>& squares, cv::Scalar color) {
    // Iterate over each square
    for (const auto& square : squares) {
        // Draw the polygon (square) using polylines
        polylines(image, square, true, color, 2, LINE_AA);
    }
}

vector<vector<Point>> locate_quadrangles(cv::Mat image, std::string output_dir)
{
    // Locate quadrangles in the image that are likely to be whiteboards
    vector<vector<Point>> squares;
    find_squares(image, squares);

    // Switch to HSV colorspace and try to find some more squares.
    Mat hsv;
    cvtColor(image, hsv, COLOR_BGR2HSV);
    find_squares(hsv, squares);
    
    imshow("gaming", image);

    // Sort squares by area
    sort(squares.begin(), squares.end(), [](const vector<Point>& c1, const vector<Point>& c2){
        return contourArea(c1, false) < contourArea(c2, false);
    });

    // Remove squares that are too small, or too big.
    // Aribitrarily, I'm saying if it's <25% or >90% of the size of the image,
    // it's too small or too big to be our target area.
    vector<vector<Point>> good_squares;
    for (int i = 0; i < squares.size(); i++) {
        int img_area = image.rows * image.cols;
        int contour_area = contourArea(squares[i]);
        if ((contour_area < img_area * 0.25) || (contour_area > img_area * 0.90))
        {
            continue;
        }
        good_squares.push_back(squares[i]);
    }

    #ifdef INKPATH_DEBUG 
    std::cout << "Good squares: " << std::to_string(good_squares.size()) << ". Bad Squares: " << std::to_string(squares.size()) << "\n";

    if (output_dir != "") {
        draw_squares(image, squares, Scalar(0, 0, 255));
        draw_squares(image, good_squares, Scalar(0, 255, 0));
        std::string opath = output_dir + "squars.jpg";
        imwrite(opath, image);
        std::cout << "Image has been written to " << opath << "\n";
    }
    #endif // INKPATH_DEBUG

    return good_squares;

    // TODO: Debug macros?
    /*
    std::cout << "Good squares: " << std::to_string(good_squares.size()) << ". Bad Squares: " << std::to_string(squares.size()) << "\n";
    draw_squares(image, good_squares);

    std::string opath = path_string + "squar_" + file_title;

    if (output_path != "") {
        imwrite(opath, image);
        std::cout << "Image has been written to " << opath << "\n";
    }
    */
}

// Extracts a quadrangle out of an image that is supposed to represent a whiteboard.
Mat get_whiteboard(Mat image, std::string output_dir)
{
    // Locate squares in an image most likely to be the whiteboard
    vector<vector<Point>> good_squares = locate_quadrangles(image, output_dir);

    // If we didn't find any good squares, then we just have to give up, because
    // whatever squares we did find are either totally wrong or encapsulate
    // the whole image anyway.
    if (good_squares.size() == 0)
    {
        std::cout << "Found no good squares :(\n";
        return image;
    }

    // Get the biggest square 
    vector<Point> best_square = good_squares[good_squares.size() - 1];

    // Sort the corners 
    sort_points_clockwise(best_square);

    // Compute the bounding box of the contour
    cv::Rect boundingBox = cv::boundingRect(best_square);

    // Clockwise order
    std::vector<cv::Point2f> dstPoints = {
        {0, 0},
        {(float) boundingBox.width - 1, 0},
        {(float) boundingBox.width - 1, (float) boundingBox.height - 1},
        {0, (float) boundingBox.height - 1},
    };

    Mat homography = findHomography(best_square, dstPoints, RANSAC);
    
    // Warp the perspective
    cv::Mat warpedImage;
    cv::warpPerspective(image, warpedImage, homography, cv::Size(boundingBox.width, boundingBox.height));

    #ifdef INKPATH_DEBUG
    if (output_dir != "") {
        std::string opath = output_dir + "warped.jpg";
        imwrite(opath, warpedImage);
        std::cout << "Image has been written to " << opath << "\n";
    }
    #endif

    return warpedImage;
}

// helper function: finds a cosine of angle between vectors from 
// pt0->pt1 and from pt0->pt2
double angle( Point pt1, Point pt2, Point pt0 )
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

// https://stackoverflow.com/a/8863060/6095682
void find_squares(Mat& image, vector<vector<Point> >& squares)
{
    // Make a border around the whole image to help with detecting boards who go
    // to the edge of the image
    int border_width = 10;
    int borderType = BORDER_CONSTANT;
    Scalar value(0, 0, 0);
    copyMakeBorder(image, image, border_width, border_width, border_width, border_width, borderType, value );

    // blur will enhance edge detection
    Mat blurred(image);
    medianBlur(image, blurred, 9);

    Mat gray0(blurred.size(), CV_8U), gray;
    vector<vector<Point> > contours;

    // find squares in every color plane of the image
    for (int c = 0; c < 3; c++)
    {
        int ch[] = {c, 0};
        mixChannels(&blurred, 1, &gray0, 1, ch, 1);

        // try several threshold levels
        const int threshold_level = 2;
        for (int l = 0; l < threshold_level; l++)
        {
            // Use Canny instead of zero threshold level!
            // Canny helps to catch squares with gradient shading
            if (l == 0)
            {
                Canny(gray0, gray, 10, 20, 3);

                // Dilate helps to remove potential holes between edge segments
                dilate(gray, gray, Mat(), Point(-1,-1));
            }
            else
            {
                    gray = gray0 >= (l+1) * 255 / threshold_level;
            }

            // Find contours and store them in a list
            findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

            // Test contours
            vector<Point> approx;
            for (size_t i = 0; i < contours.size(); i++)
            {
                    // approximate contour with accuracy proportional
                    // to the contour perimeter
                    approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

                    // Note: absolute value of an area is used because
                    // area may be positive or negative - in accordance with the
                    // contour orientation
                    if (approx.size() == 4 &&
                            fabs(contourArea(Mat(approx))) > 1000 &&
                            isContourConvex(Mat(approx)))
                    {
                            double maxCosine = 0;

                            for (int j = 2; j < 5; j++)
                            {
                                    double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
                                    maxCosine = MAX(maxCosine, cosine);
                            }

                            if (maxCosine < 0.3)
                                    squares.push_back(approx);
                    }
            }
        }
    }
}

// Skeletonization algorithm. I might mess around with this
// more down the road.
// TODO: Where did I find this? 
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
Shapes find_strokes(cv::Mat img, std::string output_dir) {
    // XXX (wdn): Does this bitwise not matter?
    cv::Mat src;
    bitwise_not(img, src);

    cv::Mat dst = cv::Mat::zeros(src.rows, src.cols, CV_8UC3);
    src = src > 1;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(src, contours, hierarchy,
        RETR_TREE, CHAIN_APPROX_SIMPLE );
    
    // Remove strokes that are too small in order to "de-noise" the image 
    // a little
    double minArea = 2.0;
    contours.erase(remove_if(contours.begin(), contours.end(),
        [minArea](const vector<Point>& contour) {
            return contourArea(contour) < minArea;
        }), contours.end());
    
    #ifdef INKPATH_DEBUG
    if (output_dir != "") {
        // iterate through all the top-level contours,
        // draw each connected component with its own random color
        for (size_t i = 0; i < contours.size(); i++) {
            Scalar color( rand()&255, rand()&255, rand()&255 ); // Random color
            drawContours(dst, contours, (int)i, color, 2, LINE_8, hierarchy, 0);
        }   
        std::string opath = output_dir + "strokes.jpg";

        imwrite(opath, dst);
        std::cout << "Image has been written to " << opath << "\n";
    }
    #endif

    return Shapes{contours, hierarchy};
}

void sort_points_clockwise(std::vector<cv::Point>& points) {
    // Step 1: Find the point closest to (0, 0)
    auto closestPoint = std::min_element(points.begin(), points.end(),
        [](const cv::Point& a, const cv::Point& b) {
            return (a.x * a.x + a.y * a.y) < (b.x * b.x + b.y * b.y);
        });

    // Step 2: Find the centroid
    cv::Point center(0, 0);
    for (const auto& point : points) {
        center.x += point.x;
        center.y += point.y;
    }
    center.x /= points.size();
    center.y /= points.size();

    // Step 3: Sort points in clockwise order relative to the centroid
    std::sort(points.begin(), points.end(),
        [&center](const cv::Point& a, const cv::Point& b) {
            double angleA = atan2(a.y - center.y, a.x - center.x);
            double angleB = atan2(b.y - center.y, b.x - center.x);
            return angleA < angleB;
        });
}
