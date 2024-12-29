#include "ipcv.h"

namespace inkp {

void draw_squares(cv::Mat& image,
                  const std::vector<std::vector<cv::Point>>& squares,
                  cv::Scalar color) {
    // Iterate over each square
    for (const auto& square : squares) {
        // Draw the polygon (square) using polylines
        cv::polylines(image, square, true, color, 2, cv::LINE_AA);
    }
}

// Locate quadrangles in the image that are likely to be whiteboards
std::vector<std::vector<cv::Point>> locate_quadrangles(cv::Mat image,
                                                       std::string output_dir) {
    std::vector<std::vector<cv::Point>> squares;

    // Also check HSV colorspace and try to find more squares.
    // XXX (wdn): We can probably just check HSV. No need to check RGB. Need to
    // do some testing.
    cv::Mat hsv;
    cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    // Copy image to avoid blurring the original image when we run find squares
    cv::Mat bgr;
    bgr = image;

    find_squares(bgr, squares);
    find_squares(hsv, squares);

    // Sort squares by area
    sort(
        squares.begin(), squares.end(),
        [](const std::vector<cv::Point>& c1, const std::vector<cv::Point>& c2) {
            return cv::contourArea(c1, false) < cv::contourArea(c2, false);
        });

    // Remove squares that are too small, or too big.
    // Aribitrarily, I'm saying if it's <25% or >90% of the size of the image,
    // it's too small or too big to be our target area.
    std::vector<std::vector<cv::Point>> good_squares;
    for (int i = 0; i < squares.size(); i++) {
        int img_area = image.rows * image.cols;
        int contour_area = cv::contourArea(squares[i]);
        if ((contour_area < img_area * 0.25) ||
            (contour_area > img_area * 0.90)) {
            continue;
        }
        good_squares.push_back(squares[i]);
    }

#ifdef INKPATH_DEBUG
    std::cout << "Good squares: " << std::to_string(good_squares.size())
              << ". Bad Squares: " << std::to_string(squares.size()) << "\n";

    if (output_dir != "") {
        draw_squares(image, squares, Scalar(0, 0, 255));
        draw_squares(image, good_squares, Scalar(0, 255, 0));
        std::string opath = output_dir + "squars.jpg";
        imwrite(opath, image);
        std::cout << "Image has been written to " << opath << "\n";
    }
#endif // INKPATH_DEBUG

    return good_squares;
}

// Extracts a quadrangle out of an image that is supposed to represent a
// whiteboard.
cv::Mat get_whiteboard(cv::Mat image, std::string output_dir) {
    // Locate squares in an image most likely to be the whiteboard
    std::vector<std::vector<cv::Point>> good_squares =
        locate_quadrangles(image, output_dir);

    // If we didn't find any good squares, then we just have to give up, because
    // whatever squares we did find are either totally wrong or encapsulate
    // the whole image anyway.
    if (good_squares.size() == 0) {
        std::cout << "Found no good squares :(\n";
        return image;
    }

    // Get the biggest square
    std::vector<cv::Point> best_square = good_squares[good_squares.size() - 1];

    // Sort the corners
    sort_points_clockwise(best_square);

    // Compute the bounding box of the contour
    cv::Rect boundingBox = cv::boundingRect(best_square);

    // Clockwise order
    std::vector<cv::Point2f> dstPoints = {
        {0, 0},
        {(float)boundingBox.width - 1, 0},
        {(float)boundingBox.width - 1, (float)boundingBox.height - 1},
        {0, (float)boundingBox.height - 1},
    };

    cv::Mat homography = findHomography(best_square, dstPoints, cv::RANSAC);

    // Warp the perspective
    cv::Mat warpedImage;
    cv::warpPerspective(image, warpedImage, homography,
                        cv::Size(boundingBox.width, boundingBox.height));

#ifdef INKPATH_DEBUG
    if (output_dir != "") {
        std::string opath = output_dir + "warped.jpg";
        imwrite(opath, warpedImage);
        std::cout << "Image has been written to " << opath << "\n";
    }
#endif

    return warpedImage;
}

// helper function: finds a cosine of angle between std::vectors from
// pt0->pt1 and from pt0->pt2
double angle(cv::Point pt1, cv::Point pt2, cv::Point pt0) {
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1 * dx2 + dy1 * dy2) /
           sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
}

// https://stackoverflow.com/a/8863060/6095682
void find_squares(cv::Mat& image,
                  std::vector<std::vector<cv::Point>>& squares) {
    // Make a border around the whole image to help with detecting boards who go
    // to the edge of the image
    int border_width = 10;
    int borderType = cv::BORDER_CONSTANT;
    cv::Scalar value(0, 0, 0);
    copyMakeBorder(image, image, border_width, border_width, border_width,
                   border_width, borderType, value);

    // blur will enhance edge detection
    cv::Mat blurred(image);
    medianBlur(image, blurred, 9);

    cv::Mat gray0(blurred.size(), CV_8U), gray;
    std::vector<std::vector<cv::Point>> contours;

    // find squares in every color plane of the image
    for (int c = 0; c < 3; c++) {
        int ch[] = {c, 0};
        mixChannels(&blurred, 1, &gray0, 1, ch, 1);

        // try several threshold levels
        const int threshold_level = 2;
        for (int l = 0; l < threshold_level; l++) {
            // Use Canny instead of zero threshold level!
            // Canny helps to catch squares with gradient shading
            if (l == 0) {
                Canny(gray0, gray, 10, 20, 3);

                // Dilate helps to remove potential holes between edge segments
                dilate(gray, gray, cv::Mat(), cv::Point(-1, -1));
            } else {
                gray = gray0 >= (l + 1) * 255 / threshold_level;
            }

            // Find contours and store them in a list
            findContours(gray, contours, cv::RETR_LIST,
                         cv::CHAIN_APPROX_SIMPLE);

            // Test contours
            std::vector<cv::Point> approx;
            for (size_t i = 0; i < contours.size(); i++) {
                // approximate contour with accuracy proportional
                // to the contour perimeter
                cv::approxPolyDP(
                    cv::Mat(contours[i]), approx,
                    cv::arcLength(cv::Mat(contours[i]), true) * 0.02, true);

                // Note: absolute value of an area is used because
                // area may be positive or negative - in accordance with the
                // contour orientation
                if (approx.size() == 4 &&
                    fabs(cv::contourArea(cv::Mat(approx))) > 1000 &&
                    cv::isContourConvex(cv::Mat(approx))) {
                    double maxCosine = 0;

                    for (int j = 2; j < 5; j++) {
                        double cosine = fabs(
                            angle(approx[j % 4], approx[j - 2], approx[j - 1]));
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
cv::Mat skeletonize(cv::Mat img_inv, std::string output_path) {
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
    if (output_path != "") {
        imwrite(output_path, skel_invert);
        std::cout << "Image has been written to " << output_path << "\n";
    }
    return skel_invert;
}

// Apply an Otsu's thresholding to the object. I found that this was
// the best function of the ones I tried
cv::Mat otsu(cv::Mat img, std::string output_path) {
    int k;
    // Upsample our image, if needed.
    cv::Mat upsampled;
    if (img.rows < 1000 || img.cols < 1000) {
        pyrUp(img, upsampled, cv::Size(img.cols * 2, img.rows * 2));
    } else {
        upsampled = img;
    }

    // otsu's thresholding after gaussian filtering
    cv::Mat gauss_thresh;
    cv::Mat blur;
    GaussianBlur(upsampled, blur, cv::Size(5, 5), 0, 0);
    threshold(blur, gauss_thresh, 0, 255, cv::THRESH_OTSU);

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
            Scalar color(rand() & 255, rand() & 255,
                         rand() & 255); // Random color
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
    auto closestPoint = std::min_element(
        points.begin(), points.end(),
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
} // namespace inkp
