#include "ipcv.h"

namespace inkp {

// Extracts a quadrangle out of an image that is supposed to represent a
// whiteboard.
cv::Mat getWhiteboard(cv::Mat image, std::string output_dir) {
    // Locate squares in an image most likely to be the whiteboard
    std::vector<std::vector<cv::Point>> goodSquares =
        filterQuadrangles(image, output_dir);

    // If we didn't find any good squares, then we just have to give up, because
    // whatever squares we did find are either totally wrong or encapsulate
    // the whole image anyway.
    if (goodSquares.size() == 0) {
        std::cout << "Found no good squares :(\n";
        return image;
    }

    // Get the biggest square
    std::vector<cv::Point> bestSquare = goodSquares[goodSquares.size() - 1];

    // Sort the corners
    sortPointsClockwise(bestSquare);

    // Compute the bounding box of the contour
    cv::Rect boundingBox = cv::boundingRect(bestSquare);

    // Clockwise order
    std::vector<cv::Point2f> dstPoints = {
        {0, 0},
        {(float)boundingBox.width - 1, 0},
        {(float)boundingBox.width - 1, (float)boundingBox.height - 1},
        {0, (float)boundingBox.height - 1},
    };

    cv::Mat homography = findHomography(bestSquare, dstPoints, cv::RANSAC);

    // Warp the perspective
    cv::Mat warpedImage;
    cv::warpPerspective(image, warpedImage, homography,
                        cv::Size(boundingBox.width, boundingBox.height));

#ifdef INKPATH_DEBUG
    if (output_dir != "") {
        std::string opath = output_dir + "warped.jpg";
        cv::imwrite(opath, warpedImage);
        std::cout << "Image has been written to " << opath << "\n";
    }
#endif

    return warpedImage;
}

// Filter quadrangles for area
std::vector<std::vector<cv::Point>> filterQuadrangles(cv::Mat image,
                                                      std::string output_dir) {
    // Also check HSV colorspace and try to find more squares.
    // XXX (wdn): We can probably just check HSV. No need to check RGB. Need to
    // do some testing.
    cv::Mat hsv;
    cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    // Copy image to avoid blurring the original image when we run find squares
    cv::Mat bgr = image.clone();

    std::vector<std::vector<cv::Point>> squares;
    findQuadrangles(bgr, squares);
    findQuadrangles(hsv, squares);

    // Sort squares by area
    sort(
        squares.begin(), squares.end(),
        [](const std::vector<cv::Point>& c1, const std::vector<cv::Point>& c2) {
            return cv::contourArea(c1, false) < cv::contourArea(c2, false);
        });

    // Remove squares that are too small, or too big.
    // Aribitrarily, I'm saying if it's <25% or >90% of the size of the image,
    // it's too small or too big to be our target area.
    std::vector<std::vector<cv::Point>> goodSquares;
    for (int i = 0; i < squares.size(); i++) {
        int imgArea = image.rows * image.cols;
        int contourArea = cv::contourArea(squares[i]);
        if ((contourArea < imgArea * 0.25) || (contourArea > imgArea * 0.90)) {
            continue;
        }
        goodSquares.push_back(squares[i]);
    }

#ifdef INKPATH_DEBUG
    std::cout << "Good squares: " << std::to_string(goodSquares.size())
              << ". Bad Squares: " << std::to_string(squares.size()) << "\n";

    if (output_dir != "") {
        // Clone the image to ensure we don't soil the original one
        cv::Mat draw_image = image.clone();
        drawSquares(draw_image, squares, cv::Scalar(0, 0, 255));
        drawSquares(draw_image, goodSquares, cv::Scalar(0, 255, 0));

        std::string opath = output_dir + "squars.jpg";
        cv::imwrite(opath, draw_image);
        std::cout << "Image has been written to " << opath << "\n";
    }
#endif // INKPATH_DEBUG

    return goodSquares;
}

// https://stackoverflow.com/a/8863060/6095682
void findQuadrangles(cv::Mat& image,
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

                    // XXX (wdn): I could roll the filter function into this one
                    if (maxCosine < 0.3)
                        squares.push_back(approx);
                }
            }
        }
    }
}

void sortPointsClockwise(std::vector<cv::Point>& points) {
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

// helper function: finds a cosine of angle between std::vectors from
// pt0->pt1 and from pt0->pt2
// https://docs.opencv.org/4.5.1/db/d00/samples_2cpp_2squares_8cpp-example.html
double angle(cv::Point pt1, cv::Point pt2, cv::Point pt0) {
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1 * dx2 + dy1 * dy2) /
           sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
}

#ifdef INKPATH_DEBUG
// Debug function that visualizes the squares found
void drawSquares(cv::Mat& image,
                 const std::vector<std::vector<cv::Point>>& squares,
                 cv::Scalar color) {
    // Iterate over each square
    for (const auto& square : squares) {
        // Draw the polygon (square) using polylines
        cv::polylines(image, square, true, color, 2, cv::LINE_AA);
    }
}
#endif

} // namespace inkp
