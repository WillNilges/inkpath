#include <getopt.h>
#include "../ipcv.h"

// A quick way to split strings separated via any character
// delimiter.
std::vector<std::string> adv_tokenizer(std::string s, char del)
{
    std::stringstream ss(s);
    std::string word;
    std::vector<std::string> tokens;
    while (!ss.eof()) {
        getline(ss, word, del);
//        cout << word << endl;
        tokens.push_back(word);
    }
    return tokens;
}

void print_help()
{
    printf("-f : file : input file\n-o : output : output file\n-h : help : print help\n");
}

void print_points(Shapes shapes)
{
    int i = 0;
    for (vector<Point> contour : shapes.contours)
    {
        cout << "CONTOUR #" << i << "\n";
        for (Point point : contour)
        {
            cout << "Point: " << point.x << ", " << point.y << "\n";
        }
        i++;
    }
}


void drawSquares(Mat& image, const vector<vector<Point>>& squares) {
    // Iterate over each square
    for (const auto& square : squares) {
        // Draw the polygon (square) using polylines
        polylines(image, square, true, Scalar(0, 255, 0), 2, LINE_AA);
    }
}

// Test function
int main(int argc, char *argv[])
{
    // CLI arguments and such
    int verbose = 0;
    int order = 0;
    std::string image_path;
    std::string output_path;
    int rc;
    /* getopt_long stores the option index here. */
    int option_index = 0;
    
    /* This contains the short command line parameters list */
    const char* getoptOptions = "f:o:h";    /* add lots of stuff here */
    
    /* This contains the long command line parameter list, it should mostly 
      match the short list                                                  */
    struct option long_options[] = {
         /* These options don’t set a flag.
             We distinguish them by their indices. */
         {"file",   required_argument, 0, 'f'},
         {"output", required_argument, 0, 'o'},
         {"help", 0, 0, 'h'},
         {0, 0, 0, 0}
    };
    
    opterr = 1;           /* Enable automatic error reporting */
    while ((rc = getopt_long_only(
                    argc,
                    argv,
                    getoptOptions,
                    long_options,
                    &option_index)) != -1) {
       /* Detect the end of the options. */
       switch (rc)
         {
         case 'f':
             image_path = optarg;
             break;
         case 'o':
             output_path = optarg;
             break;
         case 'h':
             print_help();
             break;
        default:
           fprintf (stderr, "Internal error: undefined option %0xX\n", rc);
           exit(1);
        } // End switch 
    } /* end while */

    if ((optind < argc) || image_path.empty() || output_path.empty()){
        print_help();
        exit(1);
    }

    Mat img = imread(image_path, 0);
    Mat color_img = imread(image_path, IMREAD_COLOR);
    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    // Separate the file title from the rest of the path
    std::vector<std::string> path_vec = adv_tokenizer(output_path, '/');
    std::string file_title = path_vec.back();
    path_vec.pop_back();
    std::string path_string;
    if (output_path[0] == '/')
        path_string += '/';
    for (auto &dir : path_vec)
    {
        if (!dir.empty()) {
            path_string += dir + '/';
        }
    }
    std::cout << "Using: " << path_string << file_title << "\n";

    Mat squar_input;
    squar_input = color_img;

    vector<vector<Point>> squares;
    find_squares(squar_input, squares, path_string, file_title);

    // Switch to HSV colorspace and try to find some more squares.
    Mat hsv;
    cvtColor(color_img,hsv,COLOR_BGR2HSV);
    find_squares(hsv, squares, path_string, file_title);

    /*
    for (int i = 0; i < squares.size(); i++) {
        std::cout << squares[i] << "\n";
    }
    */

    // Sort squares by area
    sort(squares.begin(), squares.end(), [](const vector<Point>& c1, const vector<Point>& c2){
        return contourArea(c1, false) < contourArea(c2, false);
    });

    // Remove contours that are too small, or too big.
    // Aribitrarily, I'm saying if it's <25% or >90% of the size of the image,
    // it's too small or too big to be our target area.
    vector<vector<Point>> good_squares;
    for (int i = 0; i < squares.size(); i++) {
        int img_area = color_img.rows * color_img.cols;
        int contour_area = contourArea(squares[i]);
        if (contour_area < img_area * 0.25) {
            continue;
        }
        if (contour_area > img_area * 0.90) {
            continue;
        }
        //std::cout << img_area << "\n";
        //std::cout << contour_area << "\n";
        good_squares.push_back(squares[i]);
    }
    std::cout << "Good squares: " << to_string(good_squares.size()) << ". Bad Squares: " << to_string(squares.size()) << "\n";
    squares = good_squares;

    drawSquares(squar_input, good_squares);

    std::string opath = path_string + "squar_" + file_title;

    if (opath != "") {
        imwrite(opath, squar_input);
        std::cout << "Image has been written to " << opath << "\n";
    }

    Mat inkpath_input;
    if (good_squares.size() > 0)
    {
        // Find the biggest contour with no contours inside it?

        // TODO: If biggest is larger than total image area minus border, then we should
        // try the second biggest image area
        
        // But for now, just get the 2nd biggest one (can't think it's 22:17)
        // FIXME: This will crash if there are no squares found
        // FIXME: There's probably a big where we're using color_img here instead of squar_input
        
        vector<Point> best_square = squares[squares.size() - 1];
        /*
        for (int i = squares.size() - 1; i > 0; i--) {
            if (contourArea(squares[i]) < (color_img.rows * color_img.cols * 0.95)) {
                best_square = squares[i];
                break;
            }
        }
        */

        // Sort the corners clockwise
        /*
        sort(best_square.begin(), best_square.end(), [](const Point& p1, const Point& p2) {
            return p1.x < p1.x;
        });
        sort(best_square.begin(), best_square.end(), [](const Point& p1, const Point& p2) {
            return p1.y < p1.y;
        });
        */

        sortPointsClockwise(best_square);
        
        // Compute the bounding box of the contour
        cv::Rect boundingBox = cv::boundingRect(best_square);
        for (int i = 0; i < best_square.size(); i++)
        {
             std::cout << best_square[i] << "\n";
        }

        std::vector<cv::Point2f> dstPoints = {
            {0, 0},
            {(float) boundingBox.width - 1, 0},
            {(float) boundingBox.width - 1, (float) boundingBox.height - 1},
            {0, (float) boundingBox.height - 1},
        };

        Mat H = findHomography(best_square, dstPoints, RANSAC);
        
        // Warp the perspective
        cv::Mat warpedImage;
        cv::warpPerspective(img, warpedImage, H, cv::Size(boundingBox.width, boundingBox.height));

        opath = path_string + "warped_" + file_title;
        if (opath != "") {
            imwrite(opath, warpedImage);
            std::cout << "Image has been written to " << opath << "\n";
        }

        inkpath_input = warpedImage;
    } else {
        std::cout << "Found no good squares :(";
        inkpath_input = img;
    }

    Mat otsu_img = otsu(inkpath_input, path_string + "otsu_" + file_title);
    Mat skel_img = skeletonize(otsu_img, path_string + "skel_" + file_title);
    Shapes shapes = find_shapes(skel_img, path_string + "shape_" + file_title);

    //print_points(shapes);


    return 0;
}

