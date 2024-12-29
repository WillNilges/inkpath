#include <getopt.h>

#include "../ipcv.h"

using namespace inkp;

// A quick way to split strings separated via any character
// delimiter.
std::vector<std::string> adv_tokenizer(std::string s, char del) {
    std::stringstream ss(s);
    std::string word;
    std::vector<std::string> tokens;
    while (!ss.eof()) {
        getline(ss, word, del);
        //        std::cout << word << endl;
        tokens.push_back(word);
    }
    return tokens;
}

void print_help() {
    printf(
        "-f : file : input file\n-o : output : output file\n-h : help : print "
        "help\n");
}

void print_points(Shapes shapes) {
    int i = 0;
    for (std::vector<cv::Point> contour : shapes.contours) {
        std::cout << "CONTOUR #" << i << "\n";
        for (cv::Point point : contour) {
            std::cout << "Point: " << point.x << ", " << point.y << "\n";
        }
        i++;
    }
}

// Test function
int main(int argc, char* argv[]) {
    // CLI arguments and such
    int verbose = 0;
    int order = 0;
    std::string image_path;
    std::string output_path;
    int rc;
    /* getopt_long stores the option index here. */
    int option_index = 0;

    /* This contains the short command line parameters list */
    const char* getoptOptions = "f:o:h"; /* add lots of stuff here */

    /* This contains the long command line parameter list, it should mostly
      match the short list                                                  */
    struct option long_options[] = {/* These options don’t set a flag.
                                        We distinguish them by their indices. */
                                    {"file", required_argument, 0, 'f'},
                                    {"output", required_argument, 0, 'o'},
                                    {"help", 0, 0, 'h'},
                                    {0, 0, 0, 0}};

    opterr = 1; /* Enable automatic error reporting */
    while ((rc = getopt_long_only(argc, argv, getoptOptions, long_options,
                                  &option_index)) != -1) {
        /* Detect the end of the options. */
        switch (rc) {
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
            fprintf(stderr, "Internal error: undefined option %0xX\n", rc);
            exit(1);
        } // End switch
    }     /* end while */

    if ((optind < argc) || image_path.empty() || output_path.empty()) {
        print_help();
        exit(1);
    }

    cv::Mat img = cv::imread(image_path, 0);
    cv::Mat color_img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
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
    for (auto& dir : path_vec) {
        if (!dir.empty()) {
            path_string += dir + '/';
        }
    }
    std::cout << "Using: " << path_string << file_title << "\n";

    // Detect a whiteboard in the image, crop, and straighten
    cv::Mat whiteboard_img = get_whiteboard(color_img, output_path);

    // Convert to grayscale for thresholding
    cv::Mat whiteboard_img_gray;
    cvtColor(whiteboard_img, whiteboard_img_gray, cv::COLOR_BGR2GRAY);

    /*
    cv::Mat hsv;
    cv::cvtColor(warpedImage, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> channels;
    split(hsv, channels);
    cv::Mat H = channels[0];
    cv::Mat S = channels[1];
    cv::Mat V = channels[2];
    cv::imwrite(path_string + "thresh_H" + file_title, H);
    cv::imwrite(path_string + "thresh_S" + file_title, S);
    cv::imwrite(path_string + "thresh_V" + file_title, V);

    // FIXME: can I check maximum contrast in H,S,V and pick that way?
    // what results in good, what results in bad?
    // Sort the channels by max contrast
    sort(channels.begin(), channels.end(), [](const cv::Mat& c1, const cv::Mat& c2){
        // Compute the mean and standard deviation of the grayscale image
        cv::Scalar c1_mean, c1_stddev, c2_mean, c2_stddev;
        cv::meanStdDev(c1, c1_mean, c1_stddev);
        cv::meanStdDev(c2, c2_mean, c2_stddev);
        // Return the standard deviation as the contrast measure
        return c1_stddev[0] < c2_stddev[0];
    });
    // FIXME: we don't always need to invert
    // FIXME: something is flipping the image
    cv::Mat inverted;
    bitwise_not(channels[2], inverted);
    */

    // Run stroke detection algorithms
    cv::Mat otsu_img =
        otsu(whiteboard_img_gray, /*path_string + "otsu_" + file_title*/ "");
    cv::Mat skel_img =
        skeletonize(otsu_img, /*path_string + "skel_" + file_title*/ "");
    Shapes shapes = find_strokes(skel_img, path_string);

    // print_points(shapes);

    return 0;
}
