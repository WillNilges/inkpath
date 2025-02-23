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
    cv::Mat whiteboard_img = getWhiteboard(color_img, output_path);

    // Do it again --- this should solve the "projector problem" where you have
    // content on a projector framed by the border of the photo. Nominally this
    // should find no more squares.
    cv::Mat whiteboard_img_2 = getWhiteboard(whiteboard_img, output_path);

    // Run stroke detection algorithms
    cv::Mat otsu_img = thresholdWithPrep(whiteboard_img_2, path_string);
    cv::Mat skel_img = skeletonize(otsu_img, path_string);
    std::vector<std::vector<cv::Point>> shapes =
        findStrokes(skel_img, path_string);

    // print_points(shapes);

    return 0;
}
