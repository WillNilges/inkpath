#include <getopt.h>
#include <iostream>

class Options {
public:
    enum METHOD {BOTH, CPU, GPU};

    bool verbose = false;
    int iters = 1;
    int artificial_upscale = 0;
    std::string timing = "";
    std::string image_path;
    std::string output_path;
    int device = 0;
    bool threshold_only = false;

    enum METHOD method;

    void print_help();
    Options(int argc, char *argv[]);
};
