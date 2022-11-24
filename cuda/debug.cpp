#include <getopt.h>
#include "ipcv.h"
#include "ipcv_gpu.h"
#include "ipcv_cuda.cuh"
#include "ipcv_cuda_adaptive_thresh.cuh"

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
    printf("-f : file : input file\n-o : output : output file\n-i : number of iterations to run (CANNOT BE MORE THAN 1 WHILE USING OUTPUT FILE)\n-h : help : print help\n");
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

void do_cpu(Mat img, std::string path_string, std::string file_title, bool verbose, bool use_adaptive)
{
    // Run on CPU
    std::string otsu_out, skel_out, shape_out;
    if (!path_string.empty() || !file_title.empty())
    {
        otsu_out = path_string + "otsu_" + file_title;
        skel_out = path_string + "skel_" + file_title;
        shape_out = path_string + "shape_" + file_title;
    }
    Mat otsu_img;
    if (use_adaptive)
    {
        otsu_out = path_string + "adaptive_" + file_title;
        otsu_img = adaptive(img, otsu_out);
    }
    else
        otsu_img = otsu(img, otsu_out);
    //Mat skel_img = skeletonize(otsu_img, skel_out);
    //Shapes shapes = find_shapes(skel_img, shape_out);

    //if (verbose)
    //    print_points(shapes);
}

void do_gpu(Mat img, std::string path_string, std::string file_title, bool verbose, cv::cuda::Stream stream1, bool use_adaptive)
{
    // Run on GPU
    std::string otsu_out, skel_out, shape_out;
    if (!path_string.empty() || !file_title.empty())
    {
        otsu_out = path_string + "gpu_otsu_" + file_title;
        skel_out = path_string + "gpu_skel_" + file_title;
        shape_out = path_string + "gpu_shape_" + file_title;
    }
    Mat gpu_otsu_img;
    if (use_adaptive)
    {
        otsu_out = path_string + "gpu_adaptive_" + file_title;
        gpu_otsu_img = adaptiveCuda(img, otsu_out, stream1);
    }
    else
        gpu_otsu_img = otsuCuda(img, otsu_out, stream1);
//    Mat gpu_otsu_img = gpu_otsu(img, otsu_out, stream1);
    //Mat gpu_skel_img = gpu_skeletonize(gpu_otsu_img, skel_out, stream1);
    //Shapes gpu_shapes = gpu_find_shapes(gpu_skel_img, shape_out);

    //if (verbose)
    //    print_points(gpu_shapes);
}

int main(int argc, char *argv[])
{
    // CLI arguments and such
    bool verbose = false;
    int order = 0;
    int iters = 1;
    int artificial_upscale = 0;
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
         {"iterations", required_argument, 0, 'i'},
         {"upscale", required_argument, 0, 'u'},
         {"verbose", 0, 0, 'v'},
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
         case 'i':
             iters = atoi(optarg);
             break;
         case 'u':
             artificial_upscale = atoi(optarg);
             break;
         case 'v':
             verbose = true;
             break;
         case 'h':
             print_help();
             break;
        default:
           fprintf (stderr, "Internal error: undefined option %0xX\n", rc);
           exit(1);
        } // End switch 
    } /* end while */

    if ((optind < argc) || image_path.empty() || (iters > 1 && !output_path.empty())){
        print_help();
        exit(1);
    }

    Mat img = imread(image_path, 0);
    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    for (int i = 0; i < artificial_upscale; i++)
    {
        std::cout << "Upscaling image pre-test x" << i+1 << "\n";
        cv::cuda::GpuMat gpu_img;
        cv::cuda::Stream stream1;
        gpu_img.upload(img);
        cv::cuda::pyrUp(gpu_img, gpu_img, stream1);
        gpu_img.download(img);
    }

    // Separate the file title from the rest of the path
    // FIXME: This feels awful, but I don't know enough about C++ to be sure.
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
    if (!path_string.empty())
        std::cout << "Using: " << path_string << file_title << "\n";
    else
        std::cout << "No output file specified.\n";

    // Timing data
    float tcpu, tcpu_adaptive, tgpu, tgpu_adaptive;
    clock_t start, end;

    std::cout << "Starting CPU...\n";

    start = clock();
    for (int i = 0; i < iters; i++)
        do_cpu(img, path_string, file_title, verbose, false);
    end = clock();
    tcpu = (float)(end - start) * 1001 / (float)CLOCKS_PER_SEC / iters;
    
    std::cout << "CPU took " << tcpu << " ms" << std::endl;

    std::cout << "Starting CPU (Adaptive)...\n";

    start = clock();
    for (int i = 0; i < iters; i++)
        do_cpu(img, path_string, file_title, verbose, true);
    end = clock();
    tcpu_adaptive = (float)(end - start) * 1001 / (float)CLOCKS_PER_SEC / iters;
    
    std::cout << "CPU (Adaptive) took " << tcpu_adaptive << " ms" << std::endl;



    cv::cuda::Stream stream1;

    // Warm-Up run
    std::cout << "Warming up GPU...\n";
    do_gpu(img, path_string, file_title, verbose, stream1, false);

    std::cout << "Starting GPU...\n";
    start = clock();
    for (int i = 0; i < iters; i++)
        do_gpu(img, path_string, file_title, verbose, stream1, false);
    end = clock();
    tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iters;

    std::cout << "GPU took " << tgpu << " ms" << std::endl;


    std::cout << "Starting GPU (Adaptive)...\n";
    start = clock();
    for (int i = 0; i < iters; i++)
        do_gpu(img, path_string, file_title, verbose, stream1, true);
    end = clock();
    tgpu_adaptive = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iters;

    std::cout << "GPU (adaptive) took " << tgpu_adaptive << " ms" << std::endl;

    // Compare times
    std::cout << "Speedup: " << tcpu/tgpu << "\n";

    return 0;
}

