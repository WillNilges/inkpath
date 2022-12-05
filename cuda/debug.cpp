#include <getopt.h>
#include <fstream>

#include "ipcv.h"
#include "ipcv_gpu.h"
#include "ipcv_cuda.cuh"
#include "ipcv_cuda_adaptive_thresh.cuh"
#include "arg.h"

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

// Run the whole algorithm

void cpu_complete(Mat img, std::string path_string, std::string file_title, bool verbose, bool use_adaptive, bool threshold_only)
{
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
        if (!path_string.empty() || !file_title.empty())
            otsu_out = path_string + "adaptive_" + file_title;
        otsu_img = adaptive(img, otsu_out);
    }
    else
        otsu_img = otsu(img, otsu_out);

    if (threshold_only)
        return;

    Mat skel_img = skeletonize(otsu_img, skel_out);
    Shapes shapes = find_shapes(skel_img, shape_out);

    //if (verbose)
    //    print_points(shapes);
}

void gpu_complete(Mat img, std::string path_string, std::string file_title, bool verbose, cv::cuda::Stream stream1, bool use_adaptive, bool threshold_only, Options args)
{
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
        if (!path_string.empty() || !file_title.empty())
            otsu_out = path_string + "gpu_adaptive_" + file_title;
        gpu_otsu_img = adaptiveCuda(img, otsu_out, 255, args.block_size, 2, stream1);
    }
    else
        gpu_otsu_img = otsuCuda(img, otsu_out, stream1);

    if (threshold_only)
        return;

    Mat gpu_skel_img = gpu_skeletonize(gpu_otsu_img, skel_out, stream1);
    Shapes gpu_shapes = gpu_find_shapes(gpu_skel_img, shape_out);

    //if (verbose)
    //    print_points(gpu_shapes);
}

int main(int argc, char* argv[])
{
    Options args(argc, argv);

    Mat img = imread(args.image_path, 0);
    if(img.empty())
    {
        std::cout << "Could not read the image: " << args.image_path << std::endl;
        return 1;
    }

    // Optionally, write the results to a file.
    std::ifstream data_check;
    std::ofstream data;
    if (!args.timing.empty())
    {
        std::cout << "Writing results to " << args.timing << ".\n";
        // Set up .csv file
        bool data_empty = false;
        data_check.open(args.timing);
        if(data_check.peek() == std::ifstream::traits_type::eof())
            data_empty = true;
        data_check.close();

        data.open(args.timing, std::ios_base::app);
        if (data_empty)
        {
            data << "device,filename,upscale_amt,time_cpu_otsu,time_cpu_adaptive,time_gpu_otsu,time_gpu_adaptive,speedup_otsu,speedup_adaptive\n";
            std::cout << "File empty. File has been initialized.\n";
        }
    }
    
    // Select device to use
    cudaError_t status;
    status = cudaSetDevice(args.device);
    if (status != cudaSuccess)
    {
        std::cout << "CUDA error: " << cudaGetErrorString(status) <<
        std::endl;
    }
    std::cout << "Using device " << args.device << ".\n";

    // Upscale the image if needed
    for (int i = 0; i < args.artificial_upscale; i++)
    {
        std::cout << "Upscaling image pre-test x" << i+1 << "\n";
        cv::cuda::GpuMat gpu_img;
        cv::cuda::Stream stream1;
        gpu_img.upload(img);
        cv::cuda::pyrUp(gpu_img, gpu_img, stream1);
        gpu_img.download(img);
    }

    // Separate the file title from the rest of the path
    std::vector<std::string> path_vec = adv_tokenizer(args.output_path, '/');
    std::string file_title = path_vec.back();
    path_vec.pop_back();
    std::string path_string;
    if (args.output_path[0] == '/')
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
    for (int i = 0; i < args.iters; i++)
        cpu_complete(img, path_string, file_title, args.verbose, false, args.threshold_only);
    end = clock();
    tcpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / args.iters;
    
    std::cout << "CPU took " << tcpu << " ms" << std::endl;

    std::cout << "Starting CPU (Adaptive)...\n";

    start = clock();
    for (int i = 0; i < args.iters; i++)
        cpu_complete(img, path_string, file_title, args.verbose, true, args.threshold_only);
    end = clock();
    tcpu_adaptive = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / args.iters;
    
    std::cout << "CPU (Adaptive) took " << tcpu_adaptive << " ms" << std::endl;

    cv::cuda::Stream stream1;

    // Warm-Up run
    std::cout << "Warming up GPU...\n";

    gpu_complete(img, path_string, file_title, args.verbose, stream1, false, args.threshold_only, args);
    gpu_complete(img, path_string, file_title, args.verbose, stream1, true, args.threshold_only, args);

    std::cout << "Starting GPU...\n";
    start = clock();
    for (int i = 0; i < args.iters; i++)
        gpu_complete(img, path_string, file_title, args.verbose, stream1, false, args.threshold_only, args);
    end = clock();
    tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / args.iters;

    std::cout << "GPU took " << tgpu << " ms" << std::endl;

    std::cout << "Starting GPU (Adaptive)...\n";
    start = clock();
    for (int i = 0; i < args.iters; i++)
        gpu_complete(img, path_string, file_title, args.verbose, stream1, true, args.threshold_only, args);
    end = clock();
    tgpu_adaptive = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / args.iters;

    std::cout << "GPU (adaptive) took " << tgpu_adaptive << " ms" << std::endl;

    // Compare times
    std::cout << "Speedup: " << tcpu/tgpu << "\n";
    std::cout << "Speedup (Adaptive): " << tcpu_adaptive/tgpu_adaptive << "\n";

    if (!args.timing.empty())
    {
        std::vector<std::string> input_path_vec = adv_tokenizer(args.image_path, '/');
        std::string input_title = input_path_vec.back();
        struct cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, args.device);

        // Check for device error
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Device error: %s\n", cudaGetErrorString(err));
            return 1;
        }

        data <<  deviceProp.name << "," << input_title << "," << args.artificial_upscale << "," << tcpu << "," << tcpu_adaptive  << "," 
            << tgpu << "," << tgpu_adaptive << "," << tcpu/tgpu 
            << "," << tcpu_adaptive/tgpu_adaptive << ","  << std::endl;
        std::cout << "Results have been saved." << std::endl;
    }

    return 0;
}

