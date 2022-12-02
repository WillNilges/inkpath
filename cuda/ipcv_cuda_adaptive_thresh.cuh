#ifndef IPCV_CUDA_ADAPTIVE_THRESH
#define IPCV_CUDA_ADAPTIVE_THRESH

// Standard OpenCV Stuff
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/types_c.h>

// OpenCV CUDA API
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafilters.hpp>
//#include <opencv2/cudev/expr/reduction.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

using namespace cv;

cv::Mat adaptiveCuda(
    cv::Mat img,
    std::string output_path,
    double maxValue,
    int blockSize,
    double delta,
    cv::cuda::Stream _stream
);

void cudaAdaptiveThreshold(
    InputArray _src,
    OutputArray _dst,
    double maxValue,
    int method,
    int type,
    int blockSize,
    double delta,
    cv::cuda::Stream _stream
);
#endif
