#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

cv::Mat otsuCuda(cv::Mat img, std::string fullFilePath, cv::cuda::Stream _stream);
double* cudaCalculateHistogram(
        cv::InputArray _input,
        long totalPixels,
        cv::cuda::Stream _stream
);
