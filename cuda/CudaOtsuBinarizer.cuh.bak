/*
MIT License

Copyright (c) 2020 Dawid Paluchowski

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
 */

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/cuda.hpp>

#pragma once
// I'm pretty sure 255 is the highest grayscale value we'll see.
#define MAXPIXVAL 255;

cv::Mat otsuCuda(std::string fullFilePath, cv::Mat loadedImage, int threadsPerBlock, int numBlocks, bool drawHistograms);
class CudaOtsuBinarizer
{
public:
    cv::cuda::GpuMat binarize(cv::cuda::GpuMat imageToBinarize);
	CudaOtsuBinarizer(int threadsPerBlock, int numBlocks, bool drawHistogram, const char* TAG = "GPU");
	virtual ~CudaOtsuBinarizer();
protected:
	int threadsPerBlock_;
	int numBlocks_;
	bool drawHistogram_;
	const char* TAG;
	virtual void showHistogram(double* histogram);
	virtual double* CudaOtsuBinarizer::cudaCalculateHistogram(cv::cuda::GpuMat rawPixels, long totalPixels);
	virtual unsigned char cudaFindThreshold(double* histogram, long int totalPixels);
	virtual unsigned char* cudaBinarize(cv::cuda::GpuMat rawPixels, long totalPixels, unsigned char threshold);
};

