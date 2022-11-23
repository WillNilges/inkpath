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

#include "CudaOtsuBinarizer.cuh"

#include <stdio.h>

// CUDA imports
#include <cuda_runtime.h>

// Can I not use macros in kernels?
__device__ unsigned int maxPixVal = MAXPIXVAL;

__global__ void kernelCalculateHistogram(unsigned int* histogram, cv::cuda::GpuMat rawPixels, long chunkSize, long totalPixels)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	int startPosition = id * chunkSize;
	for (int i = startPosition; i < (startPosition + chunkSize); i++) {
		if (i < totalPixels) {
			int pixelValue = (int)rawPixels[i];
			atomicAdd(&histogram[pixelValue], 1);
		}
	}
}

__global__ void kernelComputeClassVariances(double* histogram, double allProbabilitySum, long int totalPixels, double* betweenClassVariance)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	double firstClassProbability = 0, secondClassProbability = 0;
	double firstClassMean = 0, secondClassMean = 0;
	double firstProbabilitySum = 0;

	for (int t = 0; t <= id % maxPixVal; t++) {
		firstClassProbability += histogram[t];
		firstProbabilitySum += t * histogram[t];
	}

	secondClassProbability = 1 - firstClassProbability;

	firstClassMean = (double)firstProbabilitySum / (double)firstClassProbability;
	secondClassMean = (double)(allProbabilitySum - firstProbabilitySum) / (double)secondClassProbability;

	betweenClassVariance[id] = firstClassProbability * secondClassProbability * pow((firstClassMean - secondClassMean), 2);
}

// TODO: I think this can be replaced with a simple opencv threshold function.
__global__ void kernelBinarize(cv::cuda::GpuMat rawPixels, long totalPixels, long chunkSize, unsigned char threshold)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	int startPosition = id * chunkSize;
	for (int i = startPosition; i < (startPosition + chunkSize); i++) {
		if (i < totalPixels) {
			if ((int)rawPixels[i] >(int)threshold) {
				rawPixels[i] = 255;
			}
			else {
				rawPixels[i] = 0;
			}
		}
	}
}

// Called from debug.cpp
cv::Mat otsuCuda(std::string fullFilePath, cv::Mat img, int threadsPerBlock, int numBlocks, bool drawHistograms) {

    cv::cuda::GpuMat loadedImage;
    loadedImage.upload(img);
	CudaOtsuBinarizer* cudaBinarizer = new CudaOtsuBinarizer(threadsPerBlock, numBlocks, drawHistograms);

	cv::cuda::GpuMat gpuBinarizedImage = cudaBinarizer->binarize(loadedImage);

    // TODO: Save image

	//std::string csvTimesLog = cudaBinarizer->getBinarizerExecutionInfo(fullFilePath);
	//std::string configLog = getConfigurationInfo(threadsPerBlock, numBlocks);

	delete cudaBinarizer;

    cv::Mat binarizedImage;
    gpuBinarizedImage.download(binarizedImage);
    return binarizedImage;
}

// Constructor + Destructor
CudaOtsuBinarizer::CudaOtsuBinarizer(int threadsPerBlock, int numBlocks, bool drawHistogram, const char* TAG) {
	this->threadsPerBlock_ = threadsPerBlock;
	this->numBlocks_ = numBlocks;

	this->drawHistogram_ = drawHistogram;
	this->TAG = TAG;
}

CudaOtsuBinarizer::~CudaOtsuBinarizer() {
}

cv::cuda::GpuMat CudaOtsuBinarizer::binarize(cv::cuda::GpuMat imageToBinarize)
{
	long totalImagePixels = (long) imageToBinarize.rows * imageToBinarize.cols;

	double* histogram = cudaCalculateHistogram(imageToBinarize, totalImagePixels);
	cudaDeviceSynchronize();
	
	if (this->drawHistogram_) {
		showHistogram(histogram);
	}

	unsigned char threshold;
	threshold = cudaFindThreshold(histogram, totalImagePixels);
	cudaDeviceSynchronize();
	printf("\t[%s] Threshold value: %d\n", this->TAG, threshold);

	delete histogram;

	unsigned char* binarizedRawPixels = cudaBinarize(imageToBinarize, totalImagePixels, threshold);
	cudaDeviceSynchronize();

	std::vector<unsigned char> binarizedVector(&binarizedRawPixels[0], &binarizedRawPixels[totalImagePixels]);

	delete binarizedRawPixels;


	return new PngImage(
		imageToBinarize->getFilename(),
		imageToBinarize->getWidth(),
		imageToBinarize->getHeight(),
		binarizedVector
	);
}

void CudaOtsuBinarizer::showHistogram(double* histogram) {
	printf("\nHistogram:\n");
	double value = 0;
	for (int i = 0; i < maxPixVal; i++) {
		value = histogram[i];
		printf("\tPixel value %d -> %.5f\n", i, value);
	}
}

double* CudaOtsuBinarizer::cudaCalculateHistogram(cv::cuda::GpuMat rawPixels, long totalPixels) {
	//TODO: check cudaGetDeviceProperties function!

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	unsigned int* hostHistogram = new unsigned int[maxVal];
	for (int i = 0; i < maxVal; i++) {
		hostHistogram[i] = 0;
	}

	unsigned int* deviceHistogram;
	cudaMalloc((void **)&deviceHistogram, sizeof(unsigned int) * maxPixVal);
	cudaMemcpy(deviceHistogram, hostHistogram, sizeof(unsigned int) * maxPixVal, cudaMemcpyHostToDevice);

	unsigned char* deviceRawPixels;
	cudaMalloc((void **)&deviceRawPixels, sizeof(unsigned char) * totalPixels);
	cudaMemcpy(deviceRawPixels, rawPixels, sizeof(unsigned char) * totalPixels, cudaMemcpyHostToDevice);

	long chunkSize = ceil(totalPixels / (threadsPerBlock_ * numBlocks_)) + 1;

	cudaEventRecord(start);
	kernelCalculateHistogram<<<numBlocks_, threadsPerBlock_>>>(deviceHistogram, deviceRawPixels, chunkSize, totalPixels);
	cudaEventRecord(stop);

	cudaMemcpy(hostHistogram, deviceHistogram, sizeof(unsigned int) * maxPixVal, cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
    // TODO: Delet
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("\n\t[%s] Histogram calculated in %.6f milliseconds \n", this->TAG, milliseconds);
	binarizerTimestamp_->histogramBuildingTime += milliseconds;
    // /Delet

	cudaFree(deviceHistogram);
	cudaFree(deviceRawPixels);

	double* normalizedHistogram = new double[maxPixVal];
	long pixelsSum = 0;
	for (int v = 0; v < maxPixVal; v++) {
		normalizedHistogram[v] = (double)hostHistogram[v] / (double)totalPixels;
		pixelsSum += hostHistogram[v];
	}
	printf("\n\t[%s] Histogram pixels: %d \n", this->TAG, pixelsSum);

	delete hostHistogram;

	return normalizedHistogram;
}

unsigned char CudaOtsuBinarizer::cudaFindThreshold(double* histogram, long int totalPixels) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	int threadsPerBlock = 256;
	int numBlocks = 1;

	double allProbabilitySum = 0;
	for (int i = 0; i < maxPixVal; i++) {
		allProbabilitySum += i * histogram[i];
	}

	double* hostBetweenClassVariances = new double[maxPixVal];
	for (int i = 0; i < maxPixVal; i++) {
		hostBetweenClassVariances[i] = 0;
	}

	double* deviceHistogram;
	cudaMalloc((void **)&deviceHistogram, sizeof(double) * maxPixVal);
	cudaMemcpy(deviceHistogram, histogram, sizeof(double) * maxPixVal, cudaMemcpyHostToDevice);

	double* deviceBetweenClassVariances;
	cudaMalloc((void **)&deviceBetweenClassVariances, sizeof(double) * maxPixVal);
	cudaMemcpy(deviceBetweenClassVariances, hostBetweenClassVariances, sizeof(double) * maxPixVal, cudaMemcpyHostToDevice);

	cudaEventRecord(start);
	kernelComputeClassVariances<<<numBlocks, threadsPerBlock>>>(deviceHistogram, allProbabilitySum, totalPixels, deviceBetweenClassVariances);
	cudaEventRecord(stop);
	cudaMemcpy(hostBetweenClassVariances, deviceBetweenClassVariances, sizeof(double) * maxPixVal, cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("\n\t[%s] Threshold calculated in %.6f milliseconds \n", this->TAG, milliseconds);
	binarizerTimestamp_->thresholdFindingTime += milliseconds;

	cudaFree(deviceHistogram);
	cudaFree(deviceBetweenClassVariances);

	double maxVariance = 0;
	unsigned char currentBestThreshold = 0;
	for (int t = 0; t < maxPixVal; t++) {
		if (hostBetweenClassVariances[t] > maxVariance) {
			currentBestThreshold = (unsigned char)t;
			maxVariance = hostBetweenClassVariances[t];
		}
	}

	delete hostBetweenClassVariances;

	return currentBestThreshold;
}

unsigned char* CudaOtsuBinarizer::cudaBinarize(unsigned char * rawPixels, long totalPixels, unsigned char threshold) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	unsigned char* hostRawPixels = new unsigned char[totalPixels];

	unsigned char* deviceRawPixels;
	cudaMalloc((void **)&deviceRawPixels, sizeof(unsigned char) * totalPixels);
	cudaMemcpy(deviceRawPixels, rawPixels, totalPixels * sizeof(unsigned char), cudaMemcpyHostToDevice);

	long chunkSize = ceil(totalPixels / (threadsPerBlock_ * numBlocks_)) + 1;

	cudaEventRecord(start);
	kernelBinarize<<<numBlocks_, threadsPerBlock_>>>(deviceRawPixels, totalPixels, chunkSize, threshold);
	cudaEventRecord(stop);

	cudaMemcpy(hostRawPixels, deviceRawPixels, sizeof(unsigned char) * totalPixels, cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("\n\t[%s] Binarized in %.6f milliseconds \n", this->TAG, milliseconds);
	binarizerTimestamp_->binarizationTime += milliseconds;

	cudaFree(deviceRawPixels);

	return hostRawPixels;
}
