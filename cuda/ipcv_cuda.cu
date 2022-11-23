#include "ipcv_cuda.cuh"

#include <stdio.h>

// CUDA imports
#include <cuda_runtime.h>

#define M_COORD(i,j) (i*img.cols) + j 
#define MAXPIXVAL 256

// Go through every pixel in the image, and increment its pixel value in the histogram.
__global__ void kernelCalculateHistogram(const cv::cuda::PtrStepSz<unsigned char> input, unsigned int* deviceHistogram)
{
	int idX = blockDim.x * blockIdx.x + threadIdx.x;
    int idY = blockIdx.y * blockDim.y + threadIdx.y;

    if (idX < input.cols && idY < input.rows) {
        printf("Pixel value: %f\n", float(input(idY, idX)));
//        unsigned char * chom = input(idY, idX); // Y first, X after
//        printf("Pixel value: %c\n", chom[0]);
        //int pixelValue = (int)((double*)((unsigned char*)img + idY * imgStep))[idX];
        //printf("Pixel value: %d\n", pixelValue);
        //atomicAdd(&histogram[pixelValue], 1);
    }
}

// TODO: Port this
__global__ void kernelComputeClassVariances(double* histogram, double allProbabilitySum, long int totalPixels, double* betweenClassVariance)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	double firstClassProbability = 0, secondClassProbability = 0;
	double firstClassMean = 0, secondClassMean = 0;
	double firstProbabilitySum = 0;

	for (int t = 0; t <= id % MAXPIXVAL; t++) {
		firstClassProbability += histogram[t];
		firstProbabilitySum += t * histogram[t];
	}

	secondClassProbability = 1 - firstClassProbability;

	firstClassMean = (double)firstProbabilitySum / (double)firstClassProbability;
	secondClassMean = (double)(allProbabilitySum - firstProbabilitySum) / (double)secondClassProbability;

	betweenClassVariance[id] = firstClassProbability * secondClassProbability * pow((firstClassMean - secondClassMean), 2);
}


// Called from debug.cpp
cv::Mat otsuCuda(cv::Mat img, std::string fullFilePath, cv::cuda::Stream _stream) {
	long totalImagePixels = (long)img.rows*img.cols;

	double* histogram = cudaCalculateHistogram(img, totalImagePixels, _stream);
	cudaDeviceSynchronize();

    for (int i = 0; i < MAXPIXVAL; i++)
    {
        printf("%f", histogram[i]);
    }
    /*	
	unsigned char threshold;
	threshold = cudaFindThreshold(histogram, totalImagePixels);
	cudaDeviceSynchronize();

	delete histogram;

    cv::Mat binarized;

	unsigned char* binarizedRawPixels = cudaBinarize(imageToBinarize->getRawPixelData().data(), totalImagePixels, threshold);
	cudaDeviceSynchronize();*/
    cv::Mat binarized;

    return binarized;

}

/*double* cudaCalculateHistogram(cv::Mat hostImg, long totalPixels)
{
    // Create a blank array, representing a histogram
    unsigned int* hostHistogram = new unsigned int[MAXPIXVAL];
    for (int i = 0; i < MAXPIXVAL; i++) {
		hostHistogram[i] = 0;
	}

    // Copy histogram to device
	unsigned int* deviceHistogram;
	cudaMalloc((void **)&deviceHistogram, sizeof(unsigned int) * MAXPIXVAL);
	cudaMemcpy(deviceHistogram, hostHistogram, sizeof(unsigned int) * MAXPIXVAL, cudaMemcpyHostToDevice);

    // Copy image to device
    cv::cuda::GpuMat deviceImg;
    deviceImg.upload(hostImg);

    // Run the kernel
    printf("Rows = %d, Cols = %d\n", hostImg.rows, hostImg.cols);
    const int TILE_SIZE = 32;
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((int)ceil((float)hostImg.rows / (float)TILE_SIZE), (int)ceil((float)hostImg.cols / (float)TILE_SIZE));
    kernelCalculateHistogram<<<dimGrid, dimBlock>>>(deviceHistogram, (double*) deviceImg.data, deviceImg.step, deviceImg.rows, deviceImg.cols);

    // Copy the histogram back to the host
	cudaMemcpy(hostHistogram, deviceHistogram, sizeof(unsigned int) * MAXPIXVAL, cudaMemcpyDeviceToHost);

    // Free the device Histogram
    cudaFree(deviceHistogram);

    // Normalize the Histogram
	double* normalizedHistogram = new double[MAXPIXVAL];
	long pixelsSum = 0;
	for (int v = 0; v < MAXPIXVAL; v++) {
		normalizedHistogram[v] = (double)hostHistogram[v] / (double)totalPixels;
		pixelsSum += hostHistogram[v];
	}
    return normalizedHistogram;
}
*/

double* cudaCalculateHistogram(
        cv::InputArray _input,
        long totalPixels,
        cv::cuda::Stream _stream
){
    // Create a blank array, representing a histogram
    unsigned int* hostHistogram = new unsigned int[MAXPIXVAL];
    for (int i = 0; i < MAXPIXVAL; i++) {
		hostHistogram[i] = 0;
	}

    // Copy histogram to device
	unsigned int* deviceHistogram;
	cudaMalloc((void **)&deviceHistogram, sizeof(unsigned int) * MAXPIXVAL);
	cudaMemcpy(deviceHistogram, hostHistogram, sizeof(unsigned int) * MAXPIXVAL, cudaMemcpyHostToDevice);

    cv::cuda::GpuMat input;// = _input.getGpuMat();
    input.upload(_input);

    const int TILE_SIZE = 32;
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((int)ceil((float)input.rows / (float)TILE_SIZE), (int)ceil((float)input.cols / (float)TILE_SIZE));

    cudaStream_t stream =
        cv::cuda::StreamAccessor::getStream(_stream);
    kernelCalculateHistogram<<<dimGrid, dimBlock, 0, stream>>>(input, deviceHistogram);

    // TODO: Error handling: cudaSafeCall(cudaGetLastError());
    
    // Copy the histogram back to the host
	cudaMemcpy(hostHistogram, deviceHistogram, sizeof(unsigned int) * MAXPIXVAL, cudaMemcpyDeviceToHost);

    // Free the device Histogram
    cudaFree(deviceHistogram);

    // Normalize the Histogram
	double* normalizedHistogram = new double[MAXPIXVAL];
	long pixelsSum = 0;
	for (int v = 0; v < MAXPIXVAL; v++) {
		normalizedHistogram[v] = (double)hostHistogram[v] / (double)totalPixels;
		pixelsSum += hostHistogram[v];
	}
    return normalizedHistogram;
}
