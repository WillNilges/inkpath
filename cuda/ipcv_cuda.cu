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
    int idY = blockDim.y * blockIdx.y + threadIdx.y;

    if (idX < input.cols && idY < input.rows) {
        int pixelValue = int(input(idY, idX));
        atomicAdd(&deviceHistogram[pixelValue], 1);
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
cv::Mat otsuCuda(cv::Mat img, std::string output_path, cv::cuda::Stream _stream) {
	long totalImagePixels = (long)img.total();

	double* histogram = cudaCalculateHistogram(img, totalImagePixels, _stream);
	cudaDeviceSynchronize();

    /*
    for (int i = 0; i < MAXPIXVAL; i++)
    {
        printf("%f ", histogram[i]);
    }
    printf("\n");*/

	unsigned char threshold;
	threshold = cudaFindThreshold(histogram, totalImagePixels, _stream);
	cudaDeviceSynchronize();

    // TODO: Compare OpenCV CPU otsu threshold value with mine
    //printf("Threshold is: %d\n", threshold);

    cv::Mat hostBinarized;
    cv::cuda::GpuMat deviceBinarized;
    deviceBinarized.upload(img);
    cv::cuda::threshold(deviceBinarized, deviceBinarized, (double) threshold, MAXPIXVAL-1, cv::THRESH_BINARY, _stream);
    deviceBinarized.download(hostBinarized);

    if (!output_path.empty()) {
        imwrite(output_path, hostBinarized);
#ifdef DIAG
        std::cout << "Image has been written to " << output_path << "\n";
#endif
    }

    return hostBinarized;

}

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

    // Copy image to device
    cv::cuda::GpuMat input;
    input.upload(_input);

    const int TILE_SIZE = 32;
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((int)ceil((float)input.size().width / (float)TILE_SIZE), (int)ceil((float)input.size().height / (float)TILE_SIZE));

    cudaStream_t stream =
        cv::cuda::StreamAccessor::getStream(_stream);
    kernelCalculateHistogram<<<dimGrid, dimBlock, 0, stream>>>(input, deviceHistogram);

    // TODO: Error handling: cudaSafeCall(cudaGetLastError());
    
    // Copy the histogram back to the host
	cudaMemcpy(hostHistogram, deviceHistogram, sizeof(unsigned int) * MAXPIXVAL, cudaMemcpyDeviceToHost);

    // Free the device Histogram
    cudaFree(deviceHistogram);

    /*
    for (int i = 0; i < 256; i++)
    {
        printf("%d ", hostHistogram[i]);
    }

    printf("\n----Chom---- \n");*/

    // Normalize the Histogram
	double* normalizedHistogram = new double[MAXPIXVAL];
	long pixelsSum = 0;
	for (int v = 0; v < MAXPIXVAL; v++) {
		normalizedHistogram[v] = (double)hostHistogram[v] / (double)totalPixels;
		pixelsSum += hostHistogram[v];
	}
    return normalizedHistogram;
}

// Embarrassingly parallel shit right here
// Pretty much just copypasta'd
unsigned char cudaFindThreshold(double* histogram, long int totalPixels, cv::cuda::Stream _stream)
{
    // Set up kernel (this is a quick one)
	int threadsPerBlock = 256;
	int numBlocks = 1;

    // Total up all the probablilities?
	double allProbabilitySum = 0;
	for (int i = 0; i < MAXPIXVAL; i++) {
		allProbabilitySum += i * histogram[i];
	}

    // Set up array to hold the variances on host
	double* hostBetweenClassVariances = new double[MAXPIXVAL];
	for (int i = 0; i < MAXPIXVAL; i++) {
		hostBetweenClassVariances[i] = 0;
	}

    // Copy histogram to device again
    // TODO: Optimize this? Does it even matter? It's hardly any data. 
	double* deviceHistogram;
	cudaMalloc((void **)&deviceHistogram, sizeof(double) * MAXPIXVAL);
	cudaMemcpy(deviceHistogram, histogram, sizeof(double) * MAXPIXVAL, cudaMemcpyHostToDevice);

    // Copy variance array to device
	double* deviceBetweenClassVariances;
	cudaMalloc((void **)&deviceBetweenClassVariances, sizeof(double) * MAXPIXVAL);
	cudaMemcpy(deviceBetweenClassVariances, hostBetweenClassVariances, sizeof(double) * MAXPIXVAL, cudaMemcpyHostToDevice);

    // Perform computation
    cudaStream_t stream =
        cv::cuda::StreamAccessor::getStream(_stream);
	kernelComputeClassVariances<<<numBlocks, threadsPerBlock>>>(deviceHistogram, allProbabilitySum, totalPixels, deviceBetweenClassVariances);

    // Copy interclass variances back to host
	cudaMemcpy(hostBetweenClassVariances, deviceBetweenClassVariances, sizeof(double) * MAXPIXVAL, cudaMemcpyDeviceToHost);

	cudaFree(deviceHistogram);
	cudaFree(deviceBetweenClassVariances);

    // Find the highest variance (TODO: Invert this?)
	double maxVariance = 0;
	unsigned char currentBestThreshold = 0;
	for (int t = 0; t < MAXPIXVAL; t++) {
		if (hostBetweenClassVariances[t] > maxVariance) {
			currentBestThreshold = (unsigned char)t;
			maxVariance = hostBetweenClassVariances[t];
		}
	}

	delete hostBetweenClassVariances;

	return currentBestThreshold;
}
