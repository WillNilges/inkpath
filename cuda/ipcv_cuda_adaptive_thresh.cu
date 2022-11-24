#include "ipcv_cuda_adaptive_thresh.cuh"

// CUDA imports
#include <cuda_runtime.h>


__global__ void kernelBuildTab(unsigned char* tab, uchar imaxval, int idelta, int type)
{
	int idX = blockDim.x * blockIdx.x + threadIdx.x;
    if (idX < 768)
    {
        if( type == CV_THRESH_BINARY )
            tab[idX] = (uchar)(idX - 255 > -idelta ? imaxval : 0);
        else if( type == CV_THRESH_BINARY_INV )
            tab[idX] = (uchar)(idX - 255 <= -idelta ? imaxval : 0);
        //else // TODO: Throw error
            //CV_Error( CV_StsBadFlag, "Unknown/unsupported threshold type" );
    }
}

    //kernelThreshold<<<dimGrid, dimBlock>>>(deviceSrc, deviceMean, deviceDst, deviceTab, size);

__global__ void kernelThreshold(
    const cv::cuda::PtrStepSz<unsigned char> src,
    const cv::cuda::PtrStepSz<unsigned char> mean,
    cv::cuda::PtrStepSz<unsigned char> dst,
    unsigned char* tab,
    Size size
)
{
	int idX = blockDim.x * blockIdx.x + threadIdx.x;
	int idY = blockDim.y * blockIdx.y + threadIdx.y;

    if (idX < size.width && idY < size.height)
    {
        const uchar* sdata = src.ptr(idY);
        const uchar* mdata = mean.ptr(idY);
        uchar* ddata = dst.ptr(idY);

        ddata[idX] = tab[sdata[idX] - mdata[idX] + 255];
    }
}


// Called from debug.cpp
cv::Mat adaptiveCuda(cv::Mat img, std::string output_path, cv::cuda::Stream _stream) {
    // Binarize the image using OpenCV
    cv::Mat hostBinarized;
    cudaAdaptiveThreshold(img, hostBinarized, 255, ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 5, 2, _stream);

    if (!output_path.empty()) {
        imwrite(output_path, hostBinarized);
#ifdef DIAG
        std::cout << "Image has been written to " << output_path << "\n";
#endif
    }

    return hostBinarized;
}

void cudaAdaptiveThreshold(
    InputArray _src, OutputArray _dst, double maxValue,
    int method, int type, int blockSize, double delta,
    cv::cuda::Stream _stream
){
    Mat host_src = _src.getMat();
    CV_Assert( host_src.type() == CV_8UC1 );
    CV_Assert( blockSize % 2 == 1 && blockSize > 1 );
    Size size = host_src.size();

    // Move src to the GPU.
    cv::cuda::GpuMat dev_src;
    dev_src.upload(host_src);

    _dst.create( size, host_src.type() );
    Mat host_dst = _dst.getMat();
    cv::cuda::GpuMat dev_dst;
    dev_dst.upload(host_dst);

    if( maxValue < 0 )
    {
        host_dst = Scalar(0);
        return;
    }

    Mat host_mean;

    if( host_src.data != host_dst.data )
        host_mean = host_dst;

    if (method == ADAPTIVE_THRESH_MEAN_C)
        // Not implemented. Dunno if it ever will be, bozo.
        CV_Error( CV_StsBadFlag, "Unknown/unsupported adaptive threshold method" );
        /*boxFilter( src, mean, src.type(), Size(blockSize, blockSize),
                   Point(-1,-1), true, BORDER_REPLICATE|BORDER_ISOLATED );*/
    else if (method == ADAPTIVE_THRESH_GAUSSIAN_C)
    {
        /*
        Mat host_srcfloat, host_meanfloat;
        host_src.convertTo(host_srcfloat,CV_32F);

        cv::cuda::GpuMat dev_srcfloat, dev_meanfloat;
        dev_srcfloat.upload(host_srcfloat);
        dev_meanfloat=dev_srcfloat;

        // Gaussian filtering
        cv::Ptr<cv::cuda::Filter> gauss_filter =
            cv::cuda::createGaussianFilter(
                dev_srcfloat.type(),
                -1,
                Size(blockSize, blockSize),
                0,
                0,
                BORDER_REPLICATE,
                -1
            );
        gauss_filter->apply(dev_srcfloat, dev_meanfloat, _stream);

        dev_meanfloat.download(host_meanfloat);
        host_meanfloat.convertTo(host_mean, host_src.type());*/

        // Gaussian filtering
        cv::cuda::GpuMat dev_src, dev_mean;
        dev_src.upload(host_src);
        dev_mean.upload(host_mean);
        cv::Ptr<cv::cuda::Filter> gauss_filter =
            cv::cuda::createGaussianFilter(dev_src.type(), -1, Size(5, 5), 0, 0);
        gauss_filter->apply(dev_src, dev_mean, _stream);
        dev_mean.download(host_mean);
    }
    else
        CV_Error( CV_StsBadFlag, "Unknown/unsupported adaptive threshold method" );

    uchar imaxval = saturate_cast<uchar>(maxValue);
    int idelta = type == THRESH_BINARY ? cvCeil(delta) : cvFloor(delta);
    int magicNumber = 768; // I have literally no idea why this is 768.
    uchar hostTab[768];

    // Allocate space for tab
	unsigned char* deviceTab;
	cudaMalloc((void **)&deviceTab, sizeof(unsigned char) * magicNumber);
	//cudaMemcpy(deviceTab, hostTab, sizeof(unsigned char) * magicNumber, cudaMemcpyHostToDevice);

    // Run kernel to build the tab
    cudaStream_t stream =
        cv::cuda::StreamAccessor::getStream(_stream);
	kernelBuildTab<<<1, magicNumber, 0, stream>>>(deviceTab, idelta, imaxval, type);
    cudaDeviceSynchronize();

    // Copy finished tab to host (not necessary)
    cudaMemcpy(hostTab, deviceTab, sizeof(unsigned char) * magicNumber, cudaMemcpyDeviceToHost);

    for (int i = 0; i < magicNumber; i++)
    {
        printf("%d ", hostTab[i]);
    }

    if( host_src.isContinuous() && host_mean.isContinuous() && host_dst.isContinuous() )
    {
        size.width *= size.height;
        size.height = 1;
    }

    // Copy the matricies to the device
    cv::cuda::GpuMat deviceSrc, deviceMean, deviceDst;
    deviceSrc.upload(host_src);
    deviceMean.upload(host_mean);
    deviceDst.upload(host_dst);

    imwrite("/tmp/chom.png", host_src);

    // Set up and run the kernel.
    const int TILE_SIZE = 32;
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((int)ceil((float)size.width / (float)TILE_SIZE), (int)ceil((float)size.height / (float)TILE_SIZE));
    kernelThreshold<<<dimGrid, dimBlock, 0, stream>>>(deviceSrc, deviceMean, deviceDst, deviceTab, size);
    cudaDeviceSynchronize();

    // Copy finished product back to host
    deviceDst.download(host_dst);

    
    // Free tab
    cudaFree(deviceTab);
}
