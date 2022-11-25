/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

/*
===============================================================================

This code was heavily inspiried by https://github.com/opencv/opencv/blob/1363496c1106606684d40447f5d1149b2c66a9f8/modules/imgproc/src/thresh.cpp

I've taken the liberty of naively porting the logic to CUDA in an attempt to make
it faster. The CUDA code and anything else not copypasta'd from OpenCV falls
under Inkpath's GPL 2 license.

Cheers,
- Willard
*/

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
    cudaAdaptiveThreshold(
        img, 
        img,
        255,
        ADAPTIVE_THRESH_GAUSSIAN_C,
        THRESH_BINARY,
        3,
        2,
        _stream
    );

    if (!output_path.empty()) {
        imwrite(output_path, img);
#ifdef DIAG
        std::cout << "Image has been written to " << output_path << "\n";
#endif
    }

    return img;
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

    // Pre-apply gaussian blur
    cv::Ptr<cv::cuda::Filter> gauss_filter =
        cv::cuda::createGaussianFilter(
        dev_src.type(), 
        dev_src.type(), 
        Size(blockSize+2, blockSize+2),
        0,
        0,
        0,
        0
    );
    gauss_filter->apply(dev_src, dev_src, _stream);

    // Create dest matrix.
    _dst.create( size, host_src.type() );
    Mat host_dst = _dst.getMat();

    // Move dst to the GPU.
    cv::cuda::GpuMat dev_dst;
    dev_dst.upload(host_dst);

    if( maxValue < 0 )
    {
        host_dst = Scalar(0);
        return;

    }

    Mat host_mean;
    cv::cuda::GpuMat dev_mean;

    // Compare src and destination
    if( host_src.data != host_dst.data )
        host_mean = host_dst;

    if (method == ADAPTIVE_THRESH_MEAN_C)
        // Not implemented. Dunno if it ever will be, bozo.
        CV_Error( CV_StsBadFlag, "Unknown/unsupported adaptive threshold method" );
        /*boxFilter( src, mean, src.type(), Size(blockSize, blockSize),
                   Point(-1,-1), true, BORDER_REPLICATE|BORDER_ISOLATED );*/
    else if (method == ADAPTIVE_THRESH_GAUSSIAN_C)
    {
        // Gaussian filtering
        // Convert data to float
        cv::cuda::GpuMat dev_srcfloat, dev_meanfloat;
        dev_src.convertTo(dev_srcfloat, CV_32F);
        dev_meanfloat = dev_srcfloat;

        cv::Ptr<cv::cuda::Filter> gauss_filter =
            cv::cuda::createGaussianFilter(
            dev_srcfloat.type(), 
            dev_meanfloat.type(), 
            Size(blockSize, blockSize),
            0,
            0,
            BORDER_REPLICATE,
            BORDER_REPLICATE
        ); // TODO: border default
        gauss_filter->apply(dev_srcfloat, dev_meanfloat, _stream);

        // Convert back to normal type
        dev_meanfloat.convertTo(dev_mean, dev_src.type());
    }
    else
        CV_Error( CV_StsBadFlag, "Unknown/unsupported adaptive threshold method" );

    // Build the spooky magic array
    uchar imaxval = saturate_cast<uchar>(maxValue);
    int idelta = type == THRESH_BINARY ? cvCeil(delta) : cvFloor(delta);
    int magicNumber = 768; // I have literally no idea why this is 768.
    uchar hostTab[magicNumber];

    // Allocate space for tab
	unsigned char* deviceTab;
	cudaMalloc((void **)&deviceTab, sizeof(uchar) * magicNumber);
	cudaMemcpy(deviceTab, hostTab, sizeof(uchar) * magicNumber, cudaMemcpyHostToDevice);

    cudaStream_t stream =
        cv::cuda::StreamAccessor::getStream(_stream);

    // Run kernel to build the tab
	kernelBuildTab<<<1, magicNumber, 0, stream>>>(deviceTab, imaxval, idelta, type);
    cudaDeviceSynchronize();

    /*
    if( host_src.isContinuous() && host_mean.isContinuous() && host_dst.isContinuous() )
    {
        size.width *= size.height;
        size.height = 1;
    }
    */

    // Set up and run the kernel.
    const int TILE_SIZE = 32;
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((int)ceil((float)size.width / (float)TILE_SIZE), (int)ceil((float)size.height / (float)TILE_SIZE));
    kernelThreshold<<<dimGrid, dimBlock, 0, stream>>>(dev_src, dev_mean, dev_dst, deviceTab, size);
    cudaDeviceSynchronize();

    // Copy finished product back to host
    dev_dst.download(host_dst);

    // Free tab
    cudaFree(deviceTab);
}
