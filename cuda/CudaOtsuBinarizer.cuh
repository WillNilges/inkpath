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

#include "model/PngImage.h"
#include "model/ExecutionTimestamp.h"

#pragma once
class CudaOtsuBinarizer
{
public:
	PngImage* binarize(PngImage* imageToBinarize);
	std::string getBinarizerExecutionInfo(std::string fileName);
	CudaOtsuBinarizer(int threadsPerBlock, int numBlocks, bool drawHistogram, const char* TAG = "GPU");
	virtual ~CudaOtsuBinarizer();
protected:
	int threadsPerBlock_;
	int numBlocks_;
	ExecutionTimestamp* binarizerTimestamp_;
	bool drawHistogram_;
	const char* TAG;
	virtual void showHistogram(double* histogram);
	virtual double* cudaCalculateHistogram(unsigned char* rawPixels, long totalPixels);
	virtual unsigned char cudaFindThreshold(double* histogram, long int totalPixels);
	virtual unsigned char* cudaBinarize(unsigned char* rawPixels, long totalPixels, unsigned char threshold);
};

