#include <iostream>
#include "image_gradient_cuda.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>

__constant__ float c_sobel3x3_X[9];
__constant__ float c_sobel3x3_Y[9];
__constant__ float c_sobel5x5_X[25];
__constant__ float c_sobel5x5_Y[25];

__global__ void ComputeImageGradientsKernel(const unsigned char* input, float* Ix, float* Iy, int width, int height, FilterSize filterSize)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < width && y < height)
	{
		const float* c_sobelX, * c_sobelY;
		if (filterSize == FilterSize::Size3x3)
		{
			c_sobelX = c_sobel3x3_X;
			c_sobelY = c_sobel3x3_Y;
		}
		else if (filterSize == FilterSize::Size5x5)
		{
			c_sobelX = c_sobel5x5_X;
			c_sobelY = c_sobel5x5_Y;
		}
		else
		{
			return;
		}
		float gx = 0.0f, gy = 0.0f;
		const int padSize = static_cast<int>(filterSize) / 2;
		const int paddedWidth = width + 2 * padSize;
		const int idx = (y + padSize) * paddedWidth + (x + padSize);
		int sobelIdxY = 0;
		for (int j = idx - paddedWidth * padSize; j <= idx + paddedWidth * padSize; j += paddedWidth)
		{
			int sobelIdxX = 0;
			for (int i = j - padSize; i <= j + padSize; ++i)
			{
				gx += input[i] * c_sobelX[sobelIdxY * (2 * padSize + 1) + sobelIdxX];
				gy += input[i] * c_sobelY[sobelIdxY * (2 * padSize + 1) + sobelIdxX];
				++sobelIdxX;
			}
			++sobelIdxY;
		}
		Ix[y * width + x] = gx;
		Iy[y * width + x] = gy;
	}
}

static void LoadSobelFilterCoefficients(FilterSize filterSize)
{
	if (filterSize == FilterSize::Size3x3)
	{
		constexpr unsigned int rawFilterSize = 9;
		constexpr float h_sobel3x3_X[9] = {
			-1, 0, 1,
			-2, 0, 2,
			-1, 0, 1
		};

		constexpr float h_sobel3x3_Y[9] = {
			-1, -2, -1,
			 0,  0,  0,
			 1,  2,  1
		};
		cudaMemcpyToSymbol(c_sobel3x3_X, h_sobel3x3_X, sizeof(float) * rawFilterSize);
		cudaMemcpyToSymbol(c_sobel3x3_Y, h_sobel3x3_Y, sizeof(float) * rawFilterSize);
	}
	else if (filterSize == FilterSize::Size5x5)
	{
		constexpr unsigned int rawFilterSize = 25;
		constexpr float h_sobel5x5_X[rawFilterSize] =
		{
			-5, -4,  0,  4,  5,
			-8, -10, 0, 10,  8,
		   -10, -20, 0, 20, 10,
			-8, -10, 0, 10,  8,
			-5, -4,  0,  4,  5
		};
		constexpr float h_sobel5x5_Y[rawFilterSize] =
		{
			-5, -8, -10, -8, -5,
			-4,-10, -20,-10, -4,
			 0,  0,   0,  0,  0,
			 4, 10,  20, 10,  4,
			 5,  8,  10,  8,  5
		};
		cudaMemcpyToSymbol(c_sobel5x5_X, h_sobel5x5_X, sizeof(float) * rawFilterSize);
		cudaMemcpyToSymbol(c_sobel5x5_Y, h_sobel5x5_Y, sizeof(float) * rawFilterSize);
	}
}

void ComputeImageGradients(const cv::Mat& img, float* h_Ix, float* h_Iy, FilterSize filterSize, cv::BorderTypes borderType, int width, int height)
{
	cv::Mat imgPadded;
	const int padSize = static_cast<int>(filterSize) / 2;
	cv::copyMakeBorder(img, imgPadded, padSize, padSize, padSize, padSize, borderType);

	LoadSobelFilterCoefficients(filterSize);

	unsigned char* d_input;
	cudaMalloc(&d_input, sizeof(unsigned char) * (width + 2 * padSize) * (height + 2 * padSize));
	cudaMemcpy(d_input, imgPadded.data, sizeof(unsigned char) * (width + 2 * padSize) * (height + 2 * padSize), cudaMemcpyHostToDevice);

	float* d_Ix, * d_Iy;
	cudaMalloc(&d_Ix, sizeof(float) * width * height);
	cudaMalloc(&d_Iy, sizeof(float) * width * height);
	
	const dim3 threadsPerBlock(16, 16);
	const dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
	ComputeImageGradientsKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_Ix, d_Iy, width, height, filterSize);

	cudaDeviceSynchronize();
	
	cudaMemcpy(h_Ix, d_Ix, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_Iy, d_Iy, sizeof(float) * width * height, cudaMemcpyDeviceToHost);

	cudaFree(d_input);
	cudaFree(d_Ix);
	cudaFree(d_Iy);
}