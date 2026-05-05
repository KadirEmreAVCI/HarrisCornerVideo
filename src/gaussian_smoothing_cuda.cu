#include "gaussian_smoothing_cuda.cuh"
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__constant__ float c_gaussian3x3[9];
__constant__ float c_gaussian5x5[25];

__global__ void ApplyGaussianSmoothingKernel(
    const float* input,
    float* output,
    int width,
    int height,
    FilterSize filterSize)
{
    const int outX = blockIdx.x * blockDim.x + threadIdx.x;
    const int outY = blockIdx.y * blockDim.y + threadIdx.y;

    const int padSize = static_cast<int>(filterSize) / 2;
    const int filterWidth = 2 * padSize + 1;

    const int paddedWidth = width + 2 * padSize;
    const int paddedHeight = height + 2 * padSize;

    const float* c_gaussian = nullptr;

    if (filterSize == FilterSize::Size3x3)
    {
        c_gaussian = c_gaussian3x3;
    }
    else if (filterSize == FilterSize::Size5x5)
    {
        c_gaussian = c_gaussian5x5;
    }
    else
    {
        return;
    }

    extern __shared__ float sharedTile[];

    const int sharedWidth = blockDim.x + 2 * padSize;
    const int sharedHeight = blockDim.y + 2 * padSize;

    const int tileStartX = blockIdx.x * blockDim.x;
    const int tileStartY = blockIdx.y * blockDim.y;

    // Load input tile + halo into shared memory
    for (int localY = threadIdx.y; localY < sharedHeight; localY += blockDim.y)
    {
        for (int localX = threadIdx.x; localX < sharedWidth; localX += blockDim.x)
        {
            const int globalX = tileStartX + localX;
            const int globalY = tileStartY + localY;

            if (globalX < paddedWidth && globalY < paddedHeight)
            {
                sharedTile[localY * sharedWidth + localX] =
                    input[globalY * paddedWidth + globalX];
            }
            else
            {
                sharedTile[localY * sharedWidth + localX] = 0.0f;
            }
        }
    }

    __syncthreads();

    if (outX >= width || outY >= height)
        return;

    float sum = 0.0f;

    const int sharedCenterX = threadIdx.x + padSize;
    const int sharedCenterY = threadIdx.y + padSize;

    for (int fy = -padSize; fy <= padSize; ++fy)
    {
        for (int fx = -padSize; fx <= padSize; ++fx)
        {
            const int sharedX = sharedCenterX + fx;
            const int sharedY = sharedCenterY + fy;

            const int filterX = fx + padSize;
            const int filterY = fy + padSize;

            sum += sharedTile[sharedY * sharedWidth + sharedX] *
                c_gaussian[filterY * filterWidth + filterX];
        }
    }

    output[outY * width + outX] = sum;
}

void LoadGaussianFilterCoefficients(FilterSize filterSize)
{
	if (filterSize == FilterSize::Size3x3)
	{
		constexpr unsigned int rawFilterSize = 9;
		constexpr float h_gaussian3x3[9] = {
			1 / 16.f, 2 / 16.f, 1 / 16.f,
			2 / 16.f, 4 / 16.f, 2 / 16.f,
			1 / 16.f, 2 / 16.f, 1 / 16.f
		};
		cudaMemcpyToSymbol(c_gaussian3x3, h_gaussian3x3, sizeof(float) * rawFilterSize);
	}
	else if (filterSize == FilterSize::Size5x5)
	{
		constexpr unsigned int rawFilterSize = 25;
		constexpr float h_gaussian5x5[rawFilterSize] =
		{
			1 / 256.f, 4 / 256.f, 6 / 256.f, 4 / 256.f, 1 / 256.f,
			4 / 256.f,16 / 256.f,24 / 256.f,16 / 256.f, 4 / 256.f,
			6 / 256.f,24 / 256.f,36 / 256.f,24 / 256.f, 6 / 256.f,
			4 / 256.f,16 / 256.f,24 / 256.f,16 / 256.f, 4 / 256.f,
			1 / 256.f, 4 / 256.f, 6 / 256.f, 4 / 256.f, 1 / 256.f
		};
		cudaMemcpyToSymbol(c_gaussian5x5, h_gaussian5x5, sizeof(float) * rawFilterSize);
	}
}

static void ApplyPaddingToMatrixComponent(const float* input, cv::Mat& imgOutput, int width, int height, cv::BorderTypes borderType, int padSize)
{
	cv::Mat imgInput(height, width, CV_32F, const_cast<float*>(input));
	cv::copyMakeBorder(imgInput, imgOutput, padSize, padSize, padSize, padSize, borderType);
	CV_Assert(imgOutput.type() == CV_32F && imgOutput.isContinuous());
}

void ApplyGaussianSmoothing(const float* h_Ixx, const float* h_Iyy, const float* h_Ixy, float* h_Sxx, float* h_Syy, float* h_Sxy, int width, int height, FilterSize filterSize)
{
	cv::Mat imgIxxPadded, imgIyyPadded, imgIxyPadded;
	const int padSize = static_cast<int>(filterSize) / 2;

	ApplyPaddingToMatrixComponent(h_Ixx, imgIxxPadded, width, height, cv::BORDER_REPLICATE, padSize);
	ApplyPaddingToMatrixComponent(h_Iyy, imgIyyPadded, width, height, cv::BORDER_REPLICATE, padSize);
	ApplyPaddingToMatrixComponent(h_Ixy, imgIxyPadded, width, height, cv::BORDER_REPLICATE, padSize);

	float* const h_IxxPadded = imgIxxPadded.ptr<float>();
	float* const h_IyyPadded = imgIyyPadded.ptr<float>();
	float* const h_IxyPadded = imgIxyPadded.ptr<float>();

	LoadGaussianFilterCoefficients(filterSize);

	float* d_IxxPadded, * d_IyyPadded, * d_IxyPadded, * d_Sxx, * d_Syy, * d_Sxy;
	
	const size_t paddedDataSize = sizeof(float) * (width + 2 * padSize) * (height + 2 * padSize);
	cudaMalloc(&d_IxxPadded, paddedDataSize);
	cudaMalloc(&d_IyyPadded, paddedDataSize);
	cudaMalloc(&d_IxyPadded, paddedDataSize);
	
	const size_t unpaddedDataSize = sizeof(float) * width * height;
	cudaMalloc(&d_Sxx, unpaddedDataSize);
	cudaMalloc(&d_Syy, unpaddedDataSize);
	cudaMalloc(&d_Sxy, unpaddedDataSize);
	
	cudaMemcpy(d_IxxPadded, h_IxxPadded, paddedDataSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_IyyPadded, h_IyyPadded, paddedDataSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_IxyPadded, h_IxyPadded, paddedDataSize, cudaMemcpyHostToDevice);

	const dim3 block(16, 16);
	const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    int sharedWidth = block.x + 2 * padSize;
    int sharedHeight = block.y + 2 * padSize;

    size_t sharedMemorySize = sharedWidth * sharedHeight * sizeof(float);

	ApplyGaussianSmoothingKernel << < grid, block, sharedMemorySize >> > (d_IxxPadded, d_Sxx, width, height, filterSize);
	ApplyGaussianSmoothingKernel << < grid, block, sharedMemorySize >> > (d_IyyPadded, d_Syy, width, height, filterSize);
	ApplyGaussianSmoothingKernel << < grid, block, sharedMemorySize >> > (d_IxyPadded, d_Sxy, width, height, filterSize);

	cudaDeviceSynchronize();

	cudaMemcpy(h_Sxx, d_Sxx, unpaddedDataSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_Syy, d_Syy, unpaddedDataSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_Sxy, d_Sxy, unpaddedDataSize, cudaMemcpyDeviceToHost);

	cudaFree(d_IxxPadded);
	cudaFree(d_IyyPadded);
	cudaFree(d_IxyPadded);
	cudaFree(d_Sxx);
	cudaFree(d_Syy);
	cudaFree(d_Sxy);
}