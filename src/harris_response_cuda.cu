#include "harris_response_cuda.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>

__constant__ float c_k;

__global__ void CalculateHarrisResponseKernel(const float* Sxx, const float* Syy, const float* Sxy, float* harrisResponse, int width, int height)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < width && y < height)
	{
		const int idx = y * width + x;
		const float SxxVal = Sxx[idx];
		const float SyyVal = Syy[idx];
		const float SxyVal = Sxy[idx];

		float det = SxxVal * SyyVal - SxyVal * SxyVal;
		float trace = SxxVal + SyyVal;

		harrisResponse[idx] = det - c_k * trace * trace;
	}
}

void CalculateHarrisResponse(const float* h_Sxx, const float* h_Syy, const float* h_Sxy, float* h_harrisResponse, int width, int height, cudaStream_t* streams, int nStreams)
{
	constexpr float h_k = 0.04;
	cudaMemcpyToSymbol(c_k, &h_k, sizeof(float));

	float* d_Sxx, * d_Syy, * d_Sxy, *d_harrisResponse;
	cudaMalloc(&d_Sxx, sizeof(float) * width * height);
	cudaMalloc(&d_Syy, sizeof(float) * width * height);
	cudaMalloc(&d_Sxy, sizeof(float) * width * height);
	cudaMalloc(&d_harrisResponse, sizeof(float) * width * height);

	const int chunkHeight = (height + (nStreams - 1)) / nStreams;
	constexpr dim3 block{ 16, 16 };
	for (int i = 0; i < nStreams; ++i)
	{
		const int startRow = i * chunkHeight;
		const int endRow = std::min((i + 1) * chunkHeight, height);
		const int currentChunkHeight = endRow - startRow;
		if (currentChunkHeight <= 0)
		{
			continue;
		}
		const int offset = startRow * width;

		cudaMemcpyAsync(d_Sxx + offset, h_Sxx + offset, sizeof(float) * width * currentChunkHeight, cudaMemcpyHostToDevice, streams[i]);
		cudaMemcpyAsync(d_Syy + offset, h_Syy + offset, sizeof(float) * width * currentChunkHeight, cudaMemcpyHostToDevice, streams[i]);
		cudaMemcpyAsync(d_Sxy + offset, h_Sxy + offset, sizeof(float) * width * currentChunkHeight, cudaMemcpyHostToDevice, streams[i]);
	}

	for (int i = 0; i < nStreams; ++i)
	{
		const int startRow = i * chunkHeight;
		const int endRow = std::min((i + 1) * chunkHeight, height);
		const int currentChunkHeight = endRow - startRow;
		if (currentChunkHeight <= 0)
		{
			continue;
		}
		const int offset = startRow * width;
		const dim3 grid((width + (block.x - 1)) / block.x, (currentChunkHeight + (block.y - 1)) / block.y);
		constexpr int SHARED_MEM_SIZE{ 0 };
		CalculateHarrisResponseKernel << <grid, block, SHARED_MEM_SIZE, streams[i] >> > (d_Sxx + offset, d_Syy + offset, d_Sxy + offset, d_harrisResponse + offset, width, currentChunkHeight);
	}

	for (int i = 0; i < nStreams; ++i)
	{
		const int startRow = i * chunkHeight;
		const int endRow = std::min((i + 1) * chunkHeight, height);
		const int currentChunkHeight = endRow - startRow;
		if (currentChunkHeight <= 0)
		{
			continue;
		}
		const int offset = startRow * width;
		cudaMemcpyAsync(h_harrisResponse + offset, d_harrisResponse + offset, sizeof(float) * width * currentChunkHeight, cudaMemcpyDeviceToHost, streams[i]);
	}

	for (int i = 0; i < nStreams; ++i)
	{
		cudaStreamSynchronize(streams[i]);
	}

	cudaFree(d_Sxx);
	cudaFree(d_Syy);
	cudaFree(d_Sxy);
	cudaFree(d_harrisResponse);
}
