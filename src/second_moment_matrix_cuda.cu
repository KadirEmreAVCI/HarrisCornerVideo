#include <iostream>
#include "second_moment_matrix_cuda.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>

__global__ void ComputeSecondMomentMatrixKernel(const float* Ix, const float* Iy, float* Ixx, float* Iyy, float* Ixy, int width, int height)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < width && y < height)
	{
		const int idx = width * y + x;
		const float ix = Ix[idx], iy = Iy[idx];
		Ixx[idx] = ix * ix;
		Iyy[idx] = iy * iy;
		Ixy[idx] = ix * iy;
	}
}

void ComputeSecondMomentMatrix(const float* h_Ix, const float* h_Iy, float* h_Ixx, float* h_Iyy, float* h_Ixy, int width, int height)
{
	float* d_Ix, * d_Iy, * d_Ixx, * d_Iyy, * d_Ixy;

	cudaMalloc(&d_Ix, sizeof(float) * width * height);
	cudaMalloc(&d_Iy, sizeof(float) * width * height);
	cudaMalloc(&d_Ixx, sizeof(float) * width * height);
	cudaMalloc(&d_Iyy, sizeof(float) * width * height);
	cudaMalloc(&d_Ixy, sizeof(float) * width * height);

	constexpr int NSTREAMS = 4, SHARED_MEM_SIZE = 0;
	cudaStream_t streams[NSTREAMS];
	const int chunkHeight = (height + (NSTREAMS - 1)) / NSTREAMS;
	for (int i = 0; i < NSTREAMS; ++i)
	{
		cudaStreamCreate(&streams[i]);
		
		const int startRow = i * chunkHeight;
		const int endRow = std::min((i + 1) * chunkHeight, height);
		const int currentChunkHeight = endRow - startRow;
		const int offset = startRow * width;
		const size_t copiedDataSize = sizeof(float) * currentChunkHeight * width;

		cudaMemcpyAsync(d_Ix + offset, h_Ix + offset, copiedDataSize, cudaMemcpyHostToDevice, streams[i]);
		cudaMemcpyAsync(d_Iy + offset, h_Iy + offset, copiedDataSize, cudaMemcpyHostToDevice, streams[i]);

		const dim3 block(16, 16);
		const dim3 grid(((width + block.x - 1) / block.x), ((currentChunkHeight + block.y - 1) / block.y));
		ComputeSecondMomentMatrixKernel << <grid, block, SHARED_MEM_SIZE, streams[i]>> > (d_Ix + offset, d_Iy + offset, d_Ixx + offset, d_Iyy + offset, d_Ixy + offset, width, currentChunkHeight);

		cudaMemcpyAsync(h_Ixx + offset, d_Ixx + offset, copiedDataSize, cudaMemcpyDeviceToHost, streams[i]);
		cudaMemcpyAsync(h_Iyy + offset, d_Iyy + offset, copiedDataSize, cudaMemcpyDeviceToHost, streams[i]);
		cudaMemcpyAsync(h_Ixy + offset, d_Ixy + offset, copiedDataSize, cudaMemcpyDeviceToHost, streams[i]);
	}

	for (int i = 0; i < NSTREAMS; ++i)
	{
		cudaStreamSynchronize(streams[i]);
		cudaStreamDestroy(streams[i]);
	}

	cudaFree(d_Ix);
	cudaFree(d_Iy);
	cudaFree(d_Ixx);
	cudaFree(d_Iyy);
	cudaFree(d_Ixy);

	/*float* d_Ix, * d_Iy, * d_Ixx, * d_Iyy, * d_Ixy;

	cudaMalloc(&d_Ix, sizeof(float) * width * height);
	cudaMalloc(&d_Iy, sizeof(float) * width * height);
	cudaMalloc(&d_Ixx, sizeof(float) * width * height);
	cudaMalloc(&d_Iyy, sizeof(float) * width * height);
	cudaMalloc(&d_Ixy, sizeof(float) * width * height);

	cudaMemcpy(d_Ix, h_Ix, sizeof(float) * width * height, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Iy, h_Iy, sizeof(float) * width * height, cudaMemcpyHostToDevice);

	const dim3 threadsPerBlock(16, 16);
	const dim3 numBlocks(((width + threadsPerBlock.x - 1) / threadsPerBlock.x), ((height + threadsPerBlock.y - 1) / threadsPerBlock.y));
	ComputeSecondMomentMatrixKernel << <numBlocks, threadsPerBlock >> > (d_Ix, d_Iy, d_Ixx, d_Iyy, d_Ixy, width, height);

	cudaDeviceSynchronize();

	cudaMemcpy(h_Ixx, d_Ixx, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_Iyy, d_Iyy, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_Ixy, d_Ixy, sizeof(float) * width * height, cudaMemcpyDeviceToHost);

	cudaFree(d_Ix);
	cudaFree(d_Iy);
	cudaFree(d_Ixx);
	cudaFree(d_Iyy);
	cudaFree(d_Ixy);*/
}