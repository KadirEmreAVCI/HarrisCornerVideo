#include <iostream>
#include "second_moment_matrix_cuda.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>

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
	cudaFree(d_Ixy);
}