#include <iostream>
#include "grayscale_cuda.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>

constexpr float RED_WEIGHT{ 0.299f };
constexpr float GREEN_WEIGHT{ 0.587f };
constexpr float BLUE_WEIGHT{ 0.114f };

__constant__ float c_red_weight, c_green_weight, c_blue_weight;

__global__ void ConvertToGrayscaleKernel(const unsigned char* input, unsigned char* output, int width, int height, int channel)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < width && y < height)
	{
		const int idx = y * width + x;
		int channelIdx = idx * channel;
		output[idx] = c_blue_weight * input[channelIdx++] + c_green_weight * input[channelIdx++] + c_red_weight * input[channelIdx++];
	}
}

void ConvertBGRToGray(const cv::Mat& imgBGR, cv::Mat& imgGray, int width, int height, int channels, cudaStream_t* streams, int nStreams)
{
	cudaMemcpyToSymbol(c_red_weight, &RED_WEIGHT, sizeof(float));
	cudaMemcpyToSymbol(c_green_weight, &GREEN_WEIGHT, sizeof(float));
	cudaMemcpyToSymbol(c_blue_weight, &BLUE_WEIGHT, sizeof(float));

	const unsigned char* h_imgBGR = imgBGR.data;
	unsigned char* h_imgGrayscale;
	unsigned char* d_imgBGR, *d_imgGrayscale;
	
	cudaMallocHost(&h_imgGrayscale, sizeof(unsigned char) * width * height);
	cudaMalloc(&d_imgBGR, sizeof(unsigned char) * (width * height * channels));
	cudaMalloc(&d_imgGrayscale, sizeof(unsigned char) * (width * height));
	
	constexpr int SHARED_MEM_SIZE = 0;
	const int chunkHeight = (height + (nStreams - 1)) / nStreams;
	const dim3 block(16, 16);
	const dim3 grid(((width + block.x - 1) / block.x), ((chunkHeight + block.y - 1) / block.y));
	for (int i = 0; i < nStreams; ++i)
	{
		
		const int startRow = i * chunkHeight;
		const int endRow = std::min((i + 1) * chunkHeight, height);
		int currentChunkHeight = endRow - startRow;

		const int offsetBGR = startRow * width * channels;
		const size_t copiedDataSizeBGR = sizeof(unsigned char) * width * currentChunkHeight * channels;
		cudaMemcpyAsync(d_imgBGR + offsetBGR, h_imgBGR + offsetBGR, copiedDataSizeBGR, cudaMemcpyHostToDevice, streams[i]);
		
		const int offsetGray = startRow * width;
		const size_t copiedDataSizeGray = sizeof(unsigned char) * width * currentChunkHeight;
		ConvertToGrayscaleKernel << <grid, block, SHARED_MEM_SIZE, streams[i] >> > (d_imgBGR + offsetBGR, d_imgGrayscale + offsetGray, width, currentChunkHeight, channels);
		cudaMemcpyAsync(h_imgGrayscale + offsetGray, d_imgGrayscale + offsetGray, copiedDataSizeGray, cudaMemcpyDeviceToHost, streams[i]);
	}
	cudaDeviceSynchronize();
	
	/*cudaMemcpy(d_imgBGR, h_imgBGR, sizeof(unsigned char) * (width * height * channels), cudaMemcpyHostToDevice);

	const dim3 threadsPerBlock(16, 16);
	const dim3 numBlocks(((width + threadsPerBlock.x - 1) / threadsPerBlock.x), ((height + threadsPerBlock.y - 1) / threadsPerBlock.y));
	ConvertToGrayscaleKernel<<<numBlocks, threadsPerBlock>>>(d_imgBGR, d_imgGrayscale, width, height, channels);

	cudaDeviceSynchronize();

	cudaMemcpy(h_imgGrayscale, d_imgGrayscale, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);*/

	imgGray.create(height, width, CV_8UC1);
	std::memcpy(imgGray.data, h_imgGrayscale, width * height * sizeof(unsigned char));

	cudaFree(d_imgBGR);
	cudaFree(d_imgGrayscale);
	cudaFreeHost(h_imgGrayscale);
}