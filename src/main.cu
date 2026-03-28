#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>

constexpr float RED_WEIGHT{ 0.299f };
constexpr float GREEN_WEIGHT{ 0.587f };
constexpr float BLUE_WEIGHT{ 0.114f };

__constant__ float c_red_weight, c_green_weight, c_blue_weight;

__global__ void convertToGrayscale(unsigned char* input, unsigned char* output, int width, int height, int channel)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < width && y < height)
	{
		const int idx = y * width + x;
		output[idx] = c_blue_weight * input[idx * channel] + c_green_weight * input[idx * channel + 1] + c_red_weight * input[idx * channel + 2];
	}
}

int main()
{
	cv::Mat img = cv::imread("C:/Users/User/Workspaces/VisualStudio/VS2026/HarrisCornerVideo/data/Lenna.png");
	if (img.empty())
	{
		std::cout << "Could not read the image: " << std::endl;
	}
	else
	{
		std::cout << "Image loaded successfully: " << img.cols << "x" << img.rows << std::endl;
		std::cout << "Image channels = " << img.channels() << std::endl;
		
		unsigned char * d_imgBGR, *d_imgGrayscale;
		unsigned char* h_imgGrayscale = new unsigned char[img.cols * img.rows];

		cudaMalloc(&d_imgBGR, sizeof(unsigned char) * (img.cols * img.rows * img.channels()));
		cudaMalloc(&d_imgGrayscale, sizeof(unsigned char) * (img.cols * img.rows));
		cudaMemcpy(d_imgBGR, img.data, sizeof(unsigned char) * (img.cols * img.rows * img.channels()), cudaMemcpyHostToDevice);

		cudaMemcpyToSymbol(c_red_weight, &RED_WEIGHT, sizeof(float));
		cudaMemcpyToSymbol(c_green_weight, &GREEN_WEIGHT, sizeof(float));
		cudaMemcpyToSymbol(c_blue_weight, &BLUE_WEIGHT, sizeof(float));

		const dim3 threadsPerBlock(16, 16);	
		const dim3 numBlocks(((img.cols + threadsPerBlock.x - 1) / threadsPerBlock.x), ((img.rows + threadsPerBlock.y - 1) / threadsPerBlock.y));
		convertToGrayscale<<<numBlocks, threadsPerBlock>>>(d_imgBGR, d_imgGrayscale, img.cols, img.rows, img.channels());

		cudaError_t err = cudaDeviceSynchronize();
		if (err != cudaSuccess)
		{
			std::cout << "Kernel error: " << cudaGetErrorString(err) << std::endl;
			cudaFree(d_imgBGR);
			cudaFree(d_imgGrayscale);
			delete[] h_imgGrayscale;
			return -1;
		}

		cudaMemcpy(h_imgGrayscale, d_imgGrayscale, img.cols * img.rows, cudaMemcpyDeviceToHost);
		
		cv::Mat grayImg(img.rows, img.cols, CV_8UC1, h_imgGrayscale);
		cv::imshow("Original", img);
		cv::imshow("Grayscale", grayImg);
		cv::waitKey(0);
		
		cudaFree(d_imgBGR);
		cudaFree(d_imgGrayscale);
		delete[] h_imgGrayscale;
	}
}