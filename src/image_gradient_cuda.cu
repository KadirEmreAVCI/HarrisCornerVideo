#include <iostream>
#include "image_gradient_cuda.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>

__constant__ unsigned char c_sobel3x3[9];
__constant__ unsigned char c_sobel5x5[25];

void ComputeImageGradient(const cv::Mat& img,
	unsigned char* h_Ix,
	unsigned char* h_Iy,
	FilterSize filterSize,
	cv::BorderTypes borderType,
	int width,
	int height)
{
	cv::Mat imgPadded;
	const int padSize = static_cast<int>(filterSize) / 2;
	cv::copyMakeBorder(
		img,
		imgPadded,
		padSize, padSize,        // top, bottom
		padSize, padSize,        // left, right
		borderType
	);

	cv::imshow("Padded", imgPadded);
	cv::waitKey(0);
}