#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include "grayscale_cuda.cuh"
#include "image_gradient_cuda.cuh"
#include "second_moment_matrix_cuda.cuh"
#include "gaussian_smoothing_cuda.cuh"
#include "harris_response_cuda.cuh"
#include "finding_corners_cuda.cuh"

constexpr FilterSize filterSize = FilterSize::Size3x3;

int main()
{
	cv::Mat imgOrig = cv::imread("C:/Users/User/Workspaces/VisualStudio/VS2026/HarrisCornerVideo/data/Lenna.png");
	if (imgOrig.empty())
	{
		std::cout << "Could not read the image: " << std::endl;
	}
	else
	{
		const unsigned width = imgOrig.cols;
		const unsigned height = imgOrig.rows;
		const unsigned channels = imgOrig.channels();

		std::cout << "Image loaded successfully: " << width << "x" << height << std::endl;
		std::cout << "Image channels = " << channels << std::endl;
		
		cv::Mat imgGrayscale;
		ConvertBGRToGray(imgOrig, imgGrayscale, width, height, channels);
		
		cv::imshow("Original", imgOrig);
		cv::imshow("Grayscale", imgGrayscale);
		cv::waitKey(0);

		float* h_Ix, * h_Iy;
		cudaMallocHost(&h_Ix, sizeof(float) * width * height);
		cudaMallocHost(&h_Iy, sizeof(float) * width * height);

		ComputeImageGradients(imgGrayscale, h_Ix, h_Iy, filterSize, cv::BORDER_REPLICATE, width, height);

		float* h_Ixx, * h_Iyy, * h_Ixy;
		cudaMallocHost(&h_Ixx, sizeof(float) * width * height);
		cudaMallocHost(&h_Iyy, sizeof(float) * width * height);
		cudaMallocHost(&h_Ixy, sizeof(float) * width * height);
		
		ComputeSecondMomentMatrix(h_Ix, h_Iy, h_Ixx, h_Iyy, h_Ixy, width, height);

		cudaFreeHost(h_Ix);
		cudaFreeHost(h_Iy);

		float* h_Sxx, * h_Syy, * h_Sxy;
		cudaMallocHost(&h_Sxx, sizeof(float) * width * height);
		cudaMallocHost(&h_Syy, sizeof(float) * width * height);
		cudaMallocHost(&h_Sxy, sizeof(float) * width * height);

		ApplyGaussianSmoothing(h_Ixx, h_Iyy, h_Ixy, h_Sxx, h_Syy, h_Sxy, width, height, filterSize);

		cudaFreeHost(h_Ixx);
		cudaFreeHost(h_Iyy);
		cudaFreeHost(h_Ixy);

		float* h_harrisResponse;
		cudaMallocHost(&h_harrisResponse, sizeof(float) * width * height);

		CalculateHarrisResponse(h_Sxx, h_Syy, h_Sxy, h_harrisResponse, width, height);

		cudaFreeHost(h_Sxx);
		cudaFreeHost(h_Syy);
		cudaFreeHost(h_Sxy);

		PixelCoord* h_corners;
		unsigned int h_cornerCount = 0;
		cudaMallocHost(&h_corners, sizeof(PixelCoord) * width * height);
		
		FindCorners(h_harrisResponse, width, height, h_corners, h_cornerCount);

		cv::Mat output = imgOrig.clone();

		for (unsigned int i = 0; i < h_cornerCount; ++i)
		{
			const auto& c = h_corners[i];
			cv::circle(output, cv::Point(c.x, c.y), 2, cv::Scalar(0, 0, 255), 1);
		}

		cv::imshow("Corners", output);
		cv::waitKey(0);
	}
}