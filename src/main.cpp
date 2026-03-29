#include <opencv2/opencv.hpp>
#include <iostream>
#include "grayscale_cuda.cuh"
#include "image_gradient_cuda.cuh"
#include "second_moment_matrix.cuh"

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

		float* const h_Ix = new float[width * height];
		float* const h_Iy = new float[width * height];
		
		ComputeImageGradients(imgGrayscale, h_Ix, h_Iy, FilterSize::Size3x3, cv::BORDER_REPLICATE, width, height);

		float* const h_Ixx = new float[width * height];
		float* const h_Iyy = new float[width * height];
		float* const h_Ixy = new float[width * height];
		
		ComputeSecondMomentMatrix(h_Ix, h_Iy, h_Ixx, h_Iyy, h_Ixy, width, height);

		delete[] h_Ix;
		delete[] h_Iy;
		delete[] h_Ixx;
		delete[] h_Iyy;
		delete[] h_Ixy;
	}
}