#include <opencv2/opencv.hpp>
#include <iostream>
#include "grayscale_cuda.cuh"

int main()
{
	cv::Mat img = cv::imread("C:/Users/User/Workspaces/VisualStudio/VS2026/HarrisCornerVideo/data/Lenna.png");
	if (img.empty())
	{
		std::cout << "Could not read the image: " << std::endl;
	}
	else
	{
		const unsigned width = img.cols;
		const unsigned height = img.rows;
		const unsigned channels = img.channels();

		std::cout << "Image loaded successfully: " << width << "x" << height << std::endl;
		std::cout << "Image channels = " << channels << std::endl;
		
		unsigned char* h_imgGrayscale = new unsigned char[width * height];

		ConvertBGRToGray(img.data, h_imgGrayscale, width, height, channels);
		
		cv::Mat grayImg(height, width, CV_8UC1, h_imgGrayscale);
		cv::imshow("Original", img);
		cv::imshow("Grayscale", grayImg);
		cv::waitKey(0);
		
		delete[] h_imgGrayscale;
	}
}