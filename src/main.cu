#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>

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
		const dim3 threadsPerBlock(16, 16);	
		const dim3 numBlocks(
			((img.cols + threadsPerBlock.x - 1) / threadsPerBlock.x),
			((img.rows + threadsPerBlock.y - 1) / threadsPerBlock.y));
	}
}