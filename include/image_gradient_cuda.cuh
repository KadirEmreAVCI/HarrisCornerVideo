#pragma once
#include <opencv2/core.hpp>
#include <cstddef>

enum class FilterSize
{
	Size3x3 = 3,
	Size5x5 = 5,
	Size7x7 = 7
};

enum class PaddingType
{
	ZeroPadding,
	ReplicatePadding,
	ReflectPadding
};

void ComputeImageGradient(const cv::Mat& img, 
	unsigned char* h_Ix, 
	unsigned char* h_Iy, 
	FilterSize filterSize,
	cv::BorderTypes borderType,
	int width,
	int height);