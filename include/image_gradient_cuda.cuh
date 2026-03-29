#pragma once
#include <opencv2/core.hpp>
#include <cstddef>

enum class FilterSize
{
	Size3x3 = 3,
	Size5x5 = 5
};

enum class PaddingType
{
	ZeroPadding,
	ReplicatePadding,
	ReflectPadding
};

void ComputeImageGradients(const cv::Mat& img,
	float* h_Ix,
	float* h_Iy,
	FilterSize filterSize,
	cv::BorderTypes borderType,
	int width,
	int height);