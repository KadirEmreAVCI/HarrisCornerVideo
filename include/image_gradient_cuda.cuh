#pragma once
#include <opencv2/core.hpp>
#include <cstddef>
#include "utils.h"

void ComputeImageGradients(const cv::Mat& img,
	float* h_Ix,
	float* h_Iy,
	FilterSize filterSize,
	cv::BorderTypes borderType,
	int width,
	int height);