#pragma once
#include "utils.h"
void ApplyGaussianSmoothing(const float* h_Ixx, 
							const float* h_Iyy, 
							const float* h_Ixy, 
							float* h_Sxx, 
							float* h_Syy, 
							float* h_Sxy, 
							int width, 
							int height, 
							FilterSize filterSize);