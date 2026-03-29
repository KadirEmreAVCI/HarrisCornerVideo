#pragma once

void ComputeSecondMomentMatrix(	const float* h_Ix, 
								const float* h_Iy, 
								float* h_Ixx, 
								float* h_Iyy, 
								float* h_Ixy, 
								int width, 
								int height);