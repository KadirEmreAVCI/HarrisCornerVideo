#pragma once
#include <cuda_runtime.h>

void CalculateHarrisResponse(const float* Sxx, const float* Syy, const float* Sxy, float* harrisResponse, int width, int height, cudaStream_t* streams, int nStreams);
