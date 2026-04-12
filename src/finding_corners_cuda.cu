#include "finding_corners_cuda.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <limits>

__global__ void FindCornersKernel(const float* harrisResponse, PixelCoord* corners, unsigned int* cornerCount, float threshold, int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1)
        return;

    const int idx = y * width + x;
    const float center = harrisResponse[idx];

    if (center <= threshold)
        return;

    for (int j = -1; j <= 1; ++j)
    {
        for (int i = -1; i <= 1; ++i)
        {
            if (i == 0 && j == 0)
                continue;

            const int neighborIdx = (y + j) * width + (x + i);
            if (harrisResponse[neighborIdx] > center)
                return;
        }
    }

    const unsigned int outIdx = atomicAdd(cornerCount, 1u);
    corners[outIdx].x = x;
    corners[outIdx].y = y;
}

static float FindThreshold(const float* harrisResponse, int width, int height, float thresholdRatio)
{
    float maxResponse = std::numeric_limits<float>::lowest();
    for (int i = 0; i < width * height; ++i)
    {
        if (harrisResponse[i] > maxResponse)
            maxResponse = harrisResponse[i];
    }
    return maxResponse * thresholdRatio;
}
void FindCorners(const float* h_harrisResponse, int width, int height, PixelCoord* h_corners, unsigned int& cornerCount)
{
    constexpr float thresholdRatio = 0.01f;
    const float threshold = FindThreshold(h_harrisResponse, width, height, thresholdRatio);

    float* d_harrisResponse;
    PixelCoord* d_corners;
    unsigned int* d_cornerCount;
    cudaMalloc(&d_corners, sizeof(PixelCoord) * width * height);
    cudaMalloc(&d_cornerCount, sizeof(unsigned int));
    cudaMalloc(&d_harrisResponse, sizeof(float) * width * height);
    
    cudaMemcpy(d_harrisResponse, h_harrisResponse, sizeof(float) * width * height, cudaMemcpyHostToDevice);
    cudaMemset(d_cornerCount, 0, sizeof(unsigned int));

    constexpr dim3 block{ 16, 16 };
    const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1)/ block.y);

    FindCornersKernel << <grid, block >> > (d_harrisResponse, d_corners, d_cornerCount, threshold, width, height);

    cudaDeviceSynchronize();

    unsigned int h_cornerCount = 0;
    cudaMemcpy(&h_cornerCount, d_cornerCount, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cornerCount = h_cornerCount;
    if (cornerCount > 0)
    {
        cudaMemcpy(h_corners, d_corners, sizeof(PixelCoord) * cornerCount, cudaMemcpyDeviceToHost);
    }
    
    cudaFree(d_harrisResponse);
    cudaFree(d_corners);
    cudaFree(d_cornerCount);
}