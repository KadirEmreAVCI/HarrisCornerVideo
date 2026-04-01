#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

struct PixelCoord
{
    int x;
    int y;
};

void FindCorners(const float* h_harrisResponse, int width, int height, PixelCoord* h_corners, unsigned int& h_cornerCount);