#pragma once
#include <opencv2/opencv.hpp>
#include <cstddef>
#include <cuda_runtime.h>

void ConvertBGRToGray(const cv::Mat& imgBGR,
    cv::Mat& imgGray,
    int width,
    int height,
    int channels,
    cudaStream_t* streams,
    int nStreams);