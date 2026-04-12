#pragma once
#include <opencv2/opencv.hpp>
#include <cstddef>

void ConvertBGRToGray(const cv::Mat& imgBGR,
    cv::Mat& imgGray,
    int width,
    int height,
    int channels);