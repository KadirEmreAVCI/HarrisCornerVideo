#include "gaussian_smoothing_cuda.cuh"
#include <opencv2/opencv.hpp>

void ApplyPaddingToMatrixComponent(const float* input, cv::Mat& imgOutput, int width, int height, cv::BorderTypes borderType, int padSize)
{
	cv::Mat imgInput(height, width, CV_32F, const_cast<float*>(input));
	cv::copyMakeBorder(imgInput, imgOutput, padSize, padSize, padSize, padSize, borderType);
	CV_Assert(imgOutput.type() == CV_32F && imgOutput.isContinuous());
}

void ApplyGaussianSmoothing(const float* h_Ixx, const float* h_Iyy, const float* h_Ixy, float* h_Sxx, float* h_Syy, float* h_Sxy, int width, int height, FilterSize filterSize)
{
	cv::Mat imgIxxPadded, imgIyyPadded, imgIxyPadded;
	const int padSize = static_cast<int>(filterSize) / 2;

	ApplyPaddingToMatrixComponent(h_Ixx, imgIxxPadded, width, height, cv::BORDER_REPLICATE, padSize);
	ApplyPaddingToMatrixComponent(h_Iyy, imgIyyPadded, width, height, cv::BORDER_REPLICATE, padSize);
	ApplyPaddingToMatrixComponent(h_Ixy, imgIxyPadded, width, height, cv::BORDER_REPLICATE, padSize);

	float* const h_IxxPadded = imgIxxPadded.ptr<float>();
	float* const h_IyyPadded = imgIyyPadded.ptr<float>();
	float* const h_IxyPadded = imgIxyPadded.ptr<float>();
}