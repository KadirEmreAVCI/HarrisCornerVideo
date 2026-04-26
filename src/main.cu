#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>

#include "grayscale_cuda.cuh"
#include "image_gradient_cuda.cuh"
#include "second_moment_matrix_cuda.cuh"
#include "gaussian_smoothing_cuda.cuh"
#include "harris_response_cuda.cuh"
#include "finding_corners_cuda.cuh"

constexpr FilterSize filterSize = FilterSize::Size3x3;

int main()
{
    const std::string inputVideoPath =
        "C:/Users/User/Workspaces/VisualStudio/VS2026/HarrisCornerVideo/data/avengers_input_480p.mp4";
    const std::string outputVideoPath =
        "C:/Users/User/Workspaces/VisualStudio/VS2026/HarrisCornerVideo/output/avengers_output_480p.mp4";

    cv::VideoCapture cap(inputVideoPath);
    if (!cap.isOpened())
    {
        std::cerr << "Could not open input video: " << inputVideoPath << std::endl;
        return -1;
    }

    const int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    if (fps <= 0.0)
        fps = 30.0;

    const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    cv::VideoWriter writer(outputVideoPath, fourcc, fps, cv::Size(width, height));

    if (!writer.isOpened())
    {
        std::cerr << "Could not open output video: " << outputVideoPath << std::endl;
        return -1;
    }

    std::cout << "Video opened successfully: " << width << "x" << height
        << "  fps=" << fps << std::endl;

    // Reusable pinned host buffers
    float* h_Ix = nullptr;
    float* h_Iy = nullptr;
    float* h_Ixx = nullptr;
    float* h_Iyy = nullptr;
    float* h_Ixy = nullptr;
    float* h_Sxx = nullptr;
    float* h_Syy = nullptr;
    float* h_Sxy = nullptr;
    float* h_harrisResponse = nullptr;
    PixelCoord* h_corners = nullptr;

    const size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height);

    cudaMallocHost(&h_Ix, sizeof(float) * pixelCount);
    cudaMallocHost(&h_Iy, sizeof(float) * pixelCount);

    cudaMallocHost(&h_Ixx, sizeof(float) * pixelCount);
    cudaMallocHost(&h_Iyy, sizeof(float) * pixelCount);
    cudaMallocHost(&h_Ixy, sizeof(float) * pixelCount);

    cudaMallocHost(&h_Sxx, sizeof(float) * pixelCount);
    cudaMallocHost(&h_Syy, sizeof(float) * pixelCount);
    cudaMallocHost(&h_Sxy, sizeof(float) * pixelCount);

    cudaMallocHost(&h_harrisResponse, sizeof(float) * pixelCount);
    cudaMallocHost(&h_corners, sizeof(PixelCoord) * pixelCount);

    cv::Mat frame;
    cv::Mat grayFrame;
    cv::Mat outputFrame;

    unsigned int frameIndex = 0;

    while (cap.read(frame))
    {
        if (frame.empty())
            break;

        if (frame.cols != width || frame.rows != height)
        {
            std::cerr << "Frame size changed during video stream. Unsupported." << std::endl;
            break;
        }

        const unsigned channels = frame.channels();
        unsigned int h_cornerCount = 0;

        ConvertBGRToGray(frame, grayFrame, width, height, channels);

        ComputeImageGradients(grayFrame, h_Ix, h_Iy, filterSize, cv::BORDER_REPLICATE, width, height);

        ComputeSecondMomentMatrix(h_Ix, h_Iy, h_Ixx, h_Iyy, h_Ixy, width, height);

        ApplyGaussianSmoothing(h_Ixx, h_Iyy, h_Ixy, h_Sxx, h_Syy, h_Sxy, width, height, filterSize);

        CalculateHarrisResponse(h_Sxx, h_Syy, h_Sxy, h_harrisResponse, width, height);

        FindCorners(h_harrisResponse, width, height, h_corners, h_cornerCount);

        outputFrame = frame.clone();

        for (unsigned int i = 0; i < h_cornerCount; ++i)
        {
            const PixelCoord& c = h_corners[i];
            cv::circle(outputFrame, cv::Point(c.x, c.y), 2, cv::Scalar(0, 0, 255), 1);
        }

        writer.write(outputFrame);

        cv::imshow("Harris Corners Video", outputFrame);
        const int key = cv::waitKey(1);
        if (key == 27) // ESC
            break;

        ++frameIndex;
        if (frameIndex % 30 == 0)
        {
            std::cout << "Processed frame count: " << frameIndex << std::endl;
        }
    }

    cudaFreeHost(h_Ix);
    cudaFreeHost(h_Iy);

    cudaFreeHost(h_Ixx);
    cudaFreeHost(h_Iyy);
    cudaFreeHost(h_Ixy);

    cudaFreeHost(h_Sxx);
    cudaFreeHost(h_Syy);
    cudaFreeHost(h_Sxy);

    cudaFreeHost(h_harrisResponse);
    cudaFreeHost(h_corners);

    cap.release();
    writer.release();
    cv::destroyAllWindows();

    std::cout << "Finished. Output saved to: " << outputVideoPath << std::endl;
    return 0;
}