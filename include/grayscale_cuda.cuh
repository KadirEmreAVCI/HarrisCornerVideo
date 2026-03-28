#pragma once

#include <cstddef>

void ConvertBGRToGray(const unsigned char* h_inputBGR,
    unsigned char* h_outputGray,
    int width,
    int height,
    int channels);