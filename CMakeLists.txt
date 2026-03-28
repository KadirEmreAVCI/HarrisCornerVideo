cmake_minimum_required(VERSION 3.24)

project(HarrisCornerVideo LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(HarrisCornerVideo
    src/main.cpp
    src/grayscale_cuda.cu
)

target_include_directories(HarrisCornerVideo PRIVATE
    include
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(HarrisCornerVideo PRIVATE
    ${OpenCV_LIBS}
)

set_target_properties(HarrisCornerVideo PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 89
)