# HarrisCornerVideo

GPU-accelerated Harris corner detector for video using CUDA and OpenCV.

This project reads a video, computes Harris corner responses on each frame using CUDA kernels (grayscale conversion, image gradients, second-moment matrix, Gaussian smoothing, Harris response and corner finding), draws corners on frames and writes an output video.

Repository root:
`C:\Users\User\Workspaces\VisualStudio\VS2026\HarrisCornerVideo`

Default input / output paths (see `src/main.cu`):
- Input: `data/hacettepe_demo_input.mp4`
- Output: `data/hacettepe_demo_output.mp4`

Requirements
- Microsoft Visual Studio Community 2026 (18.4.2) or later with C++ and CUDA development support
- NVIDIA GPU with CUDA drivers compatible with your CUDA Toolkit
- CUDA Toolkit (project references CUDA 13.2 build customizations in the workspace)
- OpenCV (built for the same architecture / toolset as your Visual Studio build)
- CMake >= 3.24

Build (recommended — Visual Studio)
1. Open Visual Studio and choose __Open Folder__ -> select the repository root.
2. Visual Studio will configure the CMake project automatically. Select the CMake target `HarrisCornerVideo` in the top bar and the desired configuration (e.g., Release).
3. Build with __Build__ -> __Build All__ or press __Ctrl+Shift+B__.
4. Run from Visual Studio (select the target and press __Ctrl+F5__) or run the produced executable from the build output.

Build (command line — Developer PowerShell)
1. Open __Developer PowerShell for VS 2026__ (or a terminal with Visual Studio environment variables).
2. Create and enter a build directory: