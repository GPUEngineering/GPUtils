# ---- cmake for CUDA
cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(yourCoolProject
        DESCRIPTION "An uber cool project for GPGPU devices."
        HOMEPAGE_URL "https://github.com/yourCoolProject"
        LANGUAGES CXX)
# ----
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)  # required for calling cuda kernels from cuda kernels
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.3/bin/nvcc")  # or, wherever your cuda toolkit is stored
set(CMAKE_CUDA_ARCHITECTURES 86)
enable_language(CUDA)
# ----
add_library(example_compiler_flags INTERFACE)
target_compile_features(example_compiler_flags
        INTERFACE
        cxx_std_20
)
set(CMAKE_CXX_EXTENSIONS OFF)
# ----
add_library(example_developer_flags INTERFACE)
set(cxx_flags -Wall)
set(cuda_flags -arch=sm_60 -std=c++20 -Xcompiler=-Wall -Xcudafe=--display_error_number)
target_compile_options(example_developer_flags
        INTERFACE
        # flags for CXX builds
        $<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>
        # flags for CUDA builds
        $<$<COMPILE_LANGUAGE:CUDA>:${cuda_flags}>
)
target_link_libraries(example_compiler_flags
        INTERFACE
        $<BUILD_INTERFACE:example_developer_flags>
)
# ----
include(FetchContent)
FetchContent_Declare(
        gputils
        URL https://github.com/GPUEngineering/GPUtils/archive/refs/tags/v0.0.1.zip
)
FetchContent_MakeAvailable(GPUtils)
FetchContent_GetProperties(
        gputils
        SOURCE_DIR GPUTILS_SOURCE_DIR
)
# ----
add_executable(example_main)
target_sources(example_main
        PRIVATE
        main.cu
)
target_link_libraries(example_main  # links GPUtils dependencies
        PRIVATE
        example_compiler_flags
        cublas
        cusolver
        cudadevrt
)
target_include_directories(example_main
        PRIVATE
        "${GPUTILS_SOURCE_DIR}/include"  # includes GPUtils directory
        "${PROJECT_SOURCE_DIR}/src"
        "${PROJECT_BINARY_DIR}"
)
# ----
