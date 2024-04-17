# ---- cmake for CUDA with googleTest
cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(GPUtils
        DESCRIPTION "Easy use of vectors and matrices on GPGPU devices."
        HOMEPAGE_URL "https://github.com/GPUEngineering/GPUtils"
        LANGUAGES CXX
)
# ----
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)  # required for calling cuda kernels from cuda kernels
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.3/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES 86)
enable_language(CUDA)
# ----
add_library(device_compiler_flags INTERFACE)
target_compile_features(device_compiler_flags
        INTERFACE
        cxx_std_20
)
set(CMAKE_CXX_EXTENSIONS OFF)
# ----
add_library(developer_flags INTERFACE)
set(cxx_flags -Wall)
set(cuda_flags -arch=sm_60 -std=c++20 -Xcompiler=-Wall -Xcudafe=--display_error_number)
target_compile_options(developer_flags
        INTERFACE
        # flags for CXX builds
        $<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>
        # flags for CUDA builds
        $<$<COMPILE_LANGUAGE:CUDA>:${cuda_flags}>
)
target_link_libraries(device_compiler_flags
        INTERFACE
        $<BUILD_INTERFACE:developer_flags>
)
# ----
add_subdirectory(test)  # comment out to not build tests
# ----
add_executable(main)
target_sources(main  # add files
        PRIVATE
        main.cu
)
target_link_libraries(main
        PRIVATE
        device_compiler_flags
        cublas
        cusolver
        cudadevrt
)
target_include_directories(main
        PRIVATE
        "${PROJECT_BINARY_DIR}"
        "${PROJECT_SOURCE_DIR}/include"
)
# ----
