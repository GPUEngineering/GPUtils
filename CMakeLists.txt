# ====================================================================
# GPUtils
# ====================================================================
cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.29")
    cmake_policy(SET CMP0135 NEW)
endif()
# ----
# Set C++ version and SM architecture
if (NOT DEFINED CPPVERSION)
    set(CPPVERSION 20) # A40: 20, Orin: 17
endif()
if (NOT DEFINED SM_ARCH)
    set(SM_ARCH 86)# A40: 86, Orin: 87
endif()
# ----
project(GPUtils
        DESCRIPTION "Easy use of vectors and matrices on GPGPU devices."
        HOMEPAGE_URL "https://github.com/GPUEngineering/GPUtils"
        LANGUAGES CXX
        )
# ----
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)  # required for calling cuda kernels from cuda kernels
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES ${SM_ARCH})
set(CMAKE_CUDA_STANDARD ${CPPVERSION})
set(CMAKE_CXX_STANDARD ${CPPVERSION})
set(CMAKE_CUDA_FLAGS "-std=c++${CPPVERSION}")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; "-std=c++${CPPVERSION}")
enable_language(CUDA)
# ----
add_library(gputils_compiler_flags INTERFACE)
target_compile_features(gputils_compiler_flags INTERFACE cxx_std_${CPPVERSION})
set(CMAKE_CXX_EXTENSIONS OFF)
# ----
add_library(developer_flags INTERFACE)
set(cxx_flags -Wall)
set(cuda_flags -arch=sm_${SM_ARCH} -std=c++${CPPVERSION} -Xcompiler=-Wall -Xcudafe=--display_error_number -g)
target_compile_options(developer_flags
        INTERFACE
        # flags for CXX builds
        $<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>
        # flags for CUDA builds
        $<$<COMPILE_LANGUAGE:CUDA>:${cuda_flags}>
        )
target_link_libraries(gputils_compiler_flags INTERFACE $<BUILD_INTERFACE:developer_flags>)
# ----
add_executable(gputils_main)
target_sources(gputils_main
        PRIVATE
        main.cu
        )
target_link_libraries(gputils_main
        PRIVATE
        gputils_compiler_flags
        cublas
        cusolver
        cudadevrt
        )
target_include_directories(gputils_main
        PRIVATE
        "${PROJECT_BINARY_DIR}"
        "${PROJECT_SOURCE_DIR}/include"
        )
# ----
if(NOT GPUTILS_BUILD_TEST)
    set(GPUTILS_BUILD_TEST OFF)  # Set to ON for local testing (or add `-DGPUTILS_BUILD_TEST=ON` to your CMake profile)
endif()
if (GPUTILS_BUILD_TEST)
    add_subdirectory(test)
endif()
unset(GPUTILS_BUILD_TEST CACHE)
# ----
