# ====================================================================
# GPUtils
# ====================================================================
cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
cmake_policy(SET CMP0135 NEW)
project(GPUtils
        DESCRIPTION "Easy use of vectors and matrices on GPGPU devices."
        HOMEPAGE_URL "https://github.com/GPUEngineering/GPUtils"
        LANGUAGES CXX
        )


option(CPPVERSION "C++ version" 20) # A40: 20, Orin: 17
option(SM_ARCH "SM architecture" 86) # A40: 86, Orin: 87

set (cppversion ${CPPVERSION})
set (cppstd "c++${CPPVERSION}")
set (cxxstd cxx_std_${CPPVERSION})

# ----

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)  # required for calling cuda kernels from cuda kernels
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.4/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES ${SM_ARCH})
set(CMAKE_CUDA_STANDARD ${cppversion})
set(CMAKE_CXX_STANDARD ${cppversion})
set(CMAKE_CUDA_FLAGS "-std=${cppstd}")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=${cppstd})
enable_language(CUDA)

# ----

add_library(device_compiler_flags INTERFACE)
target_compile_features(device_compiler_flags INTERFACE ${cxxstd})
set(CMAKE_CXX_EXTENSIONS OFF)

# ----

add_library(developer_flags INTERFACE)
set(cxx_flags -Wall)
set(cuda_flags -arch=sm_${SM_ARCH} -std=${cppstd} -Xcompiler=-Wall -Xcudafe=--display_error_number -g)
target_compile_options(developer_flags
        INTERFACE
        # flags for CXX builds
        $<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>
        # flags for CUDA builds
        $<$<COMPILE_LANGUAGE:CUDA>:${cuda_flags}>
        )
target_link_libraries(device_compiler_flags INTERFACE $<BUILD_INTERFACE:developer_flags>)

# ----


# ====================================================================
# comment out for release
# ====================================================================
add_executable(main)
target_sources(main
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
add_subdirectory(test)
# ----
