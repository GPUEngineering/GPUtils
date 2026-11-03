cmake_minimum_required(VERSION 3.20)
project(GPUtilsBenchmarks LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
add_executable(gputils_benchmark benchmark.cu)
target_compile_features(gputils_benchmark PRIVATE cxx_std_17 cuda_std_17)
target_include_directories(gputils_benchmark PRIVATE ../include)
target_link_libraries(gputils_benchmark PRIVATE CUDA::cublas CUDA::cusolver)
target_compile_definitions(gputils_benchmark PRIVATE
    GPUTILS_BENCH_CONFIG="$<CONFIG>")
