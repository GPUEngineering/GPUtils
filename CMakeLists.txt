cmake_minimum_required(VERSION 3.22)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.3/bin/nvcc")

project(bigBoyConnexion LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++20)
set(CMAKE_CUDA_FLAGS "-std=c++20")



add_executable(bigBoyConnexion main.cu)
target_link_libraries(bigBoyConnexion -lcublas -lcurand -lcusolver)

add_custom_target(run
        COMMAND ./bigBoyConnexion
        DEPENDS bigBoyConnexion
        WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
        )
