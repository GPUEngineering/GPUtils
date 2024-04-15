#!/bin/bash
set -euxo pipefail

tests() {
    # Run CUDA/C++ gtests
    # ------------------------------------

    # -- create build files
    cmake -S . -B ./build -Wno-dev

    # -- build files in build folder
    cmake --build ./build

    # -- run tests
    ctest --test-dir ./build/test --output-on-failure

    # -- run compute sanitizer
    cd ./build/test
    /usr/local/cuda-12.3/bin/compute-sanitizer --tool memcheck --leak-check=full ./device_test
}


main() {
    tests
}

main
