#!/bin/bash
set -euxo pipefail

tests() {
    # ------------------------------------
    # Run tensor gtests
    # ------------------------------------

    # -- create build files
    cmake -S . -B ./build -Wno-dev

    # -- build files in build folder
    cmake --build ./build

    # -- run tests
    ctest --test-dir ./build/test --output-on-failure

    # -- run compute sanitizer
    cd ./build/test
    mem=$(/usr/local/cuda-12.3/bin/compute-sanitizer --tool memcheck --leak-check=full ./device_test)
    grep "0 errors" <<< "$mem"
    cd ../..

    # ------------------------------------
    # Run example executable
    # ------------------------------------

    # -- create build files
    cd example
    cmake -S . -B ./build -Wno-dev

    # -- build files in build folder
    cmake --build ./build

    # -- run main.cu
    ./build/example_main

    # -- run compute sanitizer
    cd ./build
    mem=$(/usr/local/cuda-12.3/bin/compute-sanitizer --tool memcheck --leak-check=full ./example_main)
    grep "0 errors" <<< "$mem"
}


main() {
    tests
}

main
