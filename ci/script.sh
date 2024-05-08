#!/bin/bash
set -euxo pipefail


tests() {
    # Where are we? (A40 or Orin?)
    cpp_version=17 # default
    sm_arch=86 # default
    hwInfoOrin=`lshw | grep Orin` ||
    if [ ! -z "$(hwInfoOrin)" ]; then
      echo "Running on Orin";
      sm_arch=87
      cpp_version=17
    else
      echo "Not running on Orin";
      sm_arch=86
      cpp_version=20
    fi

    # ------------------------------------
    # Run tensor gtests
    # ------------------------------------

    # -- create build files
    cmake -DCPPVERSION=${cpp_version} -DSM_ARCH=${sm_arch} -S . -B ./build -Wno-dev

    # -- build files in build folder
    cmake --build ./build

    # -- run tests
    ctest --test-dir ./build/test --output-on-failure

    if [ ! -z "$(hwInfoOrin)" ]; then
      return;
    fi

    # -- run compute sanitizer
    cd ./build/test
    mem=$(/usr/local/cuda/bin/compute-sanitizer --tool memcheck --leak-check=full ./device_test)
    grep "0 errors" <<< "$mem"
    cd ../..

    # ------------------------------------
    # Run example executable
    # ------------------------------------

    # -- create build files
    cd example
    cmake  -DCPPVERSION=${cpp_version} -DSM_ARCH=${sm_arch} -S . -B ./build -Wno-dev

    # -- build files in build folder
    cmake --build ./build

    # -- run main.cu
    ./build/example_main

    # -- run compute sanitizer
    cd ./build
    mem=$(/usr/local/cuda/bin/compute-sanitizer --tool memcheck --leak-check=full ./example_main)
    grep "0 errors" <<< "$mem"
}


main() {
    tests
}

main
