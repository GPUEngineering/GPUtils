#!/bin/bash
set -euxo pipefail


tests() {
    # Where are we? (A40 or Orin?)
    cpp_version=17 # default
    sm_arch=86 # default
    hwInfoOrin=`lshw | grep Orin` ||
    if [ -n "${hwInfoOrin}" ]; then
      echo "Running on Orin";
      sm_arch=87
      cpp_version=17
    else
      echo "Not running on Orin";
      sm_arch=86
      cpp_version=20
    fi


    # ------------------------------------
    # Run Python tests first
    # ------------------------------------
    pushd python
    export PYTHONPATH=.
    which python
    python -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install .
    python -W ignore test/test.py -v
    deactivate
    popd

    # ------------------------------------
    # Run tensor gtests
    # ------------------------------------

    # -- create build files
    cmake -DCPPVERSION=${cpp_version} -DSM_ARCH=${sm_arch} -DGPUTILS_BUILD_TEST=ON -S . -B ./build -Wno-dev

    # -- build files in build folder
    cmake --build ./build

    # -- run tests
    ctest --test-dir ./build/test --output-on-failure

    if [ -z "${hwInfoOrin}" ]; then

      # -- run compute sanitizer
      pushd ./build/test
      mem=$(/usr/local/cuda/bin/compute-sanitizer --tool memcheck --leak-check=full ./gputils_test)
      grep "0 errors" <<< "$mem"
      popd

      # ------------------------------------
      # Run example executable
      # ------------------------------------

      # -- create build files
      cd example
      cmake -DCPPVERSION=${cpp_version} -DSM_ARCH=${sm_arch} -S . -B ./build -Wno-dev

      # -- build files in build folder
      cmake --build ./build

      # -- run main.cu
      ./build/example_main

      # -- run compute sanitizer
      cd ./build
      mem=$(/usr/local/cuda/bin/compute-sanitizer --tool memcheck --leak-check=full ./example_main)
      grep "0 errors" <<< "$mem"
    fi
}


main() {
    tests
}

main
