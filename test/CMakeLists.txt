# ----
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
# for Windows: prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
# ----
enable_testing()
add_executable(device_test)

target_sources(device_test  # add files
        PRIVATE
        testTensor.cu
)
target_link_libraries(device_test
    PRIVATE
    device_compiler_flags
    cublas
    cusolver
    cudadevrt
    GTest::gtest_main)
target_include_directories(device_test
    PRIVATE
    "${PROJECT_BINARY_DIR}"
    "${PROJECT_SOURCE_DIR}/include"
)
include(GoogleTest)
gtest_discover_tests(device_test)
# ----
