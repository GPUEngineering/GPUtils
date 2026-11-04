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
add_executable(gputils_test)
target_sources(gputils_test  # add files
        PRIVATE
        testTensor.cu
        testFileIO.cu
        hostAllocationFailure.cpp
        sessionAcrossTranslationUnits.cu
)
target_link_libraries(gputils_test
    PRIVATE
    gputils_compiler_flags
    cublas
    cusolver
    cudadevrt
    GTest::gtest_main)
target_include_directories(gputils_test
    PRIVATE
    "${PROJECT_BINARY_DIR}"
    "${PROJECT_SOURCE_DIR}/include"
)
include(GoogleTest)
gtest_discover_tests(gputils_test)

add_executable(gputils_file_exchange exchangeTensorFiles.cu)
target_link_libraries(gputils_file_exchange PRIVATE gputils_compiler_flags cublas cusolver cudadevrt)
target_include_directories(gputils_file_exchange PRIVATE "${PROJECT_SOURCE_DIR}/include")
# ----
