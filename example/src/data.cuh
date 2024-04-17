#include <device.cuh>


Context context;  // Create only one context for the project!
size_t rows = 3;  // Number of rows of matrix A
std::vector<float> A = {1., 2.,
                        3., 4.,
                        5., 6.};  // Matrix A in row-major storage
std::vector<float> b = {7., 8.};  // Vector b
DeviceMatrix<float> d_A(context, rows, A, MatrixStorageMode::rowMajor);  // Matrix A living on your device
DeviceVector<float> d_b(context, b);  // Vector b living on your device
