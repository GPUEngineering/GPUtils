#include <gputils.cuh>


size_t rows = 3;  // Number of rows of matrix A
std::vector<float> A = {1., 2.,
                        3., 4.,
                        5., 6.};  // Matrix A in row-major storage
std::vector<float> b = {7., 8.};  // Vector b
DeviceMatrix<float> d_A(rows, A, rowMajor);  // Matrix A living on your device (provided in row-major)
DeviceVector<float> d_b(b);  // Vector b living on your device
