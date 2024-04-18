#include <tensor.cuh>


size_t rows = 3;  // Number of rows of matrix A
size_t cols = 2;  // Number of cols of matrix A
std::vector<float> A = {1., 2.,
                        3., 4.,
                        5., 6.};  // Matrix A in row-major storage
std::vector<float> b = {7., 8.};  // Vector b
DTensor<float> d_A(A, rows, cols);  // Matrix A living on your device (provided in row-major)
DTensor<float> d_b(b, cols);  // Vector b living on your device
