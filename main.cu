#include <vector>
#include <iostream>
#include <cublas_v2.h>
#include "include/tensor.cuh"
#include <memory>

#define real_t double


int main() {
    size_t m = 4;
    size_t n = 3;
    size_t k = 2;
    std::vector<real_t> data{1, 2, 3,
                             4, 5, 6,
                             7, 8, 9,
                             10, 11, 12,
                             13, 14, 15,
                             16, 17, 18,
                             19, 20, 21,
                             22, 23, 24};
    DTensor<real_t> A(data, m, n, k, rowMajor);
    std::cout << A << "\n";

    return 0;
}
