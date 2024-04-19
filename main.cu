#include <vector>
#include <iostream>
#include <cublas_v2.h>
#include "include/tensor.cuh"

#define real_t double


int main() {
    std::vector<real_t> aData{10.0, -2.0, -3.0,
                         7.0, 20.0, -1.0,
                          1.89, 60.0, -1.6,
                          -4.5, 20.0, -1.1};
    DTensor<real_t> A(aData, 3, 2, 2);
    DTensor<real_t> At = A.tr();
    std::cout << A;
    std::cout << At;
    return 0;
}
