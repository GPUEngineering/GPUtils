#include <vector>
#include <iostream>
#include <cublas_v2.h>
#include "include/tensor.cuh"

#define real_t double


int main() {
    std::vector<real_t> aData{10.0, 2.0, 3.0,
                         2.0, 20.0, -1.0,
                         3.0, -1.0, 30.0};
    DTensor<real_t> A(aData, 3, 3, 1);
    CholeskyFactoriser<real_t> chol(A);
    chol.factorise();
    std::cout << A;
    return 0;
}
