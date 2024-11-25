#include <random>
#include <algorithm>
#include <iterator>
#include <vector>
#include <iostream>
#include <cublas_v2.h>
#include "include/tensor.cuh"
#include <memory>

#define real_t double

int main() {

    std::vector<real_t> aData = {10.0, 2.0, 3.0,
                            2.0, 20.0, -1.0,
                            3.0, -1.0, 30.0};
    DTensor<real_t> A(3, 3, 2);
    DTensor<real_t> A0(A, 2, 0, 0);
    DTensor<real_t> A1(A, 2, 1, 1);
    A0.upload(aData);
    A1.upload(aData);
    CholeskyBatchFactoriser<real_t> chol(A);
    chol.factorise();
    std::cout << chol.status()(0);

    return 0;
}
