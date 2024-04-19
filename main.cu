#include <vector>
#include <iostream>
#include <cublas_v2.h>
#include "include/tensor.cuh"
#include <memory>

#define real_t double


int main() {
    std::vector<real_t> aData{1, 2, 3, 4, 5, 6, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 1};
    DTensor<real_t> A(aData, 3, 2, 3);
    Svd<real_t> svd(A, true);
    svd.factorise();
    DTensor<real_t> S = svd.singularValues();
    std::cout << S;
//    std::cout << svd.leftSingularVectors().value();
    return 0;
}
