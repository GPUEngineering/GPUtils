#include <vector>
#include <iostream>
#include <cublas_v2.h>
#include "include/gputils.cuh"
#include "include/tensor.cuh"

#define real_t double


int main() {

    std::vector<real_t> aData = {1, 2, 3, 4, 5, 6,
                                 7, 8, 9, 10, 11, 12,
                                 13, 14, 15, 16, 17, 18};
    std::vector<real_t> bData = {6, 5, 4, 3, 2, 1,
                                 7, 6, 5, 4, 3, 2,
                                 1, 2, 1, 5, -6, 8};
    Tenzor<real_t> A(aData, 2, 3, 3);
    Tenzor<real_t> B(bData, 3, 2, 3);
    Tenzor<real_t> C(2, 2, 3, true);
    C.addAB(A, B);

    std::cout << A;
    std::cout << B;
    std::cout << C;
    return 0;
}
