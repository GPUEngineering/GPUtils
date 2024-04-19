#include <vector>
#include <iostream>
#include <cublas_v2.h>
#include "include/tensor.cuh"
#include <memory>

#define real_t double


int main() {
    std::vector<real_t> aData{1, 4, 2, 5, 3, 6,
                              1, 4, 2, 5, 3, 7};
    DTensor<real_t> A(aData, 2, 3, 2);

//    std::cout << "A = " << A << " ------ \n";

    Nullspace null(A);

    return 0;
}
