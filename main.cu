#include <vector>
#include <iostream>
#include <cublas_v2.h>
#include "include/tensor.cuh"

#define real_t double


int main() {
    std::vector<real_t> bData{1.0, 2.0, 3.0,
                         6.0, 7.0, 8.0,
                         6.0, 7.0, 8.0,
                         6.0, 7.0, 8.0,
                         6.0, 7.0, 8.0,
                         6.0, 7.0, 8.0,
                         6.0, 7.0, 8.0,
                         6.0, 7.0, 8.0,};
    Tenzor<real_t> B(bData, 8, 3);
    Svd<real_t> svd(B);
    svd.factorise();
    std::cout << svd.singularValues();
    return 0;
}
