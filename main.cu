#include <vector>
#include <iostream>
#include <cublas_v2.h>
#include "include/gputils.cuh"
#include "include/tensor.cuh"

#define real_t double


int main() {

    std::vector<real_t> tData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 10};
    Tenzor<real_t> tenz(2, 3, 2);
    tenz.upload(tData);


    Tenzor<real_t> r(tenz, 1, 0, 1); // r = [:, :, 0:0]
    std::cout << r;

    Tenzor<real_t> f(tenz, 0, 0, 3); // r = [:, :, 0:0]
    std::cout << f;
    return 0;
}
