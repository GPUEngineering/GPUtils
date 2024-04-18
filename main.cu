#include <vector>
#include <iostream>
#include <cublas_v2.h>
#include "include/gputils.cuh"
#include "include/tensor.cuh"

#define real_t double


int main() {

    std::vector<real_t> tData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6};
    Tenzor<real_t> tenz(2, 3, 2);
    tenz.upload(tData);

    std::cout << tenz << std::endl;
    std::vector<real_t> v;
    tenz.download(v);
    std::cout << v[1] << std::endl;

    return 0;
}
