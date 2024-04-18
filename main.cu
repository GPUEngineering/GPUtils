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

//    std::cout << tenz << std::endl;
    std::vector<real_t> v;
    tenz.download(v);
//    std::cout << v[1] << std::endl;

    Tenzor<real_t> other(2, 3, 2);
    tenz.deviceCopyTo(other);
//    std::cout << other << std::endl;

    other *= 10.;
//    std::cout << other << std::endl;

    tenz += other;
//    std::cout << tenz << std::endl;

    other *= 0.24;
    tenz -= other;


    Tenzor<real_t> y;
    y = tenz;
    std::cout << y;

    Tenzor<real_t> z(y);
    std::cout << z;

    std::cout << z.normF();
    return 0;
}
