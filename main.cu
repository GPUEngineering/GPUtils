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

    size_t m = 10;
    size_t n = 6;
    std::vector<double> v(m*n);
    v.reserve(m*n);
    std::iota(v.begin(), v.end(), 1);
    DTensor<double> a = DTensor<double>(v, m, n);

    auto ga = GivensAnnihilator<double>(a);
    ga.annihilate(0, 1, 2);

    std::cout << a;


    return 0;
}
