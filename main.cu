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

    auto a = DTensor<double>(v, m, n, 1);
    auto b = DTensor<double>(a);
    size_t i_givens = 1, j_givens = 9;
    double c = 0.1;
    double s = 0.9;
    a.applyLeftGivensRotation(i_givens, j_givens, c, s);

    std::cout << a;


    return 0;
}
