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

    auto a = DTensor<double>::createRandomTensor(10, 6, 1, -2, 2);
    auto b = DTensor<double>(a);
    a.applyRightGivensRotation(2, 4, 0.5, 0.5);

    auto c = a - b;
    std::cout << c;


    return 0;
}
