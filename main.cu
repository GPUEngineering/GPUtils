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
//    cudaStream_t stream1;
//    cudaStreamCreate(&stream1);
//    cublasSetStream(Session::getInstance().cuBlasHandle(), stream1);

    cudaStream_t s1;
    cudaStreamCreate(&s1);

    auto a = DTensor<float>::createRandomTensor(2000, 200, 1, -2, 2);
    Svd svd(a);
    svd.factorise();

    std::cout << svd.singularValues();


    return 0;
}
