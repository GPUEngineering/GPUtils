#include <vector>
#include <iostream>
#include <cublas_v2.h>
#include "include/tensor.cuh"
#include <memory>

#define real_t double


int main() {
    size_t m = 3;
    size_t n = 4;
    std::vector<real_t> mat1{1, 1, 1,
                             3, 2, 2,
                             5, 3, 3,
                             7, 3, 3};
    std::vector<real_t> mat2{1, 8, 8,
                             3.5, 2.1, 2.1,
                             -4, 9.4, 9.4,
                             6.3, 5.5, 5.5};
    DTensor<real_t> mats(m, n, 2);
    DTensor<real_t> mats1(mats, 2, 0, 0);
    DTensor<real_t> mats2(mats, 2, 1, 1);
    mats1.upload(mat1, rowMajor);
    mats2.upload(mat2, rowMajor);
    std::cout << mats << "\n";

    std::vector<real_t> vec1{1, 3.4, -2.1, 0};
    std::vector<real_t> vec2{2.2, -3.3, 11.0, 0};
    DTensor<real_t> vecs(m+1, 1, 2);
    DTensor<real_t> vecs1(vecs, 2, 0, 0);
    DTensor<real_t> vecs2(vecs, 2, 1, 1);
    vecs1.upload(vec1, rowMajor);
    vecs2.upload(vec2, rowMajor);
    std::cout << vecs << "\n";

    Nullspace N(mats);
    N.nullspace().project(vecs);
    std::cout << vecs << "\n";

    return 0;
}
