#include <vector>
#include <iostream>
#include <cublas_v2.h>
#include "include/tensor.cuh"
#include <memory>

#define real_t double


int main() {
    size_t m = 3;
    size_t n = 4;
//    size_t k = 2;
    std::vector<real_t> mat{1, -2, 3, 4,
                            1, 2, -3, 4,
                            -1, 3, 5, -7};
    DTensor<real_t> mats(m, n, 1);
//    DTensor<real_t> mats1(mats, 2, 0, 0);
//    DTensor<real_t> mats2(mats, 2, 1, 1);
    mats.upload(mat, rowMajor);
//    mats2.upload(mat, rowMajor);

    std::vector<real_t> vec{1, 2, 3, 0};  // padded with zero
    DTensor<real_t> vecs(n, 1, 1);
//    DTensor<real_t> vecs1(vecs, 2, 0, 0);
//    DTensor<real_t> vecs2(vecs, 2, 1, 1);
    vecs.upload(vec, rowMajor);
//    vecs2.upload(vec, rowMajor);

    Nullspace N(mats);
    N.nullspace().project(vecs);
//    std::cout << vecs << "\n";

    return 0;
}
