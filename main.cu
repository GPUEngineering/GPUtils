#include <vector>
#include <iostream>
#include <cublas_v2.h>
#include "include/tensor.cuh"
#include <memory>

#define real_t double


int main() {
    size_t m = 3;
    size_t n = 7;
    std::vector<real_t> mat{1, -2, 3, 4, -1, -1, -1,
                            1, 2, -3, 4, -1, -1, -1,
                            -1, 3, 5, -7, -1, -1, -1};
    DTensor mats(m, n, 1);
    mats.upload(mat, rowMajor);

    Nullspace ns = Nullspace(mats);

    std::vector<real_t> vec{1, 2, 3, 4, 5, 6, 7};
    DTensor vecs(vec, n);

    ns.project(vecs);

    std::cout << mats << "\n";
    std::cout << vecs << "\n";

    DTensor op(m, 1, 1);
    op.addAB(mats, vecs);
    std::cout << op << "\n";

    return 0;
}
