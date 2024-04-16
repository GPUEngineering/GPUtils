#include <vector>
#include <iostream>
#include <cublas_v2.h>
#include "include/device.cuh"

#define real_t double


int main() {
    Context context;
    size_t nRows = 2, nCols = 3, nMats = 2;
    std::vector<real_t> aData = {1.0, 2.0, 3.0,
                                 6.0, 7.0, 8.0};
    DeviceMatrix<real_t> A(context, nRows, aData, rowMajor);

    std::vector<real_t> bData = {-1.0, -2.0, -3.0,
                                 25.0, -7.0, -8.0};
    DeviceMatrix<real_t> B(context, nRows, bData, rowMajor);

    CoolTensor<real_t> T(context, nRows, nCols, nMats);
    T.pushBack(A);
    T.pushBack(B);
    std::cout << T;

    std::vector<real_t*> rawData = T.raw();
    for (real_t *p : rawData) {
        std::cout << "::" << p << "::\n";
    }

    return 0;
}
