#include <vector>
#include <iostream>
#include <cublas_v2.h>
#include "include/device.cuh"

#define real_t double


int main() {
    Context context;
    size_t nRows = 2, nCols = 3, nMats = 4;
    std::vector<real_t> aData = {1.0, 2.0, 3.0,
                                 6.0, 7.0, 8.0};
    DeviceMatrix<real_t> A(context, nRows, aData, rowMajor);

    std::vector<real_t> bData = {-1.0, -2.0, -3.0,
                                 25.0, -7.0, -8.0};
    DeviceMatrix<real_t> B(context, nRows, bData, rowMajor);

    DeviceTensor<real_t> T(context, nRows, nCols, nMats);
    T.pushBack(A);
    T.pushBack(B);
    T.pushBack(B);
    T.pushBack(B);
    std::cout << T;

    /* NOTE: A DeviceTensor holds only POINTERS to the actual matrices;
     * For whatever we want to do with the actual matrices, we should
     * manipulate them outside the tensor (e.g., if we want to upload/
     * download data, or apply some matrix-specific method).
     * The tensor should be used for things that require a BUNDLE
     * or matrices, e.g., (Ai, Bi)-products, LS, etc.
     *
     * Moreover, we need:
     * 1. To make Context a global singleton!
     * 2. To make a constructor (shallow) that will allow to cast a
     *    DeviceVector as a DeviceMatrix!
     */

    std::vector<real_t *> rawData = T.raw();
    for (real_t *p: rawData) {
        std::cout << "::" << p << "::\n";
    }

    return 0;
}
