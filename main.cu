#include <vector>
#include <iostream>
#include <cublas_v2.h>
#include "include/device.cuh"

#define real_t double


int main() {
    Context context;
    std::vector<float> od{1,1,1,3,4};
    DeviceVector<float> o(context, od);
    DeviceMatrix<float> p(context, o);
    std::cout << "p = " << p << std::endl;
    auto pt = p.tr();
    auto pp = p * pt;
    std::cout << pp;

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

    std::cout << "We have " << T.numMatrices() << " matrices\n";

    DeviceVector<real_t *> rawDataDevice = T.devicePointersToMatrices();
    std::cout << rawDataDevice;
    return 0;
}
