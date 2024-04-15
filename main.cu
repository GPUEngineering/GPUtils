#include <vector>
#include <iostream>
#include <cublas_v2.h>
#include "include/device.cuh"

#define real_t double


int main() {
    Context context;

    size_t k = 8;
    std::vector<real_t> bData{1.0, 2.0, 3.0,
                              6.0, 7.0, 8.0,
                              6.0, 7.0, 8.0,
                              6.0, 7.0, 8.0,
                              6.0, 7.0, 8.0,
                              6.0, 7.0, 8.0,
                              6.0, 7.0, 8.0,
                              6.0, 7.0, 8.0,};
    DeviceMatrix<real_t> B(context, k, bData, MatrixStorageMode::rowMajor);
    SvdFactoriser<real_t> svdEngine(context, B, true, false);
    std::cout << "status = " << svdEngine.factorise() << std::endl;

    /* ~~~ print results ~~~ */
    std::cout << "B = " << B;
    std::cout << "S = " << svdEngine.singularValues();
    std::cout << "V' = " << svdEngine.rightSingularVectors();
    auto U = svdEngine.leftSingularVectors();
    if (U) std::cout << "U = " << U.value();
    std::cout << "rank B = " << svdEngine.rank() << std::endl;

    size_t rows = 2;
    std::vector<real_t> aVals = {1., 2., 3., 4., 5., 6.};
    std::vector<real_t> xVals = {1., 2., 3.};
    DeviceMatrix<real_t> A(context, rows, aVals, MatrixStorageMode::rowMajor);
    DeviceVector<real_t> x(context, xVals);

    auto b = A * x;
    std::cout << "b = " << b;


    return 0;
}
