#include <vector>
#include <iostream>
#include <cublas_v2.h>
#include "include/device_vector.cuh"
#define real_t float


int main() {
    Context context;

    size_t k = 8;
    std::vector<float> bData{1.0f, 2.0f, 3.0f,
                             6.0f, 7.0f, 8.0f,
                             6.0f, 7.0f, 8.0f,
                             6.0f, 7.0f, 8.0f,
                             6.0f, 7.0f, 8.0f,
                             6.0f, 7.0f, 8.0f,
                             6.0f, 7.0f, 8.0f,
                             6.0f, 7.0f, 8.0f,};

    DeviceMatrix<real_t> B(context, k, bData, MatrixStorageMode::rowMajor);
    SvdFactoriser<real_t> svdEngine(context, B, true, false);
    svdEngine.factorise();

    /* ~~~ print results ~~~ */
    std::cout << "B = " << B;
    std::cout << "S = " << svdEngine.singularValues();
    std::cout << "V' = " << svdEngine.rightSingularVectors();
    auto U = svdEngine.leftSingularVectors();
    if (U) std::cout << "U = " << U.value();

    std::vector<real_t> bVals = {1., 2., 3.};
    DeviceVector<real_t> b(context, bVals);
    real_t scalar = 2.;
    b *= scalar;
    std::cout << "b = " << b;
    return 0;
}
