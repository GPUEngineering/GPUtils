#include <vector>
#include <iostream>
#include <cublas_v2.h>
#include "include/device_vector.cuh"


int main() {
    Context context;

    size_t k = 8;
    std::vector<double> bData{1.0f, 2.0f, 3.0f,
                             6.0f, 7.0f, 8.0f,
                             6.0f, 7.0f, 8.0f,
                             6.0f, 7.0f, 8.0f,
                             6.0f, 7.0f, 8.0f,
                             6.0f, 7.0f, 8.0f,
                             6.0f, 7.0f, 8.0f,
                             6.0f, 7.0f, 8.0f,};
    DeviceMatrix<double> B(&context, k, bData, MatrixStorageMode::rowMajor);
    SvdFactoriser<double> svdEngine(&context, B, true, false);
    std::cout << "status = " << svdEngine.factorise() << std::endl;

    /* ~~~ print results ~~~ */
    std::cout << "B = " << B;
    std::cout << "S = " << svdEngine.singularValues();
    std::cout << "V' = " << svdEngine.rightSingularVectors();
    auto U = svdEngine.leftSingularVectors();
    if (U) std::cout << "U = " << U.value();

    return 0;
}
