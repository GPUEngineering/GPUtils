#include <vector>
#include <iostream>
#include <memory>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "include/device_vector.cuh"
#include <iomanip>


int main() {
    Context context;

    size_t k = 4;
    std::vector<float> bData{1.0f, 2.0f, 3.0f,
                             6.0f, 7.0f, 8.0f,
                             6.0f, 7.0f, 8.0f,
                             6.0f, 7.0f, 8.0f,};
    DeviceMatrix<float> B(&context, k, bData, MatrixStorageMode::rowMajor);
    SvdFactoriser<float> svdEngine(&context, B, true, false);
    svdEngine.factorise();

    /* ~~~ print results ~~~ */
    std::cout << " B = " << B;
    std::cout << "S = " << *svdEngine.singularValues();
    std::cout << "V' = " << *svdEngine.rightSingularVectors();
    auto U = svdEngine.leftSingularVectors();
    if (U) std::cout << "U = " << *U;
    std::cout << " B = " << B;
    return 0;
}
