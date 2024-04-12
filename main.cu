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
    std::vector<float> bData{1.5f, 2.0f, 3.0f,
                             6.0f, 7.0f, 8.0f,
                             -6.0f, 7.0f, 8.0f,
                             11.0f, 112.45f, 13.0f};
    DeviceMatrix<float> B(&context, k, bData, MatrixStorageMode::rowMajor);
    std::cout << " B = " << B;
    SvdFactoriser<float> svdEngine(&context, B);
    svdEngine.factorise();

    DeviceVector<float> *s = svdEngine.singularValues();
    std::cout << "S = " << *s;
    DeviceMatrix<float> *vtr = svdEngine.rightSingularVectors();
    std::cout << "V' = " << *vtr;
    return 0;
}
