#include <vector>
#include <iostream>
#include <cublas_v2.h>
#include "include/device.cuh"

#define real_t double


int main() {
    Context context;

    size_t j = 3;
    std::vector<real_t> bbData{10.0, 2.0, 1.0,
                               2.0, 70.0, 1.5,
                               1.0, 1.5, 11.0};
    DeviceMatrix<real_t> B(context, j, bbData, MatrixStorageMode::rowMajor);
    CholeskyFactoriser<real_t> choleskiser(context, B);
    choleskiser.factorise();
    std::cout << B;

    std::vector<real_t> bData = {-1.0, -2.0, 10.0};
    DeviceVector<real_t> b(context, bData);
    std::cout << "status = " <<  choleskiser.solve(b) << std::endl << std::endl;
    std::cout << b;
    return 0;
}
