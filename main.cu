#include <vector>
#include <iostream>
#include <cublas_v2.h>
#include "include/device.cuh"

#define real_t double


int main() {
    Context context;

//    size_t rows = 4;
//    std::vector<real_t> mat{1, 2, 3,
//                       6, 7, 8,
//                       9, 10, 11,
//                       12, 13, 14};
//    DeviceMatrix<real_t> A1(context, rows, mat, MatrixStorageMode::rowMajor);
//    DeviceMatrix<real_t> A2(A1);
//    std::vector<real_t> vec{1, 2, 3, 4, 5, 6};
//    DeviceVector<real_t> b1(context, vec);
//    DeviceVector<real_t> b2(b1);
//    // push DeviceMatrices
//    DeviceTensor<real_t> mats(context, 6, 3, 2);
//    mats.pushBack(A1);
//    mats.pushBack(A2);
//    mats.upload();
//    // push DeviceVectors
//    DeviceTensor<real_t> vecs(context, 6, 1, 2);
//    vecs.pushBack(b1);
//    vecs.pushBack(b2);
//    vecs.upload();
//
//    std::cout << mats << "\n";


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
