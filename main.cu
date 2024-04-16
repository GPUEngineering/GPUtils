#include <vector>
#include <iostream>
#include <cublas_v2.h>
#include "include/device.cuh"

#define real_t double


int main() {
    Context context;

    size_t k = 6;
    std::vector<real_t> mat{1.0, 2.0, 3.0,
                            6.0, 7.0, 8.0,
                            9.0, 10.0, 11.0,
                            12.0, 13.0, 14.0,
                            15.0, 16.0, 17.0,
                            18.0, 19.0, 20.0};
    DeviceMatrix<real_t> A1(context, k, mat, MatrixStorageMode::rowMajor);
    DeviceMatrix<real_t> A2(A1);

    std::vector<real_t> vec{1., 2., 3., 4., 5., 6.};
    DeviceVector<real_t> b1(context, vec);
    DeviceVector<real_t> b2(b1);

    DeviceTensor<real_t> myMats(context, 6, 3, 2);
    myMats.pushBack(A1);
    myMats.pushBack(A2);
    myMats.upload();

    DeviceTensor<real_t> myVecs(context, 6, 1, 2);
    myVecs.pushBack(b1);
    myVecs.pushBack(b2);
    myVecs.upload();

    //------------------------------

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
