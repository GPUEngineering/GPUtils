#include <vector>
#include <iostream>
#include <cublas_v2.h>
#include "include/device.cuh"

#define real_t float


int main() {
    Context context;

    size_t kk = 6;
    std::vector<real_t> mat{1.0, 2.0, 3.0,
                            6.0, 7.0, 8.0,
                            9.0, 10.0, 11.0,
                            12.0, 13.0, 14.0,
                            15.0, 16.0, 17.0,
                            18.0, 19.0, 20.0};
    DeviceMatrix<real_t> A1(context, kk, mat, MatrixStorageMode::rowMajor);
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

    return 0;
}
