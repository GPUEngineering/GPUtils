#include <vector>
#include <iostream>
#include <cublas_v2.h>
#include "include/gputils.cuh"

#define real_t double


int main() {

//    Context context;
    std::vector<real_t> a1Data = {1.0, 2.0, 3.0,
                                  6.0, 7.0, 8.0,
                                  6.0, 7.0, 8.0,
                                  6.0, 7.0, 8.0};
    std::vector<real_t> a2Data = {5.0, 2.0, 3.0,
                                  6.0, 7.0, 9.0,
                                  6.0, -7.0, 8.0,
                                  6.0, 0.0, 8.0};
    DeviceMatrix<real_t> A1(4, a1Data, rowMajor);
    DeviceMatrix<real_t> A2(4, a2Data, rowMajor);
    DeviceTensor<real_t> A(4, 3, 2);
    A.pushBack(A1);
    A.pushBack(A2);


    std::vector<real_t> b1Data = {5.0, 12.0,
                                  7.0, 17.0,
                                  11.0, 97.0};
    std::vector<real_t> b2Data = {5.0, 12.0,
                                  7.0, 17.0,
                                  11.0, 97.0};
    DeviceMatrix<real_t> B1(3, b1Data, rowMajor);
    DeviceMatrix<real_t> B2(3, b2Data, rowMajor);
    DeviceTensor<real_t> B(3, 2, 2);
    B.pushBack(B1);
    B.pushBack(B2);

    DeviceMatrix<real_t> C1(4, 2);
    DeviceMatrix<real_t> C2(4, 2);
    DeviceTensor<real_t> C(4, 2, 2);
    C.pushBack(C1);
    C.pushBack(C2);
    C.addAB(A, B);

    return 0;
}
