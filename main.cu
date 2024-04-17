#include <vector>
#include <iostream>
#include <cublas_v2.h>
#include "include/gputils.cuh"

#define real_t float


int main() {
    Context context;

    std::vector<real_t> aData = {1.0, 2.0, 3.0,
                                 6.0, 7.0, 8.0,
                                 6.0, 7.0, 8.0,
                                 6.0, 7.0, 8.0,
                                 6.0, 7.0, 8.0,
                                 6.0, 7.0, 8.0,
                                 6.0, 7.0, 8.0,
                                 6.0, 7.0, 8.0,};
    DeviceMatrix<real_t> A(context, 8, aData, rowMajor);
    std::cout << A;
    Nullspace ns(context, A);
    DeviceMatrix<real_t> N = ns.get();
    std::cout << N;


//    DeviceMatrix<real_t> B(context, nRows, bData, rowMajor);
//    DeviceTensor<real_t> LHS(context, nRows, nCols, nMats);
//    LHS.pushBack(A);
//    LHS.pushBack(B);
//
//    std::vector<real_t> xData = {1.5, 1.0};
//    std::vector<real_t> yData = {1.5, -1.0};
//    DeviceVector<real_t> x(context, xData);
//    DeviceVector<real_t> y(context, yData);
//    DeviceMatrix<real_t> xMat(context, x);
//    DeviceMatrix<real_t> yMat(context, y);
//    DeviceTensor<real_t> RHS(context, nCols, 1, nMats);
//    RHS.pushBack(xMat);
//    RHS.pushBack(yMat);
//    LHS.leastSquares(RHS);
//
//    DeviceVector<real_t> s = xMat.asVector();
//    DeviceVector<real_t> s_slice(s, 0, 0);
//    std::cout << s_slice;
//
//    std::cout << RHS;
//
//    std::cout << A;
//    std::cout << x;
//    std::cout << y;

    return 0;
}
