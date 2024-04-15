#include <vector>
#include <iostream>
#include <cublas_v2.h>
#include "include/device.cuh"

#define real_t float


int main() {
    Context context;

    size_t kk = 6;
    std::vector<real_t> bbData{1.0, 2.0, 3.0,
                               6.0, 7.0, 8.0,
                               9.0, 10.0, 11.0,
                               12.0, 13.0, 14.0,
                               15.0, 16.0, 17.0,
                               18.0, 19.0, 20.0};
    DeviceMatrix<real_t> BB(context, kk, bbData, MatrixStorageMode::rowMajor);
    auto copiedRows = BB.getRows(1, 4);
    std::cout << copiedRows << "\n ---4";


    return 0;
}
