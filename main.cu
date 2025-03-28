#include "include/tensor.cuh"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

void xyz() {
    /* Write to binary file */
    auto r = DTensor<double>::createRandomTensor(3, 6, 4, -1, 1);
    auto r2 = DTensor<double>::createRandomTensor(300, 600, 4, -1, 1);
    std::string fName = "tensor.bt"; // binary tensor file extension: .bt

    /* Parse binary file */
    auto recov = DTensor<double>::parseFromFile(fName);
    auto err = r - recov;
    std::cout << "max error : " << err.maxAbs() << std::endl;
    std::cout << "Memory: " << std::setprecision(3)
            << (float) Session::getInstance().totalAllocatedBytes() / 1e6
            << " MB" << std::endl;
}


int main() {
    Session::getInstance(5);
    xyz();
    std::cout << "Memory (outside): " << std::setprecision(3)
            << (float) Session::getInstance().totalAllocatedBytes() / 1e6
            << " MB" << std::endl;
    return 0;
}
