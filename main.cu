#include "include/tensor.cuh"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

void xyz() {
    /* Write to binary file */
    DTensor<double> r = DTensor<double>::createRandomTensor(3, 6, 4, -1, 1).setStreamIdx(1);
    std::string fName = "abcd.bt"; // binary tensor file extension: .bt
    r.saveToFile(fName);

    /* Parse binary file */
    auto recov = DTensor<double>::parseFromFile(fName);

    std::cout << r;
    std::cout << recov;

    auto err = r - recov;
    std::cout << "max error : " << err.maxAbs() << std::endl;
    std::cout << "Memory: " << std::setprecision(3)
            << (float) Session::getInstance().totalAllocatedBytes() / 1e6
            << " MB" << std::endl;
}


int main() {
    Session::setStreams(5);
    xyz();
    std::cout << "Memory (outside): " << std::setprecision(3)
            << (float) Session::getInstance().totalAllocatedBytes() / 1e6
            << " MB" << std::endl;
    return 0;
}
