#include "include/tensor.cuh"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>


int main() {
    /* Write to binary file */
    auto r = DTensor<double>::createRandomTensor(3, 6, 4, -1, 1);
    std::string fName = "tensor.bt"; // binary tensor file extension: .bt
    r.saveToFile(fName);

    /* Parse binary file */
    auto recov = DTensor<double>::parseFromFile(fName);
    auto err = r - recov;
    std::cout << "max error : " << err.maxAbs();

    return 0;
}
