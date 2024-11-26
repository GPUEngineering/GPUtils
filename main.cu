#include "include/tensor.cuh"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>


int main() {
    auto z = DTensor<double>::parseFromTextFile("../test/data/my.dtensor",
                                                StorageMode::rowMajor);
    std::cout << z;
    std::cout << " -- ";
    return 0;
}
