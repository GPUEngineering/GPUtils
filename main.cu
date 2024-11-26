#include "include/tensor.cuh"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>


int main() {
    auto z = DTensor<size_t>::parseFromTextFile("../test/data/my.dtensor",
                                                StorageMode::rowMajor);
    std::cout << z;
    z.saveToFile("hohoho.dtensor");
    return 0;
}
