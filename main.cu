#include "include/tensor.cuh"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>


int main() {
    auto z = DTensor<double>::parseFromTextFile("../test/data/my.dtensor");
    std::cout << z;
    return 0;
}
