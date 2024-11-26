#include "include/tensor.cuh"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

template<typename T>
struct data_t {
    size_t numRows;
    size_t numCols;
    size_t numMats;
    std::vector<T> data;
};

template<typename T>
data_t<T> vectorFromFile(std::string path_to_file) {
    data_t<T> dataStruct;
    std::ifstream file;
    file.open(path_to_file, std::ios::in);

    std::string line;
    getline(file, line); dataStruct.numRows = atoi(line.c_str());
    getline(file, line); dataStruct.numCols = atoi(line.c_str());
    getline(file, line); dataStruct.numMats = atoi(line.c_str());

    size_t numElements = dataStruct.numRows * dataStruct.numCols * dataStruct.numMats;
    std::vector<T> vecDataFromFile(numElements);

    size_t i = 0;
    while (getline(file, line)) {
        if constexpr (std::is_same_v<T, int>) {
            vecDataFromFile[i] = atoi(line.c_str());
        } else if constexpr (std::is_same_v<T, double>) {
            vecDataFromFile[i] = std::stod(line.c_str());
        } else if constexpr (std::is_same_v<T, float>) {
            vecDataFromFile[i] = std::stof(line.c_str());
        }
        if (i == numElements - 1) break;
        i++;
    }
    file.close();

    dataStruct.data = vecDataFromFile;
    return dataStruct;
}

int main() {
    auto z = vectorFromFile<double>("../test/data/my.dtensor");
    for (size_t i = 0; i < 3; i++) std::cout << z.data[i] << ", ";
    DTensor<double> dz(z.data, z.numRows, z.numCols, z.numMats);
    std::cout << "\n\n";

    auto q = vectorFromFile<int>("../test/data/my.dtensor");
    for (size_t i = 0; i < 3; i++) std::cout << q.data[i] << ", ";

    return 0;
}
