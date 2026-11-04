// Test-only bridge: exercise the real C++ reader/writer from Python tests.
#include "tensor.cuh"
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

template<typename T>
void exchangeFile(const std::string &mode, const std::string &path, size_t rows, size_t cols, size_t mats) {
    std::vector<T> expected(rows * cols * mats);
    for (size_t k = 0; k < mats; ++k)
        for (size_t c = 0; c < cols; ++c)
            for (size_t r = 0; r < rows; ++r)
                expected[r + rows * (c + cols * k)] = T(r + 10 * c + 100 * k) + T(.25);
    if (mode == "write") {
        DTensor<T> tensor(expected, rows, cols, mats);
        tensor.saveToFile(path);
    } else if (mode == "verify") {
        auto tensor = DTensor<T>::parseFromFile(path);
        if (tensor.numRows() != rows || tensor.numCols() != cols || tensor.numMats() != mats)
            throw std::runtime_error("C++ reader returned an incorrect shape");
        std::vector<T> actual;
        tensor.download(actual);
        if (actual != expected) throw std::runtime_error("C++ reader returned incorrect element values");
    } else {
        throw std::invalid_argument("mode must be write or verify");
    }
}

int main(int argc, char **argv) {
    try {
        if (argc != 7) throw std::invalid_argument("usage: gputils_file_exchange write|verify float|double path rows cols mats");
        const size_t rows = std::stoull(argv[4]), cols = std::stoull(argv[5]), mats = std::stoull(argv[6]);
        // This executable serves tiny deterministic test fixtures only.
        if (!rows || !cols || !mats || rows > 16 || cols > 16 || mats > 16)
            throw std::invalid_argument("test fixture dimensions must be in [1,16]");
        if (std::string(argv[2]) == "float") exchangeFile<float>(argv[1], argv[3], rows, cols, mats);
        else if (std::string(argv[2]) == "double") exchangeFile<double>(argv[1], argv[3], rows, cols, mats);
        else throw std::invalid_argument("dtype must be float or double");
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
