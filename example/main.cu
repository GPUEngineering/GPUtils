#include <device.cuh>
#include "src/data.cuh"


int main() {
    auto d_c = d_A * d_b;  // Matrix-vector multiplication on your device
    std::cout << d_c << "\n";  // Print result from your device

    return 0;
}
