#include <tensor.cuh>
#include "src/data.cuh"


int main() {
    auto d_c = d_A * d_b;  // Matrix-vector multiplication on your device
    std::cout << d_c << std::endl;  /* Print result
                                * `Tensor [3 x 1 x 1]:
                                * >> layer: 0
                                * 39,
                                * 54,
                                * 69,`
                                * from your device. */
    return 0;
}
